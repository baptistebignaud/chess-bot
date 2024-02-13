import torch
from torch import nn
from collections import OrderedDict
from torch.nn import Linear, MultiheadAttention, Embedding
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
from torch.nn import Sequential


@dataclass
class ModelArgs:
    # (row, column, [piece_id, isyourpiece, isattacked, castleK, castleQ])
    input_dim = (8, 8, 5)
    embed_dim = 8 * 16 - input_dim[-1] + 1
    vocab_size: int = 8
    n_layers: int = 4
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    device: torch.device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
        # rsqrt: 1 / sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * (
            (hidden_dim + args.multiple_of - 1) // args.multiple_of
        )

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.w2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mqa = MultiheadAttention(args.dim, args.n_heads)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.feed_forward = FeedForward(args)
        self.feed_forward_norm = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x: torch.Tensor):
        out = self.attention_norm(x)
        out = x + self.mqa(x, x, x)[0]
        out = out + self.feed_forward(self.feed_forward_norm(out))
        return out


class TurboChessBot(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.l_layers = OrderedDict()
        self.input_size = args.input_dim
        self.board_size = 8
        self.output_size = 64
        self.device = args.device

        # args.dim = np.prod(args.input_dim[:-1]) * (args.embed_dim + 5)
        args.dim = args.embed_dim + args.input_dim[-1] - 1

        self.embedding = Embedding(args.vocab_size, args.embed_dim).to(self.device)

        # self.l_layers["embedding"] = Linear(self.input_size, args.dim)
        for i in range(args.n_layers):
            self.l_layers[f"gpt layer {i}"] = TransformerBlock(args).to(args.device)

        self.l_layers["last_layer"] = Linear(args.dim, self.output_size).to(self.device)
        self.seq = nn.Sequential(self.l_layers).to(self.device)

    def forward(
        self, position: torch.Tensor, return_probs: bool = False
    ) -> torch.Tensor:
        # Position (batch_size, 8, 8, 5)
        position = position.to(self.device)
        # Embedding (batch_size, 8, 8, 1) -> (batch_size, 8, 8, embed_dim)
        piece_embed = self.embedding(position[:, :, :, 0])

        # Position (batch_size, 8, 8, 5) -> (batch_size, 8, 8, embed_dim+4)
        position = torch.cat([piece_embed, position[:, :, :, 1:]], dim=-1)

        # Position (batch_size, 8, 8, embed_dim+4) -> (batch_size, 8*8, embed_dim+4)
        position = position.view(position.shape[0], self.board_size**2, -1)

        # output (batch_size, 8, 8, embed_dim+4) -> (batch_size, 64, 64)
        output = self.seq(position)

        # output (batch_size, 64, 64) -> (batch_size, 4096)
        output = output.view(output.shape[0], -1)

        if return_probs:
            # Apply softmax to get the probability of a movement
            output = nn.Softmax(dim=-1)(output)

        return output


class TurboChessBotActorCritic(nn.Module):
    def __init__(
        self,
        model_args: ModelArgs = ModelArgs(),
        critic_layers: list[int] = [1024, 256],
    ):
        super().__init__()

        self.actor = TurboChessBot(model_args)
        dim = 64 * 64
        seq = []
        for layer in critic_layers:
            seq.append(Linear(dim, layer))
            dim = layer
        seq.append(Linear(dim, 1))
        self.critic = Sequential(*seq).to(model_args.device)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, position: torch.Tensor):
        actions = self.actor.forward(position)
        values = self.critic.forward(actions)
        values = 2 * (nn.Sigmoid()(values) - 1 / 2)
        return actions, values


if __name__ == "__main__":
    args = ModelArgs()
    dims = [1] + list(args.input_dim)
    input = torch.ones(*dims, dtype=torch.int32)
    bot = TurboChessBot(args)
    # print(bot(input))
    output = bot(input)
    # print(output)
    # print(output.shape)
