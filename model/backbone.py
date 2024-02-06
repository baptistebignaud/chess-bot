import torch
from torch import nn
from collections import OrderedDict
from torch.nn import Linear, MultiheadAttention
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    # turn + 2 * (Pawns + (knight, bishop, rock) + queen + king)
    input_dim = 1 + 2 * (8 * 5 + 3 * 2 * 2 + 2 + 5)
    dim: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32


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


class GPTBlock(nn.Module):
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
        self.output_size = 3 * 16
        self.l_layers["embedding"] = Linear(self.input_size, args.dim)
        for i in range(args.n_layers):
            self.l_layers[f"gpt layer {i}"] = GPTBlock(args)

        self.l_layers["last_layer"] = Linear(args.dim, self.output_size)
        self.seq = nn.Sequential(self.l_layers)

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        output = self.seq(position)

        # Probabilities of which piece to move (batch_size, 16)
        probs = output[:, : self.output_size // 3]
        probs = nn.Softmax(dim=1)(probs)

        # Move to do for the ith piece (batch_size, 2 * 16)
        move = output[:, self.output_size // 3 + 1 :]
        move = nn.Sigmoid()(move)
        move = 14 * (
            move - 1 / 2
        )  # Renormalize in [-7,7]^2 that is the amplitude in chessboard

        output = torch.cat([probs, move], dim=1)
        return output


if __name__ == "__main__":
    args = ModelArgs()
    input = torch.ones([500, args.dim])
    llama = GPTBlock(args)

    input = torch.ones([500, args.input_dim])
    bot = TurboChessBot(args)
    print(bot(input))
    print(bot(input).shape)
