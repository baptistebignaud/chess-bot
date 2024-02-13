from torch.nn import Linear
import torch
from torch import nn
from torch.nn import Sequential
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import sys
from dataclasses import dataclass
from typing import Tuple

sys.path.append("../")
from model.backbone import ModelArgs, TurboChessBot
from utils import AverageMeter


@dataclass
class TrainingParams:
    buffer_size = int(1e5)  # Replay buffer size
    batch_size = 16  # Minibatch size
    gamma = 0.99  # Discount factor
    tau = 0.005  # Soft update of target parameters
    lr_actor = 3e-4  # Learning rate of the actor
    lr_critic = 3e-4  # Learning rate of the critic
    weight_decay = 0  # L2 weight decay
    alpha = 0.0025  # Entropy weight parameter
    epochs: int = 20
    device: torch.device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )


class ActorCritic(nn.Module):
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
        self.model_args = model_args

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, position: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actions = self.actor.forward(position)
        values = self.critic.forward(actions)
        values = 2 * (nn.Sigmoid()(values) - 1 / 2)
        actions_norm = nn.Softmax(dim=1)(actions)
        return actions_norm, values


# For the training cf. https://github.com/suragnair/alpha-zero-general/blob/master/othello/pytorch/NNet.py
class ChessAgent:
    def __init__(
        self,
        model_args: ModelArgs = ModelArgs(),
        critic_layers: list[int] = [1024, 256],
        training_params: TrainingParams = TrainingParams(),
    ):
        self.agent = ActorCritic(model_args, critic_layers).to(model_args.device)
        self.training_params = training_params
        self.model_args = model_args

    def forward(self, intput: torch.Tensor) -> torch.Tensor:
        return self.agent.forward(intput)

    def get_input_dim(self):
        return self.agent.model_args.input_dim

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.agent.parameters())

        for epoch in range(self.training_params.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.agent.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.training_params.batch_size)

            t = tqdm(range(batch_count), desc="Training Net")
            for _ in t:
                sample_ids = np.random.randint(
                    len(examples), size=self.training_params.batch_size
                )
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.IntTensor(np.array(boards).astype(np.float64))
                boards = boards.squeeze()

                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                boards, target_pis, target_vs = (
                    boards.to(self.training_params.device),
                    target_pis.to(self.training_params.device),
                    target_vs.to(self.training_params.device),
                )

                # compute output
                out_pi, out_v = self.agent(boards)
                print("out_pi: ", out_pi.argmax(dim=1))
                print("target_pi: ", target_pis, "\n")
                print("out_v: ", out_v)
                print("target_v: ", target_vs, "\n\n")

                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def loss_pi(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        # print(targets - outputs.view(-1))
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = torch.IntTensor(board)
        board = board.to(self.training_params.device)
        self.agent.eval()
        with torch.no_grad():
            pi, v = self.agent(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(
            {
                "state_dict": self.agent.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = self.training_params.device
        checkpoint = torch.load(filepath, map_location=map_location)
        self.agent.load_state_dict(checkpoint["state_dict"])


if __name__ == "__main__":
    agent = ChessAgent()
    n_elements = 1500
    boards = [np.ones((8, 8, 5))] * n_elements
    pis = [np.ones(4096)] * n_elements
    vs = [np.ones(1)] * n_elements
    examples = [[b, p, v] for b, p, v in zip(boards, pis, vs)]
    # print(examples)
    agent.train(examples)
