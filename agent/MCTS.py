from chess_env import chessEnv, softmax
import numpy as np
from dataclasses import dataclass
from model.backbone import TurboChessBotActorCritic
import math
import chess
from chess import Move


@dataclass
class MCTSArgs:
    EPS = 10e-8
    numIters = 1000
    numEps = (
        50  # Number of complete self-play games to simulate during a new iteration.
    )
    tempThreshold = 150  #
    numMCTSSims = 25  # Number of games moves for MCTS to simulate.
    cpuct = 1


# cf. https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py
class MCTS:
    def __init__(
        self,
        env: chessEnv,
        agent: TurboChessBotActorCritic,
        args: MCTSArgs = MCTSArgs(),
    ):
        self.env = env
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        self.args = args
        self.agent = agent

    def getActionProb(self, board: chess.Board, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        board.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        s = f"{board.board_fen()}//{board.outcome()}"
        # s = board.board_fen()
        for _ in range(self.args.numMCTSSims):
            self.search(board.copy())
        counts = [
            self.Nsa[(s, chessEnv.get_position_from_id(a))]
            if (s, chessEnv.get_position_from_id(a)) in self.Nsa
            else 0
            for a in range(self.env.getActionSize())
        ]
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        # if counts.sum() == 0:
        #     pass
        probs = softmax(np.array(counts)).tolist()
        # print("probs", probs)
        return probs

    def search(self, board: chess.Board):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current board
        """
        s = f"{board.board_fen()}{board.outcome()}"
        # s = board.board_fen()
        # print(s, "\n\n")
        # If too many moves
        if s not in self.Es:
            outcome = board.outcome()
            if outcome is not None:
                if outcome.winner is None:
                    outcome = 0
                elif outcome == board.turn:
                    outcome = 1
                else:
                    outcome = -1

            self.Es[s] = outcome
        # print(self.Es[s], board.outcome(), s)
        if self.Es[s] is not None:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node
            input = chessEnv.get_model_input(board)
            actions, v = self.agent.forward(input)
            actions = actions.cpu().detach().numpy()
            actions = chessEnv.filter_valid_moves(board, actions)
            self.Ps[s] = {}
            for a in range(len(actions)):
                self.Ps[s][chessEnv.get_position_from_id(a)] = actions[a]

            legal_moves = chessEnv.get_legal_moves(board)
            valids = chessEnv.legal_moves_to_inputs(
                legal_moves, self.env.getActionSize()
            )
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v
        valids = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1
        # pick the action with the highest upper confidence bound

        # print((self.Ps[s] != 0).sum())
        # print(self.Ps[s].max())
        # print((self.Ps[s] != 0).sum())
        # print("\n\n")
        for ind in range(self.env.getActionSize()):
            a = chessEnv.get_position_from_id(ind)
            if valids[ind] != 0:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(
                        self.Ns[s]
                    ) / (1 + self.Nsa[(s, a)])
                    u = u.item()
                else:
                    u = (
                        self.args.cpuct
                        * self.Ps[s][a]
                        * math.sqrt(self.Ns[s] + self.args.EPS)
                    )  # Q = 0 ?
                    u = u.item()
                # print("cur best", cur_best, best_act)
                if u > cur_best:
                    cur_best = u
                    best_act = a

        # a_id = best_act
        a = best_act
        # a = chessEnv.get_position_from_id(a_id)
        # print(board)
        # print("\n\n")

        # print(a)

        # Next state
        board.push(Move.from_uci(a))
        v = self.search(board)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
