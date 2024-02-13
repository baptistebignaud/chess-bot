from agent import ChessAgent, MCTS, MCTSArgs
from chess_env import chessEnv
import random
import chess
import numpy as np
from chess import Move
from tqdm import tqdm
from collections import deque
from random import shuffle
import sys

sys.setrecursionlimit(100000)


class TrainingParams:
    lr: float = 10e-3
    numIters: int = 100
    numEps: int = 20
    initial_epsilon: float = 1.0
    final_epsilon: float = 0.1
    discount_factor: float = 0.95
    tempThreshold: int = 500000
    numItersForTrainExamplesHistory: int = 20
    updateThreshold = 0.6  # During arena playoff new neural net will be accepted if threshold or more of games are won.
    maxlenOfQueue = 200000  # Number of game examples to train the neural networks.


class Training:
    def __init__(
        self,
        env,
        agent: ChessAgent,
        args: TrainingParams,
        mctsargs: MCTSArgs = MCTSArgs(),
    ):
        self.env = env
        self.agent = agent
        self.pagent = self.agent.__class__()
        self.args = args
        self.trainingHistory = []
        self.mctsargs = mctsargs
        self.mcts = MCTS(self.env, self.agent, self.mctsargs)

    def run_episode(self, fen=chess.STARTING_FEN):
        trainingExamples = []
        episodeStep = 0
        board = chess.Board(fen)

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)
            # Get probability of move with MCTS tree
            pi = self.mcts.getActionProb(board, temp)

            # [board, color, policy, value]
            board_input = chessEnv.get_model_input(board, self.agent.get_input_dim())
            # trainingExamples.append([board_input, board.turn, pi, None])
            trainingExamples.append([board_input, board.turn, pi, None])

            pi = chessEnv.filter_valid_moves(board, np.array(pi))
            move_ind = random.choices(range(len(pi)), weights=pi, k=1)[0]
            move_vect = np.zeros(len(pi))
            move_vect[move_ind] = 1
            move = chessEnv.make_move(board, move_vect, "greedy")

            # Update board
            board.push(Move.from_uci(move))

            # Get outcome of the move
            outcome = board.outcome()
            # print("outcome", outcome)
            r = chessEnv.get_reward(board.turn, outcome)

            if r is not None:
                out = []
                for x in trainingExamples:
                    board = x[0]
                    pi = x[2]
                    if r != -0.25:
                        r = r * (-1) ** (x[1] != board.turn)
                    out.append([board, pi, r])
                return out
                return [
                    [x[0], x[2], r * (-1) ** (x[1] != board.turn)]
                    for x in trainingExamples
                ]

    def learn(self):
        for i in range(1, self.args.numIters + 1):
            iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
            for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                self.mcts = MCTS(self.env, self.agent, self.mctsargs)
                iterationTrainExamples += self.run_episode()

            # Save iteration into training history
            self.trainingHistory.append(iterationTrainExamples)

            if len(self.trainingHistory) > self.args.numItersForTrainExamplesHistory:
                self.trainingHistory.pop(0)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainingHistory:
                trainExamples.extend(e)

            print(len(trainExamples))
            shuffle(trainExamples)

            pmcts = MCTS(self.env, self.pagent, self.mctsargs)

            self.agent.train(trainExamples)
            nmcts = MCTS(self.env, self.agent, self.mctsargs)


if __name__ == "__main__":
    env = chessEnv()
    agent = ChessAgent()
    args = TrainingParams()
    training = Training(env, agent, args)
    training.learn()
