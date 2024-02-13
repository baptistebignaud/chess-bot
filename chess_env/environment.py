import chess
import torch
import numpy as np
from chess import Move, Outcome
import random
import sys
from typing import Literal
from gymnasium import Env
from typing import Optional
from chess import Termination
from chessboard import display
from typing import Tuple

sys.path.append("../")
from model.backbone import TurboChessBot, ModelArgs

chess_mapping = {"p": 2, "n": 3, "b": 4, "r": 5, "q": 6, "k": 7}


def softmax(x, temperature: float = 1.0):
    """Compute softmax values for each sets of scores in x."""
    # Normalize input scores
    x_normalized = x / temperature - np.max(x / temperature, axis=-1, keepdims=True)
    # Calculate softmax probabilities
    e_x = np.exp(x_normalized)
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class chessEnv(Env):
    def __init__(self, fen: str = chess.STARTING_FEN):
        self.fen = fen
        self.reset()
        self.input_dim = ModelArgs().input_dim
        self.mapping = {"p": 2, "n": 3, "b": 4, "r": 5, "q": 6, "k": 7}
        self.action_size = int(64 * 64)

    def reset(self):
        self.board = chess.Board(self.fen)

    @staticmethod
    def can_castle(board: chess.Board, color: bool) -> Tuple[bool, bool]:
        """
        Returns a list of bool if the king cast castle king and queen side
        """
        castle = [None, None]
        if color:
            castle[0] = Move.from_uci("e1g1") in [
                elem for elem in board.generate_castling_moves()
            ]
            castle[1] = Move.from_uci("e1c1") in [
                elem for elem in board.generate_castling_moves()
            ]
        else:
            castle[0] = Move.from_uci("e8g8") in [
                elem for elem in board.generate_castling_moves()
            ]
            castle[1] = Move.from_uci("e8c8") in [
                elem for elem in board.generate_castling_moves()
            ]
        return castle

    @staticmethod
    def get_model_input(
        board: chess.Board,
        input_dim: int = ModelArgs().input_dim,
    ) -> torch.IntTensor:
        """
        Prepare the input for the DL model
        """
        # If one example is given, create a batch of (1, 8, 8, 5)
        if len(input_dim) < 4:
            input_dim = [1] + list(input_dim)
        # print(input_dim)
        model_input = np.zeros(input_dim)
        rg = range(8)
        # [piece_id, isyourpiece, isattacked, castleK, castleQ]

        # TODO : Adapt this to have geometrical symetry to the output for black and white
        # # If white
        # if turn == chess.WHITE:
        #     rg = range(8)
        # else:
        #     rg = range(7, -1, 1)

        for rank in rg:
            for nb in rg:
                castles = [0, 0]
                square = rank * 8 + nb
                piece = board.piece_at(square)
                if piece is None:
                    model_input[0, nb, rank, :] = [1, 0, 0] + castles
                    continue

                iswhite = piece.color
                symbol = piece.symbol().lower()
                piece_id = chess_mapping[symbol]

                if piece_id == 7:
                    castles = chessEnv.can_castle(board=board, color=iswhite)

                if board.turn == piece.color:
                    isyourpiece = 1
                else:
                    isyourpiece = -1
                isattacked = board.is_attacked_by((not board.turn), square)

                model_input[0, nb, rank, :] = [
                    piece_id,
                    isyourpiece,
                    isattacked,
                ] + castles

        return torch.IntTensor(model_input)

    @staticmethod
    def get_reward(
        color: chess.WHITE | chess.BLACK,
        outcome: Optional[Outcome] = None,
    ):
        # Outcome of the game
        if outcome is not None:
            winner = outcome.winner
            # Draw (penalize a bit draw)
            if winner is None:
                return -0.25
            if color != winner:
                return -1
            return 1
        # else:
        #     board_fen = board.board_fen()
        #     p, P = board_fen.count("p"), board_fen.count("P")
        #     n, N = board_fen.count("n"), board_fen.count("N")
        #     b, B = board_fen.count("b"), board_fen.count("B")
        #     r, R = board_fen.count("r"), board_fen.count("R")
        #     q, Q = board_fen.count("q"), board_fen.count("Q")
        #     reward_material = P - p + (N + B - b - n) * 3 + (R - r) * 5 + (Q - q) * 10
        #     if color == chess.BLACK:
        #         reward_material *= -1

        #     reward = reward_material
        #     return reward

    def step(self, action: Move) -> Optional[Outcome]:
        self.board.push(action)
        outcome = self.board.outcome()
        # if outcome is not None:
        #     self.reset()
        return outcome

    def getActionSize(self):
        return self.action_size

    @staticmethod
    def get_id_from_position(position: Tuple[str, str]):
        incr_to_num = ord("a")
        id_origin = 8 * (ord(position[0][0]) - incr_to_num) + int(position[0][1]) - 1
        id_origin *= 64
        id_final = 8 * (ord(position[1][0]) - incr_to_num) + int(position[1][1]) - 1
        return id_origin + id_final

    @staticmethod
    def get_legal_moves(board: chess.Board):
        legal_moves = [str(elem) for elem in board.generate_legal_moves()]
        legal_moves = [[move[:2], move[2:]] for move in legal_moves]
        legal_moves = [chessEnv.get_id_from_position(move) for move in legal_moves]
        return legal_moves

    @staticmethod
    def legal_moves_to_inputs(legal_moves: list[int], action_size: int) -> np.array:
        vect = np.zeros(action_size)
        for mov in legal_moves:
            vect[mov] = 1
        return vect

    @staticmethod
    def filter_valid_moves(board: chess.Board, output: torch.Tensor):
        if len(output.shape) >= 1:
            output = output.squeeze()
        legal_moves = chessEnv.get_legal_moves(board)

        mask = np.zeros(output.shape)
        mask[legal_moves] = 1
        output = output * mask

        # No legal moves provided
        if (output.sum() == 0) or (len(np.where(output > 0)[0]) <= 1):
            output = mask
            output /= output.sum()

        else:
            # Rescale
            output[legal_moves] = output[legal_moves] / output[legal_moves].sum()

        if output.sum() == 0:
            output = mask
            output /= output.sum()

        return output

    @staticmethod
    def get_position_from_id(id: int):
        initial_pos = id // 64
        future_pos = id % 64

        rank_i = chr(65 + initial_pos // 8).lower()
        nb_i = str(1 + initial_pos % 8)

        rank_f = chr(65 + future_pos // 8).lower()
        nb_f = str(1 + future_pos % 8)

        move = rank_i + nb_i + rank_f + nb_f
        return move

    @staticmethod
    def make_move(
        board: chess.Board,
        output: np.array,
        sampling_stategy: Literal["greedy", "sampling"] = "sampling",
    ) -> chess.Board:
        """
        Make move having the output of the DL model
        It samples the piece to move, then get the movement to do. If the move is legit, it does it and returns True
        does nothing and returns False otherwise

        Args:
            output: The output of the DL model
            sampling_stategy: Which strategy to pick the move
        Returns:
            bool
        """
        if len(output.shape) > 1:
            output = output.squeeze()

        # For black reverse the outcome so that it is invariant
        if board.turn == chess.BLACK:
            output = output[::-1]
        output = chessEnv.filter_valid_moves(board, output)

        # Get piece to move
        if sampling_stategy == "sampling":
            piece_index = random.choices(np.arange(len(output)), output, k=1)[0]
        else:
            piece_index = output.argmax()

        move = chessEnv.get_position_from_id(piece_index)
        return move

    def play_game(self, playerW=None, playerB=None):
        valid_fen = "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"

        game_board = display.start()

        while True:
            display.check_for_quit()
            display.update(valid_fen, game_board)

            # board flip interface
            if not game_board.flipped:
                display.flip(game_board)


if __name__ == "__main__":
    environment = chessEnv()
    # print(environment.board)
    args = ModelArgs()
    dims = [1] + list(args.input_dim)
    input = torch.ones(*dims, dtype=torch.int32)

    bot = TurboChessBot(args)

    # environment.make_move(move)
    # print(move.shape)
    # print(move)
    print(environment.play_game())
    exit()

    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("d4")
    board.push_san("Na6")
    board.push_san("d5")
    board.push_san("c5")
    board.push_san("d5c6")
    board.push_san("Ba3")
    board.push_san("Ba6")
    board.push_san("Nh6")
    board.push_san("b3")
    board.push_san("b6")
    board.push_san("h3")
    board.push_san("Bb7")
    board.push_san("h4")
    board.push_san("Qc7")
    board.push_san("h5")

    # board.push_san("e8g8")
    print("\n\n\n")
    print([elem for elem in board.generate_castling_moves()])
    # print(
    #     move
    #     for move in board.generate_castling_moves()
    #     if board.is_kingside_castling(move)
    # )
    # print("O-O" in board.generate_legal_moves())
    # board.push_san("O-O")
    print(board)
