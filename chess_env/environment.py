import chess
import torch
import numpy as np
from chess import Move, BB_EMPTY
import random
import sys
from typing import Literal


sys.path.append("../")
from model.backbone import TurboChessBot, ModelArgs


# TODO: Handle if a pawn is promoted


class chessEnv:
    def __init__(self, fen: str = chess.STARTING_FEN):
        self.fen = fen
        self.reset()
        self.input_dim = ModelArgs().input_dim

    @staticmethod
    def get_piece_id(piece_index):
        if piece_index <= 8:
            return 1
        elif piece_index <= 10:
            return 2
        elif piece_index <= 12:
            return 3
        elif piece_index <= 14:
            return 4
        elif piece_index == 15:
            return 5
        return 6

    def get_piece_column(self, piece_index):
        if piece_index <= 8 and self.board.turn:
            return piece_index

        if piece_index <= 8 and not self.board.turn:
            return 8 - piece_index + 1

        # Left knight
        elif (piece_index == 9 and self.board.turn) or (
            piece_index == 10 and not self.board.turn
        ):
            return 2

        # Right knight
        elif (piece_index == 10 and self.board.turn) or (
            piece_index == 9 and not self.board.turn
        ):
            return 7

        # Left rock
        elif (piece_index == 13 and self.board.turn) or (
            piece_index == 14 and not self.board.turn
        ):
            return 1

        # Right rock
        elif (piece_index == 14 and self.board.turn) or (
            piece_index == 13 and not self.board.turn
        ):
            return 8

        # Queen
        elif piece_index == 15:
            return 4

        # King
        elif piece_index == 16:
            return 5

    def reset(self):
        self.board = chess.Board(self.fen)

    def can_castle(self, color: bool) -> (bool, bool):
        """
        Returns a list of bool if the king cast castle king and queen side
        """
        castle = [None, None]
        if color:
            castle[0] = Move.from_uci("e1g1") in [
                elem for elem in self.board.generate_castling_moves()
            ]
            castle[1] = Move.from_uci("e1c1") in [
                elem for elem in self.board.generate_castling_moves()
            ]
        else:
            castle[0] = Move.from_uci("e8g8") in [
                elem for elem in self.board.generate_castling_moves()
            ]
            castle[1] = Move.from_uci("e8c8") in [
                elem for elem in self.board.generate_castling_moves()
            ]
        return castle

    def get_model_input(self):
        """
        Prepare the input for the DL model
        """

        model_input = np.zeros(self.input_dim)

        ###### Kings

        # White caslting
        indx = 1 + (self.input_dim - 1) // 2
        model_input[indx - 3 : indx - 1] = self.can_castle(color=chess.WHITE)

        # White king is checked (cf. https://python-chess.readthedocs.io/en/latest/_modules/chess.html#Board.is_checkmate) is_check function
        king = self.board.king(chess.WHITE)
        model_input[indx] = bool(
            BB_EMPTY
            if king is None
            else self.board.attackers_mask(not chess.WHITE, king)
        )

        # Black caslting
        model_input[-3:-1] = self.can_castle(color=chess.BLACK)
        # Black king is checked
        king = self.board.king(chess.BLACK)
        model_input[-1] = bool(
            BB_EMPTY
            if king is None
            else self.board.attackers_mask(not chess.BLACK, king)
        )

        return torch.Tensor(model_input)

    def find_closest_piece(self, piece_index: int):
        """
        Find closer pawn to the output of DL model
        Args:
            piece_index: 1:8 -> Pawn || 9:10 -> Knight || 11:12 -> Bishop || 13:14 -> Rock || 15 -> Queen || 16 -> King
        """
        # Get available pieces on the board
        pieces = np.array(
            self.board.pieces(
                chessEnv.get_piece_id(piece_index), self.board.turn
            ).tolist()
        )
        locations = np.where(pieces == 1)[0]
        locations = np.array([(1 + loc % 8, loc // 8 + 1) for loc in locations])

        # If no available piece
        if len(locations) == 0:
            return

        # Handle bishop (special case)
        if piece_index == 11 or piece_index == 12:
            if (piece_index == 11 and self.board.turn) or (
                piece_index == 12 and not self.board.turn
            ):
                color = "black"  # black bishop
            else:
                color = "white"  # white bishop

            locations_is_black = np.array(
                [(1 + 8 * (loc[1] - 1) + loc[0] - 1) % 2 for loc in locations]
            )

            if color == "black":
                if len(locations[np.where(locations_is_black == 1)[0]]) == 0:
                    return
                return locations[np.where(locations_is_black == 1)[0]][0]
            else:
                if len(locations[np.where(locations_is_black == 0)[0]]) == 0:
                    return
                return locations[np.where(locations_is_black == 0)[0]][0]

        # Other pieces
        else:
            dist = np.abs(locations[:, 0] - self.get_piece_column(piece_index))

        location = locations[np.argmin(dist)]
        return location

    def make_move(
        self,
        output: np.array,
        sampling_stategy: Literal["greedy", "sampling"] = "sampling",
    ) -> bool:
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
        # print(self.board.has_kingside_castling_rights(self.board.turn))
        # print(self.board.has_queenside_castling_rights(self.board.turn))
        if len(output.shape) > 1:
            output = output.squeeze()

        probs = output[: len(output) // 3]
        move = output[len(output) // 3 :]

        # Get piece to move
        if sampling_stategy == "sampling":
            piece_index = random.choices(np.arange(len(probs)), probs, k=1)[0]
        else:
            piece_index = probs.argmax()

        # Get move to do
        move = move[2 * piece_index : 2 * piece_index + 2].round()

        # Find closest piece to do the move
        piece = self.find_closest_piece(piece_index + 1)

        if piece is None:
            return False

        # White to move
        if self.board.turn:
            move = (
                chr(int(65 + piece[0] - 1)).lower()
                + str(int(piece[1]))
                + chr(int(65 + piece[0] + move[0] - 1)).lower()
                + str(int(piece[1] + move[1]))
            )

        # Black to move
        else:
            move = (
                chr(int(65 + piece[0] - 1)).lower()
                + str(int(piece[1]))
                + chr(int(65 + piece[0] - move[0]) - 1).lower()
                + str(int(piece[1] - move[1]))
            )
        print("move", move)

        try:
            move = Move.from_uci(move)
        except chess.InvalidMoveError:
            return False
        if move in self.board.generate_legal_moves():
            self.board.push(move)
            return True

        return False


if __name__ == "__main__":
    environment = chessEnv()
    move = torch.Tensor([0.0] * 2 + [0, 0.56] + [0.56] * 2 * 14)
    move = [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Probabilities
    move += [0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Pawns
    move += [0, 0, 0, 0]  # Knights
    move += [0, 0, 0, 0]  # Bishops
    move += [0, 0, 0, 0]  # Rocks
    move += [0, 0]  # Queen
    move += [0, 0]  # King
    move = torch.Tensor(move)
    # print(environment.board)
    args = ModelArgs()
    input = torch.ones([1, args.dim])

    input = torch.ones([1, args.input_dim])
    bot = TurboChessBot(args)

    move = bot(input).detach()

    # print(move)

    print(environment.make_move(move.numpy()), "\n\n")
    print(environment.make_move(move.numpy()))
    print(environment.get_model_input())

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
