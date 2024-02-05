import chess
import torch
import numpy as np
from chess import Move
from torch.linalg import vector_norm

import sys

sys.path.append("../")
from model.backbone import TurboChessBot, ModelArgs


# TODO: Handle if a pawn is promoted


class chessEnv:
    def __init__(self, fen: str = chess.STARTING_FEN):
        self.fen = fen
        self.reset()

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

    def make_move(self, move: torch.Tensor):
        if len(move.shape) > 1:
            move = move.squeeze()
        dst = np.array(
            [
                vector_norm(move[2 * indx : 2 * indx + 2]).item()
                for indx in np.arange(0, len(move) // 2, 1)
            ]
        )
        piece_index = dst.argmax().item()
        # piece_index = move[2 * np.arange(0, len(move) // 2, 1)].argmax().item()
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
                + chr(int(65 + piece[0] + move[0].item() - 1)).lower()
                + str(int(piece[1] + move[1].item()))
            )

        # Black to move
        else:
            move = (
                chr(int(65 + piece[0] - 1)).lower()
                + str(int(piece[1]))
                + chr(int(65 + piece[0] - move[0].item()) - 1).lower()
                + str(int(piece[1] - move[1].item()))
            )

        try:
            move = Move.from_uci(move)

        except chess.InvalidMoveError:
            return False
        print(move)
        if move in self.board.generate_legal_moves():
            self.board.push(move)
            return True

        return False


if __name__ == "__main__":
    environment = chessEnv()
    move = torch.Tensor([0.0] * 2 + [0, 0.56] + [0.56] * 2 * 14)

    inpt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Pawns
    inpt += [0, 0, 0, 0]  # Knights
    inpt += [0, 0, 0, 0]  # Bishops
    inpt += [0, 2, 0, 0]  # Rocks
    inpt += [0, 0]  # Queen
    inpt += [0, 0]  # King
    move = torch.Tensor(inpt)
    # print(environment.board)
    args = ModelArgs()
    input = torch.ones([1, args.dim])

    input = torch.ones([1, args.input_dim])
    bot = TurboChessBot(args)
    # move = bot(input)
    print(environment.make_move(move), "\n\n")
    print(environment.make_move(move))
