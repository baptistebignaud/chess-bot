# chess-bot

## Input and output

We define as input the concatenation of pieces positions on the board as follow defining pieces from left to right, from bottom to top, if a piece is taken it will be (0,0)

input = [turn] + [pawn_w_i, knight_w_i, bishop_w_i, rock_w_i, queen_w, king_w]+[pawn_b_i, knight_b_i, bishop_b_i, rock_b_i, queen_b, king_b]

For the input, we will define:\
pawn_w_i = (x,y, kn, bsh, qu)\
with x, y being respectively the position on the x and y axis on the board of ith pawn of white\
kn being a boolean if the ith pawn has been promoted to knight\
bsh being a boolean if the ith pawn has been promoted to bishop\
qu being a boolean if the ith pawn has been promoted to queen\
(same for black with pawn_b_i)

Then for knight, bishop, rock and queen the can be define as:\
pos = (x,y)

For the king, the input will be defined as triplet:\
(x,y, castK, castQ, ischeck) being the position concatenated with the booleans stating if the king can castle king and queen side, concatenated if the king is checked.

For simplicity and symetry, we will define the output as:\
[probs]+[pawn_1_M, pawn_2_M, pawn_3_M, pawn_4_M, pawn_5_M, pawn_6_M, pawn_7_M, pawn_8_M, kniwght_1_M, kniwght_2_M, bishop_1_M, bishop_2_M, rock_1_M, rock_2_M, queen_M, king_M]

With probs being the probability of each piece to move.

With for each value being the move to do. Note that each piece are numbered increasly from the left (white and black). In a given position, since we do not know where pieces come from, we will do the assumption that the piece is the one closer from original position; except for bishop in which we can define it (light or dark squares bishop)

We will define castling as a movement of the king\
King-side castling will be [...,2,0]
Queen-side castling will be [...,-2,0]

For a given position, one can not differentiate for knights, rocks and pawns where they are coming from.
Thus, we will say that for knights, we will define as input the knight that is the closer to the position of the initial position (e.g for white if nb5 we will say that it is coming from the knight originally in b1)
Same for rocks.
