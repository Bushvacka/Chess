from torch.utils.data import Dataset, DataLoader

from chess import (
    Board,
    Move,
    pgn,
    square_file,
    square_rank,
    square_mirror,
    WHITE,
    BLACK,
)
import chess
import numpy as np
import logging
import pickle

DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
ACTION_SIZE = 4672
"""
A move in chess may be described in two parts: selecting the piece to move, and then
selecting among the legal moves for that piece. We represent the policy π(a|s) by a 8 × 8 × 73
stack of planes encoding a probability distribution over 4,672 possible moves. Each of the 8×8
positions identifies the square from which to “pick up” a piece. The first 56 planes encode
possible ‘queen moves’ for any piece: a number of squares [1..7] in which the piece will be
moved, along one of eight relative compass directions {N, NE, E, SE, S, SW, W, NW}. The
next 8 planes encode possible knight moves for that piece. The final 9 planes encode possible
underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop or
rook respectively. Other pawn moves or captures from the seventh rank are promoted to a
queen.
"""


def move_to_action(move: Move, turn: bool) -> int:
    from_square = move.from_square if turn == WHITE else square_mirror(move.from_square)
    to_square = move.to_square if turn == WHITE else square_mirror(move.to_square)

    from_file, from_rank = square_file(from_square), square_rank(from_square)
    to_file, to_rank = square_file(to_square), square_rank(to_square)

    delta_file, delta_rank = to_file - from_file, to_rank - from_rank

    # Underpromotions (64 - 72)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        plane = 56 + 8 + 3 * (delta_file + 1) + move.promotion - 2

    # Knight moves (56 - 63)
    elif (delta_file, delta_rank) in KNIGHT_MOVES:
        plane = 56 + KNIGHT_MOVES.index((delta_file, delta_rank))

    # Queen Moves (0 - 55)
    else:
        direction = DIRECTIONS.index((np.sign(delta_file), np.sign(delta_rank)))
        distance = max(abs(delta_file), abs(delta_rank))

        plane = (7 * direction) + (distance - 1)

    return from_square * 64 + plane


def action_to_move(action: int, turn: bool) -> Move:
    from_square = action // 64
    from_file, from_rank = square_file(from_square), square_file(from_square)

    plane = action % 64

    # Queen Moves
    if plane < 56:
        pass
    # Knight Moves
    elif plane < 64:
        pass
    # Underpromotions:
    else:
        plane -= 64

        delta_file = (plane // 3) - 1

        to_file, to_rank = from_file + delta_file, from_rank + 1
        to_square = to_file + to_rank * 8

        promotion = (plane % 3) + 2

    return Move(from_square, to_square, promotion)


def get_canonical_form(board: Board) -> Board:
    return board if board.turn == WHITE else board.mirror()


class ChessDataset(Dataset):
    def __init__(self) -> None:
        self.boards: list[Board] = []
        self.policies: list[np.ndarray] = []
        self.evaluations: list[float] = []

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, idx: int) -> tuple[Board, np.ndarray, float]:
        return self.boards[idx], self.policies[idx], self.evaluations[idx]

    def add(self, board: Board, policy: np.ndarray, evaluation: float) -> None:
        self.boards.append(board)
        self.policies.append(policy)
        self.evaluations.append(evaluation)

    def save_to_file(self, file_name: str) -> None:
        with open(file_name, "wb+") as f:
            pickle.dump((self.boards, self.policies, self.evaluations), f)

    def load_from_file(self, file_name: str) -> None:
        with open(file_name, "rb+") as f:
            self.boards, self.policies, self.evaluations = pickle.load(f)

    def load_from_pgn(self, file_name: str) -> None:
        logging.info(f"Loading games from {file_name}...")
        with open(file_name) as f:
            # Iterate through all of the games in the file
            while (game := pgn.read_game(f)) is not None:
                board = game.board()
                result = game.headers.get("Result")

                # Evaluation from the POV of white
                if result == "1-0":
                    evaluation = 1.0
                elif result == "0-1":
                    evaluation = -1.0
                else:
                    evaluation = 0.0

                # Imitate the player's policy for each mainline move
                for move in game.mainline_moves():
                    action = move_to_action(move, board.turn)
                    policy = np.zeros(ACTION_SIZE)
                    policy[action] = 1.0

                    canonical_board = get_canonical_form(board)

                    # Add the position to the dataset
                    self.add(canonical_board, policy, evaluation)

                    # Flip the evaluation for the next player
                    evaluation *= -1

                    # Move to the next position
                    board = board.push(move)
