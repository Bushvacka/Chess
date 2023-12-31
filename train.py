from torch.utils.data import Dataset, DataLoader

from chess import Board, pgn, WHITE, BLACK

import numpy as np

import pickle

ACTION_SIZE = 4672


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
