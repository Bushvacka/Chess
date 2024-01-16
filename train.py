import logging
import os
from concurrent.futures import Future, ProcessPoolExecutor

import h5py
import numpy as np
import torch
from chess import Board, pgn
from torch.utils.data import Dataset

from mcts import run_mcts
from model import ResNet
from utils import (
    ACTION_SIZE,
    action_to_move,
    get_canonical_form,
    get_model_form,
    move_to_action,
)


class ChessDataset(Dataset):
    def __init__(self) -> None:
        self.boards: list[torch.Tensor] = []
        self.policies: list[np.ndarray] = []
        self.evaluations: list[float] = []

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.ndarray, float]:
        return (
            self.boards[idx],
            self.policies[idx],
            self.evaluations[idx],
        )

    def add(self, board: torch.Tensor, policy: np.ndarray, evaluation: float) -> None:
        self.boards.append(board)
        self.policies.append(policy)
        self.evaluations.append(evaluation)

    def extend(
        self, training_examples: list[tuple[torch.Tensor, np.ndarray, float]]
    ) -> None:
        for board, policy, evaluation in training_examples:
            self.add(board, policy, evaluation)

    def trim(self, size: int) -> None:
        self.boards = self.boards[-size:]
        self.policies = self.policies[-size:]
        self.evaluations = self.evaluations[-size:]

    def save(self, path: str) -> None:
        logging.info(f"Saving {len(self.boards)} examples to {path}")
        with h5py.File(path, "w") as f:
            f.create_dataset("boards", data=[board.numpy() for board in self.boards])
            f.create_dataset("policies", data=self.policies)
            f.create_dataset("evaluations", data=self.evaluations)

    def load(self, path: str) -> None:
        logging.info(f"Loading examples from {path}")
        with h5py.File(path, "r") as f:
            self.boards = [torch.from_numpy(board) for board in f["boards"][:]]
            self.policies = f["policies"][:]
            self.evaluations = f["evaluations"][:]

    def load_pgn_directory(self, directory: str) -> None:
        for file_name in os.listdir(directory):
            if file_name.endswith(".pgn"):
                self.load_pgn(os.path.join(directory, file_name))

    def load_pgn(self, path: str) -> int:
        logging.info(f"Loading games from {path}")

        games_read = 0

        with open(path) as f:
            # Iterate through all of the games in the file
            while (game := pgn.read_game(f)) is not None:
                games_read += 1

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
                    action = move_to_action(move, board)
                    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
                    policy[action] = 1.0

                    canonical_board = get_canonical_form(board)

                    # Add the position to the dataset
                    self.add(get_model_form(canonical_board), policy, evaluation)

                    # Flip the evaluation for the next player
                    evaluation *= -1

                    # Move to the next position
                    board.push(move)

        logging.info(f"Loaded {games_read} games")
        return games_read


import time


def self_play(
    model_path: str, num_games: int = 1
) -> list[tuple[torch.Tensor, np.ndarray, float]]:
    # Initialize and load the model
    model = ResNet(device="cpu")
    model.load(model_path)

    training_examples = []  # (board, policy, value)

    for i in range(num_games):
        logging.info(f"Playing game {i + 1}/{num_games}")

        # Stores (board, policy)
        temp_examples = []

        board = Board()

        # Play the game till termination
        while not board.is_game_over():
            action, policy = run_mcts(model, board)

            temp_examples.append((get_model_form(get_canonical_form(board)), policy))

            move = action_to_move(action, board)

            board.push(move)

        result = board.result()

        # Evaluation from the POV of white
        if result == "1-0":
            evaluation = 1.0
        elif result == "0-1":
            evaluation = -1.0
        else:
            evaluation = 0.0

        # Add the training examples to the dataset
        for board, policy in temp_examples:
            training_examples.append((board, policy, evaluation))

            # Flip the evaluation for the next player
            evaluation *= -1

    return training_examples


def learn(model: ResNet, dataset: ChessDataset, iterations: int = 100, num_workers: int  = 1):
    for i in range(iterations):
        logging.info(f"Iteration {i + 1}/{iterations}")

        # Generate new examples
        futures: list[Future] = []

        with ProcessPoolExecutor() as executor:
            for _ in range(num_workers):
                futures.append(
                    executor.submit(self_play, "resources/models/model.pth", 80 // num_workers)
                )

        for future in futures:
            dataset.extend(future.result())

        # Trim old examples
        dataset.trim(300e3)

        # Save the updated dataset
        dataset.save("resources/examples/dataset.h5")

        # Train the model on the new examples
        model.fit(dataset, epochs=4)