import logging
import math

import numpy as np
from chess import Board

from model import ResNet
from train import ACTION_SIZE, STEP_HISTORY, move_to_action

DIRICHLET_NOISE = 0.3
NOISE_WEIGHT = 0.25

INITIAL_EXPLORATION = 1.25
BASE_EXPLORATION = 19652


class Node:
    def __init__(self, board: Board, prior: float) -> None:
        self.num_visits = 0
        self.total_value = 0
        self.prior = prior
        self.board = board
        self.children = {}

    def get_Q(self) -> float:
        return self.total_value / self.num_visits if self.num_visits > 0 else 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


class MCTS:
    def __init__(self, model: ResNet, num_simulations: int = 800) -> None:
        self.model = model
        self.num_simulations = num_simulations

    def predict(self, board: Board, temperature: float = 0) -> int:
        root = Node(board, 0)

        # Add noise to the root
        self.expand(root)
        self.add_exploration_noise(root)

        # Run simulations
        for i in range(self.num_simulations):
            logging.info(f"Simulation {i + 1}/{self.num_simulations}")
            self.search(root)

        # Select an action
        visits = [(child.num_visits, action) for action, child in root.children.items()]

        if temperature == 0:
            _, action = max(visits)
        else:
            visits = [(n ** (1 / temperature), action) for n, action in visits]
            total = sum(n for n, _ in visits)
            probs = [n / total for n, _ in visits]
            action = np.random.choice([action for _, action in visits], p=probs)

        return action

    def search(self, node: Node) -> float:
        if node.is_leaf():
            value = self.expand(node)
        else:
            # Select child with the best UCB
            best_child = max(
                node.children.values(), key=lambda child: self.ucb_score(node, child)
            )

            # Continue search
            value = self.search(best_child)

            # Update
            node.num_visits += 1
            node.total_value += value

        # Backpropagate
        return -value

    def expand(self, node: Node) -> float:
        # Rollout
        policy, value = self.model.predict(node.board)

        # Expand
        for move in node.board.legal_moves:
            board = node.board.copy(stack=STEP_HISTORY - 1)
            board.push(move)

            action = move_to_action(move, node.board)

            node.children[action] = Node(board, policy[action])

        return value

    def ucb_score(self, parent: Node, child: Node):
        Cpuct = (
            math.log((parent.num_visits + BASE_EXPLORATION + 1) / BASE_EXPLORATION)
            + INITIAL_EXPLORATION
        )
        Usa = child.prior * math.sqrt(parent.num_visits) / (1 + child.num_visits)

        return child.get_Q() + Cpuct * Usa

    def add_exploration_noise(self, node: Node):
        actions = node.children.keys()

        noises = np.random.gamma(DIRICHLET_NOISE, 1, ACTION_SIZE)

        for action, noise in zip(actions, noises):
            node.children[action].prior = (
                node.children[action].prior * (1 - NOISE_WEIGHT) + noise * NOISE_WEIGHT
            )
