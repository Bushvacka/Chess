import math

import numpy as np
from chess import Board

from model import ResNet
from utils import ACTION_SIZE, move_to_action

DIRICHLET_NOISE = 0.3
NOISE_WEIGHT = 0.25

INITIAL_EXPLORATION = 2
BASE_EXPLORATION = 19652


class Node:
    def __init__(self, board: Board, prior: float) -> None:
        self.num_visits: int = 0
        self.total_value: float = 0.0
        self.prior: float = prior
        self.board: Board = board
        self.children: dict[int, Node] = {}

    def get_Q(self) -> float:
        return self.total_value / self.num_visits if self.num_visits > 0 else 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


def run_mcts(
    model: ResNet,
    board: Board,
    num_simulations: int = 800,
    temperature: float = 0,
) -> tuple[int, np.ndarray]:
    root = Node(board, 0)

    # Add noise to the root
    expand(root, model)
    add_exploration_noise(root)

    # Run simulations
    for _ in range(num_simulations):
        search(root, model)

    # Generate policy
    policy = np.zeros(ACTION_SIZE)
    policy[list(root.children.keys())] = [
        child.num_visits for child in root.children.values()
    ]
    policy /= sum(policy)

    # Select an action
    action = select_action(root, temperature)

    return action, policy


def select_action(node: Node, temperature: float) -> int:
    visits = [(child.num_visits, action) for action, child in node.children.items()]

    if temperature == 0:
        _, action = max(visits)
    else:  # Softmax sample
        probs = np.array([math.exp(n) for n, _ in visits])
        probs /= sum(probs)
        action = np.random.choice([action for _, action in visits], p=probs)

    return action


def search(node: Node, model: ResNet) -> float:
    if node.is_leaf():
        value = expand(node, model)
    else:
        # Select child with the best UCB
        best_child = max(
            node.children.values(), key=lambda child: ucb_score(node, child)
        )

        # Continue search
        value = search(best_child, model)

        # Update
        node.num_visits += 1
        node.total_value += value

    # Backpropagate
    return -value


def expand(node: Node, model: ResNet) -> float:
    # Rollout
    policy, value = model.predict(node.board)

    # Expand
    for move in node.board.legal_moves:
        board = node.board.copy()
        board.push(move)

        action = move_to_action(move, node.board)

        node.children[action] = Node(board, policy[action])

    return value


def ucb_score(parent: Node, child: Node):
    Cpuct = (
        math.log((parent.num_visits + BASE_EXPLORATION + 1) / BASE_EXPLORATION)
        + INITIAL_EXPLORATION
    )
    Usa = child.prior * (math.sqrt(parent.num_visits) / (1 + child.num_visits))

    return child.get_Q() + Cpuct * Usa


def add_exploration_noise(node: Node):
    actions = node.children.keys()

    noises = np.random.gamma(DIRICHLET_NOISE, 1, ACTION_SIZE)

    for action, noise in zip(actions, noises):
        node.children[action].prior = (
            node.children[action].prior * (1 - NOISE_WEIGHT) + noise * NOISE_WEIGHT
        )
