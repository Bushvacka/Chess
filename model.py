import logging

import numpy as np
import torch
import torch.nn.functional as F
from chess import Board
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from train import get_canonical_form, get_legal_actions, get_model_form

INITIAL_LR = 1e-1
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9

VALUE_WEIGHT = 1e-2
TRAIN_SPLIT = 0.9


class ResBlock(nn.Module):
    def __init__(
        self,
        num_channels: int = 256,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            num_channels, num_channels, kernel_size, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = F.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y += identity
        y = F.relu(y)

        return y


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 40,
        num_channels: int = 256,
        depth: int = 20,
        num_actions: int = 4672,
        device: str = "cuda",
        name: str = "ResNet",
    ) -> None:
        super().__init__()

        self.writer = SummaryWriter(
            f"./resources/logs/{name} d={depth} p={in_channels} lr={INITIAL_LR} c={WEIGHT_DECAY} m={MOMENTUM} v={VALUE_WEIGHT}"
        )
        self.step = 0
        self.num_actions = num_actions
        self.device = device

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

        self.residual_blocks = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(depth)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, self.num_actions),
            nn.Softmax(dim=1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 8, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, 1),
            nn.Tanh(),
        )

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.conv(x)

        for block in self.residual_blocks:
            y = block(y)

        p = self.policy_head(y)
        v = self.value_head(y)

        return p, v

    def fit(self, dataset: Dataset, epochs: int = 10, batch_size: int = 512) -> None:
        # Split dataset into training and validation sets
        train_dataset, test_dataset = random_split(
            dataset,
            [TRAIN_SPLIT, 1 - TRAIN_SPLIT],
        )

        train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize the optimizer and learning rate scheduler
        optimizer = optim.SGD(
            self.parameters(),
            lr=INITIAL_LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = MultiStepLR(optimizer, milestones=[300, 600, 800, 1000], gamma=0.1)

        # Define loss functions
        policy_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.MSELoss()

        self.train()

        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            for x, p, v in train:
                optimizer.zero_grad()

                # Convert to single-precision floating point
                x: torch.Tensor = x.float()
                p: torch.Tensor = p.float()
                v: torch.Tensor = v.float()

                # Move to device
                x, p, v = x.to(self.device), p.to(self.device), v.to(self.device)

                p_hat, v_hat = self(x)

                # Calculate losses
                policy_loss = policy_criterion(p, p_hat)
                value_loss = value_criterion(v.unsqueeze(1), v_hat)

                loss: torch.Tensor = policy_loss + VALUE_WEIGHT * value_loss

                # Log losses
                self.writer.add_scalar("loss/policy", policy_loss, self.step)
                self.writer.add_scalar("loss/value", value_loss, self.step)
                self.writer.add_scalar("loss/total", loss, self.step)
                self.writer.add_scalar("lr", optimizer.param_groups[0]["lr"], self.step)
                self.step += 1

                # Backpropagate
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Log epoch accuracy
            self.writer.add_scalar("accuracy", self.evaluate(test_dataset), epoch)

    def predict(self, board: Board) -> tuple[np.ndarray, float]:
        canonical_board = get_canonical_form(board)

        x = get_model_form(canonical_board)

        # Convert to single-precision floating point
        x = x.float()

        # Move to device
        x = x.to(self.device)

        self.eval()

        with torch.no_grad():
            policy, value = self(x.unsqueeze(0))

            policy = policy.numpy()
            value = value.item

            # Mask illegal actions
            policy *= get_legal_actions(board)

            # Normalize
            if sum(policy) > 0:
                policy /= sum(policy)
            else:
                logging.warning("All actions masked")
                policy += 1 / self.num_actions

        return policy, value

    def evaluate(self, dataset: Dataset, batch_size: int = 512) -> float:
        test = DataLoader(dataset, batch_size=batch_size)

        correct_predictions = 0

        self.eval()

        with torch.no_grad():
            for x, p, _ in test:
                # Convert to single-precision floating point
                x: torch.Tensor = x.float()
                p: torch.Tensor = p.float()

                # Move to device
                x, p = x.to(self.device), p.to(self.device)

                p_hat, _ = self(x)

                # Calculate how many predictions were correct
                correct_predictions += (p.argmax(dim=1) == p_hat.argmax(dim=1)).sum()

        return correct_predictions / len(test)

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
