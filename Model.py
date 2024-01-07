import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

LEARNING_RATE = 1e-1
WEIGHT_DECAY = 1e-3


class ResBlock(nn.Module):
    def __init__(
        self,
        num_channels: int = 256,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        return F.relu(x + y)


class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 40,
        num_channels: int = 256,
        depth: int = 20,
        num_policies: int = 4672,
        device: str = "cuda",
        name: str = "ResNet",
    ) -> None:
        super().__init__()

        self.writer = SummaryWriter(
            f"./resources/logs/{name} d={depth} p={in_channels} lr={LEARNING_RATE} c={WEIGHT_DECAY}"
        )
        self.step = 0

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 5, padding=2),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
        )

        self.residual_blocks = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(depth)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, num_policies),
            nn.Softmax(dim=1),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 4, kernel_size=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, 1),
            nn.Tanh(),
        )

        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.conv(x)

        for block in self.residual_blocks:
            y = block(y)

        p = self.policy_head(y)
        v = self.value_head(y)

        return p, v

    def fit(self, dataset: Dataset, epochs: int = 10, batch_size: int = 512) -> None:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(
            self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        policy_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.MSELoss()

        self.train()

        for epoch in range(epochs):
            logging.info(f"Epoch {epoch + 1}/{epochs}")
            for x, p, v in loader:
                optimizer.zero_grad()

                # Convert to single-precision floating point
                x, p, v = x.float(), p.float(), v.float()

                # Move to device
                x, p, v = x.to(self.device), p.to(self.device), v.to(self.device)

                p_hat, v_hat = self(x)

                policy_loss = policy_criterion(p, p_hat)
                value_loss = value_criterion(v, v_hat.squeeze())

                loss = policy_loss + value_loss

                # Log losses to TensorBoard
                self.writer.add_scalar("loss/policy", policy_loss, self.step)
                self.writer.add_scalar("loss/value", value_loss, self.step)
                self.step += 1

                loss.backward()
                optimizer.step()

    def predict(self, x: torch.Tensor) -> tuple[np.ndarray, float]:
        assert self.device == "cpu", "Please predict on CPU"

        x = x.float()  # Convert to single-precision floating point
        self.eval()

        with torch.no_grad():
            y = self(x.unsqueeze(0))
            p, v = y[0].squeeze(), y[1].squeeze()
            return p.numpy(), v.item()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))
