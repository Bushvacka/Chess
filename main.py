import logging

from mcts import MCTS
from model import ResNet
from train import ACTION_SIZE, NUM_PLANES, ChessDataset

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(message)s", level=logging.INFO, datefmt="%H:%M:%S"
)


def main():
    dataset = ChessDataset()
    # dataset.load_pgn_directory("resources/examples")
    # dataset.save("resources/examples/dataset.h5")
    dataset.load("resources/examples/dataset.h5")

    model = ResNet(
        in_channels=NUM_PLANES, num_actions=ACTION_SIZE, depth=12, device="cpu"
    )
    model.fit(dataset, epochs=4, batch_size=800)
    model.save("resources/models/model.pth")


if __name__ == "__main__":
    main()
