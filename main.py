import logging

from model import ResNet
from train import ACTION_SIZE, NUM_PLANES, ChessDataset

logging.basicConfig(
    format="%(asctime)s %(message)s", level=logging.INFO, datefmt="%H:%M:%S"
)


def main():
    dataset = ChessDataset()
    dataset.load_file("resources/examples/dataset.h5")

    model = ResNet(in_channels=NUM_PLANES, depth=7, num_policies=ACTION_SIZE)
    model.fit(dataset, epochs=4, batch_size=800)
    model.save("resources/models/model.pth")


if __name__ == "__main__":
    main()
