import logging

from model import ResNet
from train import ChessDataset, learn

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(message)s", level=logging.INFO, datefmt="%H:%M:%S"
)


def main():
    dataset = ChessDataset()
    dataset.load_pgn_directory("resources/examples")
    dataset.save("resources/examples/dataset.h5")

    model = ResNet(name="Self Play")
    model.fit(dataset, epochs=10, batch_size=1024)

    learn(model, dataset)


if __name__ == "__main__":
    main()
