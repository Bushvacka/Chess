import logging

import train
from model import ResNet

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def main():
    dataset = train.ChessDataset()
    dataset.load_from_pgn("examples/SaintLouis2023.pgn")
    dataset.save("examples/dataset.pkl")
    model = ResNet()
    model.fit(dataset)
    model.save("examples/model.pth")


if __name__ == "__main__":
    main()
