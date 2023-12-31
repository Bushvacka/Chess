import train


def main():
    dataset = train.ChessDataset()
    dataset.load_from_pgn("examples/SaintLouis2023.pgn")


if __name__ == "__main__":
    main()
