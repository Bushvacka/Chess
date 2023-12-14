# Chess Engine

An attempt to create a chess engine based on Deepmind's [AlphaGo Zero paper](https://arxiv.org/abs/1712.01815) and inspired by [AlphaZero General](https://github.com/suragnair/alpha-zero-general). Uses self-play based reinforcement learning to train a neural network to act as a policy predictor for use in a Monte-Carlo Tree Search algorithm which finds the optimal move for a given board state.

## Setup

1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/Bushvacka/Chess.git
    ```

2. Navigate to the project directory:
    ```bash
    cd chess
    ```

3. Create an enviroment:
    ```bash
    python -m venv env
    ```

4. Activate the environment:
    ```bash
    source env/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Representation

In order to make a board suitable for input to a neural network, a board state is encoded using 6 planes which store piece locations and color (positive or negative).
 - Pawns   - Plane 0
 - Knights - Plane 1
 - Bishops - Plane 2
 - Rooks   - Plane 3
 - Queens  - Plane 4
 - Kings   - Plane 5

I represent the action space for chess using a vector of length 4096, which denotes moving a piece from any square on the board to any other square (64 x 64 = 4096). The actual action size for chess is [4672](https://arxiv.org/pdf/1712.01815.pdf#page=13), the extra actions coming as a result of pawn underpromotions, which I am too lazy to deal with.

