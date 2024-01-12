# Chess Engine

Based on DeepMind's [AlphaGo Zero paper](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf) and [AlphaZero paper](https://arxiv.org/abs/1712.01815). Uses a mix of supervised learning and self-play based reinforcement learning to train a network to act as a policy and value predictor. This network is used in a Monte-Carlo Tree Search algorithm to generate optimal moves for a given board state.

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
    python -m venv .venv
    ```

4. Activate the environment:
    ```bash
    .\.venv\Scripts\activate
    ```

3. Optionally, install a GPU-compatible version of [pytorch](https://pytorch.org/):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4. Install the rest of the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```


## Datasets

The games used for supervised learning were downloaded from the standard category of the [FICS Games Database](https://www.ficsgames.org/download.html).