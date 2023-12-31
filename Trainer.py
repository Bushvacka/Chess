import os, random, threading
import numpy as np
from pickle import Pickler, Unpickler
from tqdm import tqdm

from MCTS import MCTS
from Model import Model
import Chess

NUMBER_OF_ITERATIONS = 100
EPISODES_PER_ITERATION = 80
NUM_THREADS = 4
NUMBER_OF_GAMES_TO_PLAY = 10

TEMPERATURE_THRESHOLD = 30
REPLACE_THRESHOLD = 0.6

MAX_NUMBER_OF_EXAMPLES = 300000

LOAD_MODEL = False
LOAD_EXAMPLES = False
LOAD_PGN = True
SKIP_FIRST_ITERATION = True

EXAMPLE_FOLDER = "examples"
EXAMPLE_FILE_NAME = "example"
EXAMPLE_FILE_EXTENSION = ".examples"

PGN_FOLDER = "examples"
PGN_FILE_NAME = "SaintLouis2023"
PGN_FILE_EXTENSION = ".pgn"

class Trainer():
    """
    Trains a model using self-play
    Parameters:
        model: Model to be trained
    """

    def __init__(self, model: Model):
        self.model = model
        self.competitor_model = self.model.__class__()
        self.mcts = MCTS(self.model)
        self.training_examples = {}

    def executeEpisode(self):
        """
        Executes a full game of self-play. Adds the board states and policies to the training examples.
        """
        mcts = MCTS(self.model)
        training_examples = []

        board = Chess.get_init_board()
        current_player = Chess.WHITE
        number_of_moves = 0

        result = Chess.get_game_ended(board, current_player) 

        while result == 0: # Run until the game ends
            # Preform MCTS to obtain the policy for the board
            canonical_board = Chess.get_canonical_form(board, current_player)

            # More likely to explore at the beginning of the game, more likely to exploit at the end
            temperature = 1 if number_of_moves < TEMPERATURE_THRESHOLD else 0

            # Get the policy
            policy = mcts.getActionProbabilities(canonical_board, temperature)

            # Pick an action and get the corresponding move
            action = np.random.choice(Chess.ACTION_SIZE, p=policy)
            move = Chess.action_to_move(action, current_player)
            
            # Save the board as a training example
            training_examples.append([canonical_board, current_player, policy]) # TODO: Implement symmetries
           
            # Play the move
            board, current_player = Chess.get_next_state(board, move)
            number_of_moves += 1

            # Update status of the game
            result = Chess.get_game_ended(board, current_player)

        # Fill in values for each training example based on the winner and add them to the training examples
        for canonical_board, current_player, policy in training_examples:
            value = 1.0 if result == current_player else -1.0
            fen = Chess.get_fen(canonical_board)
            
            # If the board is not in the training examples, add it
            if fen not in self.training_examples:
                self.training_examples[fen] = [canonical_board, policy, value, 1]
            else: # Otherwise, update the policy, value, and count
                self.training_examples[fen][1] += policy
                self.training_examples[fen][2] += value
                self.training_examples[fen][3] += 1
    
    def generateTrainingExamples(self, num_episodes):
        for _ in tqdm(range(num_episodes), desc=f"Episodes({threading.current_thread().name})"):    
            self.executeEpisode() # Execute a self-play game

    def learn(self):
        """
        Iterates NUMBER_OF_ITERATIONS times, each time playing EPISODES_PER_ITERATION self-play games and then 
        retraining the model using the training_examples. The new model replaces the old one if it's win fraction is greater
        than REPLACE_THRESHOLD.
        """

        if LOAD_EXAMPLES:
            self.training_examples = self.load_training_examples_from_file()
        elif LOAD_PGN:
            self.training_examples = self.load_training_examples_from_pgn()

        if LOAD_MODEL:
            self.model.loadCheckpoint()

        for iteration in tqdm(range(NUMBER_OF_ITERATIONS), desc="Iterations"):
            if not (iteration == 0 and SKIP_FIRST_ITERATION):
                threads: list[threading.Thread] = []
                for _ in range(NUM_THREADS):
                    # Create threads to generate training examples
                    threads.append(threading.Thread(target=self.generateTrainingExamples, args=(EPISODES_PER_ITERATION//NUM_THREADS,)))
                
                # Start generating examples
                for thread in threads:
                    thread.start()
                
                # Wait for all threads to finish
                for thread in threads:
                    thread.join()

            # TODO: Remove the oldest examples if necessary
            num_examples = len(self.training_examples)
            if num_examples > MAX_NUMBER_OF_EXAMPLES:
                print("Removing old examples")
                # self.training_examples = self.training_examples[num_examples - MAX_NUMBER_OF_EXAMPLES:num_examples]

            # Save training examples
            self.saveTrainingExamples()

            # Save a copy of the old model
            self.model.saveCheckpoint()
            self.competitor_model.loadCheckpoint()

            # Normalize and shuffle training examples
            training_examples = []
            for canonical_board, policy, value, count in self.training_examples.values():
                training_examples.append((Chess.get_model_representation(canonical_board), policy / count, value / count))
            random.shuffle(training_examples)

            # Train the new model
            self.competitor_model.train_nn(training_examples)

            if not (iteration == 0 and SKIP_FIRST_ITERATION):
                # Compare the new model with the old model
                competitor_mcts = MCTS(self.competitor_model)
                mcts = MCTS(self.model)

                competitor_agent = lambda canonical_board: np.argmax(competitor_mcts.getActionProbabilities(canonical_board, temperature=0))
                agent = lambda canonical_board: np.argmax(mcts.getActionProbabilities(canonical_board, temperature=0))

                competitor_wins, old_wins, draws = self.playGames(competitor_agent, agent) # Play them against eachother
                
                print(f"Competitor: {competitor_wins}\nOriginal: {old_wins}")

                if competitor_wins >= old_wins: # Replace the model with the competitor
                    print("Replacing with competitor")
                    self.competitor_model.saveCheckpoint()
                    self.model.loadCheckpoint()
                else: # The competitor was not good enough, load the original model
                    print("Keeping original")
                    self.model.loadCheckpoint()
            else:
                print("Replacing with competitor")
                self.competitor_model.saveCheckpoint()
                self.model.loadCheckpoint()
    
    def playGame(self, agent1, agent2):
        """
        Plays a game between agent1 and agent2 to completion. Returns 1 if agent1 wins, -1 if
        agent2 wins, and some other value if the game ended in a draw.
        """
        board = Chess.get_init_board()
        current_player = board.turn

        result = Chess.get_game_ended(board, current_player)

        while result == 0:
            # Get the canonical board
            canonical_board = Chess.get_canonical_form(board, current_player)

            # Get the action and corresponding move
            action = agent1(canonical_board) if current_player == Chess.WHITE else agent2(canonical_board)
            move = Chess.action_to_move(action, current_player)

            # Make the move
            board, current_player = Chess.get_next_state(board, move)

            # Check if the game is over
            result = Chess.get_game_ended(board, current_player) 
        
        return result if current_player == Chess.WHITE else -result

    def playGames(self, agent1, agent2):
        """
        Plays out NUMBER_OF_GAMES_TO_PLAY between agent1 and agent2.
        Returns the number of games won by agent1, the number of games won by agent2, and the number of draws.
        """
        wins1 = 0
        wins2 = 0
        draws = 0

        for _ in tqdm(range(NUMBER_OF_GAMES_TO_PLAY//2), desc="White Games"): # Play half of the games with agent1 as white
            result = self.playGame(agent1, agent2)
            if result == 1:
                wins1 += 1
            elif result == -1:
                wins2 += 1
            else:
                draws += 1

        for _ in tqdm(range(NUMBER_OF_GAMES_TO_PLAY//2), desc="Black Games"): # Play half of the games with agent2 as white
            result = self.playGame(agent2, agent1)
            if result == 1:
                wins2 += 1
            elif result == -1:
                wins1 += 1
            else:
                draws += 1

        return wins1, wins2, draws

    def saveTrainingExamples(self):
        if not os.path.exists(EXAMPLE_FOLDER):
            os.makedirs(EXAMPLE_FOLDER)
        filepath = os.path.join(EXAMPLE_FOLDER, EXAMPLE_FILE_NAME + EXAMPLE_FILE_EXTENSION)

        with open(filepath, "wb+") as f:
            Pickler(f).dump(self.training_examples)

    def load_training_examples_from_file(self):
        filepath = os.path.join(EXAMPLE_FOLDER, EXAMPLE_FILE_NAME + EXAMPLE_FILE_EXTENSION)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return Unpickler(f).load()
    
    def load_training_examples_from_pgn(self):
        filepath = os.path.join(PGN_FOLDER, PGN_FILE_NAME + PGN_FILE_EXTENSION)

        with open(filepath) as f:
            training_examples = {}

            # Load the first game in the file
            game = Chess.load_pgn_game(f)
            player = Chess.WHITE

            # Iterate through the games in the file
            while game is not None and len(training_examples) < MAX_NUMBER_OF_EXAMPLES:
                result = game.headers.get("Result")
                board = Chess.get_init_board()
                
                for move in game.mainline_moves():
                    # Ignore underpromotions
                    if move.promotion is not None and move.promotion != 5:
                        print("Uh oh, Underpromotion!")
                        board, player = Chess.get_next_state(board, move)
                        continue

                    # Get the canonical board
                    canonical_board = Chess.get_canonical_form(board, player)

                    # Mirror the policy of the GM
                    policy = np.zeros(Chess.ACTION_SIZE)
                    action = Chess.move_to_action(move, player)
                    policy[action] = 1.0

                    # Get the value of the board
                    if result == "1-0":
                        value = 1.0 if player == Chess.WHITE else -1.0
                    elif result == "0-1":
                        value = -1.0 if player == Chess.WHITE else 1.0
                    else:
                        value = 0

                    fen = Chess.get_fen(canonical_board)
                    
                    # If the board is not in the training examples, add it
                    if fen not in training_examples:
                        training_examples[fen] = [canonical_board, policy, value, 1]
                    else: # Otherwise, update the policy, value, and count
                        training_examples[fen][1] += policy
                        training_examples[fen][2] += value
                        training_examples[fen][3] += 1

                    # Make the move
                    board, player = Chess.get_next_state(board, move)

                    # Sanity check
                    if not board.is_valid():
                        print(f"Invalid Board" + board.status)

                # Get the next game        
                game = Chess.load_pgn_game(f)

            return training_examples