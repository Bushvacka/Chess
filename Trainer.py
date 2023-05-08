import os, random, time, threading
import numpy as np
from collections import deque
from pickle import Pickler, Unpickler
from tqdm import tqdm

from MCTS import MCTS
from Model import Model
import Chess




NUMBER_OF_ITERATIONS = 100
EPISODES_PER_ITERATION = 100
NUM_THREADS = 2
NUMBER_OF_GAMES_TO_PLAY = 30

TEMPERATURE_THRESHOLD = 30
REPLACE_THRESHOLD = 0.6

MAX_NUMBER_OF_EXAMPLES = 200000

MODEL_FOLDER = "checkpoints/models"
MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pth.tar"

EXAMPLE_FOLDER = "checkpoints/examples"
EXAMPLE_FILE_NAME = "example"
EXAMPLE_FILE_EXTENSION = ".examples"
LOAD_EXAMPLES = False



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
        self.training_examples = []

    def executeEpisode(self):
        """
        Executes a full game of self-play. Adds the board state as a training example after each turn.
        Returns a list of training examples of the form (canonicalBoard, currPlayer, pi,v)
        """
        mcts = MCTS(self.model)
        training_examples = []
        board = Chess.getInitBoard()
        current_player = Chess.WHITE
        number_of_moves = 0

        result = Chess.getGameEnded(board, current_player) 

        while result == 0: # Run until the game ends
            # Preform MCTS to obtain the policy for the board
            canonical_board = Chess.getCanonicalForm(board, current_player)
            temperature = 1 if number_of_moves < TEMPERATURE_THRESHOLD else 0
            policy = mcts.getActionProbabilities(canonical_board, temperature)

            # Pick an action
            action = np.random.choice(Chess.ACTION_SIZE, p=policy)
            # Get the corresponding move
            if (current_player == Chess.WHITE):
                move = Chess.getMoveFromAction(action)
            else: # Convert move back from canonical form
                move = Chess.getMirroredMoveFromAction(action)
            # Play the move
            board, current_player = Chess.getNextState(board, move)
            number_of_moves += 1

            # Save the board as a training example
            training_examples.append([canonical_board, current_player, policy, None]) # TODO: Implement symmetries

            # Update status of the game
            result = Chess.getGameEnded(board, current_player)

        # Fill in values for each training example based on the winner
        for i in range(len(training_examples)):
            example = training_examples[i] # Form: (canonicalBoard, currPlayer, pi,v)
            example_player = example[1]
            if current_player == example_player: # The player in the example is the same, and has the same result
                example[3] = result
            else: # The player in the example is not the current player, and will have the opposite result
                example[3] = result * -1
            training_examples[i] = example

        return training_examples
    
    def generateTrainingExamples(self, num_episodes):
        for _ in tqdm(range(num_episodes), desc=f"Episodes({threading.current_thread().name})"):    
            self.training_examples += self.executeEpisode() # Execute a self-play game and add the training examples

    def learn(self):
        """
        Iterates NUMBER_OF_ITERATIONS times, each time playing EPISODES_PER_ITERATION self-play games and then 
        retraining the model using the training_examples. The new model replaces the old one if it's win fraction is greater
        than REPLACE_THRESHOLD.
        """

        if LOAD_EXAMPLES:
            self.training_examples = self.loadTrainingExamples()

        for _ in tqdm(range(NUMBER_OF_ITERATIONS)):
            threads = []
            for __ in range(NUM_THREADS):
                # Create threads to generate training examples
                threads.append(threading.Thread(target=self.generateTrainingExamples, args=(EPISODES_PER_ITERATION//NUM_THREADS,)))
            
            # Start generating examples
            for thread in threads:
                thread.start()
            
            # Wait for all threads to finish
            for thread in threads:
                thread.join()

            # Remove the oldest examples if necessary
            num_examples = len(self.training_examples)
            if num_examples > MAX_NUMBER_OF_EXAMPLES:
                self.training_examples = self.training_examples[num_examples - MAX_NUMBER_OF_EXAMPLES:num_examples]

            # Save training examples
            self.saveTrainingExamples()

            # Save a copy of the old model
            self.model.saveCheckpoint(MODEL_FOLDER, MODEL_FILE_NAME + MODEL_FILE_EXTENSION)
            self.competitor_model.loadCheckpoint(MODEL_FOLDER, MODEL_FILE_NAME + MODEL_FILE_EXTENSION)

            # Shuffle training examples
            training_examples = []
            for example in self.training_examples:
                training_examples.append((Chess.integerRepresentation(example[0]), example[2], example[3]))
            random.shuffle(training_examples)
            start = time.time()
            # Train the new model
            self.competitor_model.train_nn(training_examples)
            print(f"Training time: {time.time() - start}")

            # Compare the new model with the old model
            competitor_mcts = MCTS(self.competitor_model)
            mcts = MCTS(self.model)

            competitor_agent = lambda canonical_board: np.argmax(competitor_mcts.getActionProbabilities(canonical_board, temperature=0))
            agent = lambda canonical_board: np.argmax(mcts.getActionProbabilities(canonical_board, temperature=0))

            competitor_wins, old_wins, draws = self.playGames(competitor_agent, agent) # Play them against eachother
            
            print(f"Competitor: {competitor_wins}\nOriginal: {old_wins}")
            # Either replace or keep the model based on the win percentage of the competitor
            total_wins = (competitor_wins + old_wins) * 1.0
            if (total_wins > 0):
                win_fraction = competitor_wins / total_wins
            else:
                win_fraction = 0

            if win_fraction > REPLACE_THRESHOLD: # Replace the model with the competitor
                print("Replacing with competitor")
                self.model.loadCheckpoint(MODEL_FOLDER, MODEL_FILE_NAME)
            else: # The competitor was not good enough, load the original model
                print("Keeping original")
                self.competitor_model.saveCheckpoint(MODEL_FOLDER, MODEL_FILE_NAME)
                self.model.loadCheckpoint(MODEL_FOLDER, MODEL_FILE_NAME)
    
    def playGame(self, agent1, agent2):
        """
        Plays a game between agent1 and agent2 to completion. Returns 1 if agent1 wins, -1 if
        agent2 wins, and some other value if the game ended in a draw.
        """
        board = Chess.getInitBoard()
        current_player = board.turn
        result = Chess.getGameEnded(board, current_player)

        while result == 0:
            # Get the agent's action
            canonical_board = Chess.getCanonicalForm(board, current_player)
            if (current_player == Chess.WHITE): # Agent1 to play
                action = agent1(canonical_board)
            else: # Agent2 to play
                action = agent2(canonical_board)

            # Get the corresponding move
            if (current_player == Chess.WHITE):
                move = Chess.getMoveFromAction(action)
            else: # Convert move back from canonical form
                move = Chess.getMirroredMoveFromAction(action)

            # Make the move and get the resulting board & game result
            board, current_player = Chess.getNextState(board, move)
            result = Chess.getGameEnded(board, current_player) 
        
        if current_player == Chess.WHITE:
            return result
        else:
            return result * -1

    def playGames(self, agent1, agent2):
        """
        Plays out NUMBER_OF_GAMES_TO_PLAY between agent1 and agent2.
        Returns the number of games won by agent1, the number of games won by agent2, and the number of draws (in order)
        Parameters:
            agent1: Function representing an agent which takes in a board and returns an action
            agent2: Function representing an agent which takes in a board and returns an action
        """
        wins1 = 0
        wins2 = 0
        draws = 0

        for _ in tqdm(range(NUMBER_OF_GAMES_TO_PLAY//2), desc="White"): # Play half of the games with agent1 as white
            result = self.playGame(agent1, agent2)
            if result == 1:
                wins1 += 1
            elif result == -1:
                wins2 += 1
            else:
                draws += 1

        for _ in tqdm(range(NUMBER_OF_GAMES_TO_PLAY//2), desc="White"): # Play half of the games with agent2 as white
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

    def loadTrainingExamples(self):
        filepath = os.path.join(EXAMPLE_FOLDER, EXAMPLE_FILE_NAME + EXAMPLE_FILE_EXTENSION)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return Unpickler(f).load()

    
