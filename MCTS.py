import math
import numpy as np
import Chess
from Model import Model

NUMBER_OF_MOVES_TO_SIMULATE = 30
C = 1.0 # Exploration vs. Exploitation bias for UCB calculation

class MCTS:
    """
    Implementation of Monte Carlo Tree Search.
    Based on Deepmind's AlphaZero paper (https://arxiv.org/abs/1712.01815)
    Inspired by AlphaZero General (https://github.com/suragnair/alpha-zero-general)
    Parameters:
        model: Neural Network used to predict policy, a probability vector for all possible actions
    """
    def __init__(self, model: Model):
        self.Qsa = {}  # Expected value for taking action a from state s
        self.Ns = {}  # Number of times state s has been visited
        self.Nsa = {}  # Number of times action a has been taken from state s
        self.Ps = {}  # Policy for state s

        self.model = model
    
    def getActionProbabilities(self, canonical_board, temperature):
        """
        Returns a policy vector for all actions from this state
        Parameters:
            board: Starting board for the search
            temperature: the degree of exploration vs. exploitation
            A higher temperature means that the algorithm is more likely to choose nodes with high uncertainty. 
            A lower temperature biases the selection towards nodes with high values.
        """
        for _ in range(NUMBER_OF_MOVES_TO_SIMULATE):
            self.search(canonical_board)
        
        state = Chess.stringRepresentation(canonical_board)

        Na = [self.Nsa[(state, action)] if (state, action) in self.Nsa else 0 for action in range(Chess.ACTION_SIZE)]

        if temperature == 0: # Exploit the best action
            best_action = np.array(Na).argmax()
            probabilities = [0] * len(Na)
            probabilities[best_action] = 1
            return probabilities
        else: # Modify according to temperature and renormalize
            Na = [n ** (1. / temperature) for n in Na]
            sum = np.sum(Na)
            probabilities = [n / sum for n in Na]
            return probabilities

    def search(self, canonical_board):
        state = Chess.stringRepresentation(canonical_board)

        ended = Chess.getGameEnded(canonical_board, Chess.WHITE)

        if ended != 0: # If the game is over, return the outcome
            return -ended
        
        if state not in self.Ps: # State hasn't been visited yet
            
            #Get policy and value prediction from the model
            self.Ps[state], value = self.model.predict(Chess.planeRepresentation(canonical_board))
            # Remove illegal moves and renormalize
            mask = Chess.getValidMoves(canonical_board)
            self.Ps[state] = self.Ps[state] * mask

            sum = np.sum(self.Ps[state])
            if sum > 0:
                self.Ps[state] /= sum
            else:
                print("All valid moves masked.")
                self.Ps[state] += 1/len(self.Ps[state])
                
            self.Ns[state] = 0
            return -value
        else: # State has been reached before
            # Pick the action with the highest UCT
            valid_moves = Chess.getValidMoves(canonical_board)

            best_UCT = -99999
            best_action = None
            for action in range(Chess.ACTION_SIZE):
                if valid_moves[action]:
                    UCT = self.calculateUCT(state, action)
                    if UCT > best_UCT:
                        best_UCT = UCT
                        best_action = action
            
            action = best_action
            move = Chess.getMoveFromAction(action)
            # Recursively search from the state reached by taking the best action
            next_state, next_player = Chess.getNextState(canonical_board, move)
            next_state = Chess.getCanonicalForm(next_state, next_player)
            value = self.search(next_state)

            # Update the expected value
            if (state, action) in self.Qsa:
                self.Qsa[(state, action)] = (self.Nsa[(state, action)] * self.Qsa[(state, action)] + value) / (self.Nsa[(state, action)] + 1)
            else:
                self.Qsa[(state, action)] = value

            # Update the number of visits
            self.Ns[state] += 1
            if (state, action) in self.Nsa:
                self.Nsa[(state, action)] += 1
            else:
                self.Nsa[(state, action)] = 1

            return -value
        
    def calculateUCT(self, state, action):
        """
        Returns the Upper Confidence Bound for a given state-action pair
        Parameters:
            state: Chess board
            action: Move
        """
        if (state, action) in self.Qsa:
            return self.Qsa[(state, action)] + C * math.sqrt(self.Ns[state] / self.Nsa[(state, action)])
        else:
            return C * self.Ps[state][action] * math.sqrt(self.Ns[state] + 1e-8)