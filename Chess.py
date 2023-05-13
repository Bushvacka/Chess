import numpy as np
import os
from chess import Board, Move
import chess
import chess.pgn

WHITE = chess.WHITE
BLACK = chess.BLACK

BOARD_SIZE = 8
ACTION_SIZE = 4096
"""
The action size used here comes moving a piece from any square to any square (64 x 64 = 4096)
The actual action size is 4672 (https://arxiv.org/pdf/1712.01815.pdf). The extra actions are a result
of pawn underpromotions, which I am too lazy to deal with.
"""

PGN_FOLDER = "examples"
PGN_FILE_NAME = "starting games"
PGN_FILE_EXTENSION = ".pgn"


def getInitBoard():
    """
    Returns a representation of the starting board
    """
    return Board()

def getBoardSize():
    """
    Returns the board dimensions
    """
    return (BOARD_SIZE, BOARD_SIZE)

def getNextState(board: Board, move: Move):
    """
    Returns the board after applying the action and the player to play next
    Parameters:
        board: Chess board
        player: Current player (WHITE or BLACK)
        action: Action taken by player
    """
    if isPromotion(board, move):
        move.promotion = chess.QUEEN

    new_board = board.copy()
    new_board.push(move)
    next_player = new_board.turn

    return new_board, next_player

def isPromotion(board: Board, move: Move):
    piece = board.piece_at(move.from_square)
    if piece.piece_type == chess.PAWN:
        if piece.color == WHITE and (move.to_square >= 56 and move.to_square <= 63):
            return True
        if piece.color == BLACK and (move.to_square >= 0 and move.to_square <= 7):
            return True
    return False

def getMoveFromAction(action):
    from_square = action // 64
    to_square = action % 64

    return Move(from_square, to_square)

def getMirroredMoveFromAction(action):
    from_square = action // 64
    to_square = action % 64

    mirrored_from_square = 63 - from_square
    mirrored_to_square = 63 - to_square

    return Move(mirrored_from_square, mirrored_to_square)

def actionFromMove(move: Move):
    return move.from_square * 64 + move.to_square

def mirroredActionFromMove(move: Move):
    return (63 - move.from_square) * 64 + (63 - move.to_square)

def getValidMoves(board: Board):
    """
    Returns a binary vector of actions, 1 if the move is valid, 0 otherwise.
    Parameters:
        board: Chess board
    """
    action_space = np.zeros(shape=ACTION_SIZE)
    for move in board.generate_legal_moves():
        action_space[(move.from_square * 64) + move.to_square] = 1
    return action_space

def getGameEnded(board: Board, player):
    """
    Returns 0 if game has not ended, 1 if player won, -1 if player lost, 1e-4 for draw.
    Parameters:
        board: Chess Board
        player: Current Player (WHITE or BLACK)            
    """
    result = board.result()

    if result == "*": # Not complete
        return 0
    elif result == "1-0": # White won
        return 1 if player == WHITE else -1
    elif result == "0-1": # Black won
        return -1 if player == WHITE else 1
    elif result == "1/2-1/2": # Draw
        return 1e-4

def getCanonicalForm(board: Board, player):
    """
    Returns the canonical form of the board.
    Parameters:
        board: Chess board
        player: Current player (WHITE or BLACK)
    """
    if (player == WHITE):
        return board
    else:
        new_board = board.mirror()
        new_board.apply_transform(chess.flip_horizontal)
        return new_board


def stringRepresentation(board: Board):
    """
    Returns a FEN representation of the board
    Parameters:
        board: Chess board  
    """
    return board.fen()

def planeRepresentation(board: Board):
    """
    Returns a representation of the board suitable for input to a neural network.
    The layout of the planes is similar to AlphaZero's implementation (https://shorturl.at/gBN27)
    Pawns   - Plane 0
    Knights - Plane 1
    Bishops - Plane 2
    Rooks   - Plane 3
    Queens  - Plane 4
    Kings   - Plane 5
    Positive if white, negative if black
    Parameters:
        board: Chess board
    """
    nn_board = np.zeros((6, BOARD_SIZE * BOARD_SIZE))
    for i in range(BOARD_SIZE * BOARD_SIZE):
        piece = board.piece_at(i)
        if piece != None:
            if piece.piece_type == chess.PAWN:
                nn_board[0][i] = 1 if piece.color == WHITE else -1
            elif piece.piece_type == chess.KNIGHT:
                nn_board[1][i] = 1 if piece.color == WHITE else -1
            elif piece.piece_type == chess.BISHOP:
                nn_board[2][i] = 1 if piece.color == WHITE else -1
            elif piece.piece_type == chess.ROOK:
                nn_board[3][i] = 1 if piece.color == WHITE else -1
            elif piece.piece_type == chess.QUEEN:
                nn_board[4][i] = 1 if piece.color == WHITE else -1
            elif piece.piece_type == chess.KING:
                nn_board[5][i] = 1 if piece.color == WHITE else -1
    nn_board = np.reshape(nn_board, (6, 8, 8))                                              
    return nn_board

def loadFromPGN():
    filepath = os.path.join(PGN_FOLDER, PGN_FILE_NAME + PGN_FILE_EXTENSION)
    with open(filepath) as f:
        training_examples = []
        game = chess.pgn.read_game(f)
        while game is not None and len(training_examples) < 200000:
            result = game.headers.get("Result")
            board = Board()
            current_player = board.turn
            # print(board, end="\n\n")
            for move in game.mainline_moves():
                canonical_board = getCanonicalForm(board, current_player)
                if board.turn == WHITE:
                    action = actionFromMove(move)
                else:
                    action = mirroredActionFromMove(move)

                policy = np.zeros(ACTION_SIZE)
                policy[action] = 1.0

                if current_player == WHITE:
                    if result == "1-0":
                        value = 1.0
                    elif result == "0-1":
                        value = -1.0
                    else:
                        value = 0
                else:
                    if result == "1-0":
                        value = -1.0
                    elif result == "0-1":
                        value = 1.0
                    else:
                        value = 0

                training_examples.append([canonical_board, current_player, policy, value])

                board, current_player = getNextState(board, move)
                if not board.is_valid():
                    print(f"Invalid Board" + board.status)
                # print(board, end="\n\n")
            game = chess.pgn.read_game(f)
        return training_examples

