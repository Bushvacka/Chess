from chess import Board, Move, Piece
import chess
import numpy as np

WHITE = chess.WHITE
BLACK = chess.BLACK

BOARD_SIZE = 8
ACTION_SIZE = 4096
"""
The action size used here comes moving a piece from any square to any square (64 x 64 = 4096)
The actual action size is 4672 (https://arxiv.org/pdf/1712.01815.pdf). The extra actions are a result
of pawn underpromotions, which I am too lazy to deal with.
"""

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

def integerRepresentation(board: Board):
    """
    Returns a representation of the board suitable for input to a neural network.
    Pawns   - 1
    Knights - 2
    Bishops - 3
    Rooks   - 4
    Queens  - 5
    Kings   - 6
    Positive if white, negative if black
    Parameters:
        board: Chess board
    """
    nn_board = np.zeros(BOARD_SIZE * BOARD_SIZE)
    for i in range(BOARD_SIZE * BOARD_SIZE):
        piece = board.piece_at(i)
        if piece != None:
            if piece.piece_type == chess.PAWN:
                nn_board[i] = 1 if piece.color == WHITE else -1
            elif piece.piece_type == chess.KNIGHT:
                nn_board[i] = 2 if piece.color == WHITE else -2
            elif piece.piece_type == chess.BISHOP:
                nn_board[i] = 3 if piece.color == WHITE else -3
            elif piece.piece_type == chess.ROOK:
                nn_board[i] = 4 if piece.color == WHITE else -4
            elif piece.piece_type == chess.QUEEN:
                nn_board[i] = 5 if piece.color == WHITE else -5
            elif piece.piece_type == chess.KING:
                nn_board[i] = 6 if piece.color == WHITE else -6
    nn_board = np.reshape(nn_board, getBoardSize())                                              
    return nn_board
