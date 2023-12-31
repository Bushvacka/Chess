import numpy as np
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


def get_init_board() -> Board:
    """
    Returns the starting chess board
    """
    return Board()

def get_next_state(board: Board, move: Move) -> (Board, bool):
    """
    Returns the board state after making the given move, along with the player to play next
    """

    if is_promotion(board, move) and move.promotion == None:
        move.promotion = chess.QUEEN

    new_board = board.copy()
    new_board.push(move)
    next_player = new_board.turn

    return new_board, next_player

def is_promotion(board: Board, move: Move) -> bool:
    """
    Returns true if the move is a promotion, false otherwise.
    """
    piece = board.piece_at(move.from_square)
    if piece.piece_type == chess.PAWN:
        if piece.color == WHITE and (move.to_square >= 56 and move.to_square <= 63):
            return True
        if piece.color == BLACK and (move.to_square >= 0 and move.to_square <= 7):
            return True
    return False

def action_to_move(action: int, player: bool = WHITE) -> Move:
    """
    Returns the move corresponding to an action from the perspective
    of the given plauer
    """
    from_square = (action // 64) if player else (63 - action // 64)
    to_square = (action % 64) if player else (63 - action % 64)

    return Move(from_square, to_square)

def move_to_action(move: Move, player: bool = WHITE) -> int:
    """
    Returns the action corresponding to a move from the perspective
    of the given player
    """
    from_square = move.from_square if player else 63 - move.from_square
    to_square = move.to_square if player else 63 - move.to_square

    return from_square * 64 + to_square

def get_valid_actions(board: Board) -> np.ndarray:
    """
    Returns an array of valid actions for the given board state.
    """
    valid_actions = np.zeros(shape=ACTION_SIZE)
    for move in board.generate_legal_moves():
        valid_actions[move_to_action(move)] = 1
    return valid_actions

def get_game_ended(board: Board, player: bool) -> float:
    """
    Returns 0 if game has not ended, 1 if player won, -1 if player lost, 1e-4 for draw.          
    """
    result = board.result()

    if result == "*": # Not complete
        return 0.0
    elif result == "1-0": # White won
        return 1.0 if player == WHITE else -1.0
    elif result == "0-1": # Black won
        return -1.0 if player == WHITE else 1.0
    elif result == "1/2-1/2": # Draw
        return 1e-4

def get_canonical_form(board: Board, player: bool) -> Board:
    """
    Returns the canonical form of the board.
    """
    if (player == WHITE):
        return board
    else:
        new_board = board.mirror()
        new_board.apply_transform(chess.flip_horizontal)
        return new_board


def get_fen(board: Board) -> str:
    """
    Returns a FEN representation of the board
    """
    return board.fen()

def get_model_representation(board: Board) -> np.ndarray:
    """
    Returns a representation of the board suitable for input to a neural network.
    Pawns   - Plane 0
    Knights - Plane 1
    Bishops - Plane 2
    Rooks   - Plane 3
    Queens  - Plane 4
    Kings   - Plane 5
    Each piece is represented by a 1 if white, -1 if black
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

def load_pgn_game(pgn_file) -> chess.pgn.Game:
    """
    Returns a chess game object from the given pgn file
    """
    return chess.pgn.read_game(pgn_file)