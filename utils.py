import numpy as np
import torch
from chess import (
    BLACK,
    PAWN,
    QUEEN,
    WHITE,
    Board,
    Move,
    square,
    square_file,
    square_mirror,
    square_rank,
)

DIRECTIONS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
KNIGHT_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
ACTION_SIZE = 4672

STEP_HISTORY = 3
NUM_PLANES = STEP_HISTORY * 12 + 4


def move_mirror(move: Move) -> Move:
    return Move(
        square_mirror(move.from_square), square_mirror(move.to_square), move.promotion
    )


def move_to_action(move: Move, board: Board) -> int:
    move = move if board.turn == WHITE else move_mirror(move)

    from_square = move.from_square
    to_square = move.to_square

    from_file, from_rank = square_file(from_square), square_rank(from_square)
    to_file, to_rank = square_file(to_square), square_rank(to_square)

    delta_file, delta_rank = to_file - from_file, to_rank - from_rank

    # Underpromotions (64 - 72)
    if move.promotion is not None and move.promotion != QUEEN:
        plane = 56 + 8 + 3 * (delta_file + 1) + move.promotion - 2

    # Knight moves (56 - 63)
    elif (delta_file, delta_rank) in KNIGHT_MOVES:
        plane = 56 + KNIGHT_MOVES.index((delta_file, delta_rank))

    # Queen Moves (0 - 55)
    else:
        direction = DIRECTIONS.index((np.sign(delta_file), np.sign(delta_rank)))
        distance = max(abs(delta_file), abs(delta_rank))

        plane = (7 * direction) + (distance - 1)

    return from_square * 73 + plane


def action_to_move(action: int, board: Board) -> Move:
    assert 0 <= action < ACTION_SIZE

    promotion = None

    plane = action % 73

    # Queen Moves
    if plane < 56:
        direction = plane // 7
        distance = (plane % 7) + 1

        delta_file, delta_rank = distance * np.array(DIRECTIONS[direction])
    # Knight Moves
    elif plane < 64:
        delta_file, delta_rank = KNIGHT_MOVES[plane - 56]
    # Underpromotions
    else:
        promotion = (plane - 64) % 3 + 2
        delta_file = (plane - 64) // 3 - 1
        delta_rank = 1

    # Get the square the piece is moving from
    from_square = action // 73
    from_file, from_rank = square_file(from_square), square_rank(from_square)

    # Calculate the square the piece is moving to
    to_file, to_rank = from_file + delta_file, from_rank + delta_rank
    to_square = square(to_file, to_rank)

    # Mirror the move if the player is black
    from_square = from_square if board.turn == WHITE else square_mirror(from_square)
    to_square = to_square if board.turn == WHITE else square_mirror(to_square)

    # Promote to a queen if the move is not an underpromotion and a pawn is moving to the first or last rank
    if (
        promotion == None
        and board.piece_at(from_square).piece_type == PAWN
        and to_rank in [0, 7]
    ):
        promotion = QUEEN

    return Move(from_square, to_square, promotion)


def get_canonical_form(board: Board) -> Board:
    if board.turn == WHITE:
        return board

    canonical_board = Board().mirror()

    for move in board.move_stack:
        canonical_board.push(move_mirror(move))

    return canonical_board


def get_model_form(board: Board) -> torch.Tensor:
    """
    Returns a representation of the board suitable for input to a neural network.
    """
    planes = torch.zeros((STEP_HISTORY * 12 + 4, 64), dtype=torch.float32)

    board_copy = board.copy(stack=STEP_HISTORY - 1)

    # Piece history
    for i in range(len(board_copy.move_stack) + 1):
        for square in range(64):
            if (piece := board_copy.piece_at(square)) != None:
                plane_index = i * 12 + (1 - piece.color) * 6 + (piece.piece_type - 1)
                planes[plane_index][square] = 1

        if len(board_copy.move_stack) > 0:
            board_copy.pop()

    # Castling rights
    planes[12 * STEP_HISTORY + 0] = board.has_kingside_castling_rights(WHITE)
    planes[12 * STEP_HISTORY + 1] = board.has_queenside_castling_rights(WHITE)
    planes[12 * STEP_HISTORY + 2] = board.has_kingside_castling_rights(BLACK)
    planes[12 * STEP_HISTORY + 3] = board.has_queenside_castling_rights(BLACK)

    return planes.reshape((NUM_PLANES, 8, 8))


def get_legal_actions(board: Board) -> np.ndarray:
    actions = np.zeros(ACTION_SIZE, dtype=np.float32)

    for move in board.legal_moves:
        actions[move_to_action(move, board)] = 1

    return actions


def play_game(white, black) -> str:
    board = Board()

    while not board.is_game_over():
        print(board)
        move = white(board) if board.turn == WHITE else black(board)

        board.push(move)
    return board.result()
