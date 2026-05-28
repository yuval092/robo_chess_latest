import chess
import pytest

from src.chess_game.move_planner import (
    ArmMoveCommand,
    LogicalPieceTracker,
    MovePlanner,
    RemoveFromBoardCommand,
    TeleportCommand,
)


def planner_for(board, tracker):
    return MovePlanner(board, tracker)


def test_normal_move_emits_one_arm_command():
    board = chess.Board()
    tracker = LogicalPieceTracker()
    move = chess.Move.from_uci("e2e4")

    plan = planner_for(board, tracker).plan(move)

    assert plan == [ArmMoveCommand("white_pawn_e", "e2", "e4")]
    assert board.fen() == chess.STARTING_FEN


def test_capture_emits_remove_then_arm():
    board = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    tracker = LogicalPieceTracker.empty()
    tracker.set_piece_at("e4", "white_pawn_e")
    tracker.set_piece_at("d5", "black_pawn_d")

    plan = planner_for(board, tracker).plan(chess.Move.from_uci("e4d5"))

    assert plan == [
        RemoveFromBoardCommand("black_pawn_d", "slot_00"),
        ArmMoveCommand("white_pawn_e", "e4", "d5"),
    ]


def test_castling_emits_king_then_rook_arm_moves():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    tracker = LogicalPieceTracker.empty()
    tracker.set_piece_at("e1", "white_king")
    tracker.set_piece_at("h1", "white_rook_h")

    plan = planner_for(board, tracker).plan(chess.Move.from_uci("e1g1"))

    assert plan == [
        ArmMoveCommand("white_king", "e1", "g1"),
        ArmMoveCommand("white_rook_h", "h1", "f1"),
    ]


def test_en_passant_removes_passed_over_pawn():
    board = chess.Board("4k3/8/8/3pPp2/8/8/8/4K3 w - d6 0 1")
    tracker = LogicalPieceTracker.empty()
    tracker.set_piece_at("e5", "white_pawn_e")
    tracker.set_piece_at("d5", "black_pawn_d")

    plan = planner_for(board, tracker).plan(chess.Move.from_uci("e5d6"))

    assert plan == [
        RemoveFromBoardCommand("black_pawn_d", "slot_00"),
        ArmMoveCommand("white_pawn_e", "e5", "d6"),
    ]


def test_promotion_emits_arm_and_two_teleports():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    tracker = LogicalPieceTracker.empty()
    tracker.set_piece_at("a7", "white_pawn_a")

    plan = planner_for(board, tracker).plan(chess.Move.from_uci("a7a8q"))

    assert plan == [
        ArmMoveCommand("white_pawn_a", "a7", "a8"),
        TeleportCommand("white_pawn_a", "promotion_reserve", "slot_00"),
        TeleportCommand("white_reserve_queen_1", "square", "a8"),
    ]


def test_capture_promotion_removes_capture_first():
    board = chess.Board("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    tracker = LogicalPieceTracker.empty()
    tracker.set_piece_at("a7", "white_pawn_a")
    tracker.set_piece_at("b8", "black_rook_b")

    plan = planner_for(board, tracker).plan(chess.Move.from_uci("a7b8q"))

    assert plan == [
        RemoveFromBoardCommand("black_rook_b", "slot_00"),
        ArmMoveCommand("white_pawn_a", "a7", "b8"),
        TeleportCommand("white_pawn_a", "promotion_reserve", "slot_00"),
        TeleportCommand("white_reserve_queen_1", "square", "b8"),
    ]


def test_illegal_move_rejected_before_planning():
    board = chess.Board()

    with pytest.raises(ValueError):
        MovePlanner(board, LogicalPieceTracker()).plan(chess.Move.from_uci("e2e5"))


def test_planner_never_emits_arm_commands_to_off_board_locations():
    board = chess.Board("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    tracker = LogicalPieceTracker.empty()
    tracker.set_piece_at("a7", "white_pawn_a")
    tracker.set_piece_at("b8", "black_rook_b")

    plan = MovePlanner(board, tracker).plan(chess.Move.from_uci("a7b8q"))

    for command in plan:
        if isinstance(command, ArmMoveCommand):
            assert command.src_square in chess.SQUARE_NAMES
            assert command.dst_square in chess.SQUARE_NAMES
