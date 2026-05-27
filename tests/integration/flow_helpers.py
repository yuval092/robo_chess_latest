"""
Integration test helpers for multi-move chess game flow evaluation.

Ported from the former scripts/eval_chess_game_flow.py (scripts/ removed).
"""

from __future__ import annotations

import chess
import numpy as np

from src.chess_game.board_mapper import BoardMapper
from src.chess_game.chess_service import ChessService
from src.chess_game.move_planner import (
    ArmMoveCommand,
    LogicalPieceTracker,
    MovePlanner,
    RemoveFromBoardCommand,
    TeleportCommand,
)
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry

POSITION_TOLERANCE_MM = 5.0


# ── Helpers ────────────────────────────────────────────────────────────────

def piece_position(env, piece_id: str) -> np.ndarray:
    inner = env.unwrapped if hasattr(env, "unwrapped") else env
    previous = inner.active_piece_id
    inner.set_active_piece(piece_id)
    position = inner.get_active_piece_position()
    if previous is None:
        inner.clear_active_piece()
    else:
        inner.set_active_piece(previous)
    return position


def touched_piece_ids(plan) -> set[str]:
    touched = set()
    for command in plan.commands:
        if isinstance(command, ArmMoveCommand):
            touched.add(command.piece_id)
        elif isinstance(command, RemoveFromBoardCommand):
            touched.add(command.piece_id)
        elif isinstance(command, TeleportCommand):
            touched.add(command.piece_id)
    return touched


def command_signature(command) -> str:
    if isinstance(command, ArmMoveCommand):
        return f"ARM {command.piece_id} {command.src_square}->{command.dst_square}"
    if isinstance(command, RemoveFromBoardCommand):
        return f"REMOVE {command.piece_id} graveyard={command.graveyard_slot}"
    if isinstance(command, TeleportCommand):
        return f"TELEPORT {command.piece_id} {command.destination_kind}:{command.destination_id}"
    return repr(command)


def piece_type_from_id(piece_id: str) -> str:
    parts = piece_id.split("_")
    if "reserve" in parts:
        return parts[2]
    return parts[1]


def assert_tracker_matches_board(board: chess.Board, tracker: LogicalPieceTracker) -> None:
    expected_type = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }
    for square in chess.SQUARE_NAMES:
        board_piece = board.piece_at(chess.parse_square(square))
        tracked_piece_id = tracker.piece_id_at(square)
        if board_piece is None:
            if tracked_piece_id is not None:
                raise SystemExit(
                    f"Tracker has {tracked_piece_id} on empty logical square {square}"
                )
            continue
        if tracked_piece_id is None:
            raise SystemExit(f"Tracker is missing {board_piece.symbol()} on {square}")
        color = "white" if board_piece.color == chess.WHITE else "black"
        piece_type = expected_type[board_piece.piece_type]
        if (
            not tracked_piece_id.startswith(f"{color}_")
            or piece_type_from_id(tracked_piece_id) != piece_type
        ):
            raise SystemExit(
                f"Tracker mismatch on {square}: board={board_piece.symbol()} "
                f"tracked={tracked_piece_id}"
            )


def assert_occupancy_matches_tracker(
    occupancy: PhysicalOccupancy, tracker: LogicalPieceTracker
) -> None:
    for square in chess.SQUARE_NAMES:
        tracked_piece_id = tracker.piece_id_at(square)
        physical_piece_id = occupancy.piece_at_square(square)
        if physical_piece_id != tracked_piece_id:
            raise SystemExit(
                f"Occupancy mismatch on {square}: tracker={tracked_piece_id} "
                f"physical={physical_piece_id}"
            )


def assert_physical_positions(
    env, mapper: BoardMapper, occupancy: PhysicalOccupancy, tolerance_mm: float
) -> None:
    for piece_id in PieceRegistry().starting_square_map():
        square = occupancy.square_of_piece(piece_id)
        if square is None:
            continue
        expected = mapper.square_to_piece_xyz(chess.parse_square(square))
        actual = piece_position(env, piece_id)
        error_mm = float(np.linalg.norm(actual - expected) * 1000.0)
        if error_mm > tolerance_mm:
            raise SystemExit(
                f"Physical position mismatch for {piece_id} on {square}: "
                f"{error_mm:.1f}mm > {tolerance_mm:.1f}mm"
            )


def assert_nonmoving_displacement(
    env, before: dict[str, np.ndarray], skipped: set[str], tolerance_mm: float
) -> None:
    worst_piece = None
    worst_mm = 0.0
    for piece_id, start_pos in before.items():
        if piece_id in skipped:
            continue
        displacement_mm = float(
            np.linalg.norm(piece_position(env, piece_id) - start_pos) * 1000.0
        )
        if displacement_mm > worst_mm:
            worst_piece = piece_id
            worst_mm = displacement_mm
    print(
        f"  nonmoving_max_displacement piece={worst_piece} displacement={worst_mm:.1f}mm"
    )
    if worst_mm > tolerance_mm:
        raise SystemExit(
            f"Non-moving displacement exceeded tolerance: {worst_piece} {worst_mm:.1f}mm"
        )


def print_failure_context(env, fen_before: str, uci: str, plan, result) -> None:
    print("Failure context:")
    print(f"  fen_before={fen_before}")
    print(f"  attempted_uci={uci}")
    print("  physical_plan:")
    for command in plan.commands:
        print(f"    - {command_signature(command)}")
    print(f"  physical_error={result.error}")
    for command, command_result in result.command_results:
        print(f"  failed_command={command_signature(command)}")
        if hasattr(command_result, "stage_results"):
            for stage_name, stage_result in command_result.stage_results:
                print(
                    f"    stage={stage_name} success={stage_result.success} "
                    f"error={stage_result.error_mm:.1f}mm reason={stage_result.crash_reason}"
                )
    if env is not None:
        inner = env.unwrapped if hasattr(env, "unwrapped") else env
        grip_pos = inner._utils.get_site_xpos(inner.model, inner.data, "robot0:grip")
        print(f"  active_piece={inner.active_piece_id}")
        if inner.active_piece_id is not None:
            print(f"  active_piece_pos={inner.get_active_piece_position().tolist()}")
        print(f"  grip_pos={grip_pos.tolist()}")


# ── Main flow runner ───────────────────────────────────────────────────────

def run_flow(
    moves: list[str],
    physical_executor,
    tracker: LogicalPieceTracker,
    env,
    tolerance_mm: float,
    *,
    mapper: BoardMapper | None = None,
    occupancy: PhysicalOccupancy | None = None,
    verify_agreement: bool = False,
) -> None:
    """
    Run a sequence of UCI chess moves through the full physical execution pipeline.

    Raises SystemExit on any physical failure or state consistency violation.
    """
    service = ChessService()
    occupancy_piece_ids = set(PieceRegistry().starting_square_map())
    for index, uci in enumerate(moves, start=1):
        fen_before = service.fen()
        move = service.validate_uci(uci)
        plan = MovePlanner(service.board, tracker).plan(move)
        before_positions = {}
        if env is not None:
            before_positions = {
                piece_id: piece_position(env, piece_id)
                for piece_id in occupancy_piece_ids
            }

        result = physical_executor.execute(plan)
        if not result.success:
            print_failure_context(env, fen_before, uci, plan, result)
            raise SystemExit(
                f"move {index} {uci} physical execution failed: {result.error}"
            )
        home_result = physical_executor.return_to_home()
        if not home_result.success:
            raise SystemExit(
                f"move {index} {uci} return home failed: {home_result.error}"
            )

        service.push(move)
        tracker.apply_committed_move(move, plan)
        assert_tracker_matches_board(service.board, tracker)
        if verify_agreement and occupancy is not None:
            assert_occupancy_matches_tracker(occupancy, tracker)
            if env is not None and mapper is not None:
                assert_physical_positions(env, mapper, occupancy, POSITION_TOLERANCE_MM)
        if env is not None:
            assert_nonmoving_displacement(
                env, before_positions, touched_piece_ids(plan), tolerance_mm
            )
        for command in plan.commands:
            if isinstance(command, RemoveFromBoardCommand):
                occupancy_piece_ids.discard(command.piece_id)
        print(f"move {index}: {uci} ok fen={service.fen()}")
