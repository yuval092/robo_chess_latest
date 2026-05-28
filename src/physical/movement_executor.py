"""Board-to-board arm movement executor with occupancy reconciliation."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from src.chess_game.board_mapper import BoardMapper
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_teleport import IDENTITY_QUAT, PieceTeleporter
from src.utils.io import load_config

M_TO_MM = 1000.0


@dataclass
class PhysicalMoveResult:
    success: bool
    piece_id: str
    src_square: str | None
    dst_square: str | None
    stage_results: list
    error: str | None = None


class MovementExecutor:
    """Physical board-to-board move executor for selected chess pieces."""

    def __init__(
        self, env, controller, board_mapper: BoardMapper, occupancy: PhysicalOccupancy
    ):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.controller = controller
        self.board_mapper = board_mapper
        self.occupancy = occupancy
        self.teleporter = PieceTeleporter(env, board_mapper)
        cfg = load_config("env")
        self._reconcile_xy_tol = cfg["reconcile_xy_tolerance_m"]
        self._reconcile_z_tol = cfg["reconcile_z_tolerance_m"]

    def move_piece_between_squares(
        self, piece_id: str, src_square: str, dst_square: str
    ) -> PhysicalMoveResult:
        try:
            self.occupancy.assert_piece_at(piece_id, src_square)
            self.occupancy.assert_square_empty(dst_square)
        except ValueError as exc:
            return PhysicalMoveResult(
                False, piece_id, src_square, dst_square, [], str(exc)
            )

        src_xy = self.board_mapper.square_name_to_xy(src_square)
        dst_xy = self.board_mapper.square_name_to_xy(dst_square)
        return self.move_piece_xy(
            piece_id, src_xy, dst_xy, src_square, dst_square
        )

    def move_piece_xy(
        self,
        piece_id: str,
        src_xy: np.ndarray,
        dst_xy: np.ndarray,
        src_square: str | None = None,
        dst_square: str | None = None,
    ) -> PhysicalMoveResult:
        self.env.set_active_piece(piece_id)
        sequence_result = self.controller.run_full_move(src_xy, dst_xy)

        if not sequence_result.success:
            failed_stage = sequence_result.failed_at or "move"
            reason = next(
                (sr.crash_reason for name, sr in sequence_result.stage_results if name == failed_stage),
                None,
            )
            error = f"{failed_stage}: {reason}" if reason else failed_stage
            return PhysicalMoveResult(False, piece_id, src_square, dst_square, sequence_result.stage_results, error)

        landing_failure = self._check_landing_tolerance(
            piece_id, dst_xy, src_square, dst_square, sequence_result.stage_results
        )
        if landing_failure is not None:
            return landing_failure

        self._reconcile_placement(piece_id, dst_square)
        return PhysicalMoveResult(True, piece_id, src_square, dst_square, sequence_result.stage_results)

    def _check_landing_tolerance(
        self,
        piece_id: str,
        dst_xy: np.ndarray,
        src_square: str | None,
        dst_square: str | None,
        stage_results: list,
    ) -> PhysicalMoveResult | None:
        """Return a failure result if the piece landed too far from the target, else None."""
        piece_pos = self.env.get_active_piece_position()
        expected_z = self.env.TABLE_Z + self.env.CUBE_HEIGHT / 2.0
        xy_error = float(np.linalg.norm(piece_pos[:2] - dst_xy[:2]))
        z_error = float(abs(piece_pos[2] - expected_z))
        if xy_error > self._reconcile_xy_tol:
            return PhysicalMoveResult(
                False, piece_id, src_square, dst_square, stage_results,
                f"XY_LANDING_FAILED {xy_error * M_TO_MM:.1f}mm",
            )
        if z_error > self._reconcile_z_tol:
            return PhysicalMoveResult(
                False, piece_id, src_square, dst_square, stage_results,
                f"Z_LANDING_FAILED {z_error * M_TO_MM:.1f}mm",
            )
        return None

    def _reconcile_placement(self, piece_id: str, dst_square: str | None) -> None:
        """Snap the piece to the exact target square and update occupancy."""
        if dst_square is None:
            return
        placed_xyz = self.board_mapper.square_to_piece_xyz(chess.parse_square(dst_square))
        self.teleporter.teleport_piece_to_xyz(piece_id, placed_xyz, quat=IDENTITY_QUAT)
        self.occupancy.set_piece_square(piece_id, dst_square)
