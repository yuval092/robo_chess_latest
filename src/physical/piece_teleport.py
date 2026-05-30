"""Free-joint teleport helpers for instantaneous chess piece repositioning."""

from __future__ import annotations

import re

import chess
import mujoco
import numpy as np

from src.chess_env.simulation import IDENTITY_QUAT, unwrap_env
from src.chess_game.board_mapper import BoardMapper
from src.utils.io import load_config


class PieceTeleporter:
    """Freejoint teleport helpers for chess piece bodies."""

    def __init__(self, env, board_mapper: BoardMapper | None = None):
        self.env = unwrap_env(env)
        self.board_mapper = board_mapper or BoardMapper.from_configs()
        self.chess_cfg = load_config("chess")

    def teleport_piece_to_xyz(self, piece_id: str, xyz: np.ndarray) -> None:
        joint_id = self.env.model.joint(f"piece_{piece_id}:joint").id
        qpos_start = self.env.model.jnt_qposadr[joint_id]
        dof_start  = self.env.model.jnt_dofadr[joint_id]

        xyz_slice  = slice(qpos_start,     qpos_start + 3)  # free joint: 3 position values
        quat_slice = slice(qpos_start + 3, qpos_start + 7)  # free joint: 4 quaternion values
        vel_slice  = slice(dof_start,      dof_start  + 6)  # free joint: 6 velocity DOFs

        self.env.data.qpos[xyz_slice]  = np.asarray(xyz, dtype=float)
        self.env.data.qpos[quat_slice] = IDENTITY_QUAT
        self.env.data.qvel[vel_slice]  = 0.0
        self.env.data.qacc[vel_slice]  = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

    def teleport_piece_to_square(self, piece_id: str, square: str) -> None:
        xyz = self.board_mapper.square_to_piece_xyz(chess.parse_square(square))
        self.teleport_piece_to_xyz(piece_id, xyz)

    def teleport_piece_to_graveyard(self, piece_id: str, slot_id: str) -> None:
        color = self._piece_color(piece_id)
        xyz = self._get_slot_xyz(
            self.chess_cfg["graveyards"][color],
            self.chess_cfg["reserves"]["graveyard_slot_spacing_m"],
            slot_id,
        )
        self.teleport_piece_to_xyz(piece_id, xyz)

    def teleport_piece_to_promotion_reserve(self, piece_id: str, slot_id: str) -> None:
        color = self._piece_color(piece_id)
        xyz = self._get_slot_xyz(
            self.chess_cfg["promotion_reserve"][color],
            self.chess_cfg["reserves"]["promotion_slot_spacing_m"],
            slot_id,
        )
        self.teleport_piece_to_xyz(piece_id, xyz)

    @staticmethod
    def _piece_color(piece_id: str) -> str:
        if piece_id.startswith("white_"):
            return "white"
        if piece_id.startswith("black_"):
            return "black"
        raise ValueError(f"Cannot infer color from piece id {piece_id}")

    @staticmethod
    def _parse_slot_index(slot_id: str) -> int:
        """Parse a slot id like 'slot_3' into its integer index."""
        match = re.fullmatch(r"slot_(\d+)", slot_id)
        if match is None:
            raise ValueError(f"Slot id must be in format 'slot_NN': {slot_id!r}")
        return int(match.group(1))

    def _get_slot_xyz(self, slot_layout: dict, slot_gap_m: float, slot_id: str) -> np.ndarray:
        """Return the XYZ position of a slot in a storage area.

        Slots are arranged in a grid (like a matrix). slot_layout defines the number of rows/cols
        and the origin. slot_gap_m is the distance between adjacent slot centres.
        """
        slot = self._parse_slot_index(slot_id)
        row = slot // slot_layout["cols"]
        col = slot % slot_layout["cols"]
        if row >= slot_layout["rows"]:
            raise ValueError(f"Slot {slot_id} is outside configured slot grid")
        origin = slot_layout["origin_xyz"]
        return np.array(
            [origin[0] + row * slot_gap_m, origin[1] + col * slot_gap_m, origin[2]],
            dtype=float,
        )
