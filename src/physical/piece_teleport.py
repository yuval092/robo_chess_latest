"""Free-joint teleport helpers for instantaneous chess piece repositioning."""

from __future__ import annotations

import re

import chess
import mujoco
import numpy as np

from src.chess_env.simulation import unwrap_env
from src.chess_game.board_mapper import BoardMapper
from src.utils.io import load_config

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


class PieceTeleporter:
    """Freejoint teleport helpers for chess piece bodies."""

    def __init__(self, env, board_mapper: BoardMapper | None = None):
        self.env = unwrap_env(env)
        self.board_mapper = board_mapper or BoardMapper.from_configs()
        self.chess_cfg = load_config("chess")

    def teleport_piece_to_xyz(
        self, piece_id: str, xyz: np.ndarray, quat: np.ndarray | None = None
    ) -> None:
        joint_name = f"piece_{piece_id}:joint"
        quat = IDENTITY_QUAT if quat is None else np.asarray(quat, dtype=float)
        xyz = np.asarray(xyz, dtype=float)
        joint_id = self.env.model.joint(joint_name).id
        qpos_start = self.env.model.jnt_qposadr[joint_id]
        dof_start = self.env.model.jnt_dofadr[joint_id]
        self.env.data.qpos[qpos_start : qpos_start + 3] = xyz
        self.env.data.qpos[qpos_start + 3 : qpos_start + 7] = quat
        self.env.data.qvel[dof_start : dof_start + 6] = 0.0
        self.env.data.qacc[dof_start : dof_start + 6] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

    def teleport_piece_to_square(self, piece_id: str, square: str) -> None:
        xyz = self.board_mapper.square_to_piece_xyz(chess.parse_square(square))
        self.teleport_piece_to_xyz(piece_id, xyz)

    def teleport_piece_to_graveyard(self, piece_id: str, slot_id: str) -> None:
        color = self._piece_color(piece_id)
        xyz = self._slot_xyz(
            self.chess_cfg["graveyards"][color],
            self.chess_cfg["reserves"]["graveyard_slot_spacing_m"],
            slot_id,
        )
        self.teleport_piece_to_xyz(piece_id, xyz)

    def teleport_piece_to_promotion_reserve(self, piece_id: str, slot_id: str) -> None:
        color = self._piece_color(piece_id)
        xyz = self._slot_xyz(
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
    def _slot_index(slot_id: str) -> int:
        match = re.fullmatch(r"slot_(\d+)", slot_id)
        if match is None:
            raise ValueError(f"Slot id must be in format 'slot_NN': {slot_id!r}")
        return int(match.group(1))

    def _slot_xyz(self, cfg: dict, spacing: float, slot_id: str) -> np.ndarray:
        slot = self._slot_index(slot_id)
        row = slot // cfg["cols"]
        col = slot % cfg["cols"]
        if row >= cfg["rows"]:
            raise ValueError(f"Slot {slot_id} is outside configured slot grid")
        origin = cfg["origin_xyz"]
        return np.array(
            [origin[0] + row * spacing, origin[1] + col * spacing, origin[2]],
            dtype=float,
        )
