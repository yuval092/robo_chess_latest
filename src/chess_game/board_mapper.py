from __future__ import annotations

from dataclasses import dataclass
import math

import chess
import numpy as np

from src.utils.config import load_config


@dataclass(frozen=True)
class BoardGeometry:
    center_xy: tuple[float, float]
    cell_size_m: float
    board_size: int
    table_surface_z: float
    cube_height_m: float
    table_half_x: float
    table_half_y: float


class BoardMapper:
    """Maps python-chess squares to world-frame table coordinates."""

    REQUIRED_CELL_SIZE_M = 0.08
    REQUIRED_BOARD_WIDTH_M = 0.64
    REQUIRED_TABLE_MARGIN_M = 0.03
    TOLERANCE_M = 1e-9

    def __init__(self, geometry: BoardGeometry):
        self.geometry = geometry
        self._validate_geometry()

    @classmethod
    def from_configs(cls) -> "BoardMapper":
        chess_cfg = load_config("chess")
        env_cfg = load_config("env")
        board_cfg = chess_cfg["board"]
        pieces_cfg = chess_cfg["pieces"]
        geometry = BoardGeometry(
            center_xy=tuple(board_cfg["center_xy"]),
            cell_size_m=float(board_cfg["cell_size_m"]),
            board_size=int(board_cfg["board_size"]),
            table_surface_z=float(env_cfg["table_surface_z"]),
            cube_height_m=float(pieces_cfg["cube_height_m"]),
            table_half_x=float(env_cfg["table_half_x"]),
            table_half_y=float(env_cfg["table_half_y"]),
        )
        return cls(geometry)

    @property
    def board_width_m(self) -> float:
        return self.geometry.board_size * self.geometry.cell_size_m

    @property
    def board_min_xy(self) -> np.ndarray:
        half = self.board_width_m / 2.0
        return np.array([self.geometry.center_xy[0] - half, self.geometry.center_xy[1] - half])

    @property
    def board_max_xy(self) -> np.ndarray:
        half = self.board_width_m / 2.0
        return np.array([self.geometry.center_xy[0] + half, self.geometry.center_xy[1] + half])

    def square_to_xy(self, square: chess.Square) -> np.ndarray:
        rank_index = chess.square_rank(square)
        file_index = chess.square_file(square)
        min_xy = self.board_min_xy
        return np.array(
            [
                min_xy[0] + (rank_index + 0.5) * self.geometry.cell_size_m,
                min_xy[1] + (file_index + 0.5) * self.geometry.cell_size_m,
            ],
            dtype=float,
        )

    def square_name_to_xy(self, square_name: str) -> np.ndarray:
        return self.square_to_xy(chess.parse_square(square_name))

    def square_to_piece_xyz(self, square: chess.Square) -> np.ndarray:
        xy = self.square_to_xy(square)
        z = self.geometry.table_surface_z + self.geometry.cube_height_m / 2.0
        return np.array([xy[0], xy[1], z], dtype=float)

    def all_square_centers(self) -> dict[str, np.ndarray]:
        return {chess.square_name(square): self.square_to_xy(square) for square in chess.SQUARES}

    def nearest_square(self, xy: np.ndarray) -> chess.Square:
        xy = np.asarray(xy, dtype=float)
        self.assert_on_board(xy)
        min_xy = self.board_min_xy
        offset = (xy[:2] - min_xy) / self.geometry.cell_size_m
        rank_index = min(max(int(math.floor(offset[0])), 0), self.geometry.board_size - 1)
        file_index = min(max(int(math.floor(offset[1])), 0), self.geometry.board_size - 1)
        return chess.square(file_index, rank_index)

    def assert_on_board(self, xy: np.ndarray) -> None:
        xy = np.asarray(xy, dtype=float)
        min_xy = self.board_min_xy
        max_xy = self.board_max_xy
        if not (min_xy[0] <= xy[0] <= max_xy[0] and min_xy[1] <= xy[1] <= max_xy[1]):
            raise ValueError(
                f"XY {xy[:2].tolist()} is outside chess board bounds "
                f"{min_xy.tolist()}..{max_xy.tolist()}"
            )

    def _validate_geometry(self) -> None:
        g = self.geometry
        if not math.isclose(g.cell_size_m, self.REQUIRED_CELL_SIZE_M, abs_tol=self.TOLERANCE_M):
            raise ValueError(f"Chess cell size must be exactly 0.08m, got {g.cell_size_m}")
        if g.board_size != 8:
            raise ValueError(f"Chess board size must be 8, got {g.board_size}")
        if not math.isclose(self.board_width_m, self.REQUIRED_BOARD_WIDTH_M, abs_tol=self.TOLERANCE_M):
            raise ValueError(f"Chess board width must be 0.64m, got {self.board_width_m}")

        table_min = np.array([g.center_xy[0] - g.table_half_x, g.center_xy[1] - g.table_half_y])
        table_max = np.array([g.center_xy[0] + g.table_half_x, g.center_xy[1] + g.table_half_y])
        board_min = self.board_min_xy
        board_max = self.board_max_xy
        margins = np.array(
            [
                board_min[0] - table_min[0],
                table_max[0] - board_max[0],
                board_min[1] - table_min[1],
                table_max[1] - board_max[1],
            ]
        )
        if np.any(margins + self.TOLERANCE_M < self.REQUIRED_TABLE_MARGIN_M):
            raise ValueError(f"Chess board must leave at least 0.03m table margin, got {margins.tolist()}")
