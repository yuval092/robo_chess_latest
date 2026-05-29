"""Board-to-world coordinate mapper for the RoboChess chess board."""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from src.utils.io import load_config


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

    def __init__(self, geometry: BoardGeometry):
        self.geometry = geometry

    @classmethod
    def from_configs(cls) -> BoardMapper:
        """Construct this object from project configuration files."""
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
        """Return the total board width in metres."""
        return self.geometry.board_size * self.geometry.cell_size_m

    @property
    def board_min_xy(self) -> np.ndarray:
        """Return the lower-left board corner in world XY coordinates."""
        half = self.board_width_m / 2.0
        return np.array(
            [self.geometry.center_xy[0] - half, self.geometry.center_xy[1] - half]
        )

    def square_to_xy(self, square: chess.Square) -> np.ndarray:
        """Return the world XY centre for a chess square index."""
        rank_index = chess.square_rank(square)
        file_index = chess.square_file(square)
        min_xy = self.board_min_xy
        return np.array(
            [
                min_xy[0] + (file_index + 0.5) * self.geometry.cell_size_m,
                min_xy[1] + (rank_index + 0.5) * self.geometry.cell_size_m,
            ],
            dtype=float,
        )

    def square_name_to_xy(self, square_name: str) -> np.ndarray:
        """Return the world XY centre for an algebraic square name."""
        return self.square_to_xy(chess.parse_square(square_name))

    def square_to_piece_xyz(self, square: chess.Square) -> np.ndarray:
        """Return the world XYZ piece centre for a chess square."""
        xy = self.square_to_xy(square)
        z = self.geometry.table_surface_z + self.geometry.cube_height_m / 2.0
        return np.array([xy[0], xy[1], z], dtype=float)

    def all_square_centers(self) -> dict[str, np.ndarray]:
        """Return all board square centres keyed by square name."""
        return {
            chess.square_name(square): self.square_to_xy(square)
            for square in chess.SQUARES
        }
