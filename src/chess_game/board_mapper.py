"""Board-to-world coordinate mapper for the RoboChess chess board."""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class BoardValidation:
    required_cell_size_m: float
    required_board_width_m: float
    required_table_margin_m: float
    geometry_tolerance_m: float


_DEFAULT_VALIDATION = BoardValidation(
    required_cell_size_m=0.08,
    required_board_width_m=0.64,
    required_table_margin_m=0.03,
    geometry_tolerance_m=1.0e-9,
)


class BoardMapper:
    """Maps python-chess squares to world-frame table coordinates."""

    def __init__(
        self, geometry: BoardGeometry, validation: BoardValidation = _DEFAULT_VALIDATION
    ):
        self.geometry = geometry
        self.validation = validation
        self._validate_geometry()

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
        val_cfg = board_cfg["validation"]
        validation = BoardValidation(
            required_cell_size_m=float(val_cfg["required_cell_size_m"]),
            required_board_width_m=float(val_cfg["required_board_width_m"]),
            required_table_margin_m=float(val_cfg["required_table_margin_m"]),
            geometry_tolerance_m=float(val_cfg["geometry_tolerance_m"]),
        )
        return cls(geometry, validation)

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

    @property
    def board_max_xy(self) -> np.ndarray:
        """Return the upper-right board corner in world XY coordinates."""
        half = self.board_width_m / 2.0
        return np.array(
            [self.geometry.center_xy[0] + half, self.geometry.center_xy[1] + half]
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

    def nearest_square(self, xy: np.ndarray) -> chess.Square:
        """Return the nearest chess square for a world XY coordinate."""
        xy = np.asarray(xy, dtype=float)
        self.assert_on_board(xy)
        min_xy = self.board_min_xy
        offset = (xy[:2] - min_xy) / self.geometry.cell_size_m
        file_index = min(
            max(int(math.floor(offset[0])), 0), self.geometry.board_size - 1
        )
        rank_index = min(
            max(int(math.floor(offset[1])), 0), self.geometry.board_size - 1
        )
        return chess.square(file_index, rank_index)

    def assert_on_board(self, xy: np.ndarray) -> None:
        """Raise if a world XY coordinate is outside the board."""
        xy = np.asarray(xy, dtype=float)
        min_xy = self.board_min_xy
        max_xy = self.board_max_xy
        if not (min_xy[0] <= xy[0] <= max_xy[0] and min_xy[1] <= xy[1] <= max_xy[1]):
            raise ValueError(
                f"XY {xy[:2].tolist()} is outside chess board bounds "
                f"{min_xy.tolist()}..{max_xy.tolist()}"
            )

    def _validate_geometry(self) -> None:
        """Validate loaded board geometry invariants."""
        g = self.geometry
        v = self.validation
        if not math.isclose(
            g.cell_size_m, v.required_cell_size_m, abs_tol=v.geometry_tolerance_m
        ):
            raise ValueError(
                f"Chess cell size must be exactly 0.08m, got {g.cell_size_m}"
            )
        if g.board_size != 8:
            raise ValueError(f"Chess board size must be 8, got {g.board_size}")
        if not math.isclose(
            self.board_width_m,
            v.required_board_width_m,
            abs_tol=v.geometry_tolerance_m,
        ):
            raise ValueError(
                f"Chess board width must be 0.64m, got {self.board_width_m}"
            )

        table_min = np.array(
            [g.center_xy[0] - g.table_half_x, g.center_xy[1] - g.table_half_y]
        )
        table_max = np.array(
            [g.center_xy[0] + g.table_half_x, g.center_xy[1] + g.table_half_y]
        )
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
        if np.any(margins + v.geometry_tolerance_m < v.required_table_margin_m):
            raise ValueError(
                f"Chess board must leave at least 0.03m table margin, got {margins.tolist()}"
            )
