import chess
import numpy as np
import pytest

from src.chess_game.board_mapper import BoardMapper
from src.utils.config import load_config


def test_square_centers_are_64_unique_points():
    mapper = BoardMapper.from_configs()
    centers = mapper.all_square_centers()

    assert len(centers) == 64
    assert len({tuple(np.round(xy, 9)) for xy in centers.values()}) == 64


def test_all_square_centers_inside_table():
    mapper = BoardMapper.from_configs()
    cfg = load_config("env")
    cx, cy = cfg["table_center_xy"]
    hx, hy = cfg["table_half_x"], cfg["table_half_y"]

    for xy in mapper.all_square_centers().values():
        assert cx - hx < xy[0] < cx + hx
        assert cy - hy < xy[1] < cy + hy


def test_cell_spacing_is_8cm():
    mapper = BoardMapper.from_configs()

    assert np.isclose(mapper.square_name_to_xy("a2")[0] - mapper.square_name_to_xy("a1")[0], 0.08)
    assert np.isclose(mapper.square_name_to_xy("b1")[1] - mapper.square_name_to_xy("a1")[1], 0.08)


def test_board_does_not_use_env_edge_margin():
    mapper = BoardMapper.from_configs()
    env_cfg = load_config("env")
    cx, cy = env_cfg["table_center_xy"]
    hx, hy = env_cfg["table_half_x"], env_cfg["table_half_y"]
    margin = env_cfg["edge_margin"]

    edge_margin_a1 = np.array([cx - hx + margin + 0.5 * ((2 * (hx - margin)) / 8), cy])
    assert not np.isclose(mapper.square_name_to_xy("a1")[0], edge_margin_a1[0])


def test_a1_h1_a8_h8_positions_match_orientation():
    mapper = BoardMapper.from_configs()

    assert np.allclose(mapper.square_name_to_xy("a1"), [0.600, -0.0159])
    assert np.allclose(mapper.square_name_to_xy("h1"), [0.600, 0.5441])
    assert np.allclose(mapper.square_name_to_xy("a8"), [1.160, -0.0159])
    assert np.allclose(mapper.square_name_to_xy("h8"), [1.160, 0.5441])


@pytest.mark.parametrize("square_name", ["a1", "d4", "e5", "h8"])
def test_nearest_square_roundtrip(square_name):
    mapper = BoardMapper.from_configs()
    xy = mapper.square_name_to_xy(square_name)

    assert mapper.nearest_square(xy) == chess.parse_square(square_name)


def test_near_and_far_rank_centers_match_exact_8cm_geometry():
    mapper = BoardMapper.from_configs()

    assert np.isclose(mapper.square_name_to_xy("d1")[0], 0.600)
    assert np.isclose(mapper.square_name_to_xy("d8")[0], 1.160)
