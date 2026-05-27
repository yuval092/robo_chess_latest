import gymnasium as gym
import numpy as np
import pytest

from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_game.board_mapper import BoardMapper
from src.physical.movement_executor import MovementExecutor
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry
from src.utils.args import resolve_model_paths


def piece_xyz(uw, piece):
    joint_id = uw.model.joint(piece.joint_name).id
    qpos_start = uw.model.jnt_qposadr[joint_id]
    return uw.data.qpos[qpos_start : qpos_start + 3].copy()


def make_executor(env, registry):
    model_paths = resolve_model_paths()
    controller = ModelEmbeddedController(env)
    controller.load_all(
        model_paths["transit"],
        model_paths["descend"],
        model_paths["ascend"],
    )
    return MovementExecutor(
        env,
        controller,
        BoardMapper.from_configs(),
        PhysicalOccupancy(registry.starting_square_map()),
    )


@pytest.mark.parametrize(
    ("piece_id", "src", "dst", "dst_square"),
    [
        ("white_pawn_e", "e2", "e4", __import__("chess").E4),
        ("white_knight_g", "g1", "f3", __import__("chess").F3),
        ("white_queen", "d1", "h5", __import__("chess").H5),
    ],
)
def test_move_piece_in_crowded_starting_position(piece_id, src, dst, dst_square):
    env = gym.make(
        "ChessFetchTask-v0",
        render_mode=None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
    )
    env.reset()
    registry = PieceRegistry()
    starts = {
        piece.piece_id: piece_xyz(env.unwrapped, piece)
        for piece in registry.all_pieces()
    }

    result = make_executor(env, registry).move_piece_between_squares(piece_id, src, dst)

    assert result.success, result.error
    mapper = BoardMapper.from_configs()
    moved_pos = piece_xyz(env.unwrapped, registry.by_id(piece_id))
    assert np.linalg.norm(moved_pos - mapper.square_to_piece_xyz(dst_square)) < 0.002

    for piece in registry.all_pieces():
        if piece.piece_id == piece_id:
            continue
        assert (
            np.linalg.norm(piece_xyz(env.unwrapped, piece) - starts[piece.piece_id])
            < 0.002
        )

    env.close()


def test_move_rejects_empty_source_and_occupied_destination_before_motion():
    env = gym.make(
        "ChessFetchTask-v0",
        render_mode=None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
    )
    env.reset()
    registry = PieceRegistry()
    executor = make_executor(env, registry)

    wrong_source = executor.move_piece_between_squares("white_pawn_e", "e3", "e4")
    occupied_dst = executor.move_piece_between_squares("white_pawn_e", "e2", "e7")

    assert not wrong_source.success
    assert "Expected white_pawn_e at e3" in wrong_source.error
    assert not occupied_dst.success
    assert "occupied" in occupied_dst.error
    env.close()
