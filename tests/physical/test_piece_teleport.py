import gymnasium as gym
import numpy as np

from src.chess_game.board_mapper import BoardMapper
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry
from src.physical.piece_teleport import PieceTeleporter


def piece_xyz(uw, piece_id):
    joint_name = f"piece_{piece_id}:joint"
    joint_id = uw.model.joint(joint_name).id
    qpos_start = uw.model.jnt_qposadr[joint_id]
    return uw.data.qpos[qpos_start : qpos_start + 3].copy()


def test_set_active_piece_routes_position_to_selected_piece():
    env = gym.make("ChessFetchTask-Play-v0", render_mode=None, show_chess_pieces=True)
    env.reset()
    uw = env.unwrapped

    uw.set_active_piece("white_pawn_e")

    assert np.allclose(uw.get_active_piece_position(), piece_xyz(uw, "white_pawn_e"))
    env.close()


def test_teleport_piece_to_square_sets_pose_and_zeroes_velocity():
    env = gym.make("ChessFetchTask-Play-v0", render_mode=None, show_chess_pieces=True)
    env.reset()
    uw = env.unwrapped
    mapper = BoardMapper.from_configs()
    teleporter = PieceTeleporter(env, mapper)

    teleporter.teleport_piece_to_square("white_pawn_e", "e4")

    assert np.allclose(
        piece_xyz(uw, "white_pawn_e"),
        mapper.square_to_piece_xyz(__import__("chess").E4),
    )
    joint_id = uw.model.joint("piece_white_pawn_e:joint").id
    dof_start = uw.model.jnt_dofadr[joint_id]
    assert np.allclose(uw.data.qvel[dof_start : dof_start + 6], 0.0)
    env.close()


def test_teleport_piece_to_graveyard_slot():
    env = gym.make("ChessFetchTask-Play-v0", render_mode=None, show_chess_pieces=True)
    env.reset()
    teleporter = PieceTeleporter(env)

    teleporter.teleport_piece_to_graveyard("white_pawn_e", "slot_03")

    assert np.allclose(
        piece_xyz(env.unwrapped, "white_pawn_e"), [0.640, -0.300 + 3 * 0.045, 0.015]
    )
    env.close()


def test_physical_occupancy_rejects_collisions():
    occupancy = PhysicalOccupancy(PieceRegistry().starting_square_map())

    assert occupancy.piece_at_square("e2") == "white_pawn_e"
    occupancy.assert_piece_at("white_pawn_e", "e2")
    occupancy.assert_square_empty("e4")
    occupancy.set_piece_square("white_pawn_e", "e4")
    assert occupancy.square_of_piece("white_pawn_e") == "e4"

    try:
        occupancy.set_piece_square("white_pawn_d", "e4")
    except ValueError as exc:
        assert "occupied" in str(exc)
    else:
        raise AssertionError("Expected occupied square rejection")
