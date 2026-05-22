import chess
import gymnasium as gym
import mujoco
import numpy as np

import src.chess_env
from src.chess_game.board_mapper import BoardMapper
from src.physical.piece_registry import PieceRegistry, reserve_piece_ids
from src.utils.config import load_config


def qpos_xyz(uw, joint_name):
    joint_id = uw.model.joint(joint_name).id
    qpos_start = uw.model.jnt_qposadr[joint_id]
    return uw.data.qpos[qpos_start : qpos_start + 3].copy()


def test_all_active_and_reserve_piece_bodies_exist():
    env = gym.make("ChessFetchTask-v0", render_mode=None, show_chess_pieces=True)
    env.reset()
    uw = env.unwrapped
    registry = PieceRegistry()

    for piece in registry.all_pieces():
        assert mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_BODY, piece.body_name) != -1
        assert mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_JOINT, piece.joint_name) != -1
        assert mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_GEOM, piece.cube_geom_name) != -1
        assert mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_GEOM, piece.visual_geom_name) != -1

    for piece_id, _, _ in reserve_piece_ids():
        assert mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_BODY, f"piece_{piece_id}") != -1

    env.close()


def test_piece_joint_cube_and_visual_properties():
    env = gym.make("ChessFetchTask-v0", render_mode=None, show_chess_pieces=True)
    env.reset()
    uw = env.unwrapped
    registry = PieceRegistry()
    damping = load_config("chess")["pieces"]["freejoint_damping"]

    for piece in registry.all_pieces():
        joint_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_JOINT, piece.joint_name)
        assert uw.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        dof_start = uw.model.jnt_dofadr[joint_id]
        assert np.allclose(uw.model.dof_damping[dof_start : dof_start + 6], damping)

        cube_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_GEOM, piece.cube_geom_name)
        assert np.allclose(uw.model.geom_size[cube_id], [0.015, 0.015, 0.015])
        assert uw.model.geom_contype[cube_id] != 0
        assert uw.model.geom_conaffinity[cube_id] != 0

        visual_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_GEOM, piece.visual_geom_name)
        assert uw.model.geom_contype[visual_id] == 0
        assert uw.model.geom_conaffinity[visual_id] == 0

    env.close()


def test_each_piece_type_uses_correct_mesh():
    env = gym.make("ChessFetchTask-v0", render_mode=None, show_chess_pieces=True)
    env.reset()
    uw = env.unwrapped
    registry = PieceRegistry()
    type_to_mesh = {
        "pawn": "chess_pawn_mesh",
        "rook": "chess_rook_mesh",
        "knight": "chess_knight_mesh",
        "bishop": "chess_bishop_mesh",
        "queen": "chess_queen_mesh",
        "king": "chess_king_mesh",
    }

    for piece in registry.all_pieces():
        expected_mesh = type_to_mesh[piece.piece_type]
        visual_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_GEOM, piece.visual_geom_name)
        mesh_id = uw.model.geom_dataid[visual_id]
        actual_mesh_name = mujoco.mj_id2name(uw.model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        assert actual_mesh_name == expected_mesh, (
            f"{piece.piece_id}: expected mesh {expected_mesh}, got {actual_mesh_name}"
        )

    env.close()


def test_show_chess_pieces_reset_places_active_pieces_on_starting_squares():
    env = gym.make("ChessFetchTask-v0", render_mode=None, show_chess_pieces=True)
    env.reset()
    uw = env.unwrapped
    mapper = BoardMapper.from_configs()
    registry = PieceRegistry()

    for piece in registry.all_pieces():
        expected = mapper.square_to_piece_xyz(chess.parse_square(piece.initial_square))
        actual = qpos_xyz(uw, piece.joint_name)
        assert np.linalg.norm(actual - expected) < 0.002

    env.close()


def test_default_reset_hides_active_pieces_for_legacy_object0_evaluations():
    env = gym.make("ChessFetchTask-v0", render_mode=None)
    env.reset()
    uw = env.unwrapped
    registry = PieceRegistry()

    for piece in registry.all_pieces():
        actual = qpos_xyz(uw, piece.joint_name)
        assert actual[0] > 2.0
        assert actual[2] < 0.05

    env.close()
