import chess
import gymnasium as gym
import mujoco
import numpy as np

import src.chess_env  # noqa: F401 - registers ChessFetchTask-v0
from src.chess_game.board_mapper import BoardMapper
from src.utils.io import load_config


def _geom_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def test_table_surface_geometry_and_legs() -> None:
    env = gym.make("ChessFetchTask-v0", render_mode=None)
    try:
        model = env.unwrapped.model
        cfg = load_config("env")
        assert cfg["table_half_x"] == 0.35
        assert cfg["table_half_y"] == 0.35

        for leg_name in (
            "table0_leg_far_plus",
            "table0_leg_far_minus",
            "table0_leg_near_plus",
            "table0_leg_near_minus",
        ):
            assert _geom_id(model, leg_name) != -1

        table_id = _body_id(model, "table0")
        surface_id = _geom_id(model, "table0_surface")
        surface_top = (
            model.body_pos[table_id][2]
            + model.geom_pos[surface_id][2]
            + model.geom_size[surface_id][2]
        )
        assert abs(surface_top - 0.400) < 0.001
    finally:
        env.close()


def test_board_visual_geoms_match_board_mapper() -> None:
    env = gym.make("ChessFetchTask-v0", render_mode=None)
    try:
        model = env.unwrapped.model
        mapper = BoardMapper.from_configs()
        table_center = np.array(load_config("env")["table_center_xy"])
        table_id = _body_id(model, "table0")

        max_center_error = 0.0
        for square in chess.SQUARES:
            square_name = chess.square_name(square)
            geom_id = _geom_id(model, f"board_{square_name}_visual")
            assert geom_id != -1
            assert model.geom_bodyid[geom_id] == table_id
            assert model.geom_contype[geom_id] == 0
            assert model.geom_conaffinity[geom_id] == 0

            center_xy = model.geom_pos[geom_id][:2] + table_center
            expected_xy = mapper.square_name_to_xy(square_name)
            max_center_error = max(
                max_center_error,
                float(np.linalg.norm(center_xy - expected_xy)),
            )

        assert max_center_error < 0.001
    finally:
        env.close()


def test_zone_geoms_are_world_space_visual_only_and_aligned() -> None:
    env = gym.make("ChessFetchTask-v0", render_mode=None)
    try:
        model = env.unwrapped.model
        cfg = load_config("chess")
        zones = [
            ("zone_white_graveyard", "graveyards", "white", cfg["reserves"]["graveyard_slot_spacing_m"]),
            ("zone_black_graveyard", "graveyards", "black", cfg["reserves"]["graveyard_slot_spacing_m"]),
            ("zone_white_reserve", "promotion_reserve", "white", cfg["reserves"]["promotion_slot_spacing_m"]),
            ("zone_black_reserve", "promotion_reserve", "black", cfg["reserves"]["promotion_slot_spacing_m"]),
        ]

        for geom_name, section, color, spacing in zones:
            geom_id = _geom_id(model, geom_name)
            assert geom_id != -1
            assert model.geom_bodyid[geom_id] == 0
            assert model.geom_contype[geom_id] == 0
            assert model.geom_conaffinity[geom_id] == 0

            zone_cfg = cfg[section][color]
            rows = zone_cfg["rows"]
            cols = zone_cfg["cols"]
            origin = zone_cfg["origin_xyz"]
            expected_pos = np.array(
                [
                    origin[0] + (rows - 1) * spacing / 2.0,
                    origin[1] + (cols - 1) * spacing / 2.0,
                    0.002,
                ]
            )
            expected_size = np.array(
                [
                    (rows * spacing) / 2.0 + 0.005,
                    (cols * spacing) / 2.0 + 0.005,
                    0.002,
                ]
            )
            assert np.allclose(model.geom_pos[geom_id], expected_pos, atol=1e-6)
            assert np.allclose(model.geom_size[geom_id], expected_size, atol=1e-6)
    finally:
        env.close()


def test_object0_static_stability() -> None:
    env = gym.make("ChessFetchTask-v0", render_mode=None, force_scenario="transit")
    try:
        env.reset()
        joint_id = env.unwrapped.model.joint("object0:joint").id
        qpos_start = env.unwrapped.model.jnt_qposadr[joint_id]
        start_pos = env.unwrapped.data.qpos[qpos_start : qpos_start + 3].copy()

        for _ in range(100):
            env.step(np.zeros(4))

        end_pos = env.unwrapped.data.qpos[qpos_start : qpos_start + 3].copy()
        assert np.linalg.norm(start_pos - end_pos) < 0.001
    finally:
        env.close()


def test_all_board_squares_are_kinematically_reachable() -> None:
    env = gym.make("ChessFetchTask-v0", render_mode=None)
    try:
        env.reset()
        cfg = load_config("env")
        mapper = BoardMapper.from_configs()
        max_err = 0.0

        for xy in mapper.all_square_centers().values():
            for z in (cfg["safe_z"], cfg["grasp_z"]):
                target = np.array([xy[0], xy[1], z])
                env.unwrapped._settle_arm_to_start(target)
                grip_pos = env.unwrapped._utils.get_site_xpos(
                    env.unwrapped.model,
                    env.unwrapped.data,
                    "robot0:grip",
                )
                max_err = max(max_err, float(np.linalg.norm(target - grip_pos)))

        assert max_err <= 0.005
    finally:
        env.close()


def test_grasp_xml_parameters() -> None:
    env = gym.make("ChessFetchTask-v0", render_mode=None)
    try:
        model = env.unwrapped.model
        cube_body_id = _body_id(model, "object0")
        assert abs(model.body_mass[cube_body_id] - 0.05) < 0.001

        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "robot0:l_gripper_finger_joint",
        )
        assert abs(model.actuator_gainprm[actuator_id, 0] - 20000) < 1.0
        assert abs(model.actuator_ctrlrange[actuator_id, 1] - 0.05) < 0.001
        assert abs(env.unwrapped.GRASP_Z - 0.430) < 0.001
    finally:
        env.close()


def test_hidden_object_teleport_verification() -> None:
    env = gym.make("ChessFetchTask-v0", render_mode=None, hide_object=True)
    try:
        env.reset()
        joint_id = env.unwrapped.model.joint("object0:joint").id
        qpos_start = env.unwrapped.model.jnt_qposadr[joint_id]
        actual = env.unwrapped.data.qpos[qpos_start : qpos_start + 3].copy()
        expected = np.array(load_config("env")["hidden_object_pos"])
        assert np.linalg.norm(actual - expected) < 0.001
    finally:
        env.close()
