"""Base MuJoCo simulation environment wrapping the Fetch Pick-and-Place robot."""

import threading
from pathlib import Path

import gymnasium_robotics.envs.fetch.pick_and_place as _fpp_module
import mujoco
import numpy as np
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv

from src.utils.io import load_config

_xml_path_lock = threading.Lock()


def unwrap_env(env):
    """Unwrap a gymnasium environment to its innermost environment."""
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    return inner


def reset_elapsed_steps(env) -> None:
    """Reset the first wrapper or env object that exposes _elapsed_steps."""
    current = env
    while True:
        if hasattr(current, "_elapsed_steps"):
            current._elapsed_steps = 0
            return
        if not hasattr(current, "env"):
            return
        current = current.env


class ChessSimulationEnv(MujocoFetchPickAndPlaceEnv):
    """
    Modular MuJoCo environment for RoboChess fine-tuning.

    This class extends the base FetchPickAndPlace-v4 environment to support
    a larger 64cm x 64cm chess board and configuration-driven physics constants.
    It handles low-level simulation tasks like model loading, action scaling,
    and board sampling.
    """

    def __init__(self, debug=False, **kwargs):
        self.env_cfg = load_config("env")
        self.physics_cfg = load_config("physics")
        self.debug = debug
        self._init_table_constants()
        self._init_physics_constants()
        self._load_chess_model(**kwargs)

    def _init_table_constants(self) -> None:
        """Set board geometry constants from env config."""
        self.TABLE_CENTER_XY = np.array(self.env_cfg["table_center_xy"])
        self.TABLE_HALF_X = self.env_cfg["table_half_x"]
        self.TABLE_HALF_Y = self.env_cfg["table_half_y"]
        self.TABLE_SURFACE_Z = self.env_cfg["table_surface_z"]
        self.EDGE_MARGIN = self.env_cfg["edge_margin"]
        self.MIN_GOAL_DIST = self.env_cfg["min_goal_dist"]
        self.CUBE_HEIGHT = self.env_cfg["cube_height"]

        margin = self.EDGE_MARGIN
        center_x, center_y = self.TABLE_CENTER_XY
        self.BOARD_MIN_XY = np.array([center_x - self.TABLE_HALF_X + margin, center_y - self.TABLE_HALF_Y + margin])
        self.BOARD_MAX_XY = np.array([center_x + self.TABLE_HALF_X - margin, center_y + self.TABLE_HALF_Y - margin])

    def _init_physics_constants(self) -> None:
        """Set physics and control constants from physics config."""
        self.ENV_SETUP_STEPS = self.physics_cfg["env_setup_steps"]
        self.POS_CTRL_SCALE = self.physics_cfg["pos_ctrl_scale"]
        raw_quat = np.array(self.physics_cfg["vertical_quat"])
        self.VERTICAL_QUAT = raw_quat / np.linalg.norm(raw_quat)

    def _load_chess_model(self, **kwargs) -> None:
        """Inject our chess board XML, call super().__init__, then restore the original path."""
        project_root = Path(__file__).resolve().parents[2]
        asset_path = str(project_root / "chess_env" / "assets" / "pick_and_place.xml")

        with _xml_path_lock:
            _original_path = _fpp_module.MODEL_XML_PATH
            _fpp_module.MODEL_XML_PATH = asset_path
            try:
                super().__init__(**kwargs)
                init_qpos = self.physics_cfg["initial_qpos"]
                self.initial_qpos[0] = init_qpos[0]
                self.initial_qpos[1] = init_qpos[1]
            finally:
                _fpp_module.MODEL_XML_PATH = _original_path


    def _random_board_position(self) -> np.ndarray:
        """Sample a random XY within the playable board area, at table surface Z."""
        x, y = self.np_random.uniform(self.BOARD_MIN_XY, self.BOARD_MAX_XY)
        return np.array([x, y, self.TABLE_SURFACE_Z])

    def _render_callback(self):
        """Suppress Fetch's moving target0 goal marker in chess visualizations."""
        pass

    def _set_action(self, action):
        """Translate [dx, dy, dz, gripper] action into scaled mocap delta + finger enforcement."""
        assert action.shape == (4,)
        pos_ctrl = action[:3].copy() * self.POS_CTRL_SCALE
        self._enforce_fingers()
        self._utils.mocap_set_action(
            self.model, self.data, np.concatenate([pos_ctrl, np.zeros(4)])
        )

    def _enforce_fingers(self) -> None:
        """Set finger actuator targets; in teleport mode also force joint positions directly.

        grasp_mode=True: actuator-driven only — MuJoCo's Kp controller handles contact
        forces so the cube can push back against the fingers.
        grasp_mode=False: teleport mode — joint positions are forced directly so fingers
        never drift during pure-movement phases.
        """
        target = self.finger_target_joint
        self.data.ctrl[0] = target
        self.data.ctrl[1] = target

        if not self.grasp_mode:
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint", target
            )
            self._utils.set_joint_qpos(
                self.model, self.data, "robot0:r_gripper_finger_joint", target
            )
            self._utils.set_joint_qvel(
                self.model, self.data, "robot0:l_gripper_finger_joint", 0.0
            )
            self._utils.set_joint_qvel(
                self.model, self.data, "robot0:r_gripper_finger_joint", 0.0
            )

    def _env_setup(self, initial_qpos):
        """
        Initial setup of the environment. Forces vertical orientation and lets
        physics settle before the first episode starts.
        """
        super()._env_setup(initial_qpos)

        # Force torso to optimal operating height for board reach
        torso_height = self.env_cfg["torso_height"]
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:torso_lift_joint", torso_height
        )

        # Force crane orientation immediately on setup
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT
        )
        for _ in range(self.ENV_SETUP_STEPS):
            mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
