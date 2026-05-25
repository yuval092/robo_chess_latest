"""Chess task environment: observations, rewards, reset, and scenario state."""

import logging
import os

import numpy as np

from gymnasium import spaces

from src.chess_env.simulation import ChessSimulationEnv
from src.chess_env.task_execution import GraspPlaceMixin
from src.chess_env.task_runtime import TaskRuntimeMixin
from src.chess_env.task_state import TaskStateMixin
from src.utils.io import load_config

HOME_POSTURE_JOINTS = (
    "robot0:torso_lift_joint",
    "robot0:shoulder_pan_joint",
    "robot0:shoulder_lift_joint",
    "robot0:upperarm_roll_joint",
    "robot0:elbow_flex_joint",
    "robot0:forearm_roll_joint",
    "robot0:wrist_flex_joint",
    "robot0:wrist_roll_joint",
    "robot0:l_gripper_finger_joint",
    "robot0:r_gripper_finger_joint",
)


class ChessTaskEnv(
    GraspPlaceMixin, TaskStateMixin, TaskRuntimeMixin, ChessSimulationEnv
):
    """
    Config-driven Reinforcement Learning environment for RoboChess movement.

    This class implements the V6 "Pure Movement" logic, handling scenario-based
    task definitions (Transit, Descend, Ascend), reward shaping, terminal conditions,
    and a drift curriculum to stabilize training.

    It employs the "Holding Object" trick (transfer architecture) to leverage
    pretrained transport weights without requiring piece-grasping physics.
    """

    def __init__(
        self,
        force_scenario=None,
        hide_object=True,
        show_chess_pieces=False,
        debug=False,
        drift_curriculum_steps=None,
        force_drift_limit=None,
        fixed_drift=False,
        **kwargs,
    ):
        """
        Initializes the task environment.

        Args:
            force_scenario (str): Lock the environment into 'transit', 'descend', or 'ascend'.
            hide_object (bool): If True, teleports the piece to a hidden location (used for pure movement).
            debug (bool): Enables verbose per-step state logging.
        """
        # Consume legacy params to avoid gym warnings
        kwargs.pop("sample_debug_freq", None)
        kwargs.pop("total_curriculum_steps", None)
        kwargs.pop("num_envs", None)
        kwargs.pop("curriculum_progress_override", None)

        # Load configurations
        self.env_cfg = load_config("env")
        self.physics_cfg = load_config("physics")
        self.chess_cfg = load_config("chess")

        self.force_scenario = force_scenario
        self.hide_object = hide_object
        self.show_chess_pieces = show_chess_pieces
        self.base_debug = debug
        self.debug = debug
        self.drift_curriculum_steps = (
            drift_curriculum_steps or self.env_cfg["drift_curriculum_steps"]
        )
        self.force_drift_limit = force_drift_limit
        self.fixed_drift = fixed_drift

        # Initialization Debug
        if debug:
            print(f"[DEBUG INIT] Pid {os.getpid()} | Scenario: {force_scenario}")

        self.current_scenario = None
        self.tube_center_xy = None
        self.gripper_forced_state = None
        self.goal_pos = None
        self.episode_steps = 0
        self.total_env_steps = 0
        self.episode_number = 0
        self._use_transfer_obs = False

        # Initialize base simulation
        super().__init__(debug=debug, **kwargs)

        # Override observation space to match our minimal dict (Stage 2)
        self.observation_space = spaces.Dict(
            {
                "grip_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32"),
                "grip_vel": spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32"),
                "l_finger": spaces.Box(-np.inf, np.inf, shape=(), dtype="float32"),
                "goal_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32"),
                "observation": spaces.Box(-np.inf, np.inf, shape=(7,), dtype="float32"),
                "achieved_goal": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype="float32"
                ),
                "desired_goal": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype="float32"
                ),
                "scenario_id": spaces.Discrete(
                    4
                ),  # 0: None, 1: transit, 2: descend, 3: ascend
            }
        )

        # --- Task Constants ---
        self.CUBE_HEIGHT = self.env_cfg["cube_height"]
        self.CUBE_Z = self.env_cfg["cube_z"]
        self.GRASP_Z = self.env_cfg["grasp_z"]
        self.HOVER_Z = self.env_cfg["hover_z"]
        self.SAFE_Z = self.env_cfg["safe_z"]
        self.SUCCESS_THRESHOLD = self.env_cfg["success_threshold"]
        self.DRIFT_LIMIT_START = self.env_cfg["drift_limit_start"]
        self.DRIFT_LIMIT_END = self.env_cfg["drift_limit_end"]
        self.DRIFT_CURRICULUM_STEPS = self.drift_curriculum_steps
        self.current_drift_limit = self._get_current_drift_limit()
        self.FLOOR_LIMIT = self.env_cfg["floor_limit"]
        self.HIDDEN_OBJECT_POS = np.array(self.env_cfg["hidden_object_pos"])
        self.MAX_SETTLE_STEPS = self.physics_cfg["max_settle_steps"]
        self.SETTLE_TOLERANCE = self.physics_cfg["settle_tolerance"]
        self.SETTLE_GAIN = self.physics_cfg["settle_gain"]
        self.SETTLE_STEPS_FINAL = self.physics_cfg["settle_steps_final"]
        self.HALT_VEL_THRESHOLD = self.env_cfg["halt_vel_threshold"]

        # --- Actuator Enforcement ---
        self.FINGER_OPEN_JOINT = self.env_cfg["finger_open_joint"]
        self.FINGER_CLOSED_JOINT = self.env_cfg["finger_closed_joint"]
        self.FINGER_OUTER_OFFSET = self.env_cfg["finger_outer_offset"]
        self.finger_target_joint = self.FINGER_CLOSED_JOINT

        # --- Grasp Stage State ---
        self.grasp_mode = False

        # Debug hook: set to a callable(phase: str) to receive state every physics step
        self._debug_step_callback = None
        self._debug_current_phase = "idle"

        # Home Position
        home_xy = self.env_cfg["home_position_xy"]
        self.HOME_POS = np.array([home_xy[0], home_xy[1], self.SAFE_Z])

        # Grasp Thresholds
        self.GRASP_CONTACT_APPROACH_TOLERANCE = self.env_cfg[
            "grasp_contact_approach_tolerance"
        ]
        self.GRASP_ALIGN_TOLERANCE = self.env_cfg["grasp_align_tolerance"]
        self.GRASP_CLOSE_STEPS = self.env_cfg["grasp_close_steps"]
        self.GRASP_RAMP_END = self.env_cfg["grasp_ramp_end"]
        self.EMPTY_GRASP_THRESHOLD = self.env_cfg["empty_grasp_threshold"]
        self.GRASP_HOLD_STEPS = self.env_cfg["grasp_hold_steps"]
        self.GRASP_PLUNGE_STEP_M = self.env_cfg["grasp_plunge_step_m"]
        self.GRASP_RETRACT_STEP_M = self.env_cfg["grasp_retract_step_m"]
        self.RELEASE_RAMP_STEPS = self.env_cfg["release_ramp_steps"]
        self.RELEASE_SETTLE_STEPS = self.env_cfg["release_settle_steps"]
        self.GRASP_VERIFY_XY_THRESHOLD = self.env_cfg["grasp_verify_xy_threshold"]
        self.GRASP_VERIFY_Z_THRESHOLD = self.env_cfg["grasp_verify_z_threshold"]
        self.CUBE_HELD_XY_LIMIT = self.env_cfg["cube_held_xy_limit"]
        self.CUBE_HELD_Z_LIMIT = self.env_cfg["cube_held_z_limit"]

        # Overrides for evaluation
        self.force_start_pos = None
        self.force_cube_pos = None
        self._board_mapper = None
        self._piece_registry = None
        self.active_piece_id = None
        self.active_piece_body_name = None
        self.active_piece_joint_name = None
        self._home_posture_qpos = None
        self._home_posture_mocap_pos = None
        self._home_posture_mocap_quat = None
        self._home_posture_grip_pos = None

        # --- Logger Setup ---
        log_cfg = self.env_cfg["logging"]
        log_dir = log_cfg["log_dir"]
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger(f"chess_task_{os.getpid()}")
        if not self.logger.handlers:
            if debug:
                os.makedirs(log_dir, exist_ok=True)
                handler = logging.FileHandler(
                    os.path.join(log_dir, f"env_{os.getpid()}.log")
                )
                handler.setFormatter(
                    logging.Formatter("%(asctime)s - [TASK] - %(message)s")
                )
                handler.setLevel(logging.DEBUG)
                self.logger.addHandler(handler)
            else:
                self.logger.addHandler(logging.NullHandler())
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def _get_obs(self):
        """Returns a minimal physics-state dict for status/debugging."""
        if self._use_transfer_obs:
            return self._build_transfer_observation()

        grip_pos = (
            self._grip_pos()
            .copy()
            .astype(np.float32)
        )
        grip_vel = (
            self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
            .copy()
            .astype(np.float32)
        )
        l_finger = np.float32(
            self._utils.get_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint"
            ).item()
        )
        goal_pos = (
            self.goal_pos.copy().astype(np.float32)
            if self.goal_pos is not None
            else np.zeros(3, dtype=np.float32)
        )

        # New minimal observation vector (7D: pos, vel, finger)
        obs_vec = np.concatenate([grip_pos, grip_vel, [l_finger]])

        # Scenario ID mapping
        scen_map = {None: 0, "transit": 1, "descend": 2, "ascend": 3}
        scenario_id = scen_map.get(self.current_scenario, 0)

        return {
            "observation": obs_vec,
            "achieved_goal": grip_pos,
            "desired_goal": goal_pos,
            "grip_pos": grip_pos,
            "grip_vel": grip_vel,
            "l_finger": l_finger,
            "goal_pos": goal_pos,
            "scenario_id": scenario_id,
        }

    def _build_transfer_observation(self):
        """
        Constructs the 25-dimensional transfer observation vector.

        This matches FetchPickAndPlace-v4's MultiInputPolicy observation layout.
        The "Holding Object" trick sets object_pos to grip_pos so transferred
        policies behave as if the gripper is already carrying the object.
        """
        (
            grip_pos,
            _object_pos,
            _object_rel_pos,
            _gripper_state,
            _object_rot,
            _object_velp,
            _object_velr,
            grip_velp,
            _gripper_vel,
        ) = self.generate_mujoco_observations()

        fake_object_pos = grip_pos.copy()
        goal = (
            self.goal
            if (hasattr(self, "goal") and self.goal is not None)
            else np.zeros(3)
        )
        rel_to_goal = goal.astype(np.float64) - grip_pos

        obs_vec = np.concatenate(
            [
                grip_pos,
                fake_object_pos,
                rel_to_goal,
                np.zeros(2),
                np.zeros(3),
                np.zeros(3),
                np.zeros(3),
                grip_velp,
                np.zeros(2),
            ]
        ).astype(np.float64)

        return {
            "observation": obs_vec,
            "achieved_goal": grip_pos.astype(np.float64),
            "desired_goal": goal.astype(np.float64),
        }
