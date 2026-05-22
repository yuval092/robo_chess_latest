import logging
import os
import mujoco
import numpy as np
from src.chess_env.simulation import ChessSimulationEnv
from src.chess_game.board_mapper import BoardMapper
from src.physical.piece_registry import PieceRegistry, reserve_piece_ids
from src.utils.config import load_config

class ChessTaskEnv(ChessSimulationEnv):
    """
    Config-driven Reinforcement Learning environment for RoboChess movement.
    
    This class implements the V6 "Pure Movement" logic, handling scenario-based
    task definitions (Transit, Descend, Ascend), reward shaping, terminal conditions,
    and a drift curriculum to stabilize training.
    
    It employs the "Holding Object" trick (Phase 9 architecture) to leverage
    pretrained transport weights without requiring piece-grasping physics.
    """

    def __init__(self, force_scenario=None, hide_object=True, show_chess_pieces=False, debug=False, **kwargs):
        """
        Initializes the task environment.

        Args:
            force_scenario (str): Lock the environment into 'transit', 'descend', or 'ascend'.
            hide_object (bool): If True, teleports the piece to a hidden location (used for pure movement).
            debug (bool): Enables verbose per-step state logging.
        """
        # Consume legacy params to avoid gym warnings
        kwargs.pop('drift_curriculum_steps', None)
        kwargs.pop('force_drift_limit', None)
        kwargs.pop('fixed_drift', None)
        kwargs.pop('sample_debug_freq', None)
        kwargs.pop('total_curriculum_steps', None)
        kwargs.pop('num_envs', None)
        kwargs.pop('curriculum_progress_override', None)
        
        # Load configurations
        self.env_cfg = load_config("env")
        self.physics_cfg = load_config("physics")
        self.chess_cfg = load_config("chess")

        self.force_scenario = force_scenario
        self.hide_object = hide_object
        self.show_chess_pieces = show_chess_pieces
        self.base_debug = debug
        self.debug = debug
        
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

        # Initialize base simulation
        super().__init__(debug=debug, **kwargs)

        # Override observation space to match our minimal dict (Stage 2)
        from gymnasium import spaces
        self.observation_space = spaces.Dict({
            "grip_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32"),
            "grip_vel": spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32"),
            "l_finger": spaces.Box(-np.inf, np.inf, shape=(), dtype="float32"),
            "goal_pos": spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32"),
            "observation": spaces.Box(-np.inf, np.inf, shape=(7,), dtype="float32"),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32"),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32"),
            "scenario_id": spaces.Discrete(4), # 0: None, 1: transit, 2: descend, 3: ascend
        })

        # --- Task Constants ---
        self.CUBE_HEIGHT = self.env_cfg["cube_height"]
        self.CUBE_Z = self.env_cfg["cube_z"]
        self.GRASP_Z = self.env_cfg["grasp_z"]
        self.HOVER_Z = self.env_cfg.get("hover_z", 0.460)
        self.SAFE_Z = self.env_cfg["safe_z"]
        self.SUCCESS_THRESHOLD = self.env_cfg["success_threshold"]
        self.DRIFT_LIMIT_END = self.env_cfg["drift_limit_end"]
        self.current_drift_limit = self.env_cfg["drift_limit_end"]  # Fixed; no curriculum
        self.FLOOR_LIMIT = self.env_cfg["floor_limit"]
        self.HIDDEN_OBJECT_POS = np.array(self.env_cfg["hidden_object_pos"])
        self.MAX_SETTLE_STEPS = self.physics_cfg["max_settle_steps"]
        self.SETTLE_TOLERANCE = self.physics_cfg["settle_tolerance"]
        self.SETTLE_GAIN = self.physics_cfg["settle_gain"]
        self.SETTLE_STEPS_FINAL = self.physics_cfg["settle_steps_final"]
        self.HALT_VEL_THRESHOLD = self.env_cfg.get("halt_vel_threshold", 0.0005)

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
        home_xy = self.env_cfg.get("home_position_xy", [0.680, 0.2641])
        self.HOME_POS = np.array([home_xy[0], home_xy[1], self.SAFE_Z])

        # Grasp Thresholds
        self.GRASP_CONTACT_APPROACH_TOLERANCE = self.env_cfg.get("grasp_contact_approach_tolerance", 0.001)
        self.GRASP_CLOSE_STEPS = self.env_cfg.get("grasp_close_steps", 150)
        self.GRASP_RAMP_END = self.env_cfg.get("grasp_ramp_end", 0.010)
        self.EMPTY_GRASP_THRESHOLD = self.env_cfg.get(
            "empty_grasp_threshold", self.GRASP_RAMP_END + 0.003
        )
        self.GRASP_HOLD_STEPS = self.env_cfg.get("grasp_hold_steps", 50)
        self.GRASP_PLUNGE_STEP_M = self.env_cfg.get("grasp_plunge_step_m", 0.002)
        self.GRASP_RETRACT_STEP_M = self.env_cfg.get("grasp_retract_step_m", 0.002)
        self.RELEASE_RAMP_STEPS = self.env_cfg.get("release_ramp_steps", 12)
        self.RELEASE_SETTLE_STEPS = self.env_cfg.get("release_settle_steps", 8)
        self.GRASP_VERIFY_XY_THRESHOLD = self.env_cfg.get("grasp_verify_xy_threshold", 0.015)
        self.GRASP_VERIFY_Z_THRESHOLD = self.env_cfg.get("grasp_verify_z_threshold", 0.020)
        self.CUBE_HELD_XY_LIMIT = self.env_cfg.get("cube_held_xy_limit", 0.030)
        self.CUBE_HELD_Z_LIMIT = self.env_cfg.get("cube_held_z_limit", 0.020)

        # Overrides for evaluation
        self.force_start_pos = None
        self.force_cube_pos = None
        self._board_mapper = None
        self._piece_registry = None
        self.active_piece_id = None
        self.active_piece_body_name = None
        self.active_piece_joint_name = None

        # --- Logger Setup ---
        log_cfg = self.env_cfg.get("logging", {})
        log_level = logging.DEBUG if debug else getattr(logging, log_cfg.get("level", "INFO"))
        log_dir = log_cfg.get("log_dir", "logs/env_debug")
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger(f"chess_task_{os.getpid()}")
        if not self.logger.handlers:
            if debug:
                os.makedirs(log_dir, exist_ok=True)
                handler = logging.FileHandler(os.path.join(log_dir, f"env_{os.getpid()}.log"))
                handler.setFormatter(logging.Formatter("%(asctime)s - [TASK] - %(message)s"))
                handler.setLevel(logging.DEBUG)
                self.logger.addHandler(handler)
            else:
                self.logger.addHandler(logging.NullHandler())
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def _get_obs(self):
        """Returns a minimal physics-state dict for status/debugging."""
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy().astype(np.float32)
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip").copy().astype(np.float32)
        l_finger = np.float32(self._utils.get_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint"
        ).item())
        goal_pos = self.goal_pos.copy().astype(np.float32) if self.goal_pos is not None else np.zeros(3, dtype=np.float32)
        
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

    def get_cube_position(self) -> np.ndarray:
        """Returns the active piece position, or object0 for legacy scripts."""
        if self.active_piece_joint_name is not None:
            return self.get_active_piece_position()
        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start = self.model.jnt_qposadr[obj_joint_id]
        return self.data.qpos[qpos_start : qpos_start + 3].copy()

    def get_cube_quat(self) -> np.ndarray:
        """Returns the active piece quaternion, or object0 for legacy scripts."""
        if self.active_piece_joint_name is not None:
            return self.get_active_piece_quat()
        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start = self.model.jnt_qposadr[obj_joint_id]
        return self.data.qpos[qpos_start + 3 : qpos_start + 7].copy()

    def set_active_piece(self, piece_id: str) -> None:
        if self._piece_registry is None:
            self._piece_registry = PieceRegistry()
        piece = self._piece_registry.by_id(piece_id)
        self.active_piece_id = piece.piece_id
        self.active_piece_body_name = piece.body_name
        self.active_piece_joint_name = piece.joint_name

    def clear_active_piece(self) -> None:
        self.active_piece_id = None
        self.active_piece_body_name = None
        self.active_piece_joint_name = None

    def get_active_piece_position(self) -> np.ndarray:
        if self.active_piece_joint_name is None:
            raise RuntimeError("No active chess piece selected.")
        joint_id = self.model.joint(self.active_piece_joint_name).id
        qpos_start = self.model.jnt_qposadr[joint_id]
        return self.data.qpos[qpos_start : qpos_start + 3].copy()

    def get_active_piece_quat(self) -> np.ndarray:
        if self.active_piece_joint_name is None:
            raise RuntimeError("No active chess piece selected.")
        joint_id = self.model.joint(self.active_piece_joint_name).id
        qpos_start = self.model.jnt_qposadr[joint_id]
        return self.data.qpos[qpos_start + 3 : qpos_start + 7].copy()

    def _set_freejoint_pose(self, joint_name: str, xyz: np.ndarray, quat=None) -> None:
        quat = np.array([1.0, 0.0, 0.0, 0.0]) if quat is None else np.asarray(quat, dtype=float)
        xyz = np.asarray(xyz, dtype=float)
        joint_id = self.model.joint(joint_name).id
        qpos_start = self.model.jnt_qposadr[joint_id]
        dof_start = self.model.jnt_dofadr[joint_id]
        self.data.qpos[qpos_start : qpos_start + 3] = xyz
        self.data.qpos[qpos_start + 3 : qpos_start + 7] = quat
        self.data.qvel[dof_start : dof_start + 6] = 0.0
        self.data.qacc[dof_start : dof_start + 6] = 0.0

    def _reserve_position(self, index: int, color: str, piece_type: str) -> np.ndarray:
        reserve_cfg = self.chess_cfg["promotion_reserve"][color]
        spacing = self.chess_cfg["reserves"]["promotion_slot_spacing_m"]
        origin = reserve_cfg["origin_xyz"]
        piece_type_offset = {"queen": 0, "rook": 8, "bishop": 16, "knight": 24}[piece_type]
        slot = piece_type_offset + index
        row = slot // reserve_cfg["cols"]
        col = slot % reserve_cfg["cols"]
        return np.array([origin[0] + row * spacing, origin[1] + col * spacing, origin[2]], dtype=float)

    def _hidden_piece_position(self, index: int) -> np.ndarray:
        row = index // 12
        col = index % 12
        return np.array([2.2 + row * 0.05, -0.5 + col * 0.05, self.CUBE_HEIGHT / 2.0])

    def _reset_chess_piece_bodies(self) -> None:
        if self._board_mapper is None:
            self._board_mapper = BoardMapper.from_configs()
        if self._piece_registry is None:
            self._piece_registry = PieceRegistry()

        hidden_index = 0
        import chess
        for piece in self._piece_registry.all_pieces():
            if self.show_chess_pieces:
                xyz = self._board_mapper.square_to_piece_xyz(chess.parse_square(piece.initial_square))
            else:
                xyz = self._hidden_piece_position(hidden_index)
                hidden_index += 1
            self._set_freejoint_pose(piece.joint_name, xyz)

        for piece_id, color, piece_type in reserve_piece_ids():
            joint_name = f"piece_{piece_id}:joint"
            if self.show_chess_pieces:
                reserve_index = int(piece_id.rsplit("_", 1)[1]) - 1
                xyz = self._reserve_position(reserve_index, color, piece_type)
            else:
                xyz = self._hidden_piece_position(hidden_index)
                hidden_index += 1
            self._set_freejoint_pose(joint_name, xyz)

    def _sample_goal(self):
        """Returns the goal sampled during reset_sim."""
        return self.goal_pos.copy()

    def _is_success(self, achieved_goal, desired_goal):
        """
        Determines if the gripper is within the success threshold of the goal.
        Checks both 2D (XY) and height (Z) precision.
        """
        achieved_goal = np.asarray(achieved_goal)
        desired_goal  = np.asarray(desired_goal)
        d_xy = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        d_z = abs(achieved_goal[2] - desired_goal[2])

        return float(d_xy < self.SUCCESS_THRESHOLD and d_z < self.SUCCESS_THRESHOLD)

    def _set_gripper_state(self):
        """Physically sets the joint positions of the fingers based on the target state."""
        target = self.finger_target_joint
        self._utils.set_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint", target)
        self._utils.set_joint_qpos(self.model, self.data, "robot0:r_gripper_finger_joint", target)
        self.data.qvel[self.model.joint("robot0:l_gripper_finger_joint").dofadr[0]] = 0.0
        self.data.qvel[self.model.joint("robot0:r_gripper_finger_joint").dofadr[0]] = 0.0
        
        # Sync actuator ctrl to prevent position actuator from fighting the teleport
        l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_joint")
        r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:r_gripper_finger_joint")
        self.data.ctrl[l_id] = target
        self.data.ctrl[r_id] = target
        
        mujoco.mj_forward(self.model, self.data)

    def _check_cube_held(self, grip_pos: np.ndarray) -> tuple[bool, str | None]:
        """
        Checks that the cube is still within the gripper.
        Only called when grasp_mode=True and current_scenario is 'ascend' or 'transit'.
        """
        cube_pos = self.get_cube_position()
        xy_error = np.linalg.norm(cube_pos[:2] - grip_pos[:2])
        # Cube CoM is ~15mm below grip site when held at center; allow tolerance
        z_error = abs(cube_pos[2] - (grip_pos[2] - 0.015))
        
        if xy_error > self.CUBE_HELD_XY_LIMIT:
            return False, f"CUBE_DROPPED_XY (err={xy_error*1000:.1f}mm > limit={self.CUBE_HELD_XY_LIMIT*1000:.0f}mm)"
        if z_error > self.CUBE_HELD_Z_LIMIT:
            return False, f"CUBE_DROPPED_Z (err={z_error*1000:.1f}mm > limit={self.CUBE_HELD_Z_LIMIT*1000:.0f}mm)"
        return True, None

    def _mujoco_step(self, action):
        super()._mujoco_step(action)
        if self._debug_step_callback is not None:
            self._debug_step_callback(self._debug_current_phase)

    def _settle_arm_to_start(self, arm_start_pos):
        """
        Uses the robust _move_mocap_to method to move the arm to the starting position.
        Prevents physics explosions and handles gravity sag.
        """
        self._move_mocap_to(arm_start_pos, self.VERTICAL_QUAT, max_steps=100, tolerance=self.SETTLE_TOLERANCE)
        
        # Zero velocities to ensure a stable episode start
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self):
        """
        Executes a complex reset sequence with scripted gripper transitions:
        1. Samples start/goal positions.
        2. Teleports object to a hidden or board location.
        3. Settles the arm at the starting waypoint with fingers CLOSED (stability).
        4. Executes scenario-specific scripted gripper transitions (Open/Close).
        5. Validates final finger state before handing over to RL model.
        """
        self.episode_steps = 0
        self.episode_number += 1

        # Stage 2 removed periodic debug sampling
        self.debug = self.base_debug
        
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        if self.force_scenario is not None:
            self.current_scenario = self.force_scenario
        else:
            self.current_scenario = self.np_random.choice(["transit", "descend", "ascend"])

        # NEW: reset grasp mode on every reset
        self.grasp_mode = False

        start_pos = self._sample_board_position()
        start_xy = start_pos[:2]

        # Define scenario-specific start/goal configurations
        if self.current_scenario == "transit":
            goal_pos = self._sample_board_position()
            while np.linalg.norm(goal_pos[:2] - start_xy) < self.MIN_GOAL_DIST:
                goal_pos = self._sample_board_position()
            self.tube_center_xy = None
            
            # Chess mode: always start at HOME_POS (board center); RL/eval: random or forced
            if self.force_start_pos is not None:
                arm_start_pos = self.force_start_pos.copy()
            elif self.show_chess_pieces:
                arm_start_pos = self.HOME_POS.copy()
            else:
                arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
                
            self.goal_pos = np.array([goal_pos[0], goal_pos[1], self.SAFE_Z])
        elif self.current_scenario == "descend":
            self.tube_center_xy = start_xy.copy()
            arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
            self.goal_pos = np.array([start_xy[0], start_xy[1], self.HOVER_Z])
        elif self.current_scenario == "ascend":
            self.tube_center_xy = start_xy.copy()
            arm_start_pos = np.array([start_xy[0], start_xy[1], self.HOVER_Z])
            self.goal_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])

        # NEW: Ensure self.goal is set BEFORE any call to self.render()
        # FetchEnv's _render_callback uses self.goal, which is normally set 
        # by RobotEnv.reset() AFTER _reset_sim() returns.
        if self.goal_pos is not None:
            self.goal = self.goal_pos.copy()

        # Reset parent simulation
        super()._reset_sim()

        # Place object
        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start = self.model.jnt_qposadr[obj_joint_id]
        dof_start = self.model.jnt_dofadr[obj_joint_id]
        
        if self.hide_object:
            self.data.qpos[qpos_start : qpos_start + 3] = self.HIDDEN_OBJECT_POS
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0
        elif self.force_cube_pos is not None:
            # Support force_cube_pos override
            self.data.qpos[qpos_start : qpos_start + 3] = self.force_cube_pos[:3]
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0
        else:
            self.data.qpos[qpos_start : qpos_start + 2] = start_xy
            self.data.qpos[qpos_start + 2] = self.TABLE_Z + (self.CUBE_HEIGHT / 2.0)
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0

        self._reset_chess_piece_bodies()
        mujoco.mj_forward(self.model, self.data)
        
        # Force torso to optimal height for board reach
        torso_height = self.env_cfg.get("torso_height", 0.25)
        self._utils.set_joint_qpos(self.model, self.data, "robot0:torso_lift_joint", torso_height)
        mujoco.mj_forward(self.model, self.data)
        
        # --- PHASE 1: Settle arm CLOSED for stability ---
        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT)
        self.finger_target_joint = self.FINGER_CLOSED_JOINT
        self._set_gripper_state()
        self._settle_arm_to_start(arm_start_pos)
        
        # --- PHASE 2: Scripted Transitions ---
        # Dummy action for _set_action calls during transitions
        dummy_action = np.zeros(4)

        if self.current_scenario == "descend":
            # Descend: closed --> opened (scripted transition before RL)
            self.finger_target_joint = self.FINGER_OPEN_JOINT
            self._set_gripper_state()
            self._move_mocap_to(arm_start_pos, self.VERTICAL_QUAT, max_steps=150, tolerance=0.003)
        elif self.current_scenario == "ascend":
            # Ascend: start opened --> closed (scripted transition before RL)
            # First open them
            self.finger_target_joint = self.FINGER_OPEN_JOINT
            self._set_gripper_state() # Force immediately
            self._move_mocap_to(arm_start_pos, self.VERTICAL_QUAT, max_steps=100, tolerance=0.003)
            
            # Then close them scripted
            self.finger_target_joint = self.FINGER_CLOSED_JOINT
            self._set_gripper_state() # Fix: Force close immediately to ensure convergence
            self._move_mocap_to(arm_start_pos, self.VERTICAL_QUAT, max_steps=150, tolerance=0.003)
        else:
            # Transit: stay closed
            self.finger_target_joint = self.FINGER_CLOSED_JOINT
            self._move_mocap_to(arm_start_pos, self.VERTICAL_QUAT, max_steps=150, tolerance=0.003)

        # --- PHASE 3: Final Validation ---
        l_pos = self._utils.get_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint").item()
        if abs(l_pos - self.finger_target_joint) > 0.0005:
            self.logger.error(f"Reset Failed: Finger joint at {l_pos:.6f}, target {self.finger_target_joint:.6f} (Scenario: {self.current_scenario})")
            return False 
        
        if self.debug:
            obj_pos_now = self.data.qpos[qpos_start : qpos_start + 3]
            grip_pos_now = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            arm_joints = ["robot0:shoulder_pan_joint", "robot0:shoulder_lift_joint",
                          "robot0:upperarm_roll_joint", "robot0:elbow_flex_joint",
                          "robot0:forearm_roll_joint", "robot0:wrist_flex_joint",
                          "robot0:wrist_roll_joint"]
            joint_qpos = []
            for jname in arm_joints:
                try:
                    joint_qpos.append(self._utils.get_joint_qpos(self.model, self.data, jname).item())
                except Exception:
                    joint_qpos.append(float('nan'))
            jq_str = ", ".join([f"{x:.4f}" for x in joint_qpos])
            tube_str = f"{self.tube_center_xy}" if self.tube_center_xy is not None else "N/A"
            msg = (
                f"\n{'='*80}\n"
                f"[EPISODE {self.episode_number} START] Scenario: {self.current_scenario.upper()}\n"
                f"  GripPos:    [{', '.join([f'{x:.4f}' for x in grip_pos_now])}]\n"
                f"  Goal:       [{', '.join([f'{x:.4f}' for x in self.goal_pos])}]\n"
                f"  TubeCenter: {tube_str}\n"
                f"  ObjPos:     [{', '.join([f'{x:.4f}' for x in obj_pos_now])}]\n"
                f"  ArmJoints:  [{jq_str}]\n"
                f"{'='*80}"
            )
            self.logger.debug(msg)

        return True

    def transition_validate(self, nominal_exit_pos: np.ndarray | None = None) -> dict:
        """
        Returns arm state diagnostics at a scenario transition point.
        """
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip").copy()
        speed = float(np.linalg.norm(grip_vel))
        l_finger = self._utils.get_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint").item()

        result = {
            "grip_pos": grip_pos.tolist(),
            "grip_speed_mm_s": speed * 1000,
            "is_velocity_ok": speed < self.HALT_VEL_THRESHOLD,
            "error_from_nominal_mm": None,
            "finger_state": l_finger,
        }
        if nominal_exit_pos is not None:
            result["error_from_nominal_mm"] = float(np.linalg.norm(grip_pos - nominal_exit_pos) * 1000)
        return result

    def _move_mocap_to(self, target_pos: np.ndarray, target_quat: np.ndarray,
                       max_steps: int = 150, tolerance: float = 0.001) -> bool:
        """
        Drives the physical arm so that robot0:grip site reaches target_pos.
        Automatically handles site-to-body offsets by applying deltas.
        """
        zero_action = np.zeros(4)
        should_render = (self.render_mode == "human")
        for _ in range(max_steps):
            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            error = target_pos - grip_pos
            if np.linalg.norm(error) < tolerance:
                return True
                
            self._set_action(zero_action)                   # Resets mocap to body
            # Apply error as delta to mocap_pos to pull the site toward target
            self.data.mocap_pos[0][:3] += error
            self.data.mocap_quat[0][:] = target_quat
            self._mujoco_step(None)
            if should_render:
                self.render()
            
        final_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        return bool(np.linalg.norm(final_pos - target_pos) < tolerance)

    def _step_locked_grip(self, target_pos: np.ndarray, should_render: bool) -> None:
        """Reset mocap to the body, then pull the grip site to a target pose for one step."""
        self._set_action(np.zeros(4))
        error = target_pos - self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        self.data.mocap_pos[0][:3] += error
        self.data.mocap_quat[0][:] = self.VERTICAL_QUAT
        self._mujoco_step(None)
        if should_render:
            self.render()

    def _plunge_to_z(self, xy: np.ndarray, target_z: float, step_m: float, should_render: bool) -> tuple[np.ndarray, int]:
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        commanded_z = grip_pos[2]
        steps = 0
        for _ in range(int(round(max(0.0, commanded_z - target_z) / step_m)) + 6):
            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            if grip_pos[2] <= target_z + 0.001:
                break
            commanded_z = max(target_z, commanded_z - step_m)
            self._step_locked_grip(np.array([xy[0], xy[1], commanded_z]), should_render)
            steps += 1
        return self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy(), steps

    def _retract_to_hover(
        self,
        xy_provider,
        start_z: float,
        step_m: float,
        should_render: bool,
        *,
        verify_held: bool = False,
    ) -> tuple[str | None, int]:
        commanded_z = start_z
        steps = 0
        for _ in range(int(round((self.HOVER_Z - start_z) / step_m)) + 3):
            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            if grip_pos[2] >= self.HOVER_Z - 0.001:
                break
            commanded_z = min(self.HOVER_Z, commanded_z + step_m)
            xy = xy_provider()
            self._step_locked_grip(np.array([xy[0], xy[1], commanded_z]), should_render)
            steps += 1

            if verify_held:
                cube_now = self.get_cube_position()
                grip_now = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
                if abs(cube_now[2] - (grip_now[2] - 0.015)) > self.CUBE_HELD_Z_LIMIT:
                    return "CUBE_DROPPED_DURING_RETRACT", steps
        return None, steps

    def _hold_locked_target(self, target_provider, steps: int, should_render: bool) -> None:
        for _ in range(steps):
            self._step_locked_grip(target_provider(), should_render)

    def execute_grasp(self) -> dict:
        """
        Scripted GRASP pipeline. Runs after DESCEND succeeds at HOVER_Z.
        Returns dict with 'success' bool. Arm exits at HOVER_Z with cube held.

        KEY INVARIANT: After every _set_action call, re-assert BOTH mocap_pos AND
        mocap_quat before _mujoco_step. _set_action resets mocap to physical body
        state (via reset_mocap2body_xpos), so assertions must be repeated every step.
        """
        import math

        result = {
            "success": False,
            "reason": None,
            "pre_grasp_cube_xy": None,
            "post_grasp_cube_pos": None,
            "final_xy_error_mm": 0.0,
            "final_z_error_mm": 0.0,
            "final_finger_pos": 0.0,
            "close_steps_used": 0,
        }

        # ── Phase 0: Halt & Settle ────────────────────────────────────────────────
        self._debug_current_phase = "grasp_p0_halt"
        # Zero velocity FIRST. Residual RL momentum causes oscillation if not stopped.
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._set_gripper_state()  # Restore finger ctrl after ctrl[:]=0 to keep fingers open
        mujoco.mj_forward(self.model, self.data)

        should_render = (self.render_mode == "human")

        # Speed check: arm should be stationary
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        if float(np.linalg.norm(grip_vel)) > 0.005:
            result["reason"] = "PRECONDITION_SPEED"
            return result

        # Z check: RL must have stopped at HOVER_Z (within ±15mm)
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        if abs(grip_pos[2] - self.HOVER_Z) > 0.025:
            result["reason"] = (f"PRECONDITION_Z (grip={grip_pos[2]*1000:.1f}mm, "
                                f"HOVER_Z={self.HOVER_Z*1000:.1f}mm)")
            return result

        # Fingers must be open
        l_finger = self._utils.get_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint"
        ).item()
        if l_finger < self.FINGER_OPEN_JOINT - 0.003:
            result["reason"] = f"PRECONDITION_FINGERS_NOT_OPEN (j={l_finger:.4f})"
            return result

        # ── Phase 1+2: Rotation Abort Check + Perfect Align & Verticalize ────────
        self._debug_current_phase = "grasp_p12_align"
        # Read cube position and check for dangerous diagonal orientation.
        cube_pos = self.get_cube_position().copy()
        result["pre_grasp_cube_xy"] = cube_pos[:2].copy()

        self.logger.debug(f"[GRASP] Start. Cube at {cube_pos}, Grip at {self._utils.get_site_xpos(self.model, self.data, 'robot0:grip')}")
        # Cube rotation check: diagonal cube (yaw >25°) has effective width 42.4mm,
        # exceeding max finger opening (38mm). Abort before plunge to prevent stub.
        cube_quat = self.get_cube_quat()  # (w, x, y, z)
        w, x, y, z = cube_quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = abs(math.atan2(siny_cosp, cosy_cosp))  # → [0, π]
        # Correct 4-fold symmetry: fold [0,π] into [0,π/2] quadrant, then find
        # distance to nearest square axis. A cube at 175° is equivalent to 5°.
        # The simple formula `min(yaw, abs(yaw - π/2))` FAILS above 90° —
        # e.g. 175° gives 85° (false abort) instead of the correct 5°.
        yaw_modulo   = yaw % (math.pi / 2)                        # fold → [0, π/2)
        effective_yaw = min(yaw_modulo, (math.pi / 2) - yaw_modulo)  # dist to nearest axis
        if effective_yaw > 0.436:  # > 25 degrees
            result["reason"] = f"CUBE_ROTATED (yaw={math.degrees(effective_yaw):.1f}°)"
            return result

        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        align_target = np.array([cube_pos[0], cube_pos[1], grip_pos[2]])
        if not self._move_mocap_to(align_target, self.VERTICAL_QUAT, max_steps=100, tolerance=0.001):
            result["reason"] = "ROTATION_FAILED (kinematic limit — arm cannot reach vertical at this position)"
            return result

        # ── Phase 3: Plunge (current Z → PLACE_Z) ────────────────────────────────
        self._debug_current_phase = "grasp_p3_plunge"
        place_z = self.GRASP_Z
        grip_pos, _plunge_steps = self._plunge_to_z(cube_pos[:2], place_z, self.GRASP_PLUNGE_STEP_M, should_render)
        if self.debug:
            self.logger.debug(f"[GRASP] Post-Plunge. Grip at {grip_pos}, Cube at {self.get_cube_position()}")
        if abs(grip_pos[2] - self.GRASP_Z) > 0.008:
            result["reason"] = (f"PLUNGE_FAILED (z={grip_pos[2]*1000:.1f}mm, "
                                f"target={self.GRASP_Z*1000:.1f}mm)")
            return result

        # ── Phase 4: Grasp (Finger Close, Linear Ramp) ───────────────────────────
        self._debug_current_phase = "grasp_p4_close"
        # Ramp finger target from OPEN to a secure-grip value over the configured step budget.
        # Direct jump creates a large impulse; ramping limits contact shock.
        # Track the live cube XY so the arm follows contact motion instead of fighting it.
        self.grasp_mode = True
        ramp_start    = self.FINGER_OPEN_JOINT    # 0.0181
        ramp_end      = self.GRASP_RAMP_END
        ramp_delta    = (ramp_start - ramp_end) / self.GRASP_CLOSE_STEPS
        empty_detect_threshold = self.EMPTY_GRASP_THRESHOLD
        empty_detect_start = max(8, int(self.GRASP_CLOSE_STEPS * 0.65))

        steps_used = 0
        for step in range(self.GRASP_CLOSE_STEPS):
            self.finger_target_joint = max(ramp_end, ramp_start - ramp_delta * step)
            live_cube = self.get_cube_position()
            self._step_locked_grip(np.array([live_cube[0], live_cube[1], self.GRASP_Z]), should_render)
            steps_used += 1

            # Early abort: fingers reached the secure-grip target = no cube contact.
            # With cube: fingers stall at j≈0.0141.
            if step >= empty_detect_start:
                l_now = self._utils.get_joint_qpos(
                    self.model, self.data, "robot0:l_gripper_finger_joint"
                ).item()
                if l_now < empty_detect_threshold:
                    result["reason"] = f"FINGER_CLOSED_EMPTY (j={l_now:.4f} at step {step})"
                    result["close_steps_used"] = steps_used
                    return result

        result["close_steps_used"] = steps_used

        # ── Phase 5: Hold & Verify ────────────────────────────────────────────────
        self._debug_current_phase = "grasp_p5_hold"
        # Settle to let contact impulses stabilize (prevent ringing).
        # Continue tracking live cube XY and re-asserting vertical quat.
        def live_cube_target():
            live_cube = self.get_cube_position()
            return np.array([live_cube[0], live_cube[1], self.GRASP_Z])

        self._hold_locked_target(live_cube_target, self.GRASP_HOLD_STEPS, should_render)

        cube_pos  = self.get_cube_position()
        grip_pos  = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        l_finger  = self._utils.get_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint"
        ).item()

        xy_error = float(np.linalg.norm(cube_pos[:2] - grip_pos[:2])) * 1000
        z_error  = float(abs(cube_pos[2] - grip_pos[2])) * 1000

        result["post_grasp_cube_pos"]  = cube_pos.copy()
        result["final_xy_error_mm"]    = xy_error
        result["final_z_error_mm"]     = z_error
        result["final_finger_pos"]     = l_finger

        if xy_error > self.GRASP_VERIFY_XY_THRESHOLD * 1000:
            result["reason"] = (f"VERIFY_XY_FAILED ({xy_error:.1f}mm > "
                                f"{self.GRASP_VERIFY_XY_THRESHOLD*1000:.0f}mm)")
            return result

        if z_error > self.GRASP_VERIFY_Z_THRESHOLD * 1000:
            result["reason"] = (f"VERIFY_Z_FAILED ({z_error:.1f}mm > "
                                f"{self.GRASP_VERIFY_Z_THRESHOLD*1000:.0f}mm)")
            return result

        if l_finger < empty_detect_threshold:
            result["reason"] = f"VERIFY_FINGERS_CLOSED_EMPTY (j={l_finger:.4f})"
            return result

        # ── Phase 6: Retract (PLACE_Z → HOVER_Z) ──────────────────────────────────
        self._debug_current_phase = "grasp_p6_retract"
        retract_error, _retract_steps = self._retract_to_hover(
            lambda: self.get_cube_position()[:2],
            place_z,
            self.GRASP_RETRACT_STEP_M,
            should_render,
            verify_held=True,
        )
        if retract_error:
            result["reason"] = retract_error
            return result

        result["success"] = True
        result["total_steps_used"] = (
            _plunge_steps + result["close_steps_used"] + self.GRASP_HOLD_STEPS + _retract_steps
        )
        result["post_grasp_cube_pos"] = self.get_cube_position().copy()
        return result

    def execute_place(self, dst_xy: np.ndarray) -> dict:
        """
        Scripted PLACE pipeline. Runs after DESCEND to HOVER_Z over destination.
        dst_xy: 2D destination XY (board square center).
        Returns dict with 'success' bool. Arm exits at HOVER_Z, cube placed, grasp_mode=False.
        """
        result = {
            "success": False,
            "reason": None,
            "final_cube_pos": None,
            "final_xy_error_mm": 0.0,
        }

        # ── Phase 0: Halt & Settle ────────────────────────────────────────────────
        # CRITICAL: Zero the full simulation state ([:]), not just robot DOFs [:15].
        self._debug_current_phase = "place_p0_halt"
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._set_gripper_state()  # Restore finger ctrl after ctrl[:]=0 to prevent squeeze during halt
        mujoco.mj_forward(self.model, self.data)

        should_render = (self.render_mode == "human")

        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        if float(np.linalg.norm(grip_vel)) > 0.005:
            result["reason"] = "PRECONDITION_SPEED"
            return result

        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        if abs(grip_pos[2] - self.HOVER_Z) > 0.025:
            result["reason"] = f"PRECONDITION_Z (grip={grip_pos[2]*1000:.1f}mm)"
            return result

        # ── Phase 1+2: Vertical Correction + XY Align Over Destination ───────────
        self._debug_current_phase = "place_p12_align"
        # Simultaneously correct wrist orientation and move over destination.
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        align_target = np.array([dst_xy[0], dst_xy[1], grip_pos[2]])
        if not self._move_mocap_to(align_target, self.VERTICAL_QUAT, max_steps=100, tolerance=0.001):
            result["reason"] = "ROTATION_FAILED (kinematic limit)"
            return result

        # ── Phase 3: Plunge (current Z → PLACE_Z) ────────────────────────────────
        self._debug_current_phase = "place_p3_plunge"
        place_z = self.GRASP_Z
        _, _plunge_steps = self._plunge_to_z(dst_xy, place_z, self.GRASP_PLUNGE_STEP_M, should_render)

        # Verify plunge reached place_z before releasing the cube. If the arm stalled
        # mid-descent and we open fingers here, the cube falls from height.
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        place_pos = grip_pos.copy()  # Capture for Phase 6
        if abs(grip_pos[2] - place_z) > 0.008:
            result["reason"] = (f"PLUNGE_FAILED (z={grip_pos[2]*1000:.1f}mm, "
                                f"target={place_z*1000:.1f}mm)")
            return result

        # ── Phase 4: Release (Linear Ramp Open) ───────────────────────────────────
        self._debug_current_phase = "place_p4_release"
        # Ramp from GRASP_RAMP_END (0.010, actual grip position) to OPEN (0.0181).
        # Starting from 0.0000 would actively squeeze fingers for the first ~29 steps before opening.
        release_target = place_pos.copy()
        ramp_start = self.GRASP_RAMP_END       # 0.010 — actual finger position during grip
        ramp_end   = self.FINGER_OPEN_JOINT    # 0.0181
        release_ramp_steps = self.RELEASE_RAMP_STEPS
        release_settle_steps = self.RELEASE_SETTLE_STEPS
        ramp_delta = (ramp_end - ramp_start) / release_ramp_steps

        for step in range(release_ramp_steps):
            self.finger_target_joint = min(ramp_end, ramp_start + ramp_delta * step)
            self._step_locked_grip(release_target, should_render)

        # Full open settle
        self.finger_target_joint = self.FINGER_OPEN_JOINT
        self._hold_locked_target(lambda: release_target, release_settle_steps, should_render)

        # Disable grasp mode — fingers can be teleported again in subsequent RL phases
        self.grasp_mode = False

        # ── Phase 5: Verify Placement ─────────────────────────────────────────────
        self._debug_current_phase = "place_p5_verify"
        cube_pos = self.get_cube_position()
        xy_error = float(np.linalg.norm(cube_pos[:2] - dst_xy[:2])) * 1000
        z_error  = float(abs(cube_pos[2] - (self.TABLE_Z + self.CUBE_HEIGHT / 2.0))) * 1000

        result["final_cube_pos"] = cube_pos.copy()
        result["final_xy_error_mm"] = xy_error

        if xy_error > 20.0:
            result["reason"] = f"PLACE_XY_FAILED ({xy_error:.1f}mm drift from target)"
            return result

        if z_error > 10.0:
            result["reason"] = f"PLACE_Z_FAILED ({z_error:.1f}mm — cube not flat on table)"
            return result

        # ── Phase 6: Retract (PLACE_Z → HOVER_Z) ──────────────────────────────────
        self._debug_current_phase = "place_p6_retract"
        _, _retract_steps = self._retract_to_hover(lambda: place_pos[:2], place_z, self.GRASP_RETRACT_STEP_M, should_render)

        result["success"] = True
        result["total_steps_used"] = (
            _plunge_steps + self.RELEASE_RAMP_STEPS + self.RELEASE_SETTLE_STEPS + _retract_steps
        )
        return result

    def soft_reset(self, new_scenario: str, new_goal_pos: np.ndarray,
                   nominal_exit_pos: np.ndarray,
                   nominal_xy: np.ndarray | None = None) -> tuple[dict, dict]:
        """
        Transitions to a new scenario without teleporting the arm.
        Executes: complete halt -> waypoint alignment -> gripper transition -> state update.

        Note: The caller is responsible for resetting environment wrappers (like TimeLimit).
        
        Returns:
            (obs, info): obs is the new observation, info contains transition diagnostics.
        """
        # ROBOT_DOF = 15 is correct for Fetch: 3 slides + 1 torso + 2 head + 7 arm + 2 fingers
        ROBOT_DOF = 15
        HALT_HOLD_MAX_STEPS = 30
        ALIGN_TOLERANCE_M = 0.004
        ALIGN_MAX_STEPS = 80
        ALIGN_GAIN = 1.0
        ALIGN_MAX_STEP_M = 0.012

        # Phase 1: Halt (Driving arm to a dead stop)
        self._debug_current_phase = "softreset_p1_halt"
        zero_action = np.zeros(4)
        should_render = (self.render_mode == "human")
        self.data.qvel[:ROBOT_DOF] = 0.0
        self.data.qacc[:ROBOT_DOF] = 0.0
        mujoco.mj_forward(self.model, self.data)
        halt_steps = 0
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        if np.linalg.norm(grip_vel) >= self.HALT_VEL_THRESHOLD:
            for _ in range(HALT_HOLD_MAX_STEPS):
                grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
                if np.linalg.norm(grip_vel) < self.HALT_VEL_THRESHOLD:
                    break
                self._set_action(zero_action)
                self._mujoco_step(None)
                if should_render:
                    self.render()
                halt_steps += 1

        # Active velocity zeroing (Robot only, preserves object physics)
        self.data.qvel[:ROBOT_DOF] = 0.0
        self.data.qacc[:ROBOT_DOF] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Confirm halt (Plan Step 1c)
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        speed = float(np.linalg.norm(grip_vel))
        if speed >= self.HALT_VEL_THRESHOLD:
            raise RuntimeError(
                f"soft_reset HALT_FAILED: speed={speed*1000:.3f}mm/s >= "
                f"threshold={self.HALT_VEL_THRESHOLD*1000:.1f}mm/s"
            )

        # Phase 2: Waypoint Alignment (Physics-simulated smooth movement)
        self._debug_current_phase = "softreset_p2_align"
        converged = False
        align_steps = 0
        for loop_step in range(ALIGN_MAX_STEPS):
            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            error = nominal_exit_pos - grip_pos
            dist = np.linalg.norm(error)
            if dist < ALIGN_TOLERANCE_M:
                self.data.qvel[:ROBOT_DOF] = 0.0
                self.data.qacc[:ROBOT_DOF] = 0.0
                mujoco.mj_forward(self.model, self.data)
                converged = True
                break
            # Proportional step toward target
            step_vec = ALIGN_GAIN * error
            if np.linalg.norm(step_vec) > ALIGN_MAX_STEP_M:
                step_vec = step_vec / np.linalg.norm(step_vec) * ALIGN_MAX_STEP_M
            self.data.mocap_pos[0][:3] += step_vec
            self._mujoco_step(None)
            if should_render:
                self.render()
            align_steps = loop_step + 1

        if not converged:
            raise RuntimeError(f"soft_reset ALIGN_FAILED: did not converge to {nominal_exit_pos}")

        # Phase 3: State Update
        self._debug_current_phase = "softreset_p3_state"
        prev_scenario = self.current_scenario
        self.current_scenario = new_scenario
        self.goal_pos = new_goal_pos.copy()
        self.goal = self.goal_pos.copy() # Critical for observation desync
        self.episode_steps = 0
        
        if new_scenario in {"descend", "ascend"}:
            if nominal_xy is None:
                raise ValueError(f"nominal_xy required for {new_scenario}")
            self.tube_center_xy = nominal_xy.copy()
        else:
            self.tube_center_xy = None

        # Re-enforce vertical orientation
        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT)

        # Phase 4: Gripper Transition (Scripted finger movement)
        self._debug_current_phase = "softreset_p4_gripper"
        if not self.grasp_mode:
            needs_open = (new_scenario == "descend" and prev_scenario != "descend")
            needs_close = (new_scenario in {"ascend", "transit"} and prev_scenario == "descend")
            dummy_action = np.zeros(4)

            if needs_open:
                self.finger_target_joint = self.FINGER_OPEN_JOINT
                self._set_gripper_state()
                self._move_mocap_to(nominal_exit_pos, self.VERTICAL_QUAT, max_steps=40, tolerance=0.004)
            elif needs_close:
                self.finger_target_joint = self.FINGER_CLOSED_JOINT
                self._set_gripper_state()
                self._move_mocap_to(nominal_exit_pos, self.VERTICAL_QUAT, max_steps=40, tolerance=0.004)

            # Finger validation (same as _reset_sim Phase 3)
            l_pos = self._utils.get_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint"
            ).item()
            if abs(l_pos - self.finger_target_joint) > 0.0005:
                raise RuntimeError(
                    f"soft_reset FINGER_VALIDATION_FAILED: "
                    f"actual={l_pos:.6f}, target={self.finger_target_joint:.6f}"
                )
        # else: grasp_mode=True -> fingers stay actuator-driven, no teleportation or validation needed

        info = {
            "halt_steps": halt_steps,
            "align_steps": align_steps,
        }
        return self._get_obs(), info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Stub: scripted system does not use reward."""
        return 0.0

    def step(self, action):
        """
        Minimal physics step. The ScriptedController drives the arm via
        _move_mocap_to; this step() satisfies the Gymnasium interface only.
        """
        zero = np.zeros(4)
        zero[3] = -1.0
        self._set_action(zero)
        self._mujoco_step(zero)
        obs = self._get_obs()
        return obs, 0.0, False, False, {"is_success": 0.0, "scenario": self.current_scenario}
