"""Shared chess-env scaffolding used by both training and production.

Holds the bits Training and Production both need: config loading, observation
space, reset sequence, scenario state, piece-body teleport, soft_reset, and the
low-level mocap/finger helpers. Training adds RL reward + curriculum on top;
Production adds piece registry + scripted grasp/place on top.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager

import chess
import mujoco
import numpy as np
from gymnasium import spaces

from src.chess_env.simulation import IDENTITY_QUAT, ChessSimulationEnv
from src.chess_game.board_mapper import BoardMapper
from src.physical.piece_registry import PieceRegistry, reserve_piece_ids
from src.utils.io import load_config


PRETRAINED_OBS_SPACE = spaces.Dict(
    {
        "observation": spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float64),
        "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
        "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
    }
)

@contextmanager
def pretrained_obs_format_enabled(env) -> Generator[None, None, None]:
    """Enable pretrained obs format and restore wrapper/unwrapped state on exit."""
    unwrapped = env.unwrapped
    saved_flag = unwrapped._use_pretrained_obs_format
    saved_unwrapped_space = unwrapped.observation_space
    saved_wrapper_space = env.observation_space
    try:
        unwrapped._use_pretrained_obs_format = True
        unwrapped.observation_space = PRETRAINED_OBS_SPACE
        env.observation_space = PRETRAINED_OBS_SPACE
        yield
    finally:
        unwrapped._use_pretrained_obs_format = saved_flag
        unwrapped.observation_space = saved_unwrapped_space
        env.observation_space = saved_wrapper_space


class ChessBaseEnv(ChessSimulationEnv):
    """Shared scaffolding: configs, observation space, reset, soft_reset, helpers."""

    def __init__(
        self,
        force_scenario=None,
        hide_object=True,
        show_chess_pieces=False,
        debug=False,
        **kwargs,
    ):
        # _get_obs() is called during super().__init__() — only these three are needed early.
        self._use_pretrained_obs_format = False
        self.current_scenario = None
        self.goal_pos = None
        super().__init__(debug=debug, **kwargs)
        self._init_scenario_params(force_scenario, hide_object, show_chess_pieces, debug)
        self._init_observation_space()
        self._init_task_constants()
        self._init_logger(debug)

    def _init_scenario_params(
        self, force_scenario, hide_object, show_chess_pieces, debug
    ) -> None:
        """Set configs, scenario flags, and episode counters."""
        self.chess_cfg = load_config("chess")

        self.force_scenario = force_scenario
        self.hide_object = hide_object
        self.show_chess_pieces = show_chess_pieces

        if debug:
            print(f"[DEBUG INIT] Pid {os.getpid()} | Scenario: {force_scenario}")

        self.tube_center_xy = None
        self.episode_steps = 0
        self.total_env_steps = 0
        self.episode_number = 0

    def _init_observation_space(self) -> None:
        """Set the native chess observation space (overridden by training env for pretrained obs format)."""
        self.observation_space = spaces.Dict(
            {
                "grip_vel": spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32"),
                "observation": spaces.Box(-np.inf, np.inf, shape=(7,), dtype="float32"),
                "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32"),
                "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype="float32"),
            }
        )

    def _init_task_constants(self) -> None:
        """Set scenario Z-levels, finger limits, and runtime state from config."""
        self.GRASP_Z = self.env_cfg["grasp_z"]
        self.HOVER_Z = self.env_cfg["hover_z"]
        self.SAFE_Z = self.env_cfg["safe_z"]
        self.SUCCESS_THRESHOLD = self.env_cfg["success_threshold"]
        self.FLOOR_LIMIT = self.env_cfg["floor_limit"]
        self.HIDDEN_OBJECT_POS = np.array(self.env_cfg["hidden_object_pos"])
        self.SETTLE_TOLERANCE = self.physics_cfg["settle_tolerance"]
        self.HALT_VEL_THRESHOLD = self.env_cfg["halt_vel_threshold"]

        self.FINGER_OPEN_JOINT = self.env_cfg["finger_open_joint"]
        self.FINGER_CLOSED_JOINT = self.env_cfg["finger_closed_joint"]
        self.finger_target_joint = self.FINGER_CLOSED_JOINT
        self.grasp_mode = False

        home_xy = self.env_cfg["home_position_xy"]
        self.HOME_POS = np.array([home_xy[0], home_xy[1], self.SAFE_Z])

        self.force_start_pos = None
        self.force_cube_pos = None
        self._board_mapper = BoardMapper.from_configs()
        self._piece_registry = PieceRegistry()
        self.active_piece_id = None
        self.active_piece_joint_name = None

    def _init_logger(self, debug: bool) -> None:
        """Set up per-process file logger (debug mode) or null logger (normal mode)."""
        log_cfg = self.env_cfg["logging"]
        log_dir = log_cfg["log_dir"]

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

    # ── Observation ──────────────────────────────────────────────────────────

    def _get_obs(self):
        """Minimal physics-state dict; switches to 25-D pretrained obs format when requested."""
        if self._use_pretrained_obs_format:
            return self._build_pretrained_observation()

        grip_pos = self.get_grip_pos().copy().astype(np.float32)
        grip_vel = self.get_grip_vel().astype(np.float32)
        finger_angle = np.float32(self.get_finger_angle())
        goal_pos = (
            self.goal_pos.copy().astype(np.float32)
            if self.goal_pos is not None
            else np.zeros(3, dtype=np.float32)
        )

        obs_vec = np.concatenate([grip_pos, grip_vel, [finger_angle]])

        return {
            "observation": obs_vec,
            "achieved_goal": grip_pos,
            "desired_goal": goal_pos,
            "grip_vel": grip_vel,
        }

    def _build_pretrained_observation(self):
        """25-D observation matching FetchPickAndPlace-v4's MultiInputPolicy layout."""
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
        goal = self.goal if self.goal is not None else np.zeros(3)
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

    # ── Low-level helpers (used by both reset and grasp/place) ──────────────

    def get_grip_pos(self) -> np.ndarray:
        """Return the current world position of the gripper site."""
        return self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()

    def get_grip_vel(self) -> np.ndarray:
        """Return the current world velocity of the gripper site."""
        return self._utils.get_site_xvelp(self.model, self.data, "robot0:grip").copy()

    def get_finger_angle(self) -> float:
        """Return the current left finger joint angle.

        Both fingers are always driven to the same target simultaneously, so
        left mirrors right exactly — checking one is sufficient.
        """
        return self._utils.get_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint"
        ).item()

    def _move_grip_to(
        self,
        target_pos: np.ndarray,
        max_steps: int = 150,
        tolerance: float = 0.001,
    ) -> bool:
        """Drive the arm so robot0:grip reaches target_pos. Handles site-to-body offsets."""
        for _ in range(max_steps):
            if np.linalg.norm(target_pos - self.get_grip_pos()) < tolerance:
                return True
            self._step_grip_toward(target_pos)

        return bool(np.linalg.norm(self.get_grip_pos() - target_pos) < tolerance)

    def _step_grip_toward(self, target_pos: np.ndarray) -> None:
        """Reset mocap to the body, then pull the grip site toward target_pos for one step."""
        self._set_action(np.zeros(4))
        self.data.mocap_pos[0][:3] += target_pos - self.get_grip_pos()
        self.data.mocap_quat[0][:] = self.VERTICAL_QUAT
        self._mujoco_step(None)


    def _mujoco_step(self, action):
        super()._mujoco_step(action)
        if self.render_mode == "human":
            self.render()

    def _commit_finger_target(self):
        """Enforce fingers and forward-propagate physics.

        Use during reset/soft_reset where no physics step follows immediately.
        For in-step enforcement _apply_finger_target (called by _set_action) suffices.
        """
        self._apply_finger_target()
        mujoco.mj_forward(self.model, self.data)

    def _settle_arm_to_start(self, arm_start_pos):
        """Move the arm to the starting position; prevents physics explosions and gravity sag."""
        self._move_grip_to(
            arm_start_pos,
            max_steps=100,
            tolerance=self.SETTLE_TOLERANCE,
        )
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _sample_goal(self):
        """Returns the goal sampled during reset_sim."""
        return self.goal_pos.copy()

    def _is_success(self, achieved_goal, desired_goal):
        """Check XY and Z precision against the success threshold."""
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)
        d_xy = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        d_z = abs(achieved_goal[2] - desired_goal[2])
        return float(d_xy < self.SUCCESS_THRESHOLD and d_z < self.SUCCESS_THRESHOLD)

    # ── Chess-piece body placement ──────────────────────────────────────────

    def _set_freejoint_pose(self, joint_name: str, xyz: np.ndarray, quat=None) -> None:
        quat = IDENTITY_QUAT if quat is None else np.asarray(quat, dtype=float)
        xyz = np.asarray(xyz, dtype=float)
        joint_id = self.model.joint(joint_name).id
        qpos_start = self.model.jnt_qposadr[joint_id]
        dof_start = self.model.jnt_dofadr[joint_id]
        self.data.qpos[qpos_start : qpos_start + 3] = xyz
        self.data.qpos[qpos_start + 3 : qpos_start + 7] = quat
        self.data.qvel[dof_start : dof_start + 6] = 0.0

    def _reserve_position(self, index: int, color: str, piece_type: str) -> np.ndarray:
        reserve_cfg = self.chess_cfg["promotion_reserve"][color]
        spacing = self.chess_cfg["reserves"]["promotion_slot_spacing_m"]
        origin = reserve_cfg["origin_xyz"]
        piece_type_offset = {"queen": 0, "rook": 8, "bishop": 16, "knight": 24}[
            piece_type
        ]
        slot = piece_type_offset + index
        row = slot // reserve_cfg["cols"]
        col = slot % reserve_cfg["cols"]
        return np.array(
            [origin[0] + row * spacing, origin[1] + col * spacing, origin[2]],
            dtype=float,
        )

    def _hidden_piece_position(self, index: int) -> np.ndarray:
        row = index // 12
        col = index % 12
        return np.array([2.2 + row * 0.05, -0.5 + col * 0.05, self.PIECE_HEIGHT / 2.0])

    def _reset_chess_piece_bodies(self) -> None:
        hidden_index = self._place_board_pieces(hidden_index=0)
        self._place_reserve_pieces(hidden_index)

    def _place_board_pieces(self, hidden_index: int) -> int:
        """Place each active piece on its starting square or off-screen. Returns next hidden_index."""
        for piece in self._piece_registry.all_pieces():
            if self.show_chess_pieces:
                xyz = self._board_mapper.square_to_piece_xyz(
                    chess.parse_square(piece.initial_square)
                )
            else:
                xyz = self._hidden_piece_position(hidden_index)
                hidden_index += 1
            self._set_freejoint_pose(piece.joint_name, xyz)
        return hidden_index

    def _place_reserve_pieces(self, hidden_index: int) -> None:
        """Place each promotion reserve piece in its reserve slot or off-screen."""
        for piece_id, color, piece_type in reserve_piece_ids():
            joint_name = f"piece_{piece_id}:joint"
            if self.show_chess_pieces:
                reserve_index = int(piece_id.rsplit("_", 1)[1]) - 1
                xyz = self._reserve_position(reserve_index, color, piece_type)
            else:
                xyz = self._hidden_piece_position(hidden_index)
                hidden_index += 1
            self._set_freejoint_pose(joint_name, xyz)

    # ── Reset ────────────────────────────────────────────────────────────────

    def _reset_sim(self):
        """Episode reset: scenario selection, object placement, scripted gripper transitions."""
        self.episode_steps = 0
        self.episode_number += 1
        self.grasp_mode = False

        # Must run before super()._reset_sim() — the parent calls _sample_goal()
        # which reads self.goal_pos, so it must be set first.
        arm_start_pos, start_xy = self._setup_episode()
        super()._reset_sim()
        self._place_objects(start_xy)
        self._prepare_arm_for_episode(arm_start_pos)

        if not self._validate_finger_state():
            return False

        self._on_reset_settled(arm_start_pos)
        if self.debug:
            self._log_reset_state()
        return True

    def _setup_episode(self) -> tuple[np.ndarray, np.ndarray]:
        """Pick scenario and compute all episode positions.

        Sets self.current_scenario, self.goal_pos, self.tube_center_xy, self.goal.
        Returns (arm_start_pos, start_xy):
          arm_start_pos — where the arm settles and fingers transition during reset.
          start_xy      — board XY used to position the cube object.
        """
        self.current_scenario = (
            self.force_scenario
            if self.force_scenario is not None
            else self.np_random.choice(["transit", "descend", "ascend"])
        )

        start_xy = self._random_board_position()[:2]

        if self.current_scenario == "transit":
            goal_pos = self._random_board_position()
            while np.linalg.norm(goal_pos[:2] - start_xy) < self.MIN_GOAL_DIST:
                goal_pos = self._random_board_position()
            self.tube_center_xy = None
            self.goal_pos = np.array([*goal_pos[:2], self.SAFE_Z])

            if self.force_start_pos is not None:
                arm_start_pos = self.force_start_pos.copy()
            elif self.show_chess_pieces:
                arm_start_pos = self.HOME_POS.copy()
            else:
                arm_start_pos = np.array([*start_xy, self.SAFE_Z])
        else:  # descend or ascend — same XY tube, mirrored Z values
            self.tube_center_xy = start_xy.copy()
            arm_z, goal_z = (
                (self.SAFE_Z, self.HOVER_Z) if self.current_scenario == "descend"
                else (self.HOVER_Z, self.SAFE_Z)
            )
            arm_start_pos = np.array([*start_xy, arm_z])
            self.goal_pos = np.array([*start_xy, goal_z])

        self.goal = self.goal_pos.copy()
        return arm_start_pos, start_xy

    def _place_objects(self, start_xy: np.ndarray) -> None:
        """Place the dummy cube and all chess piece bodies, then forward-propagate."""
        if self.hide_object:
            self._set_freejoint_pose("object0:joint", self.HIDDEN_OBJECT_POS)
        elif self.force_cube_pos is not None:
            self._set_freejoint_pose("object0:joint", self.force_cube_pos[:3])
        else:
            cube_xyz = np.array([
                start_xy[0], start_xy[1],
                self.TABLE_SURFACE_Z + self.PIECE_HEIGHT / 2.0,
            ])
            self._set_freejoint_pose("object0:joint", cube_xyz)

        self._reset_chess_piece_bodies()
        mujoco.mj_forward(self.model, self.data)

        torso_height = self.env_cfg["torso_height"]
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:torso_lift_joint", torso_height
        )
        mujoco.mj_forward(self.model, self.data)

    def _prepare_arm_for_episode(self, arm_start_pos: np.ndarray) -> None:
        """Settle arm at start position with scenario-appropriate finger state."""
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT
        )
        # Always start settled with fingers closed for physics stability.
        self.finger_target_joint = self.FINGER_CLOSED_JOINT
        self._commit_finger_target()
        self._settle_arm_to_start(arm_start_pos)

        if self.current_scenario == "descend":
            self._set_fingers_and_move_to(arm_start_pos, self.FINGER_OPEN_JOINT)
        elif self.current_scenario == "ascend":
            self._set_fingers_and_move_to(arm_start_pos, self.FINGER_OPEN_JOINT, max_steps=100)
            self._set_fingers_and_move_to(arm_start_pos, self.FINGER_CLOSED_JOINT)
        else:
            self._set_fingers_and_move_to(arm_start_pos, self.FINGER_CLOSED_JOINT)

    def _set_fingers_and_move_to(
        self, target: np.ndarray, finger_state: float, max_steps: int = 150
    ) -> None:
        """Set finger target, enforce it, then move the grip to target."""
        self.finger_target_joint = finger_state
        self._commit_finger_target()
        self._move_grip_to(target, max_steps=max_steps, tolerance=0.003)

    def _validate_finger_state(self) -> bool:
        """Return False and log an error if the finger didn't reach its target."""
        finger_angle = self.get_finger_angle()
        if abs(finger_angle - self.finger_target_joint) > 0.0005:
            self.logger.error(
                f"Reset Failed: Finger joint at {finger_angle:.6f}, "
                f"target {self.finger_target_joint:.6f} (Scenario: {self.current_scenario})"
            )
            return False
        return True

    def _log_reset_state(self) -> None:
        """Log a full state dump at the start of a debug episode."""
        obj_qpos_start = self.model.jnt_qposadr[self.model.joint("object0:joint").id]
        obj_pos_now = self.data.qpos[obj_qpos_start : obj_qpos_start + 3]
        arm_joints = [
            "robot0:shoulder_pan_joint", "robot0:shoulder_lift_joint",
            "robot0:upperarm_roll_joint", "robot0:elbow_flex_joint",
            "robot0:forearm_roll_joint", "robot0:wrist_flex_joint",
            "robot0:wrist_roll_joint",
        ]
        joint_qpos = []
        for jname in arm_joints:
            try:
                joint_qpos.append(self._utils.get_joint_qpos(self.model, self.data, jname).item())
            except Exception:
                joint_qpos.append(float("nan"))

        tube_str = f"{self.tube_center_xy}" if self.tube_center_xy is not None else "N/A"
        self.logger.debug(
            f"\n{'=' * 80}\n"
            f"[EPISODE {self.episode_number} START] Scenario: {self.current_scenario.upper()}\n"
            f"  GripPos:    [{', '.join(f'{x:.4f}' for x in self.get_grip_pos())}]\n"
            f"  Goal:       [{', '.join(f'{x:.4f}' for x in self.goal_pos)}]\n"
            f"  TubeCenter: {tube_str}\n"
            f"  ObjPos:     [{', '.join(f'{x:.4f}' for x in obj_pos_now)}]\n"
            f"  ArmJoints:  [{', '.join(f'{x:.4f}' for x in joint_qpos)}]\n"
            f"{'=' * 80}"
        )

    def _on_reset_settled(self, arm_start_pos: np.ndarray) -> None:
        """Hook for subclasses to react once the reset arm is settled. Default no-op."""
        return None

    # ── Soft reset (scenario swap mid-episode without teleporting the arm) ──

    def soft_reset(
        self,
        new_scenario: str,
        new_goal_pos: np.ndarray,
        nominal_exit_pos: np.ndarray,
        nominal_xy: np.ndarray | None = None,
    ) -> dict:
        """Transition to a new scenario without teleporting the arm."""
        self._halt_arm()
        self._align_to_exit_pos(nominal_exit_pos)

        prev_scenario = self.current_scenario
        self._update_scenario_state(new_scenario, new_goal_pos, nominal_xy)

        if not self.grasp_mode:
            self._transition_fingers_for_scenario(nominal_exit_pos, new_scenario, prev_scenario)

        return self._get_obs()

    def _update_scenario_state(
        self, new_scenario: str, new_goal_pos: np.ndarray, nominal_xy: np.ndarray | None
    ) -> None:
        """Switch scenario context: goal, tube centre, wrist orientation, and step counter."""
        if new_scenario in {"descend", "ascend"} and nominal_xy is None:
            raise ValueError(f"nominal_xy required for {new_scenario}")

        self.current_scenario = new_scenario
        self.goal_pos = new_goal_pos.copy()
        self.goal = self.goal_pos.copy()
        self.episode_steps = 0
        self.tube_center_xy = nominal_xy.copy() if new_scenario in {"descend", "ascend"} else None
        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT)

    def _zero_robot_dynamics(self) -> None:
        """Zero robot joint velocity and acceleration, then forward-propagate.

        Only zeros robot DOFs (not object joints) so piece physics is preserved.
        """
        # 3 slides + 1 torso + 2 head + 7 arm + 2 fingers
        robot_dof = 15
        self.data.qvel[:robot_dof] = 0.0
        self.data.qacc[:robot_dof] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _halt_arm(self) -> None:
        """Zero robot velocity and wait up to 30 steps for the arm to stop."""
        self._zero_robot_dynamics()

        for _ in range(30):
            if np.linalg.norm(self.get_grip_vel()) < self.HALT_VEL_THRESHOLD:
                break
            self._set_action(np.zeros(4))
            self._mujoco_step(None)

        self._zero_robot_dynamics()

        speed = float(np.linalg.norm(self.get_grip_vel()))
        if speed >= self.HALT_VEL_THRESHOLD:
            raise RuntimeError(
                f"soft_reset HALT_FAILED: speed={speed * 1000:.3f}mm/s >= "
                f"threshold={self.HALT_VEL_THRESHOLD * 1000:.1f}mm/s"
            )

    def _align_to_exit_pos(self, nominal_exit_pos: np.ndarray) -> None:
        """Nudge arm to nominal_exit_pos via capped physics steps."""
        ALIGN_MAX_STEPS = 80
        ALIGN_MAX_STEP_M = 0.012
        ALIGN_TOLERANCE_M = 0.004
        for _ in range(ALIGN_MAX_STEPS):
            error = nominal_exit_pos - self.get_grip_pos()
            if np.linalg.norm(error) < ALIGN_TOLERANCE_M:
                self._zero_robot_dynamics()
                return
            step_vec = error / max(np.linalg.norm(error), 1e-9) * min(np.linalg.norm(error), ALIGN_MAX_STEP_M)
            self.data.mocap_pos[0][:3] += step_vec
            self._mujoco_step(None)

        raise RuntimeError(f"soft_reset ALIGN_FAILED: did not converge to {nominal_exit_pos}")

    def _transition_fingers_for_scenario(
        self, nominal_exit_pos: np.ndarray, new_scenario: str, prev_scenario: str
    ) -> None:
        """Open or close fingers for the new scenario, then validate the result."""
        needs_open = new_scenario == "descend" and prev_scenario != "descend"
        needs_close = new_scenario in {"ascend", "transit"} and prev_scenario == "descend"

        if needs_open:
            self._set_fingers_and_move_to(nominal_exit_pos, self.FINGER_OPEN_JOINT, max_steps=40)
        elif needs_close:
            self._set_fingers_and_move_to(nominal_exit_pos, self.FINGER_CLOSED_JOINT, max_steps=40)

        finger_angle = self.get_finger_angle()
        if abs(finger_angle - self.finger_target_joint) > 0.0005:
            raise RuntimeError(
                f"soft_reset FINGER_VALIDATION_FAILED: "
                f"actual={finger_angle:.6f}, target={self.finger_target_joint:.6f}"
            )
