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

from src.chess_env.simulation import ChessSimulationEnv
from src.chess_game.board_mapper import BoardMapper
from src.physical.piece_registry import PieceRegistry, reserve_piece_ids
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

TRANSFER_OBS_SPACE = spaces.Dict(
    {
        "observation": spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float64),
        "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
        "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
    }
)


def enable_transfer_obs(env) -> None:
    """Enable transfer observations on both the wrapper and unwrapped env."""
    unwrapped = env.unwrapped
    unwrapped._use_transfer_obs = True
    unwrapped.observation_space = TRANSFER_OBS_SPACE
    env.observation_space = TRANSFER_OBS_SPACE


@contextmanager
def transfer_obs_enabled(env) -> Generator[None, None, None]:
    """Enable transfer observations and restore wrapper/unwrapped state on exit."""
    unwrapped = env.unwrapped
    saved_flag = unwrapped._use_transfer_obs
    saved_unwrapped_space = unwrapped.observation_space
    saved_wrapper_space = env.observation_space
    try:
        enable_transfer_obs(env)
        yield
    finally:
        unwrapped._use_transfer_obs = saved_flag
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
        self.env_cfg = load_config("env")
        self.physics_cfg = load_config("physics")
        self.chess_cfg = load_config("chess")

        self.force_scenario = force_scenario
        self.hide_object = hide_object
        self.show_chess_pieces = show_chess_pieces
        self.base_debug = debug
        self.debug = debug

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

        super().__init__(debug=debug, **kwargs)

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
                "scenario_id": spaces.Discrete(4),
            }
        )

        self.CUBE_HEIGHT = self.env_cfg["cube_height"]
        self.CUBE_Z = self.env_cfg["cube_z"]
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

        self._debug_step_callback = None
        self._debug_current_phase = "idle"

        home_xy = self.env_cfg["home_position_xy"]
        self.HOME_POS = np.array([home_xy[0], home_xy[1], self.SAFE_Z])

        # Overrides typically used by evaluation / scripted control
        self.force_start_pos = None
        self.force_cube_pos = None
        self._board_mapper = None
        self._piece_registry = None
        self.active_piece_id = None
        self.active_piece_body_name = None
        self.active_piece_joint_name = None

        log_cfg = self.env_cfg["logging"]
        log_dir = log_cfg["log_dir"]
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger(f"chess_task_{os.getpid()}")
        if not self.logger.handlers:
            if debug:
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
        """Minimal physics-state dict; switches to 25-D transfer obs when requested."""
        if self._use_transfer_obs:
            return self._build_transfer_observation()

        grip_pos = self._grip_pos().copy().astype(np.float32)
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

        obs_vec = np.concatenate([grip_pos, grip_vel, [l_finger]])
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

    # ── Low-level helpers (used by both reset and grasp/place) ──────────────

    def _grip_pos(self) -> np.ndarray:
        """Return the current world position of the gripper site."""
        return self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()

    def _move_mocap_to(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        max_steps: int = 150,
        tolerance: float = 0.001,
    ) -> bool:
        """Drive the arm so robot0:grip reaches target_pos. Handles site-to-body offsets."""
        zero_action = np.zeros(4)
        should_render = self.render_mode == "human"
        for _ in range(max_steps):
            grip_pos = self._grip_pos()
            error = target_pos - grip_pos
            if np.linalg.norm(error) < tolerance:
                return True

            self._set_action(zero_action)
            self.data.mocap_pos[0][:3] += error
            self.data.mocap_quat[0][:] = target_quat
            self._mujoco_step(None)
            if should_render:
                self.render()

        final_pos = self._grip_pos()
        return bool(np.linalg.norm(final_pos - target_pos) < tolerance)

    def _step_locked_grip(self, target_pos: np.ndarray, should_render: bool) -> None:
        """Reset mocap to the body, then pull the grip site to a target pose for one step."""
        self._set_action(np.zeros(4))
        error = target_pos - self._grip_pos()
        self.data.mocap_pos[0][:3] += error
        self.data.mocap_quat[0][:] = self.VERTICAL_QUAT
        self._mujoco_step(None)
        if should_render:
            self.render()

    def _mujoco_step(self, action):
        super()._mujoco_step(action)
        if self._debug_step_callback is not None:
            self._debug_step_callback(self._debug_current_phase)

    def _set_gripper_state(self):
        """Physically set the joint positions of the fingers based on the target state."""
        target = self.finger_target_joint
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint", target
        )
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:r_gripper_finger_joint", target
        )
        self.data.qvel[self.model.joint("robot0:l_gripper_finger_joint").dofadr[0]] = (
            0.0
        )
        self.data.qvel[self.model.joint("robot0:r_gripper_finger_joint").dofadr[0]] = (
            0.0
        )

        # Sync actuator ctrl so the position actuator doesn't fight the teleport
        l_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_joint"
        )
        r_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:r_gripper_finger_joint"
        )
        self.data.ctrl[l_id] = target
        self.data.ctrl[r_id] = target

        mujoco.mj_forward(self.model, self.data)

    def _settle_arm_to_start(self, arm_start_pos):
        """Move the arm to the starting position; prevents physics explosions and gravity sag."""
        self._move_mocap_to(
            arm_start_pos,
            self.VERTICAL_QUAT,
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
        quat = (
            np.array([1.0, 0.0, 0.0, 0.0])
            if quat is None
            else np.asarray(quat, dtype=float)
        )
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
        return np.array([2.2 + row * 0.05, -0.5 + col * 0.05, self.CUBE_HEIGHT / 2.0])

    def _reset_chess_piece_bodies(self) -> None:
        if self._board_mapper is None:
            self._board_mapper = BoardMapper.from_configs()
        if self._piece_registry is None:
            self._piece_registry = PieceRegistry()

        hidden_index = 0
        for piece in self._piece_registry.all_pieces():
            if self.show_chess_pieces:
                xyz = self._board_mapper.square_to_piece_xyz(
                    chess.parse_square(piece.initial_square)
                )
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

    # ── Reset ────────────────────────────────────────────────────────────────

    def _reset_sim(self):
        """Episode reset: scenario selection, object placement, scripted gripper transitions."""
        self.episode_steps = 0
        self.episode_number += 1

        self.debug = self.base_debug
        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        if self.force_scenario is not None:
            self.current_scenario = self.force_scenario
        else:
            self.current_scenario = self.np_random.choice(
                ["transit", "descend", "ascend"]
            )

        self.grasp_mode = False

        start_pos = self._random_board_position()
        start_xy = start_pos[:2]

        if self.current_scenario == "transit":
            goal_pos = self._random_board_position()
            while np.linalg.norm(goal_pos[:2] - start_xy) < self.MIN_GOAL_DIST:
                goal_pos = self._random_board_position()
            self.tube_center_xy = None

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

        if self.goal_pos is not None:
            self.goal = self.goal_pos.copy()

        super()._reset_sim()

        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start = self.model.jnt_qposadr[obj_joint_id]
        dof_start = self.model.jnt_dofadr[obj_joint_id]

        if self.hide_object:
            self.data.qpos[qpos_start : qpos_start + 3] = self.HIDDEN_OBJECT_POS
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0
        elif self.force_cube_pos is not None:
            self.data.qpos[qpos_start : qpos_start + 3] = self.force_cube_pos[:3]
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0
        else:
            self.data.qpos[qpos_start : qpos_start + 2] = start_xy
            self.data.qpos[qpos_start + 2] = self.TABLE_SURFACE_Z + (self.CUBE_HEIGHT / 2.0)
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0

        self._reset_chess_piece_bodies()
        mujoco.mj_forward(self.model, self.data)

        torso_height = self.env_cfg["torso_height"]
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:torso_lift_joint", torso_height
        )
        mujoco.mj_forward(self.model, self.data)

        # Phase 1: settle arm CLOSED for stability
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT
        )
        self.finger_target_joint = self.FINGER_CLOSED_JOINT
        self._set_gripper_state()
        self._settle_arm_to_start(arm_start_pos)

        # Phase 2: scripted finger transitions
        if self.current_scenario == "descend":
            self.finger_target_joint = self.FINGER_OPEN_JOINT
            self._set_gripper_state()
            self._move_mocap_to(
                arm_start_pos, self.VERTICAL_QUAT, max_steps=150, tolerance=0.003
            )
        elif self.current_scenario == "ascend":
            self.finger_target_joint = self.FINGER_OPEN_JOINT
            self._set_gripper_state()
            self._move_mocap_to(
                arm_start_pos, self.VERTICAL_QUAT, max_steps=100, tolerance=0.003
            )
            self.finger_target_joint = self.FINGER_CLOSED_JOINT
            self._set_gripper_state()
            self._move_mocap_to(
                arm_start_pos, self.VERTICAL_QUAT, max_steps=150, tolerance=0.003
            )
        else:
            self.finger_target_joint = self.FINGER_CLOSED_JOINT
            self._move_mocap_to(
                arm_start_pos, self.VERTICAL_QUAT, max_steps=150, tolerance=0.003
            )

        # Phase 3: validate final finger state
        l_pos = self._utils.get_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint"
        ).item()
        if abs(l_pos - self.finger_target_joint) > 0.0005:
            self.logger.error(
                f"Reset Failed: Finger joint at {l_pos:.6f}, target {self.finger_target_joint:.6f} (Scenario: {self.current_scenario})"
            )
            return False

        self._on_reset_settled(arm_start_pos)

        if self.debug:
            obj_pos_now = self.data.qpos[qpos_start : qpos_start + 3]
            grip_pos_now = self._grip_pos()
            arm_joints = [
                "robot0:shoulder_pan_joint",
                "robot0:shoulder_lift_joint",
                "robot0:upperarm_roll_joint",
                "robot0:elbow_flex_joint",
                "robot0:forearm_roll_joint",
                "robot0:wrist_flex_joint",
                "robot0:wrist_roll_joint",
            ]
            joint_qpos = []
            for jname in arm_joints:
                try:
                    joint_qpos.append(
                        self._utils.get_joint_qpos(self.model, self.data, jname).item()
                    )
                except Exception:
                    joint_qpos.append(float("nan"))
            jq_str = ", ".join([f"{x:.4f}" for x in joint_qpos])
            tube_str = (
                f"{self.tube_center_xy}" if self.tube_center_xy is not None else "N/A"
            )
            msg = (
                f"\n{'=' * 80}\n"
                f"[EPISODE {self.episode_number} START] Scenario: {self.current_scenario.upper()}\n"
                f"  GripPos:    [{', '.join([f'{x:.4f}' for x in grip_pos_now])}]\n"
                f"  Goal:       [{', '.join([f'{x:.4f}' for x in self.goal_pos])}]\n"
                f"  TubeCenter: {tube_str}\n"
                f"  ObjPos:     [{', '.join([f'{x:.4f}' for x in obj_pos_now])}]\n"
                f"  ArmJoints:  [{jq_str}]\n"
                f"{'=' * 80}"
            )
            self.logger.debug(msg)

        return True

    def _on_reset_settled(self, arm_start_pos: np.ndarray) -> None:
        """Hook for subclasses to react once the reset arm is settled. Default no-op."""
        return None

    # ── Soft reset (scenario swap mid-episode without teleporting the arm) ──

    def transition_validate(self, nominal_exit_pos: np.ndarray | None = None) -> dict:
        """Return arm state diagnostics at a scenario transition point."""
        grip_pos = self._grip_pos()
        grip_vel = self._utils.get_site_xvelp(
            self.model, self.data, "robot0:grip"
        ).copy()
        speed = float(np.linalg.norm(grip_vel))
        l_finger = self._utils.get_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint"
        ).item()

        result = {
            "grip_pos": grip_pos.tolist(),
            "grip_speed_mm_s": speed * 1000,
            "is_velocity_ok": speed < self.HALT_VEL_THRESHOLD,
            "error_from_nominal_mm": None,
            "finger_state": l_finger,
        }
        if nominal_exit_pos is not None:
            result["error_from_nominal_mm"] = float(
                np.linalg.norm(grip_pos - nominal_exit_pos) * 1000
            )
        return result

    def soft_reset(
        self,
        new_scenario: str,
        new_goal_pos: np.ndarray,
        nominal_exit_pos: np.ndarray,
        nominal_xy: np.ndarray | None = None,
    ) -> tuple[dict, dict]:
        """Transition to a new scenario without teleporting the arm."""
        # Fetch: 3 slides + 1 torso + 2 head + 7 arm + 2 fingers
        ROBOT_DOF = 15
        HALT_HOLD_MAX_STEPS = 30
        ALIGN_TOLERANCE_M = 0.004
        ALIGN_MAX_STEPS = 80
        ALIGN_GAIN = 1.0
        ALIGN_MAX_STEP_M = 0.012

        # Phase 1: drive arm to a dead stop
        self._debug_current_phase = "softreset_p1_halt"
        zero_action = np.zeros(4)
        should_render = self.render_mode == "human"
        self.data.qvel[:ROBOT_DOF] = 0.0
        self.data.qacc[:ROBOT_DOF] = 0.0
        mujoco.mj_forward(self.model, self.data)
        halt_steps = 0
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        if np.linalg.norm(grip_vel) >= self.HALT_VEL_THRESHOLD:
            for _ in range(HALT_HOLD_MAX_STEPS):
                grip_vel = self._utils.get_site_xvelp(
                    self.model, self.data, "robot0:grip"
                )
                if np.linalg.norm(grip_vel) < self.HALT_VEL_THRESHOLD:
                    break
                self._set_action(zero_action)
                self._mujoco_step(None)
                if should_render:
                    self.render()
                halt_steps += 1

        self.data.qvel[:ROBOT_DOF] = 0.0
        self.data.qacc[:ROBOT_DOF] = 0.0
        mujoco.mj_forward(self.model, self.data)

        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        speed = float(np.linalg.norm(grip_vel))
        if speed >= self.HALT_VEL_THRESHOLD:
            raise RuntimeError(
                f"soft_reset HALT_FAILED: speed={speed * 1000:.3f}mm/s >= "
                f"threshold={self.HALT_VEL_THRESHOLD * 1000:.1f}mm/s"
            )

        # Phase 2: waypoint alignment via physics-simulated smooth movement
        self._debug_current_phase = "softreset_p2_align"
        converged = False
        align_steps = 0
        for loop_step in range(ALIGN_MAX_STEPS):
            grip_pos = self._grip_pos()
            error = nominal_exit_pos - grip_pos
            dist = np.linalg.norm(error)
            if dist < ALIGN_TOLERANCE_M:
                self.data.qvel[:ROBOT_DOF] = 0.0
                self.data.qacc[:ROBOT_DOF] = 0.0
                mujoco.mj_forward(self.model, self.data)
                converged = True
                break
            step_vec = ALIGN_GAIN * error
            if np.linalg.norm(step_vec) > ALIGN_MAX_STEP_M:
                step_vec = step_vec / np.linalg.norm(step_vec) * ALIGN_MAX_STEP_M
            self.data.mocap_pos[0][:3] += step_vec
            self._mujoco_step(None)
            if should_render:
                self.render()
            align_steps = loop_step + 1

        if not converged:
            raise RuntimeError(
                f"soft_reset ALIGN_FAILED: did not converge to {nominal_exit_pos}"
            )

        # Phase 3: state update
        self._debug_current_phase = "softreset_p3_state"
        prev_scenario = self.current_scenario
        self.current_scenario = new_scenario
        self.goal_pos = new_goal_pos.copy()
        self.goal = self.goal_pos.copy()
        self.episode_steps = 0

        if new_scenario in {"descend", "ascend"}:
            if nominal_xy is None:
                raise ValueError(f"nominal_xy required for {new_scenario}")
            self.tube_center_xy = nominal_xy.copy()
        else:
            self.tube_center_xy = None

        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT
        )

        # Phase 4: scripted gripper transition
        self._debug_current_phase = "softreset_p4_gripper"
        if not self.grasp_mode:
            needs_open = new_scenario == "descend" and prev_scenario != "descend"
            needs_close = (
                new_scenario in {"ascend", "transit"} and prev_scenario == "descend"
            )

            if needs_open:
                self.finger_target_joint = self.FINGER_OPEN_JOINT
                self._set_gripper_state()
                self._move_mocap_to(
                    nominal_exit_pos, self.VERTICAL_QUAT, max_steps=40, tolerance=0.004
                )
            elif needs_close:
                self.finger_target_joint = self.FINGER_CLOSED_JOINT
                self._set_gripper_state()
                self._move_mocap_to(
                    nominal_exit_pos, self.VERTICAL_QUAT, max_steps=40, tolerance=0.004
                )

            l_pos = self._utils.get_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint"
            ).item()
            if abs(l_pos - self.finger_target_joint) > 0.0005:
                raise RuntimeError(
                    f"soft_reset FINGER_VALIDATION_FAILED: "
                    f"actual={l_pos:.6f}, target={self.finger_target_joint:.6f}"
                )

        info = {"halt_steps": halt_steps, "align_steps": align_steps}
        return self._get_obs(), info
