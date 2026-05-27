"""Piece state and home-posture helpers for ChessTaskEnv."""

from __future__ import annotations

import chess
import mujoco
import numpy as np

from src.chess_game.board_mapper import BoardMapper
from src.physical.piece_registry import PieceRegistry, reserve_piece_ids

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


class TaskStateMixin:
    """Provide active-piece, cube, and home posture helpers."""

    def get_cube_position(self) -> np.ndarray:
        """Returns the selected active piece position."""
        return self.get_active_piece_position()

    def get_cube_quat(self) -> np.ndarray:
        """Returns the selected active piece quaternion."""
        return self.get_active_piece_quat()

    def set_active_piece(self, piece_id: str) -> None:
        """Run set active piece logic."""
        if self._piece_registry is None:
            self._piece_registry = PieceRegistry()
        piece = self._piece_registry.by_id(piece_id)
        self.active_piece_id = piece.piece_id
        self.active_piece_body_name = piece.body_name
        self.active_piece_joint_name = piece.joint_name

    def clear_active_piece(self) -> None:
        """Run clear active piece logic."""
        self.active_piece_id = None
        self.active_piece_body_name = None
        self.active_piece_joint_name = None

    def get_active_piece_position(self) -> np.ndarray:
        """Run get active piece position logic."""
        if self.active_piece_joint_name is None:
            raise RuntimeError("No active chess piece selected.")
        joint_id = self.model.joint(self.active_piece_joint_name).id
        qpos_start = self.model.jnt_qposadr[joint_id]
        return self.data.qpos[qpos_start : qpos_start + 3].copy()

    def get_active_piece_quat(self) -> np.ndarray:
        """Run get active piece quat logic."""
        if self.active_piece_joint_name is None:
            raise RuntimeError("No active chess piece selected.")
        joint_id = self.model.joint(self.active_piece_joint_name).id
        qpos_start = self.model.jnt_qposadr[joint_id]
        return self.data.qpos[qpos_start + 3 : qpos_start + 7].copy()

    def _set_freejoint_pose(self, joint_name: str, xyz: np.ndarray, quat=None) -> None:
        """Run  set freejoint pose logic."""
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
        """Run  reserve position logic."""
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
        """Run  hidden piece position logic."""
        row = index // 12
        col = index % 12
        return np.array([2.2 + row * 0.05, -0.5 + col * 0.05, self.CUBE_HEIGHT / 2.0])

    def _reset_chess_piece_bodies(self) -> None:
        """Run  reset chess piece bodies logic."""
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

    def _sample_goal(self):
        """Returns the goal sampled during reset_sim."""
        return self.goal_pos.copy()

    def _is_success(self, achieved_goal, desired_goal):
        """
        Determines if the gripper is within the success threshold of the goal.
        Checks both 2D (XY) and height (Z) precision.
        """
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)
        d_xy = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        d_z = abs(achieved_goal[2] - desired_goal[2])

        return float(d_xy < self.SUCCESS_THRESHOLD and d_z < self.SUCCESS_THRESHOLD)

    def _set_gripper_state(self):
        """Physically sets the joint positions of the fingers based on the target state."""
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

        # Sync actuator ctrl to prevent position actuator from fighting the teleport
        l_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_joint"
        )
        r_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:r_gripper_finger_joint"
        )
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
            return (
                False,
                f"CUBE_DROPPED_XY (err={xy_error * 1000:.1f}mm > limit={self.CUBE_HELD_XY_LIMIT * 1000:.0f}mm)",
            )
        if z_error > self.CUBE_HELD_Z_LIMIT:
            return (
                False,
                f"CUBE_DROPPED_Z (err={z_error * 1000:.1f}mm > limit={self.CUBE_HELD_Z_LIMIT * 1000:.0f}mm)",
            )
        return True, None

    def _mujoco_step(self, action):
        """Run  mujoco step logic."""
        super()._mujoco_step(action)
        if self._debug_step_callback is not None:
            self._debug_step_callback(self._debug_current_phase)

    def _settle_arm_to_start(self, arm_start_pos):
        """
        Uses the robust _move_mocap_to method to move the arm to the starting position.
        Prevents physics explosions and handles gravity sag.
        """
        self._move_mocap_to(
            arm_start_pos,
            self.VERTICAL_QUAT,
            max_steps=100,
            tolerance=self.SETTLE_TOLERANCE,
        )

        # Zero velocities to ensure a stable episode start
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _capture_home_posture(self) -> None:
        """Remember the canonical reset posture used between chess moves."""
        self._home_posture_qpos = {}
        for joint_name in HOME_POSTURE_JOINTS:
            joint = self.model.joint(joint_name)
            self._home_posture_qpos[joint_name] = float(
                self.data.qpos[joint.qposadr[0]]
            )
        self._home_posture_mocap_pos = self.data.mocap_pos[0][:3].copy()
        self._home_posture_mocap_quat = self.data.mocap_quat[0].copy()
        self._home_posture_grip_pos = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()

    def reset_arm_to_home_posture(self) -> dict:
        """
        Restore the exact reset-time home joint posture after returning to home.

        Moving to home by end-effector XYZ alone can leave redundant wrist/roll
        joints in different configurations. This normalizes those joints once the
        gripper is safely back at home and no piece is held.
        """
        result = {"success": False, "reason": None, "final_error_mm": 0.0}

        if self.grasp_mode:
            result["reason"] = "HOME_POSTURE_RESET_BLOCKED_HELD_PIECE"
            return result

        if self._home_posture_qpos is None:
            result["reason"] = "HOME_POSTURE_NOT_CAPTURED"
            return result

        self._debug_current_phase = "home_posture_reset"
        self.current_scenario = "transit"
        self.tube_center_xy = None
        self.goal_pos = self.HOME_POS.copy()
        self.goal = self.goal_pos.copy()
        self.finger_target_joint = self.FINGER_CLOSED_JOINT
        self.grasp_mode = False

        should_render = self.render_mode == "human"
        l_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_joint"
        )
        r_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:r_gripper_finger_joint"
        )

        # Snapshot current state so we can interpolate smoothly to target
        start_qpos = {
            joint_name: float(self.data.qpos[self.model.joint(joint_name).qposadr[0]])
            for joint_name in self._home_posture_qpos
        }
        start_mocap_pos = self.data.mocap_pos[0][:3].copy()
        start_mocap_quat = self.data.mocap_quat[0].copy()

        N_STEPS = 10
        for i in range(N_STEPS):
            t = (i + 1) / N_STEPS
            for joint_name, target in self._home_posture_qpos.items():
                joint = self.model.joint(joint_name)
                self.data.qpos[joint.qposadr[0]] = start_qpos[joint_name] + t * (
                    target - start_qpos[joint_name]
                )
                self.data.qvel[joint.dofadr[0]] = 0.0

            mocap_pos = start_mocap_pos + t * (
                self._home_posture_mocap_pos - start_mocap_pos
            )
            mocap_quat = start_mocap_quat + t * (
                self._home_posture_mocap_quat - start_mocap_quat
            )
            mocap_quat /= np.linalg.norm(mocap_quat)
            self.data.mocap_pos[0][:3] = mocap_pos
            self.data.mocap_quat[0][:] = mocap_quat
            self.data.ctrl[l_id] = self.FINGER_CLOSED_JOINT
            self.data.ctrl[r_id] = self.FINGER_CLOSED_JOINT
            mujoco.mj_forward(self.model, self.data)
            if should_render:
                self.render()

        # Final snap to exact targets and zero residual dynamics
        for joint_name, qpos in self._home_posture_qpos.items():
            joint = self.model.joint(joint_name)
            self.data.qpos[joint.qposadr[0]] = qpos
            self.data.qvel[joint.dofadr[0]] = 0.0
            self.data.qacc[joint.dofadr[0]] = 0.0

        self.data.mocap_pos[0][:3] = self._home_posture_mocap_pos
        self.data.mocap_quat[0][:] = self._home_posture_mocap_quat
        self.data.ctrl[l_id] = self.FINGER_CLOSED_JOINT
        self.data.ctrl[r_id] = self.FINGER_CLOSED_JOINT
        mujoco.mj_forward(self.model, self.data)

        final_grip = self._utils.get_site_xpos(
            self.model, self.data, "robot0:grip"
        ).copy()
        result["final_error_mm"] = float(
            np.linalg.norm(final_grip - self._home_posture_grip_pos) * 1000.0
        )
        result["success"] = result["final_error_mm"] < 1.0
        if not result["success"]:
            result["reason"] = (
                f"HOME_POSTURE_RESET_FAILED ({result['final_error_mm']:.3f}mm)"
            )
        return result
