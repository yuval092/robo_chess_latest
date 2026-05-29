"""Production environment: chess piece registry plus scripted grasp/place.

Used by the play loop (main.py). The arm is driven via execute_grasp / execute_place
and the embedded SAC controller, not via gym step() — there's no RL reward path here.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from src.chess_env.base_env import HOME_POSTURE_JOINTS, ChessBaseEnv
from src.physical.piece_registry import PieceRegistry


class ChessProductionEnv(ChessBaseEnv):
    """Adds chess-piece state tracking plus scripted grasp/place pipelines."""

    def __init__(self, force_scenario="transit", **kwargs):
        super().__init__(force_scenario=force_scenario, **kwargs)

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

        self._home_posture_qpos = None
        self._home_posture_mocap_pos = None
        self._home_posture_mocap_quat = None
        self._home_posture_grip_pos = None

    # ── Hook: capture posture when reset arm lands at HOME_POS ──────────────

    def _on_reset_settled(self, arm_start_pos: np.ndarray) -> None:
        if np.linalg.norm(arm_start_pos - self.HOME_POS) < 1e-9:
            self._capture_home_posture()

    # ── Active piece / cube state ───────────────────────────────────────────

    def get_cube_position(self) -> np.ndarray:
        """Return the selected active piece position."""
        return self.get_active_piece_position()

    def get_cube_quat(self) -> np.ndarray:
        """Return the selected active piece quaternion."""
        return self.get_active_piece_quat()

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

    def _check_cube_held(self, grip_pos: np.ndarray) -> tuple[bool, str | None]:
        """Verify the cube is still in the gripper. Called when grasp_mode is on."""
        cube_pos = self.get_cube_position()
        xy_error = np.linalg.norm(cube_pos[:2] - grip_pos[:2])
        # Cube CoM sits ~15mm below the grip site when held at center
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

    # ── Home posture capture / restore ──────────────────────────────────────

    def _capture_home_posture(self) -> None:
        """Snapshot the canonical reset posture used between chess moves."""
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
        """Restore the exact reset-time joint posture after the gripper returns to home.

        Moving home by XYZ alone can leave redundant wrist/roll joints in different
        configurations; this snaps them back to the canonical posture.
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

    # ── Scripted plunge / retract helpers used by grasp & place ─────────────

    def _plunge_to_z(
        self, xy: np.ndarray, target_z: float, step_m: float, should_render: bool
    ) -> tuple[np.ndarray, int]:
        grip_pos = self._grip_pos()
        commanded_z = grip_pos[2]
        steps = 0
        for _ in range(int(round(max(0.0, commanded_z - target_z) / step_m)) + 6):
            grip_pos = self._grip_pos()
            if grip_pos[2] <= target_z + 0.001:
                break
            commanded_z = max(target_z, commanded_z - step_m)
            self._step_locked_grip(np.array([xy[0], xy[1], commanded_z]), should_render)
            steps += 1
        return self._grip_pos(), steps

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
            grip_pos = self._grip_pos()
            if grip_pos[2] >= self.HOVER_Z - 0.001:
                break
            commanded_z = min(self.HOVER_Z, commanded_z + step_m)
            xy = xy_provider()
            self._step_locked_grip(np.array([xy[0], xy[1], commanded_z]), should_render)
            steps += 1

            if verify_held:
                cube_now = self.get_cube_position()
                grip_now = self._grip_pos()
                if abs(cube_now[2] - (grip_now[2] - 0.015)) > self.CUBE_HELD_Z_LIMIT:
                    return "CUBE_DROPPED_DURING_RETRACT", steps
        return None, steps

    def _hold_locked_target(
        self, target_provider, steps: int, should_render: bool
    ) -> None:
        for _ in range(steps):
            self._step_locked_grip(target_provider(), should_render)

    # ── Scripted grasp pipeline ─────────────────────────────────────────────

    def execute_grasp(self) -> dict:
        """Scripted GRASP pipeline. Runs after DESCEND succeeds at HOVER_Z.

        KEY INVARIANT: after every _set_action call, re-assert BOTH mocap_pos AND
        mocap_quat before _mujoco_step. _set_action resets mocap to physical body
        state, so the assertions must be repeated every step.
        """
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

        # ── Phase 0: Halt & Settle ────────────────────────────────────────────
        self._debug_current_phase = "grasp_p0_halt"
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._set_gripper_state()
        mujoco.mj_forward(self.model, self.data)

        should_render = self.render_mode == "human"

        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        if float(np.linalg.norm(grip_vel)) > 0.005:
            result["reason"] = "PRECONDITION_SPEED"
            return result

        grip_pos = self._grip_pos()
        if abs(grip_pos[2] - self.HOVER_Z) > 0.025:
            result["reason"] = (
                f"PRECONDITION_Z (grip={grip_pos[2] * 1000:.1f}mm, "
                f"HOVER_Z={self.HOVER_Z * 1000:.1f}mm)"
            )
            return result

        l_finger = self._utils.get_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint"
        ).item()
        if l_finger < self.FINGER_OPEN_JOINT - 0.003:
            result["reason"] = f"PRECONDITION_FINGERS_NOT_OPEN (j={l_finger:.4f})"
            return result

        # ── Phase 1+2: Rotation Abort Check + Perfect Align ──────────────────
        self._debug_current_phase = "grasp_p12_align"
        cube_pos = self.get_cube_position().copy()
        result["pre_grasp_cube_xy"] = cube_pos[:2].copy()

        self.logger.debug(
            f"[GRASP] Start. Cube at {cube_pos}, Grip at {self._grip_pos()}"
        )
        # Diagonal cube (yaw >25°) has effective width 42.4mm, exceeding max finger
        # opening (38mm). Abort before plunge to prevent stub.
        cube_quat = self.get_cube_quat()
        w, x, y, z = cube_quat
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = abs(math.atan2(siny_cosp, cosy_cosp))
        # 4-fold symmetry: fold [0,π] into [0,π/2], then find distance to nearest axis.
        # The naïve `min(yaw, abs(yaw - π/2))` fails above 90° (e.g. 175° → 85° false abort).
        yaw_modulo = yaw % (math.pi / 2)
        effective_yaw = min(yaw_modulo, (math.pi / 2) - yaw_modulo)
        if effective_yaw > 0.436:
            result["reason"] = f"CUBE_ROTATED (yaw={math.degrees(effective_yaw):.1f}°)"
            return result

        grip_pos = self._grip_pos()
        align_target = np.array([cube_pos[0], cube_pos[1], grip_pos[2]])
        if not self._move_mocap_to(
            align_target,
            self.VERTICAL_QUAT,
            max_steps=150,
            tolerance=self.GRASP_ALIGN_TOLERANCE,
        ):
            final_pos = self._grip_pos()
            align_error_mm = float(np.linalg.norm(final_pos - align_target) * 1000.0)
            result["reason"] = (
                "ROTATION_FAILED "
                f"(align error {align_error_mm:.1f}mm > {self.GRASP_ALIGN_TOLERANCE * 1000:.0f}mm)"
            )
            return result

        # ── Phase 3: Plunge (current Z → PLACE_Z) ────────────────────────────
        self._debug_current_phase = "grasp_p3_plunge"
        place_z = self.GRASP_Z
        grip_pos, _plunge_steps = self._plunge_to_z(
            cube_pos[:2], place_z, self.GRASP_PLUNGE_STEP_M, should_render
        )
        if self.debug:
            self.logger.debug(
                f"[GRASP] Post-Plunge. Grip at {grip_pos}, Cube at {self.get_cube_position()}"
            )
        if abs(grip_pos[2] - self.GRASP_Z) > 0.008:
            result["reason"] = (
                f"PLUNGE_FAILED (z={grip_pos[2] * 1000:.1f}mm, "
                f"target={self.GRASP_Z * 1000:.1f}mm)"
            )
            return result

        # ── Phase 4: Grasp (Finger Close, Linear Ramp) ───────────────────────
        # Ramp finger target from OPEN to a secure-grip value over the step budget.
        # Direct jump creates a large impulse; ramping limits contact shock.
        self._debug_current_phase = "grasp_p4_close"
        self.grasp_mode = True
        ramp_start = self.FINGER_OPEN_JOINT
        ramp_end = self.GRASP_RAMP_END
        ramp_delta = (ramp_start - ramp_end) / self.GRASP_CLOSE_STEPS
        empty_detect_threshold = self.EMPTY_GRASP_THRESHOLD
        empty_detect_start = max(8, int(self.GRASP_CLOSE_STEPS * 0.65))

        steps_used = 0
        for step in range(self.GRASP_CLOSE_STEPS):
            self.finger_target_joint = max(ramp_end, ramp_start - ramp_delta * step)
            live_cube = self.get_cube_position()
            self._step_locked_grip(
                np.array([live_cube[0], live_cube[1], self.GRASP_Z]), should_render
            )
            steps_used += 1

            # Early abort: fingers reached the secure-grip target with no cube contact.
            if step >= empty_detect_start:
                l_now = self._utils.get_joint_qpos(
                    self.model, self.data, "robot0:l_gripper_finger_joint"
                ).item()
                if l_now < empty_detect_threshold:
                    result["reason"] = (
                        f"FINGER_CLOSED_EMPTY (j={l_now:.4f} at step {step})"
                    )
                    result["close_steps_used"] = steps_used
                    return result

        result["close_steps_used"] = steps_used

        # ── Phase 5: Hold & Verify ───────────────────────────────────────────
        self._debug_current_phase = "grasp_p5_hold"

        def live_cube_target():
            live_cube = self.get_cube_position()
            return np.array([live_cube[0], live_cube[1], self.GRASP_Z])

        self._hold_locked_target(live_cube_target, self.GRASP_HOLD_STEPS, should_render)

        cube_pos = self.get_cube_position()
        grip_pos = self._grip_pos()
        l_finger = self._utils.get_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint"
        ).item()

        xy_error = float(np.linalg.norm(cube_pos[:2] - grip_pos[:2])) * 1000
        z_error = float(abs(cube_pos[2] - grip_pos[2])) * 1000

        result["post_grasp_cube_pos"] = cube_pos.copy()
        result["final_xy_error_mm"] = xy_error
        result["final_z_error_mm"] = z_error
        result["final_finger_pos"] = l_finger

        if xy_error > self.GRASP_VERIFY_XY_THRESHOLD * 1000:
            result["reason"] = (
                f"VERIFY_XY_FAILED ({xy_error:.1f}mm > "
                f"{self.GRASP_VERIFY_XY_THRESHOLD * 1000:.0f}mm)"
            )
            return result

        if z_error > self.GRASP_VERIFY_Z_THRESHOLD * 1000:
            result["reason"] = (
                f"VERIFY_Z_FAILED ({z_error:.1f}mm > "
                f"{self.GRASP_VERIFY_Z_THRESHOLD * 1000:.0f}mm)"
            )
            return result

        if l_finger < empty_detect_threshold:
            result["reason"] = f"VERIFY_FINGERS_CLOSED_EMPTY (j={l_finger:.4f})"
            return result

        # ── Phase 6: Retract (PLACE_Z → HOVER_Z) ─────────────────────────────
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
            _plunge_steps
            + result["close_steps_used"]
            + self.GRASP_HOLD_STEPS
            + _retract_steps
        )
        result["post_grasp_cube_pos"] = self.get_cube_position().copy()
        return result

    # ── Scripted place pipeline ─────────────────────────────────────────────

    def execute_place(self, dst_xy: np.ndarray) -> dict:
        """Scripted PLACE pipeline. Runs after DESCEND to HOVER_Z over destination.

        dst_xy: 2D destination XY (board square center).
        Arm exits at HOVER_Z, cube placed, grasp_mode=False.
        """
        result = {
            "success": False,
            "reason": None,
            "final_cube_pos": None,
            "final_xy_error_mm": 0.0,
        }

        # ── Phase 0: Halt & Settle ───────────────────────────────────────────
        # CRITICAL: zero the full simulation state ([:]), not just robot DOFs.
        self._debug_current_phase = "place_p0_halt"
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._set_gripper_state()
        mujoco.mj_forward(self.model, self.data)

        should_render = self.render_mode == "human"

        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        if float(np.linalg.norm(grip_vel)) > 0.005:
            result["reason"] = "PRECONDITION_SPEED"
            return result

        grip_pos = self._grip_pos()
        if abs(grip_pos[2] - self.HOVER_Z) > 0.025:
            result["reason"] = f"PRECONDITION_Z (grip={grip_pos[2] * 1000:.1f}mm)"
            return result

        # ── Phase 1+2: Vertical Correction + XY Align Over Destination ──────
        self._debug_current_phase = "place_p12_align"
        grip_pos = self._grip_pos()
        align_target = np.array([dst_xy[0], dst_xy[1], grip_pos[2]])
        if not self._move_mocap_to(
            align_target,
            self.VERTICAL_QUAT,
            max_steps=150,
            tolerance=self.GRASP_ALIGN_TOLERANCE,
        ):
            final_pos = self._grip_pos()
            align_error_mm = float(np.linalg.norm(final_pos - align_target) * 1000.0)
            result["reason"] = (
                "ROTATION_FAILED "
                f"(align error {align_error_mm:.1f}mm > {self.GRASP_ALIGN_TOLERANCE * 1000:.0f}mm)"
            )
            return result

        # ── Phase 3: Plunge (current Z → PLACE_Z) ────────────────────────────
        self._debug_current_phase = "place_p3_plunge"
        place_z = self.GRASP_Z
        _, _plunge_steps = self._plunge_to_z(
            dst_xy, place_z, self.GRASP_PLUNGE_STEP_M, should_render
        )

        # Verify plunge reached place_z before releasing. If we stalled mid-descent
        # and open here, the cube falls from height.
        grip_pos = self._grip_pos()
        place_pos = grip_pos.copy()
        if abs(grip_pos[2] - place_z) > 0.008:
            result["reason"] = (
                f"PLUNGE_FAILED (z={grip_pos[2] * 1000:.1f}mm, "
                f"target={place_z * 1000:.1f}mm)"
            )
            return result

        # ── Phase 4: Release (Linear Ramp Open) ──────────────────────────────
        # Ramp from GRASP_RAMP_END (actual grip position) to OPEN. Starting from
        # 0.0 would actively squeeze fingers before opening.
        self._debug_current_phase = "place_p4_release"
        release_target = place_pos.copy()
        ramp_start = self.GRASP_RAMP_END
        ramp_end = self.FINGER_OPEN_JOINT
        release_ramp_steps = self.RELEASE_RAMP_STEPS
        release_settle_steps = self.RELEASE_SETTLE_STEPS
        ramp_delta = (ramp_end - ramp_start) / release_ramp_steps

        for step in range(release_ramp_steps):
            self.finger_target_joint = min(ramp_end, ramp_start + ramp_delta * step)
            self._step_locked_grip(release_target, should_render)

        self.finger_target_joint = self.FINGER_OPEN_JOINT
        self._hold_locked_target(
            lambda: release_target, release_settle_steps, should_render
        )

        self.grasp_mode = False

        # ── Phase 5: Verify Placement ────────────────────────────────────────
        self._debug_current_phase = "place_p5_verify"
        cube_pos = self.get_cube_position()
        xy_error = float(np.linalg.norm(cube_pos[:2] - dst_xy[:2])) * 1000
        z_error = (
            float(abs(cube_pos[2] - (self.TABLE_Z + self.CUBE_HEIGHT / 2.0))) * 1000
        )

        result["final_cube_pos"] = cube_pos.copy()
        result["final_xy_error_mm"] = xy_error

        if xy_error > 20.0:
            result["reason"] = f"PLACE_XY_FAILED ({xy_error:.1f}mm drift from target)"
            return result

        if z_error > 10.0:
            result["reason"] = (
                f"PLACE_Z_FAILED ({z_error:.1f}mm — cube not flat on table)"
            )
            return result

        # ── Phase 6: Retract (PLACE_Z → HOVER_Z) ─────────────────────────────
        self._debug_current_phase = "place_p6_retract"
        _, _retract_steps = self._retract_to_hover(
            lambda: place_pos[:2], place_z, self.GRASP_RETRACT_STEP_M, should_render
        )

        result["success"] = True
        result["total_steps_used"] = (
            _plunge_steps
            + self.RELEASE_RAMP_STEPS
            + self.RELEASE_SETTLE_STEPS
            + _retract_steps
        )
        return result
