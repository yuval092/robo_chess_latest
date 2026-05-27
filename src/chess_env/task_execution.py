"""Grasp and place execution pipeline for ChessTaskEnv."""

from __future__ import annotations

import math

import mujoco
import numpy as np


class GraspPlaceMixin:
    """Provide grasp/place execution helpers to ChessTaskEnv."""

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
        """
        Drives the physical arm so that robot0:grip site reaches target_pos.
        Automatically handles site-to-body offsets by applying deltas.
        """
        zero_action = np.zeros(4)
        should_render = self.render_mode == "human"
        for _ in range(max_steps):
            grip_pos = self._grip_pos()
            error = target_pos - grip_pos
            if np.linalg.norm(error) < tolerance:
                return True

            self._set_action(zero_action)  # Resets mocap to body
            # Apply error as delta to mocap_pos to pull the site toward target
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

    def _plunge_to_z(
        self, xy: np.ndarray, target_z: float, step_m: float, should_render: bool
    ) -> tuple[np.ndarray, int]:
        """Run  plunge to z logic."""
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
        """Run  retract to hover logic."""
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
        """Run  hold locked target logic."""
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

        should_render = self.render_mode == "human"

        # Speed check: arm should be stationary
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        if float(np.linalg.norm(grip_vel)) > 0.005:
            result["reason"] = "PRECONDITION_SPEED"
            return result

        # Z check: RL must have stopped at HOVER_Z (within ±15mm)
        grip_pos = self._grip_pos()
        if abs(grip_pos[2] - self.HOVER_Z) > 0.025:
            result["reason"] = (
                f"PRECONDITION_Z (grip={grip_pos[2] * 1000:.1f}mm, "
                f"HOVER_Z={self.HOVER_Z * 1000:.1f}mm)"
            )
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

        self.logger.debug(
            f"[GRASP] Start. Cube at {cube_pos}, Grip at {self._grip_pos()}"
        )
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
        yaw_modulo = yaw % (math.pi / 2)  # fold → [0, π/2)
        effective_yaw = min(
            yaw_modulo, (math.pi / 2) - yaw_modulo
        )  # dist to nearest axis
        if effective_yaw > 0.436:  # > 25 degrees
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

        # ── Phase 3: Plunge (current Z → PLACE_Z) ────────────────────────────────
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

        # ── Phase 4: Grasp (Finger Close, Linear Ramp) ───────────────────────────
        self._debug_current_phase = "grasp_p4_close"
        # Ramp finger target from OPEN to a secure-grip value over the configured step budget.
        # Direct jump creates a large impulse; ramping limits contact shock.
        # Track the live cube XY so the arm follows contact motion instead of fighting it.
        self.grasp_mode = True
        ramp_start = self.FINGER_OPEN_JOINT  # 0.0181
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

            # Early abort: fingers reached the secure-grip target = no cube contact.
            # With cube: fingers stall at j≈0.0141.
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

        # ── Phase 5: Hold & Verify ────────────────────────────────────────────────
        self._debug_current_phase = "grasp_p5_hold"

        # Settle to let contact impulses stabilize (prevent ringing).
        # Continue tracking live cube XY and re-asserting vertical quat.
        def live_cube_target():
            """Run live cube target logic."""
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
            _plunge_steps
            + result["close_steps_used"]
            + self.GRASP_HOLD_STEPS
            + _retract_steps
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

        should_render = self.render_mode == "human"

        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        if float(np.linalg.norm(grip_vel)) > 0.005:
            result["reason"] = "PRECONDITION_SPEED"
            return result

        grip_pos = self._grip_pos()
        if abs(grip_pos[2] - self.HOVER_Z) > 0.025:
            result["reason"] = f"PRECONDITION_Z (grip={grip_pos[2] * 1000:.1f}mm)"
            return result

        # ── Phase 1+2: Vertical Correction + XY Align Over Destination ───────────
        self._debug_current_phase = "place_p12_align"
        # Simultaneously correct wrist orientation and move over destination.
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

        # ── Phase 3: Plunge (current Z → PLACE_Z) ────────────────────────────────
        self._debug_current_phase = "place_p3_plunge"
        place_z = self.GRASP_Z
        _, _plunge_steps = self._plunge_to_z(
            dst_xy, place_z, self.GRASP_PLUNGE_STEP_M, should_render
        )

        # Verify plunge reached place_z before releasing the cube. If the arm stalled
        # mid-descent and we open fingers here, the cube falls from height.
        grip_pos = self._grip_pos()
        place_pos = grip_pos.copy()  # Capture for Phase 6
        if abs(grip_pos[2] - place_z) > 0.008:
            result["reason"] = (
                f"PLUNGE_FAILED (z={grip_pos[2] * 1000:.1f}mm, "
                f"target={place_z * 1000:.1f}mm)"
            )
            return result

        # ── Phase 4: Release (Linear Ramp Open) ───────────────────────────────────
        self._debug_current_phase = "place_p4_release"
        # Ramp from GRASP_RAMP_END (0.010, actual grip position) to OPEN (0.0181).
        # Starting from 0.0000 would actively squeeze fingers for the first ~29 steps before opening.
        release_target = place_pos.copy()
        ramp_start = self.GRASP_RAMP_END  # 0.010 — actual finger position during grip
        ramp_end = self.FINGER_OPEN_JOINT  # 0.0181
        release_ramp_steps = self.RELEASE_RAMP_STEPS
        release_settle_steps = self.RELEASE_SETTLE_STEPS
        ramp_delta = (ramp_end - ramp_start) / release_ramp_steps

        for step in range(release_ramp_steps):
            self.finger_target_joint = min(ramp_end, ramp_start + ramp_delta * step)
            self._step_locked_grip(release_target, should_render)

        # Full open settle
        self.finger_target_joint = self.FINGER_OPEN_JOINT
        self._hold_locked_target(
            lambda: release_target, release_settle_steps, should_render
        )

        # Disable grasp mode — fingers can be teleported again in subsequent RL phases
        self.grasp_mode = False

        # ── Phase 5: Verify Placement ─────────────────────────────────────────────
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

        # ── Phase 6: Retract (PLACE_Z → HOVER_Z) ──────────────────────────────────
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
