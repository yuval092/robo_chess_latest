"""Production environment: chess piece registry plus scripted grasp/place.

Used by the play loop (main.py). The arm is driven via execute_grasp / execute_place
and the embedded SAC controller, not via gym step() — there's no RL reward path here.
"""

from __future__ import annotations

import math

import numpy as np

from src.chess_env.base_env import ChessBaseEnv

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
        self.PIECE_HELD_XY_LIMIT = self.env_cfg["piece_held_xy_limit"]
        self.PIECE_HELD_Z_LIMIT = self.env_cfg["piece_held_z_limit"]

        self._home_posture_qpos = None
        self._home_posture_mocap_pos = None
        self._home_posture_mocap_quat = None
        self._home_posture_grip_pos = None

    # ── Hook: capture posture when reset arm lands at HOME_POS ──────────────

    def _on_reset_settled(self, arm_start_pos: np.ndarray) -> None:
        if np.linalg.norm(arm_start_pos - self.HOME_POS) < 1e-9:
            self._capture_home_posture()

    # ── Active piece state ───────────────────────────────────────────────────

    def set_active_piece(self, piece_id: str) -> None:
        piece = self._piece_registry.by_id(piece_id)
        self.active_piece_id = piece.piece_id
        self.active_piece_joint_name = piece.joint_name

    def clear_active_piece(self) -> None:
        self.active_piece_id = None
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

    def _check_piece_held(self, grip_pos: np.ndarray) -> tuple[bool, str | None]:
        """Verify the active piece is still in the gripper. Called when grasp_mode is on."""
        piece_pos = self.get_active_piece_position()
        xy_error = np.linalg.norm(piece_pos[:2] - grip_pos[:2])
        # Piece CoM sits ~15mm below the grip site when held at center
        z_error = abs(piece_pos[2] - (grip_pos[2] - 0.015))

        if xy_error > self.PIECE_HELD_XY_LIMIT:
            return (
                False,
                f"PIECE_DROPPED_XY (err={xy_error * 1000:.1f}mm > limit={self.PIECE_HELD_XY_LIMIT * 1000:.0f}mm)",
            )
        if z_error > self.PIECE_HELD_Z_LIMIT:
            return (
                False,
                f"PIECE_DROPPED_Z (err={z_error * 1000:.1f}mm > limit={self.PIECE_HELD_Z_LIMIT * 1000:.0f}mm)",
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
        self._home_posture_grip_pos = self.get_grip_pos()

    def _apply_home_posture_step(
        self,
        alpha: float,
        start_qpos: dict,
        start_mocap_pos: np.ndarray,
        start_mocap_quat: np.ndarray,
    ) -> None:
        """Move joints and mocap body alpha-fraction toward the saved home posture.

        alpha=0 leaves state unchanged; alpha=1 snaps to exact home values.
        After interpolating, fingers are committed and physics is forward-propagated.
        """
        for joint_name, target in self._home_posture_qpos.items():
            joint = self.model.joint(joint_name)
            self.data.qpos[joint.qposadr[0]] = start_qpos[joint_name] + alpha * (
                target - start_qpos[joint_name]
            )
            self.data.qvel[joint.dofadr[0]] = 0.0
            self.data.qacc[joint.dofadr[0]] = 0.0

        mocap_pos = start_mocap_pos + alpha * (self._home_posture_mocap_pos - start_mocap_pos)
        mocap_quat = start_mocap_quat + alpha * (self._home_posture_mocap_quat - start_mocap_quat)
        mocap_quat /= np.linalg.norm(mocap_quat)
        self.data.mocap_pos[0][:3] = mocap_pos
        self.data.mocap_quat[0][:] = mocap_quat
        self._commit_finger_target()
        if self.render_mode == "human":
            self.render()

    def reset_arm_to_home_posture(self) -> tuple[bool, str | None]:
        """Restore the exact reset-time joint posture after the gripper returns to home.

        Moving home by XYZ alone can leave redundant wrist/roll joints in different
        configurations; this snaps them back to the canonical posture.
        """
        if self.grasp_mode:
            return False, "HOME_POSTURE_RESET_BLOCKED_HELD_PIECE"

        if self._home_posture_qpos is None:
            return False, "HOME_POSTURE_NOT_CAPTURED"

        if self.current_scenario != "transit":
            return False, f"HOME_POSTURE_RESET_WRONG_SCENARIO ({self.current_scenario})"

        self.tube_center_xy = None
        self.goal_pos = self.HOME_POS.copy()
        self.goal = self.goal_pos.copy()
        self.finger_target_joint = self.FINGER_CLOSED_JOINT

        start_qpos = {
            joint_name: float(self.data.qpos[self.model.joint(joint_name).qposadr[0]])
            for joint_name in self._home_posture_qpos
        }
        start_mocap_pos = self.data.mocap_pos[0][:3].copy()
        start_mocap_quat = self.data.mocap_quat[0].copy()

        N_STEPS = 10
        for i in range(N_STEPS):
            self._apply_home_posture_step(
                (i + 1) / N_STEPS, start_qpos, start_mocap_pos, start_mocap_quat
            )

        error_mm = float(np.linalg.norm(self.get_grip_pos() - self._home_posture_grip_pos) * 1000.0)
        if error_mm >= 1.0:
            return False, f"HOME_POSTURE_RESET_FAILED ({error_mm:.3f}mm)"
        return True, None

    # ── Scripted plunge / retract helpers used by grasp & place ─────────────

    def _plunge_to_z(
        self, xy: np.ndarray, target_z: float, step_m: float
    ) -> np.ndarray:
        """Lower the gripper straight down over xy until it reaches target_z.

        Plunge = scripted downward motion, one step_m increment per sim step,
        XY held fixed. Returns the final grip pos.
        """
        grip_pos = self.get_grip_pos()
        commanded_z = grip_pos[2]
        for _ in range(int(round(max(0.0, commanded_z - target_z) / step_m)) + 6):
            grip_pos = self.get_grip_pos()
            if grip_pos[2] <= target_z + 0.001:
                break
            commanded_z = max(target_z, commanded_z - step_m)
            self._step_grip_toward(np.array([xy[0], xy[1], commanded_z]))
        return self.get_grip_pos()

    def _retract_to_hover(
        self,
        xy_provider,
        start_z: float,
        step_m: float,
        *,
        verify_held: bool = False,
    ) -> str | None:
        """Raise the gripper from start_z back up to HOVER_Z.

        Retract = inverse of plunge: one step_m increment per sim step upward,
        XY taken from xy_provider() each step (allows tracking a moving piece).
        Returns a failure reason string, or None on success.
        """
        commanded_z = start_z
        for _ in range(int(round((self.HOVER_Z - start_z) / step_m)) + 3):
            grip_pos = self.get_grip_pos()
            if grip_pos[2] >= self.HOVER_Z - 0.001:
                break
            commanded_z = min(self.HOVER_Z, commanded_z + step_m)
            xy = xy_provider()
            self._step_grip_toward(np.array([xy[0], xy[1], commanded_z]))

            if verify_held:
                piece_now = self.get_active_piece_position()
                grip_now = self.get_grip_pos()
                if abs(piece_now[2] - (grip_now[2] - 0.015)) > self.PIECE_HELD_Z_LIMIT:
                    return "PIECE_DROPPED_DURING_RETRACT"
        return None

    def _hold_locked_target(self, target_provider, steps: int) -> None:
        for _ in range(steps):
            self._step_grip_toward(target_provider())

    # ── Shared pipeline phase helpers ───────────────────────────────────────

    def _halt_and_check_hover_preconditions(
        self, check_fingers_open: bool
    ) -> str | None:
        """Zero sim state, reassert finger ctrl, then verify speed/Z (and optionally fingers open).

        Each pipeline starts here. Returns a reason string on failure, or None to continue.
        """
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._commit_finger_target()

        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        if float(np.linalg.norm(grip_vel)) > 0.005:
            return "PRECONDITION_SPEED"

        grip_pos = self.get_grip_pos()
        if abs(grip_pos[2] - self.HOVER_Z) > 0.025:
            return (
                f"PRECONDITION_Z (grip={grip_pos[2] * 1000:.1f}mm, "
                f"HOVER_Z={self.HOVER_Z * 1000:.1f}mm)"
            )

        if check_fingers_open:
            l_finger = self.get_finger_angle()
            if l_finger < self.FINGER_OPEN_JOINT - 0.003:
                return f"PRECONDITION_FINGERS_NOT_OPEN (j={l_finger:.4f})"

        return None

    def _align_over_xy(self, xy: np.ndarray) -> str | None:
        """Reassert vertical wrist and align the grip site over (xy) at current Z."""
        grip_pos = self.get_grip_pos()
        align_target = np.array([xy[0], xy[1], grip_pos[2]])
        if self._move_grip_to(
            align_target,
            max_steps=150,
            tolerance=self.GRASP_ALIGN_TOLERANCE,
        ):
            return None
        final_pos = self.get_grip_pos()
        align_error_mm = float(np.linalg.norm(final_pos - align_target) * 1000.0)
        return (
            "ROTATION_FAILED "
            f"(align error {align_error_mm:.1f}mm > {self.GRASP_ALIGN_TOLERANCE * 1000:.0f}mm)"
        )

    def _plunge_to_grasp_z(
        self, xy: np.ndarray
    ) -> tuple[np.ndarray, str | None]:
        """Plunge from current Z to GRASP_Z over xy. Returns (final grip pos, reason)."""
        grip_pos = self._plunge_to_z(xy, self.GRASP_Z, self.GRASP_PLUNGE_STEP_M)
        if abs(grip_pos[2] - self.GRASP_Z) > 0.008:
            return grip_pos, (
                f"PLUNGE_FAILED (z={grip_pos[2] * 1000:.1f}mm, "
                f"target={self.GRASP_Z * 1000:.1f}mm)"
            )
        return grip_pos, None

    # ── Scripted grasp pipeline ─────────────────────────────────────────────

    def execute_grasp(self) -> dict:
        """Scripted GRASP pipeline. Runs after DESCEND succeeds at HOVER_Z.

        Each phase asserts both mocap_pos AND mocap_quat before _mujoco_step,
        because _set_action resets mocap to the physical body state every call.
        """
        result = {
            "success": False,
            "reason": None,
            "pre_grasp_piece_xy": None,
            "post_grasp_piece_pos": None,
            "final_xy_error_mm": 0.0,
            "final_z_error_mm": 0.0,
            "final_finger_pos": 0.0,
        }
        if reason := self._halt_and_check_hover_preconditions(check_fingers_open=True):
            result["reason"] = reason
            return result

        piece_pos, reason = self._grasp_check_piece_orientation()
        result["pre_grasp_piece_xy"] = piece_pos[:2].copy()
        if reason:
            result["reason"] = reason
            return result
        if reason := self._align_over_xy(piece_pos[:2]):
            result["reason"] = reason
            return result

        grip_pos, reason = self._plunge_to_grasp_z(piece_pos[:2])
        if self.debug:
            self.logger.debug(
                f"[GRASP] Post-Plunge. Grip at {grip_pos}, Piece at {self.get_active_piece_position()}"
            )
        if reason:
            result["reason"] = reason
            return result

        if reason := self._grasp_close_fingers_with_ramp():
            result["reason"] = reason
            return result

        piece_pos, grip_pos, l_finger, reason = self._grasp_hold_and_verify()
        result["post_grasp_piece_pos"] = piece_pos.copy()
        result["final_xy_error_mm"] = (
            float(np.linalg.norm(piece_pos[:2] - grip_pos[:2])) * 1000
        )
        result["final_z_error_mm"] = float(abs(piece_pos[2] - grip_pos[2])) * 1000
        result["final_finger_pos"] = l_finger
        if reason:
            result["reason"] = reason
            return result

        if reason := self._grasp_retract():
            result["reason"] = reason
            return result

        result["success"] = True
        result["post_grasp_piece_pos"] = self.get_active_piece_position().copy()
        return result

    def _grasp_check_piece_orientation(self) -> tuple[np.ndarray, str | None]:
        """Read the piece pose and abort if its yaw exceeds finger clearance.

        A piece rotated >25° has effective width 42.4mm, exceeding max finger
        opening (38mm). Aborting before plunge prevents a stub.
        """
        piece_pos = self.get_active_piece_position().copy()
        self.logger.debug(
            f"[GRASP] Start. Piece at {piece_pos}, Grip at {self.get_grip_pos()}"
        )

        w, x, y, z = self.get_active_piece_quat()
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = abs(math.atan2(siny_cosp, cosy_cosp))
        # 4-fold symmetry: fold [0,π] into [0,π/2], then distance to nearest axis.
        # The naïve `min(yaw, abs(yaw - π/2))` fails above 90° (175° → false 85°).
        yaw_modulo = yaw % (math.pi / 2)
        effective_yaw = min(yaw_modulo, (math.pi / 2) - yaw_modulo)
        if effective_yaw > 0.436:
            return piece_pos, f"PIECE_ROTATED (yaw={math.degrees(effective_yaw):.1f}°)"
        return piece_pos, None

    def _grasp_close_fingers_with_ramp(self) -> str | None:
        """Ramp finger target from OPEN to GRASP_RAMP_END while tracking live piece XY.

        Direct jump creates a large impulse; ramping limits contact shock. Aborts
        early if fingers reach the secure-grip target with no piece contact.
        """
        self.grasp_mode = True
        ramp_start = self.FINGER_OPEN_JOINT
        ramp_end = self.GRASP_RAMP_END
        ramp_delta = (ramp_start - ramp_end) / self.GRASP_CLOSE_STEPS
        empty_detect_start = max(8, int(self.GRASP_CLOSE_STEPS * 0.65))

        for step in range(self.GRASP_CLOSE_STEPS):
            self.finger_target_joint = max(ramp_end, ramp_start - ramp_delta * step)
            live_piece = self.get_active_piece_position()
            self._step_grip_toward(
                np.array([live_piece[0], live_piece[1], self.GRASP_Z])
            )

            if step >= empty_detect_start:
                l_now = self.get_finger_angle()
                if l_now < self.EMPTY_GRASP_THRESHOLD:
                    return f"FINGER_CLOSED_EMPTY (j={l_now:.4f} at step {step})"

        return None

    def _grasp_hold_and_verify(self) -> tuple[np.ndarray, np.ndarray, float, str | None]:
        """Hold the grip steady on the piece, then verify XY/Z error and finger position."""

        def live_piece_target():
            live_piece = self.get_active_piece_position()
            return np.array([live_piece[0], live_piece[1], self.GRASP_Z])

        self._hold_locked_target(live_piece_target, self.GRASP_HOLD_STEPS)

        piece_pos = self.get_active_piece_position()
        grip_pos = self.get_grip_pos()
        l_finger = self.get_finger_angle()

        xy_error_mm = float(np.linalg.norm(piece_pos[:2] - grip_pos[:2])) * 1000
        z_error_mm = float(abs(piece_pos[2] - grip_pos[2])) * 1000

        if xy_error_mm > self.GRASP_VERIFY_XY_THRESHOLD * 1000:
            return piece_pos, grip_pos, l_finger, (
                f"VERIFY_XY_FAILED ({xy_error_mm:.1f}mm > "
                f"{self.GRASP_VERIFY_XY_THRESHOLD * 1000:.0f}mm)"
            )
        if z_error_mm > self.GRASP_VERIFY_Z_THRESHOLD * 1000:
            return piece_pos, grip_pos, l_finger, (
                f"VERIFY_Z_FAILED ({z_error_mm:.1f}mm > "
                f"{self.GRASP_VERIFY_Z_THRESHOLD * 1000:.0f}mm)"
            )
        if l_finger < self.EMPTY_GRASP_THRESHOLD:
            return piece_pos, grip_pos, l_finger, (
                f"VERIFY_FINGERS_CLOSED_EMPTY (j={l_finger:.4f})"
            )
        return piece_pos, grip_pos, l_finger, None

    def _grasp_retract(self) -> str | None:
        """Retract from GRASP_Z to HOVER_Z while tracking the piece and verifying it stays held."""
        return self._retract_to_hover(
            lambda: self.get_active_piece_position()[:2],
            self.GRASP_Z,
            self.GRASP_RETRACT_STEP_M,
            verify_held=True,
        )

    # ── Scripted place pipeline ─────────────────────────────────────────────

    def execute_place(self, dst_xy: np.ndarray) -> dict:
        """Scripted PLACE pipeline. Runs after DESCEND to HOVER_Z over destination.

        dst_xy: 2D destination XY (board square center).
        Arm exits at HOVER_Z, piece placed, grasp_mode=False.
        """
        result = {
            "success": False,
            "reason": None,
            "final_piece_pos": None,
            "final_xy_error_mm": 0.0,
        }
        if reason := self._halt_and_check_hover_preconditions(check_fingers_open=False):
            result["reason"] = reason
            return result

        if reason := self._align_over_xy(dst_xy):
            result["reason"] = reason
            return result

        place_pos, reason = self._plunge_to_grasp_z(dst_xy)
        if reason:
            result["reason"] = reason
            return result

        self._place_release_fingers(place_pos.copy())

        piece_pos, reason = self._place_verify_placement(dst_xy)
        result["final_piece_pos"] = piece_pos.copy()
        result["final_xy_error_mm"] = (
            float(np.linalg.norm(piece_pos[:2] - dst_xy[:2])) * 1000
        )
        if reason:
            result["reason"] = reason
            return result

        self._place_retract(place_pos[:2])

        result["success"] = True
        return result

    def _place_release_fingers(self, release_target: np.ndarray) -> None:
        """Ramp fingers from GRASP_RAMP_END to OPEN, then settle.

        Ramping from 0.0 would actively squeeze fingers before opening; starting
        from the actual grip position avoids that.
        """
        ramp_start = self.GRASP_RAMP_END
        ramp_end = self.FINGER_OPEN_JOINT
        ramp_delta = (ramp_end - ramp_start) / self.RELEASE_RAMP_STEPS

        for step in range(self.RELEASE_RAMP_STEPS):
            self.finger_target_joint = min(ramp_end, ramp_start + ramp_delta * step)
            self._step_grip_toward(release_target)

        self.finger_target_joint = self.FINGER_OPEN_JOINT
        self._hold_locked_target(lambda: release_target, self.RELEASE_SETTLE_STEPS)
        self.grasp_mode = False

    def _place_verify_placement(
        self, dst_xy: np.ndarray
    ) -> tuple[np.ndarray, str | None]:
        """Read piece pose and check XY drift against dst_xy and Z height against table surface."""
        piece_pos = self.get_active_piece_position()
        xy_error_mm = float(np.linalg.norm(piece_pos[:2] - dst_xy[:2])) * 1000
        z_error_mm = (
            float(abs(piece_pos[2] - (self.TABLE_SURFACE_Z + self.PIECE_HEIGHT / 2.0))) * 1000
        )

        if xy_error_mm > 20.0:
            return piece_pos, f"PLACE_XY_FAILED ({xy_error_mm:.1f}mm drift from target)"
        if z_error_mm > 10.0:
            return piece_pos, (
                f"PLACE_Z_FAILED ({z_error_mm:.1f}mm — piece not flat on table)"
            )
        return piece_pos, None

    def _place_retract(self, place_xy: np.ndarray) -> None:
        """Retract from GRASP_Z to HOVER_Z over the placement XY. Always succeeds."""
        self._retract_to_hover(lambda: place_xy, self.GRASP_Z, self.GRASP_RETRACT_STEP_M)
