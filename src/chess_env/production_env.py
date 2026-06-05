"""Production environment: chess piece registry plus scripted grasp/place.

Used by the play loop (main.py). The arm is driven via execute_grasp / execute_place
and the embedded SAC controller, not via gym step() — there's no RL reward path here.
"""

from __future__ import annotations

import math

import numpy as np

from src.chess_env.base_env import ChessBaseEnv
from src.utils.validation import ensure_finite_array, ensure_finite_scalar

PIECE_COM_GRIP_OFFSET = 0.015  # Piece center of mass sits ~15mm below the grip site when held at center

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
        self.HOVER_SPEED_THRESHOLD = self.env_cfg["hover_speed_threshold"]
        self.HOVER_Z_TOLERANCE = self.env_cfg["hover_z_tolerance"]
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
        self.PLACE_VERIFY_XY_THRESHOLD = self.env_cfg["place_verify_xy_threshold"]
        self.PLACE_VERIFY_Z_THRESHOLD = self.env_cfg["place_verify_z_threshold"]
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
        self.active_piece_id = piece_id
        self.active_piece_joint_name = f"piece_{piece_id}:joint"

    def clear_active_piece(self) -> None:
        self.active_piece_id = None
        self.active_piece_joint_name = None

    def _active_piece_qpos_start(self) -> int:
        if self.active_piece_joint_name is None:
            raise RuntimeError("No active chess piece selected.")
        joint_id = self.model.joint(self.active_piece_joint_name).id
        return int(self.model.jnt_qposadr[joint_id])

    def get_active_piece_position(self) -> np.ndarray:
        qpos_start = self._active_piece_qpos_start()
        return ensure_finite_array(
            "active_piece_position", self.data.qpos[qpos_start : qpos_start + 3], (3,)
        ).copy()

    def get_active_piece_quat(self) -> np.ndarray:
        qpos_start = self._active_piece_qpos_start()
        return ensure_finite_array(
            "active_piece_quat", self.data.qpos[qpos_start + 3 : qpos_start + 7], (4,)
        ).copy()

    def _check_piece_held(self, grip_pos: np.ndarray) -> str | None:
        """Verify the active piece is still in the gripper. Called when grasp_mode is on."""
        grip_pos = ensure_finite_array("grip_pos", grip_pos, (3,))
        piece_pos = self.get_active_piece_position()
        xy_error = np.linalg.norm(piece_pos[:2] - grip_pos[:2])
        z_error = abs(piece_pos[2] - (grip_pos[2] - PIECE_COM_GRIP_OFFSET))

        if xy_error > self.PIECE_HELD_XY_LIMIT:
            return f"PIECE_DROPPED_XY (err={xy_error * 1000:.1f}mm > limit={self.PIECE_HELD_XY_LIMIT * 1000:.0f}mm)"
        if z_error > self.PIECE_HELD_Z_LIMIT:
            return f"PIECE_DROPPED_Z (err={z_error * 1000:.1f}mm > limit={self.PIECE_HELD_Z_LIMIT * 1000:.0f}mm)"
        return None

    # ── Home posture capture / restore ──────────────────────────────────────

    def _capture_home_posture(self) -> None:
        """Snapshot the canonical reset posture used between chess moves."""
        self._home_posture_qpos = {}
        for joint_name in HOME_POSTURE_JOINTS:
            joint = self.model.joint(joint_name)
            self._home_posture_qpos[joint_name] = ensure_finite_scalar(
                joint_name,
                self.data.qpos[joint.qposadr[0]],
            )
        self._home_posture_mocap_pos = ensure_finite_array(
            "home_mocap_pos", self.data.mocap_pos[0][:3], (3,)
        ).copy()
        self._home_posture_mocap_quat = ensure_finite_array(
            "home_mocap_quat", self.data.mocap_quat[0], (4,)
        ).copy()
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
        mocap_quat_norm = np.linalg.norm(mocap_quat)
        if mocap_quat_norm <= 0.0 or not np.isfinite(mocap_quat_norm):
            raise RuntimeError("HOME_POSTURE_INVALID_MOCAP_QUAT")
        mocap_quat /= mocap_quat_norm
        self.data.mocap_pos[0][:3] = mocap_pos
        self.data.mocap_quat[0][:] = mocap_quat
        self._commit_finger_target()
        if self.render_mode == "human":
            self.render()

    def reset_arm_to_home_posture(self) -> str | None:
        """Restore the exact reset-time joint posture after the gripper returns to home.

        Moving home by XYZ alone can leave redundant wrist/roll joints in different
        configurations; this snaps them back to the canonical posture.
        """
        if self.grasp_mode:
            return "HOME_POSTURE_RESET_BLOCKED_HELD_PIECE"

        if self._home_posture_qpos is None:
            return "HOME_POSTURE_NOT_CAPTURED"

        if self.current_scenario != "transit":
            return f"HOME_POSTURE_RESET_WRONG_SCENARIO ({self.current_scenario})"

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
            return f"HOME_POSTURE_RESET_FAILED ({error_mm:.3f}mm)"
        return None

    # ── Shared pipeline phase helpers ───────────────────────────────────────

    def _move_z(
        self,
        xy: np.ndarray,
        target_z: float,
        step_m: float,
        *,
        verify_held: bool = False,
    ) -> str | None:
        """Move the gripper vertically toward target_z at step_m per sim step, XY fixed.

        Works for both descent (plunge) and ascent (retract) — direction is inferred
        from target_z vs current Z. Returns a drop reason if verify_held detects the
        piece fell during ascent, or None on success.
        """
        xy = ensure_finite_array("xy", xy, (2,))
        target_z = ensure_finite_scalar("target_z", target_z)
        step_m = ensure_finite_scalar("step_m", step_m)
        if step_m <= 0.0 or not np.isfinite(step_m):
            return "MOVE_Z_INVALID_STEP"
        step_target_z = self.get_grip_pos()[2]
        descending = target_z < step_target_z
        z_step = -step_m if descending else step_m
        # +6 slack steps absorb physics lag after the distance estimate.
        max_steps = int(round(abs(target_z - step_target_z) / step_m)) + 6
        for _ in range(max_steps):
            if abs(self.get_grip_pos()[2] - target_z) <= 0.001:
                break
            step_target_z += z_step
            step_target_z = max(target_z, step_target_z) if descending else min(target_z, step_target_z)
            self._step_grip_toward(np.array([xy[0], xy[1], step_target_z]))
            if verify_held:
                if abs(self.get_active_piece_position()[2] - (self.get_grip_pos()[2] - PIECE_COM_GRIP_OFFSET)) > self.PIECE_HELD_Z_LIMIT:
                    return "PIECE_DROPPED_DURING_RETRACT"
        grip_z = self.get_grip_pos()[2]
        if abs(grip_z - target_z) > 0.008:
            return f"MOVE_Z_TIMEOUT (z={grip_z * 1000:.1f}mm, target={target_z * 1000:.1f}mm)"
        return None

    def _halt(self) -> None:
        """Zero sim velocities/accelerations and reassert finger control."""
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.ctrl[:] = 0.0
        self._commit_finger_target()

    def _check_hover_preconditions(self) -> str | None:
        """Verify speed, Z height, and (when not in grasp_mode) that fingers are open."""
        grip_vel = self.get_grip_vel()
        if float(np.linalg.norm(grip_vel)) > self.HOVER_SPEED_THRESHOLD:
            return "PRECONDITION_SPEED"

        grip_pos = self.get_grip_pos()
        if abs(grip_pos[2] - self.HOVER_Z) > self.HOVER_Z_TOLERANCE:
            return (
                f"PRECONDITION_Z (grip={grip_pos[2] * 1000:.1f}mm, "
                f"HOVER_Z={self.HOVER_Z * 1000:.1f}mm)"
            )

        if not self.grasp_mode:
            finger_angle = self.get_finger_angle()
            if finger_angle < self.FINGER_OPEN_JOINT - 0.003:
                return f"PRECONDITION_FINGERS_NOT_OPEN (j={finger_angle:.4f})"

        return None

    def _align_over_xy(self, xy: np.ndarray) -> str | None:
        """Align the grip site over (xy) at current Z."""
        xy = ensure_finite_array("xy", xy, (2,))
        grip_pos = self.get_grip_pos()
        align_target = np.array([xy[0], xy[1], grip_pos[2]])
        if self._move_grip_to(align_target, tolerance=self.GRASP_ALIGN_TOLERANCE):
            return None
        final_pos = self.get_grip_pos()
        align_error_mm = float(np.linalg.norm(final_pos - align_target) * 1000.0)
        return (
            "ALIGN_FAILED "
            f"(align error {align_error_mm:.1f}mm > {self.GRASP_ALIGN_TOLERANCE * 1000:.0f}mm)"
        )

    def execute_grasp(self) -> str | None:
        """Scripted GRASP pipeline. Runs after DESCEND succeeds at HOVER_Z.

        Returns None on success or a reason string on failure.
        Each phase asserts both mocap_pos AND mocap_quat before _mujoco_step,
        because _set_action resets mocap to the physical body state every call.
        """
        self._halt()
        if reason := self._check_hover_preconditions():
            return reason

        piece_pos, reason = self._check_piece_yaw()
        if reason:
            return reason
        self.logger.debug(
            f"[GRASP] Start. Piece at {piece_pos}, Grip at {self.get_grip_pos()}"
        )
        if reason := self._align_over_xy(piece_pos[:2]):
            return reason

        if reason := self._move_z(piece_pos[:2], self.GRASP_Z, self.GRASP_PLUNGE_STEP_M):
            return reason

        self.logger.debug(
            f"[GRASP] Post-Plunge. Grip at {self.get_grip_pos()}, "
            f"Piece at {self.get_active_piece_position()}"
        )

        self.grasp_mode = True
        if reason := self._close_fingers():
            return reason

        if reason := self._hold_and_verify():
            return reason

        return self._move_z(
            self.get_grip_pos()[:2], self.HOVER_Z, self.GRASP_RETRACT_STEP_M, verify_held=True
        )

    def _check_piece_yaw(self) -> tuple[np.ndarray, str | None]:
        """Read the piece pose and abort if its yaw exceeds finger clearance.

        Yaw is the piece's rotation around the vertical (Z) axis — i.e. how much
        it has spun on the table relative to the gripper's approach direction.
        A piece rotated >25° has effective width 42.4mm, exceeding max finger
        opening (38mm). Aborting before plunge prevents a stub.
        """
        piece_pos = self.get_active_piece_position().copy()

        # Extract yaw from quaternion (rotation around Z axis).
        quat = self.get_active_piece_quat()
        quat_norm = np.linalg.norm(quat)
        if quat_norm <= 0.0 or not np.isfinite(quat_norm):
            return piece_pos, "INVALID_PIECE_QUAT"
        w, x, y, z = quat / quat_norm
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = abs(math.atan2(siny_cosp, cosy_cosp))
        # 4-fold symmetry: fold [0,π] into [0,π/2], then distance to nearest axis.
        # The naïve `min(yaw, abs(yaw - π/2))` fails above 90° (175° → false 85°).
        yaw_modulo = yaw % (math.pi / 2)
        effective_yaw = min(yaw_modulo, (math.pi / 2) - yaw_modulo)
        if effective_yaw > math.radians(25):
            return piece_pos, f"PIECE_ROTATED (yaw={math.degrees(effective_yaw):.1f}°)"
        return piece_pos, None

    def _close_fingers(self) -> str | None:
        """Move finger target from OPEN to GRASP_RAMP_END while tracking live piece XY.

        Incremental steps limit contact shock vs. a direct jump. Aborts early if
        fingers close past EMPTY_GRASP_THRESHOLD with no piece contact.
        """
        ramp_start = self.FINGER_OPEN_JOINT
        ramp_end = self.GRASP_RAMP_END
        ramp_delta = (ramp_start - ramp_end) / self.GRASP_CLOSE_STEPS
        # Skip empty-grasp detection for the first 65% of steps: fingers haven't
        # had time to close yet, so any reading would be a false positive.
        empty_detect_start = max(8, int(self.GRASP_CLOSE_STEPS * 0.65))

        for step in range(self.GRASP_CLOSE_STEPS):
            self.finger_target_joint = max(ramp_end, ramp_start - ramp_delta * step)
            live_piece = self.get_active_piece_position()
            self._step_grip_toward(
                np.array([live_piece[0], live_piece[1], self.GRASP_Z])
            )

            if step >= empty_detect_start:
                finger_angle = self.get_finger_angle()
                if finger_angle < self.EMPTY_GRASP_THRESHOLD:
                    return f"FINGER_CLOSED_EMPTY (j={finger_angle:.4f} at step {step})"

        return None

    def _hold_and_verify(self) -> str | None:
        """Hold the grip steady on the piece, then verify XY/Z error and finger position."""
        for _ in range(self.GRASP_HOLD_STEPS):
            live_piece = self.get_active_piece_position()
            self._step_grip_toward(np.array([live_piece[0], live_piece[1], self.GRASP_Z]))

        piece_pos = self.get_active_piece_position()
        grip_pos = self.get_grip_pos()
        finger_angle = self.get_finger_angle()

        xy_error_mm = float(np.linalg.norm(piece_pos[:2] - grip_pos[:2])) * 1000
        z_error_mm = float(abs(piece_pos[2] - grip_pos[2])) * 1000

        if xy_error_mm > self.GRASP_VERIFY_XY_THRESHOLD * 1000:
            return (
                f"VERIFY_XY_FAILED ({xy_error_mm:.1f}mm > "
                f"{self.GRASP_VERIFY_XY_THRESHOLD * 1000:.0f}mm)"
            )
        if z_error_mm > self.GRASP_VERIFY_Z_THRESHOLD * 1000:
            return (
                f"VERIFY_Z_FAILED ({z_error_mm:.1f}mm > "
                f"{self.GRASP_VERIFY_Z_THRESHOLD * 1000:.0f}mm)"
            )
        if finger_angle < self.EMPTY_GRASP_THRESHOLD:
            return f"VERIFY_FINGERS_CLOSED_EMPTY (j={finger_angle:.4f})"
        return None

    def execute_place(self, dst_xy: np.ndarray) -> str | None:
        """Scripted PLACE pipeline. Runs after DESCEND to HOVER_Z over destination.

        Returns None on success or a reason string on failure.
        dst_xy: 2D destination XY (board square center).
        Arm exits at HOVER_Z, piece placed, grasp_mode=False.
        """
        self._halt()
        if reason := self._check_hover_preconditions():
            return reason

        if reason := self._align_over_xy(dst_xy):
            return reason

        if reason := self._move_z(dst_xy, self.GRASP_Z, self.GRASP_PLUNGE_STEP_M):
            return reason
        place_pos = self.get_grip_pos()

        self._open_fingers(place_pos.copy())
        self.grasp_mode = False

        if reason := self._verify_placement(dst_xy):
            return reason

        if reason := self._move_z(place_pos[:2], self.HOVER_Z, self.GRASP_RETRACT_STEP_M):
            return reason
        return None

    def _open_fingers(self, hold_pos: np.ndarray) -> None:
        """Move finger target from GRASP_RAMP_END to OPEN, then settle.

        Stepping from GRASP_RAMP_END (not 0.0) avoids squeezing before opening.
        """
        ramp_start = self.GRASP_RAMP_END
        ramp_end = self.FINGER_OPEN_JOINT
        ramp_delta = (ramp_end - ramp_start) / self.RELEASE_RAMP_STEPS

        for step in range(self.RELEASE_RAMP_STEPS):
            self.finger_target_joint = min(ramp_end, ramp_start + ramp_delta * step)
            self._step_grip_toward(hold_pos)

        self.finger_target_joint = self.FINGER_OPEN_JOINT
        for _ in range(self.RELEASE_SETTLE_STEPS):
            self._step_grip_toward(hold_pos)

    def _verify_placement(self, dst_xy: np.ndarray) -> str | None:
        """Check XY drift against dst_xy and Z height against table surface."""
        dst_xy = ensure_finite_array("dst_xy", dst_xy, (2,))
        piece_pos = self.get_active_piece_position()
        xy_error_mm = float(np.linalg.norm(piece_pos[:2] - dst_xy[:2])) * 1000
        z_error_mm = float(
            abs(piece_pos[2] - (self.TABLE_SURFACE_Z + self.PIECE_HEIGHT / 2.0)) * 1000
        )

        if xy_error_mm > self.PLACE_VERIFY_XY_THRESHOLD * 1000:
            return f"PLACE_XY_FAILED ({xy_error_mm:.1f}mm drift from target)"
        if z_error_mm > self.PLACE_VERIFY_Z_THRESHOLD * 1000:
            return f"PLACE_Z_FAILED ({z_error_mm:.1f}mm — piece not flat on table)"
        return None
