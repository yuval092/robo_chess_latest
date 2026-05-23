"""
ScriptedController: Deterministic arm movement for all waypoint stages.
"""
import numpy as np
import mujoco
from dataclasses import dataclass
from typing import Optional


@dataclass
class StageResult:
    """Result of a single waypoint stage."""
    success: bool
    steps: int
    crash_reason: Optional[str]
    final_pos: np.ndarray
    error_mm: float  # Distance from target at end of stage


@dataclass
class SequenceResult:
    """Result of a full scenario chain."""
    success: bool
    stage_results: list  # List[tuple[str, StageResult]]
    failed_at: Optional[str]  # scenario name where failure occurred
    grasp_quality: Optional[dict]  # Set after execute_grasp


class ScriptedController:
    """
    Deterministic scripted controller for ChessTaskEnv.
    Drives the arm through transit/descend/ascend/grasp/place stages
    without any RL model.
    """

    # Movement parameters
    TRANSIT_TOLERANCE_M   = 0.004   # 4mm: success threshold for transit
    VERTICAL_TOLERANCE_M  = 0.004   # 4mm: success threshold for descend/ascend
    STEP_GAIN             = 1.0     # Full error applied per step (proportional)
    MIN_STEP_SIZE_M       = 0.002   # 2mm floor prevents slow final-approach creep
    MAX_STEP_SIZE_M       = 0.024   # 24mm per physics step max; keeps edge-square reachability stable
    TRANSIT_MAX_STEPS     = 300     # Sufficient for longest board diagonal at 12mm/step
    VERTICAL_MAX_STEPS    = 200     # Sufficient for 50mm (SAFE_Z → HOVER_Z)
    FLOOR_LIMIT           = 0.400   # Abort transit if grip Z drops below this
    GRASP_VERIFY_DRIFT_MM = 30.0    # Max XY drift for "cube held" check post-grasp

    def __init__(self, env, drift_limit: float = 0.010, render_fn=None, render_delay: float = 0.0):
        """
        Args:
            env: gymnasium-wrapped ChessTaskEnv (or the unwrapped env directly)
            drift_limit: Tube constraint radius in meters for descend/ascend (default 1cm)
            render_fn: Optional callback for rendering intermediate frames
            render_delay: Optional delay after each render call
        """
        # Unwrap to access ChessTaskEnv directly
        inner = env
        while hasattr(inner, 'env'):
            inner = inner.env
        self._env = inner
        self._wrapped_env = env  # Keep reference for TimeLimit reset
        self.drift_limit = drift_limit
        self._render_fn = render_fn
        self._render_delay = render_delay

    # ------------------------------------------------------------------
    # Internal movement primitive with per-step safety checks
    # ------------------------------------------------------------------

    def _run_movement_loop(
        self,
        target_pos: np.ndarray,
        *,
        tolerance: float,
        max_steps: int,
        abort_fn=None
    ) -> StageResult:
        """
        Proportional movement loop with optional per-step abort check.
        
        The abort_fn signature: (grip_pos: np.ndarray) -> (abort: bool, reason: str)
        If abort_fn is None, no safety checks are applied.
        """
        import time
        env = self._env
        for step in range(max_steps):
            grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
            error = target_pos - grip_pos
            dist = float(np.linalg.norm(error))

            if dist < tolerance:
                return StageResult(
                    success=True, steps=step, crash_reason=None,
                    final_pos=grip_pos.copy(), error_mm=dist * 1000
                )

            if abort_fn is not None:
                abort, reason = abort_fn(grip_pos)
                if abort:
                    return StageResult(
                        success=False, steps=step, crash_reason=reason,
                        final_pos=grip_pos.copy(), error_mm=dist * 1000
                    )

            # Apply capped proportional step
            step_vec = self.STEP_GAIN * error
            step_norm = float(np.linalg.norm(step_vec))
            if step_norm > self.MAX_STEP_SIZE_M:
                step_vec = step_vec / step_norm * self.MAX_STEP_SIZE_M
            elif 0.0 < step_norm < self.MIN_STEP_SIZE_M:
                step_vec = step_vec / step_norm * self.MIN_STEP_SIZE_M

            env._set_action(np.zeros(4))          # Reset mocap to current body
            env.data.mocap_pos[0][:3] += step_vec  # Apply capped delta
            env.data.mocap_quat[0][:] = env.VERTICAL_QUAT
            env._mujoco_step(None)

            if self._render_fn is not None:
                self._render_fn()
                if self._render_delay > 0:
                    time.sleep(self._render_delay)

        # Timed out — report final position
        grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
        dist = float(np.linalg.norm(target_pos - grip_pos))
        return StageResult(
            success=False, steps=max_steps, crash_reason="TIMEOUT",
            final_pos=grip_pos.copy(), error_mm=dist * 1000
        )

    # ------------------------------------------------------------------
    # Individual stage methods
    # ------------------------------------------------------------------

    def run_transit(self, target_xy: np.ndarray) -> StageResult:
        """
        Move arm horizontally from current position to [target_xy, SAFE_Z].
        Monitors: floor hit (grip Z below FLOOR_LIMIT).
        """
        env = self._env
        env._debug_current_phase = "transit"
        target = np.array([target_xy[0], target_xy[1], env.SAFE_Z])

        def transit_abort(grip_pos):
            if grip_pos[2] < self.FLOOR_LIMIT:
                return True, f"FLOOR_HIT (z={grip_pos[2]:.4f})"
            if env.grasp_mode:  # Cube drop check during transit-with-cube
                held, reason = env._check_cube_held(grip_pos)
                if not held:
                    return True, reason
            return False, None

        return self._run_movement_loop(
            target,
            tolerance=self.TRANSIT_TOLERANCE_M,
            max_steps=self.TRANSIT_MAX_STEPS,
            abort_fn=transit_abort
        )

    def run_descend(self, target_xy: np.ndarray) -> StageResult:
        """
        Move arm vertically from [target_xy, SAFE_Z] to [target_xy, HOVER_Z].
        Monitors: tube constraint (XY drift), table hit (grip below surface).
        """
        env = self._env
        env._debug_current_phase = "descend"
        target = np.array([target_xy[0], target_xy[1], env.HOVER_Z])
        tube_center = np.array([target_xy[0], target_xy[1]])

        def descend_abort(grip_pos):
            drift = float(np.linalg.norm(grip_pos[:2] - tube_center))
            if drift > self.drift_limit:
                return True, f"TUBE_BREACH (drift={drift*1000:.1f}mm > limit={self.drift_limit*1000:.0f}mm)"
            if grip_pos[2] < env.TABLE_SURFACE_Z:
                return True, f"TABLE_HIT (z={grip_pos[2]:.4f})"
            return False, None

        return self._run_movement_loop(
            target,
            tolerance=self.VERTICAL_TOLERANCE_M,
            max_steps=self.VERTICAL_MAX_STEPS,
            abort_fn=descend_abort
        )

    def run_ascend(self, target_xy: np.ndarray) -> StageResult:
        """
        Move arm vertically from [target_xy, HOVER_Z] to [target_xy, SAFE_Z].
        Monitors: tube constraint (XY drift), cube drop (if grasp_mode=True).
        """
        env = self._env
        env._debug_current_phase = "ascend"
        target = np.array([target_xy[0], target_xy[1], env.SAFE_Z])
        tube_center = np.array([target_xy[0], target_xy[1]])

        def ascend_abort(grip_pos):
            drift = float(np.linalg.norm(grip_pos[:2] - tube_center))
            if drift > self.drift_limit:
                return True, f"TUBE_BREACH (drift={drift*1000:.1f}mm > limit={self.drift_limit*1000:.0f}mm)"
            if env.grasp_mode:
                held, reason = env._check_cube_held(grip_pos)
                if not held:
                    return True, reason
            return False, None

        return self._run_movement_loop(
            target,
            tolerance=self.VERTICAL_TOLERANCE_M,
            max_steps=self.VERTICAL_MAX_STEPS,
            abort_fn=ascend_abort
        )

    def run_grasp(self) -> StageResult:
        """
        Execute the scripted grasp pipeline at current arm position.
        Precondition: arm is stationary at HOVER_Z over cube XY.
        """
        env = self._env
        grip_before = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()

        result_dict = env.execute_grasp()  # Returns dict with success/reason
        success = result_dict.get("success", False)
        grip_after = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
        dist = float(np.linalg.norm(grip_after - grip_before))

        return StageResult(
            success=success,
            steps=result_dict.get("total_steps_used", result_dict.get("close_steps_used", 0)),
            crash_reason=None if success else result_dict.get("reason", "GRASP_FAILED"),
            final_pos=grip_after.copy(),
            error_mm=dist * 1000
        )

    def run_place(self, dst_xy: np.ndarray) -> StageResult:
        """
        Execute the scripted place pipeline at current arm position.
        Precondition: arm is stationary at HOVER_Z over dst_xy with cube held.
        """
        env = self._env
        grip_before = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()

        result_dict = env.execute_place(dst_xy)
        success = result_dict.get("success", False)
        grip_after = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()

        return StageResult(
            success=success,
            steps=result_dict.get("total_steps_used", 0),
            crash_reason=None if success else result_dict.get("reason", "PLACE_FAILED"),
            final_pos=grip_after.copy(),
            error_mm=0.0
        )

    # ------------------------------------------------------------------
    # Scenario transition helper
    # ------------------------------------------------------------------

    def transition(
        self,
        new_scenario: str,
        new_goal_pos: np.ndarray,
        nominal_exit_pos: np.ndarray,
        nominal_xy=None
    ) -> dict:
        """
        Delegate to env.soft_reset() and reset the TimeLimit step counter.
        Returns the info dict from soft_reset.
        """
        obs, info = self._env.soft_reset(
            new_scenario, new_goal_pos, nominal_exit_pos, nominal_xy
        )
        # Reset the TimeLimit wrapper's step counter
        curr = self._wrapped_env
        # Try to find _elapsed_steps in wrappers
        while hasattr(curr, 'env'):
            if hasattr(curr, '_elapsed_steps'):
                curr._elapsed_steps = 0
                break
            curr = curr.env
        else:
            # Check the last one too
            if hasattr(curr, '_elapsed_steps'):
                curr._elapsed_steps = 0
                
        return info

    # ------------------------------------------------------------------
    # High-level sequence runners
    # ------------------------------------------------------------------

    def run_pick_sequence(self, src_xy: np.ndarray) -> SequenceResult:
        """
        Execute: transit → descend → grasp → ascend at src_xy.
        Returns SequenceResult with per-stage StageResults.
        """
        from src.chess_env.waypoints import exit_waypoint, SAFE_Z, HOVER_Z
        results = []

        # 1. Transit to src_xy
        transit_result = self.run_transit(src_xy)
        results.append(("transit", transit_result))
        if not transit_result.success:
            return SequenceResult(False, results, "transit", None)

        # 2. Transition to descend
        nom_exit = np.array([src_xy[0], src_xy[1], SAFE_Z])
        goal_descend = np.array([src_xy[0], src_xy[1], HOVER_Z])
        self.transition("descend", goal_descend, nom_exit, src_xy)

        # 3. Descend
        descend_result = self.run_descend(src_xy)
        results.append(("descend", descend_result))
        if not descend_result.success:
            return SequenceResult(False, results, "descend", None)

        # 4. Grasp
        grasp_result = self.run_grasp()
        results.append(("grasp", grasp_result))
        if not grasp_result.success:
            return SequenceResult(False, results, "grasp", None)

        # 5. Transition to ascend (grasp_mode already True)
        nom_exit_hover = np.array([src_xy[0], src_xy[1], HOVER_Z])
        goal_ascend = np.array([src_xy[0], src_xy[1], SAFE_Z])
        self.transition("ascend", goal_ascend, nom_exit_hover, src_xy)

        # 6. Ascend (with cube held)
        ascend_result = self.run_ascend(src_xy)
        results.append(("ascend", ascend_result))
        if not ascend_result.success:
            return SequenceResult(False, results, "ascend", None)

        return SequenceResult(True, results, None, None)

    def run_place_sequence(self, dst_xy: np.ndarray) -> SequenceResult:
        """
        Execute: transit → descend → place → ascend at dst_xy.
        Precondition: cube is currently held (grasp_mode=True).
        """
        from src.chess_env.waypoints import SAFE_Z, HOVER_Z
        results = []

        # 1. Transit to dst_xy (cube held throughout)
        transit_result = self.run_transit(dst_xy)
        results.append(("transit", transit_result))
        if not transit_result.success:
            return SequenceResult(False, results, "transit", None)

        # 2. Transition to descend (grasp_mode preserved)
        nom_exit = np.array([dst_xy[0], dst_xy[1], SAFE_Z])
        goal_descend = np.array([dst_xy[0], dst_xy[1], HOVER_Z])
        self.transition("descend", goal_descend, nom_exit, dst_xy)

        # 3. Descend (cube still held, grasp_mode=True)
        descend_result = self.run_descend(dst_xy)
        results.append(("descend", descend_result))
        if not descend_result.success:
            return SequenceResult(False, results, "descend", None)

        # 4. Place
        place_result = self.run_place(dst_xy)
        results.append(("place", place_result))
        if not place_result.success:
            return SequenceResult(False, results, "place", None)

        # 5. Transition to ascend (grasp_mode now False)
        nom_exit_hover = np.array([dst_xy[0], dst_xy[1], HOVER_Z])
        goal_ascend = np.array([dst_xy[0], dst_xy[1], SAFE_Z])
        self.transition("ascend", goal_ascend, nom_exit_hover, dst_xy)

        # 6. Ascend (empty gripper)
        ascend_result = self.run_ascend(dst_xy)
        results.append(("ascend", ascend_result))
        if not ascend_result.success:
            return SequenceResult(False, results, "ascend", None)

        return SequenceResult(True, results, None, None)

    def run_full_move(self, src_xy: np.ndarray, dst_xy: np.ndarray) -> SequenceResult:
        """
        Execute full pick-and-place: pick from src_xy, place at dst_xy.
        Returns combined SequenceResult.
        """
        pick_result = self.run_pick_sequence(src_xy)
        if not pick_result.success:
            return pick_result

        place_result = self.run_place_sequence(dst_xy)
        # Merge stage results
        combined_results = pick_result.stage_results + place_result.stage_results
        return SequenceResult(
            success=place_result.success,
            stage_results=combined_results,
            failed_at=place_result.failed_at,
            grasp_quality=place_result.grasp_quality
        )
