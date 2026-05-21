# Stage 3: ScriptedController

## Goal

Create `src/chess_env/controller.py` containing the `ScriptedController` class. This class wraps `ChessTaskEnv` and provides high-level scripted movement for all waypoint stages: transit, descend, ascend, and full pick-and-place sequences. It adds per-step safety monitoring that was previously inside `task.py::step()`.

---

## Preconditions

- Stage 2 is complete: env loads cleanly, RL observation/reward code removed.
- `scripts/verify_physics.py` still passes all 5 tests.

---

## 3.1 — Design Principles

### Separation of Concerns

| Layer | Responsibility |
|-------|---------------|
| `ChessTaskEnv` | MuJoCo physics, mocap control, scripted primitives (`_move_mocap_to`, `execute_grasp`, `execute_place`, `soft_reset`) |
| `ScriptedController` | Waypoint sequencing, per-step safety monitoring, result reporting |

The controller does **not** duplicate the physics layer. It delegates all movement to existing `task.py` methods. Its value is sequencing and monitoring.

### Movement Loop Architecture

For transit/descend/ascend stages, the controller implements its own movement loop using the same 4-step pattern (`_set_action → mocap_pos → mocap_quat → _mujoco_step`) rather than calling the black-box `_move_mocap_to`. This allows per-step safety checks to be injected naturally.

Why not reuse `_move_mocap_to`? `_move_mocap_to` returns only True/False after completion — it cannot abort early or report the crash reason mid-loop. The controller needs the per-step visibility.

For grasp/place stages, the controller delegates entirely to `execute_grasp()` and `execute_place()` which already have their own internal safety logic.

---

## 3.2 — New File: `src/chess_env/controller.py`

### Full Implementation

```python
"""
ScriptedController: Deterministic arm movement for all waypoint stages.
"""
import numpy as np
import mujoco
from dataclasses import dataclass, field
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
    stage_results: list  # List[StageResult]
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
    MAX_STEP_SIZE_M       = 0.008   # 8mm per physics step max (prevents overshoot)
    TRANSIT_MAX_STEPS     = 400     # Generous limit for long board diagonals
    VERTICAL_MAX_STEPS    = 200     # Sufficient for 90mm (SAFE_Z → HOVER_Z)
    FLOOR_LIMIT           = 0.400   # Abort transit if grip Z drops below this
    GRASP_VERIFY_DRIFT_MM = 30.0    # Max XY drift for "cube held" check post-grasp

    def __init__(self, env, drift_limit: float = 0.010):
        """
        Args:
            env: gymnasium-wrapped ChessTaskEnv (or the unwrapped env directly)
            drift_limit: Tube constraint radius in meters for descend/ascend (default 1cm)
        """
        # Unwrap to access ChessTaskEnv directly
        inner = env
        while hasattr(inner, 'env'):
            inner = inner.env
        self._env = inner
        self._wrapped_env = env  # Keep reference for TimeLimit reset
        self.drift_limit = drift_limit

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
            if np.linalg.norm(step_vec) > self.MAX_STEP_SIZE_M:
                step_vec = step_vec / np.linalg.norm(step_vec) * self.MAX_STEP_SIZE_M

            env._set_action(np.zeros(4))          # Reset mocap to current body
            env.data.mocap_pos[0][:3] += step_vec  # Apply capped delta
            env.data.mocap_quat[0][:] = env.VERTICAL_QUAT
            env._mujoco_step(None)

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
        target = np.array([target_xy[0], target_xy[1], env.SAFE_Z])

        def transit_abort(grip_pos):
            if grip_pos[2] < self.FLOOR_LIMIT:
                return True, f"FLOOR_HIT (z={grip_pos[2]:.4f})"
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

        result_dict = env.execute_grasp()  # Returns dict with status/error_code
        success = (result_dict.get("status") == "success")
        grip_after = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
        dist = float(np.linalg.norm(grip_after - grip_before))

        return StageResult(
            success=success,
            steps=result_dict.get("total_steps", 0),
            crash_reason=None if success else result_dict.get("error_code", "GRASP_FAILED"),
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
        success = (result_dict.get("status") == "success")
        grip_after = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()

        return StageResult(
            success=success,
            steps=result_dict.get("total_steps", 0),
            crash_reason=None if success else result_dict.get("error_code", "PLACE_FAILED"),
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
        while hasattr(curr, '_elapsed_steps'):
            curr._elapsed_steps = 0
            break
        while hasattr(curr, 'env') and not hasattr(curr, '_elapsed_steps'):
            curr = curr.env
            if hasattr(curr, '_elapsed_steps'):
                curr._elapsed_steps = 0
                break
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
```

---

## 3.3 — Changes to `execute_grasp` and `execute_place` Return Values

The current `execute_grasp()` and `execute_place()` return dicts that include a `"status"` key and an `"error_code"` key. Verify that the actual return format in `task.py` matches what the ScriptedController expects.

**Check the current return format:**
```bash
grep -n "return\|status\|error_code" src/chess_env/task.py | grep -A2 "execute_grasp\|execute_place"
```

If the current return dict uses different key names (e.g., `"result"` instead of `"status"`), update the `run_grasp` and `run_place` methods in the controller to match the actual keys. Do not modify `execute_grasp` or `execute_place` themselves.

---

## 3.4 — Minor Modification to `task.py`: `_move_mocap_to` Tolerance

The existing `_move_mocap_to` has `tolerance=0.001` (1mm). This is the same tolerance used by:
- `execute_grasp` Phase 2 (XY alignment)
- `execute_grasp` Phase 3 (plunge, per-step target)
- `soft_reset` Phase 4 (finger validation align)

Keep `_move_mocap_to` exactly as-is. The ScriptedController uses `TRANSIT_TOLERANCE_M=0.004` (4mm) which is more lenient than the 1mm internal tolerance — this is intentional since transit positioning to within 4mm is sufficient for the waypoint system.

---

## 3.5 — `src/utils/args.py` — Shared Argparse Module

Create this file now (ahead of Stage 4 scripts) so Stage 4 can import it.

### File: `src/utils/args.py`

```python
"""
Shared argparse utilities for all RoboChess evaluation scripts.
"""
import argparse


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add visualization and common flags to any ArgumentParser."""
    parser.add_argument(
        "--visualize", action="store_true",
        help="Open a MuJoCo viewer window (requires display)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Sleep seconds between steps when visualizing (default: 0)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable environment debug logging"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--drift-limit", type=float, default=0.010,
        help="Tube constraint radius in meters for descend/ascend (default: 0.010)"
    )
    return parser


def make_env(args, force_scenario=None):
    """
    Create the gymnasium environment using parsed args.
    Handles render_mode selection based on --visualize flag.
    """
    import gymnasium as gym
    import src.chess_env  # trigger registration

    render_mode = "human" if args.visualize else None
    env = gym.make(
        "ChessFetchTask-v0",
        render_mode=render_mode,
        force_scenario=force_scenario,
        debug=args.debug,
    )
    return env
```

---

## 3.6 — Validation Steps

### Check 1: Controller Imports Cleanly

```bash
python -c "
from src.chess_env.controller import ScriptedController, StageResult, SequenceResult
print('PASS: imports clean')
"
```

**Expected**: `PASS` with no ImportError.

### Check 2: Controller Instantiates from Env

```bash
python -c "
import gymnasium as gym
import src.chess_env
from src.chess_env.controller import ScriptedController

env = gym.make('ChessFetchTask-v0', force_scenario='transit')
env.reset()
ctrl = ScriptedController(env, drift_limit=0.010)
print(f'PASS: controller created. env type: {type(ctrl._env).__name__}')
env.close()
"
```

**Expected**: `PASS: controller created. env type: ChessTaskEnv`.

### Check 3: Single Transit Stage

```bash
python -c "
import gymnasium as gym
import numpy as np
import src.chess_env
from src.chess_env.controller import ScriptedController

env = gym.make('ChessFetchTask-v0', force_scenario='transit')
env.reset()
ctrl = ScriptedController(env, drift_limit=0.010)

target_xy = np.array([0.88, 0.2641])  # Board center
result = ctrl.run_transit(target_xy)
print(f'Transit result: success={result.success}, steps={result.steps}, error={result.error_mm:.1f}mm')
assert result.success, f'Transit failed: {result.crash_reason}'
assert result.error_mm < 5.0, f'Transit error too large: {result.error_mm:.1f}mm'
print('PASS')
env.close()
"
```

**Expected**: Transit to board center succeeds within ~400 steps, error < 5mm.

### Check 4: Descend Stage

```bash
python -c "
import gymnasium as gym
import numpy as np
import src.chess_env
from src.chess_env.controller import ScriptedController
from src.chess_env.waypoints import SAFE_Z, HOVER_Z

env = gym.make('ChessFetchTask-v0', force_scenario='descend')
env.reset()  # Arm starts near [x, y, SAFE_Z] for descend scenario
ctrl = ScriptedController(env, drift_limit=0.010)

# Get current XY from obs
inner = env.unwrapped
grip_pos = inner._utils.get_site_xpos(inner.model, inner.data, 'robot0:grip')
target_xy = grip_pos[:2].copy()

result = ctrl.run_descend(target_xy)
print(f'Descend result: success={result.success}, steps={result.steps}, error={result.error_mm:.1f}mm, z={result.final_pos[2]:.4f}')
assert result.success, f'Descend failed: {result.crash_reason}'
assert abs(result.final_pos[2] - HOVER_Z) < 0.005, f'Z error: {result.final_pos[2]:.4f} vs {HOVER_Z}'
print('PASS')
env.close()
"
```

**Expected**: Descend succeeds, final Z within 5mm of HOVER_Z (0.460).

### Check 5: Full Pick Sequence

```bash
python -c "
import gymnasium as gym
import numpy as np
import src.chess_env
from src.chess_env.controller import ScriptedController

env = gym.make('ChessFetchTask-v0', force_scenario='transit', hide_object=False)
env.reset()
ctrl = ScriptedController(env, drift_limit=0.010)
inner = env.unwrapped

# Use cube position as source
src_pos = inner.get_cube_position()
src_xy = src_pos[:2]
print(f'Cube at: {src_pos}')

result = ctrl.run_pick_sequence(src_xy)
print(f'Pick sequence: success={result.success}, failed_at={result.failed_at}')
for name, r in result.stage_results:
    status = 'OK' if r.success else f'FAIL ({r.crash_reason})'
    print(f'  {name:10s}: {status}, steps={r.steps}, error={r.error_mm:.1f}mm')

if result.success:
    print('PASS: Pick sequence completed')
else:
    print('FAIL: Pick sequence did not complete')
env.close()
"
```

**Expected**: All stages complete (success=True). If grasp fails, debug by checking HOVER_Z alignment.

### Check 6: `args.py` Smoke Test

```bash
python -c "
from src.utils.args import add_common_args, make_env
import argparse
p = argparse.ArgumentParser()
p = add_common_args(p)
args = p.parse_args([])
print(f'Defaults: visualize={args.visualize}, delay={args.delay}, n_episodes={args.n_episodes}')
print('PASS')
"
```

**Expected**: Prints defaults and `PASS`.

---

## 3.7 — Parameter Tuning Notes

### `MAX_STEP_SIZE_M = 0.008`

Transit moves can cover ~60cm diagonal (654mm). At 8mm/step, this takes ~82 steps. With `TRANSIT_MAX_STEPS=400`, there is 5× headroom. If transit times out on long diagonals, increase `TRANSIT_MAX_STEPS` or `MAX_STEP_SIZE_M`.

### `TRANSIT_TOLERANCE_M = 0.004` (4mm)

4mm is looser than `_move_mocap_to`'s default 1mm. Transit only needs to place the arm close enough that the subsequent `soft_reset` alignment step will correct any residual error. The alignment step uses 3mm tolerance; 4mm transit tolerance is tight enough to guarantee the alignment step converges quickly.

### `VERTICAL_TOLERANCE_M = 0.004` (4mm)

For descend and ascend, the arm moves ~90mm (SAFE_Z→HOVER_Z or back). The vertical motion is nearly pure Z with the proportional controller keeping XY locked. 4mm Z tolerance means the arm stops within 4mm of the exact HOVER_Z or SAFE_Z. This is sufficient for the grasp pipeline (which re-aligns in Phase 2 of `execute_grasp`).

### Drift Limit Default: `drift_limit=0.010` (1cm)

The 1cm tube constraint matches the final training curriculum value. During evaluation, this can be tightened to 0.005 (5mm) to test strict alignment. For initial testing, use 1cm.

---

## 3.8 — Known Edge Cases

### Near-Corner Positions

The arm base is at X=0.60. The near board corners are at X=0.58, Y=±0.30 from table center. These are approximately 0.30m from the arm base in lateral distance. Transit to these positions may require large shoulder rotation. Verify in Check 3 that corners work, not just center.

If a near corner fails, the `MAX_STEP_SIZE_M` limit may need to be reduced (slower approach to avoid overshoot) or the arm base X in Stage 1 may need a small adjustment.

### `grasp_mode` During Transit with Cube

When `run_place_sequence` calls `run_transit`, `grasp_mode=True`. The `run_transit` abort function currently only checks `FLOOR_HIT`. It should also check cube drop. Update the transit abort function when transitioning with cube:

```python
def run_transit(self, target_xy: np.ndarray) -> StageResult:
    env = self._env
    target = np.array([target_xy[0], target_xy[1], env.SAFE_Z])

    def transit_abort(grip_pos):
        if grip_pos[2] < self.FLOOR_LIMIT:
            return True, f"FLOOR_HIT (z={grip_pos[2]:.4f})"
        if env.grasp_mode:  # Cube drop check during transit-with-cube
            held, reason = env._check_cube_held(grip_pos)
            if not held:
                return True, reason
        return False, None

    return self._run_movement_loop(...)
```

The `run_transit` implementation above already includes this cube-drop check (see the full implementation in 3.2).

---

## 3.9 — Summary of New Files

| File | Type | Contents |
|------|------|---------|
| `src/chess_env/controller.py` | New | `ScriptedController`, `StageResult`, `SequenceResult` |
| `src/utils/args.py` | New | `add_common_args`, `make_env` |

**Stage 3 is complete when Check 5 (Full Pick Sequence) passes consistently over 5 consecutive attempts.**
