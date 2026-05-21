# Stage 5: Cleanup and Bug Fixes

## Goal

Remove the training module, fix all known pre-existing bugs, update the test suite so all tests pass, and make a final verification pass to confirm the system is clean, deterministic, and fully functional without any RL dependency.

---

## Preconditions

- Stages 1–4 are complete.
- All Stage 4 validation checks pass.
- `scripts/verify_physics.py` passes all tests.
- `eval_stages.py --stages transit,descend,ascend --n-episodes 10` shows > 80% success for all stages.

---

## 5.1 — Remove Training Module

### Files to Delete

```
src/training/__init__.py
src/training/callbacks.py
src/training/curriculum.py
src/training/replay_buffer.py   (if exists)
src/training/                   (entire directory)
configs/training.yaml           (RL training config)
models/                         (if it contains only RL model files)
checkpoints/                    (all RL model checkpoints)
```

**Check before deleting:**
```bash
grep -r "from src.training\|import training\|src/training" src/ scripts/ tests/
```

If any remaining file imports from `src.training`, update those imports first.

```bash
grep -r "training.yaml\|load_config.*training" src/ scripts/
```

If `configs/training.yaml` is loaded anywhere, update that code to not require it. The `load_config` function should be tolerant of a missing training.yaml or the training config should be left in place as an archive (renamed to `training.yaml.archive`).

### Staging (Archive First, Delete Later)

Before deleting, archive the RL components in case they are needed for future re-introduction:
```bash
mkdir -p archive/rl_system
cp -r src/training archive/rl_system/
cp configs/training.yaml archive/rl_system/ 2>/dev/null || true
# Add archive/ to .gitignore
```

Then delete:
```bash
rm -rf src/training/
```

---

## 5.2 — Fix: `test_waypoints.py` Failing Assertions

### Root Cause

Tests in `tests/chess_env/test_waypoints.py` assert that the descend scenario goal Z is `0.430`, but the actual `HOVER_Z` constant is `0.460`. These tests were written when `hover_z=0.430` and were not updated when the config changed.

### Fix

Find all assertions in the file that use `0.430` for descend-scenario goal positions:

```python
# Current (WRONG):
assert result == [x, y, 0.430]     # in test_derive_goal_pos
assert result == [x, y, 0.430]     # in test_derive_goal_pos_all_scenarios
assert result == [x, y, 0.430]     # in test_exit_waypoint_all_scenarios
```

Replace with the correct value from the waypoints module:

```python
# Fixed:
from src.chess_env.waypoints import HOVER_Z
assert result == pytest.approx([x, y, HOVER_Z])
```

Using `pytest.approx` instead of direct equality check is safer for floating-point Z values.

**Exact edit**:

File: `tests/chess_env/test_waypoints.py`

1. Add import at top:
   ```python
   from src.chess_env.waypoints import SAFE_Z, HOVER_Z, GRASP_Z
   ```

2. Replace all `0.430` assertions:
   - `z=0.430` in descend test → `z=HOVER_Z` (for `derive_goal_pos("descend", ...)`)
   - `z=0.430` in exit waypoint test → `z=HOVER_Z` (for `exit_waypoint("descend", ...)`)

3. Verify fix:
   ```bash
   python -m pytest tests/chess_env/test_waypoints.py -v
   ```
   **Expected**: All 9 tests pass.

---

## 5.3 — Fix: `execute_grasp` Phase 6 `release_target` Scoping Bug

### Root Cause

In `task.py::execute_grasp()`, Phase 6 (Retract, GRASP_Z → HOVER_Z) uses the variable `release_target` to specify the target XY during retract. However, `release_target` is defined inside Phase 4's loop as `plunge_target.copy()` and is therefore technically in Phase 4's scope. The code works incidentally because Python closures retain the last assigned value after the loop, but this is fragile.

### Fix

At the end of Phase 3 (Plunge, just before Phase 4), capture the final plunge target position explicitly:

**Find** the end of Phase 3 loop (around line 600 in current task.py):
```python
# End of Phase 3 plunge loop
# The loop exits when grip_pos[2] <= GRASP_Z + 0.001

# --- ADD THIS LINE after the plunge loop exits: ---
grasp_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
```

**Find** the end of Phase 4's linear ramp loop. After the loop, `close_target` (or `plunge_target`) is used. Rename to use `grasp_pos` for clarity:

In Phase 4, find:
```python
error = close_target - self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
```

In Phase 6, find:
```python
plunge_target = np.array([release_target[0], release_target[1], target_z])
```

Replace with:
```python
plunge_target = np.array([grasp_pos[0], grasp_pos[1], target_z])
```

This makes the retract XY position explicit and independent of Phase 4's loop variable.

**Verify fix**:
```bash
python -c "
import gymnasium as gym
import numpy as np
import src.chess_env
env = gym.make('ChessFetchTask-v0', force_scenario='descend', hide_object=False)
env.reset()
inner = env.unwrapped
grip_pos = inner._utils.get_site_xpos(inner.model, inner.data, 'robot0:grip')
target_xy = grip_pos[:2].copy()

# Ensure arm is at HOVER_Z and fingers are open
inner.finger_target_joint = inner.FINGER_OPEN_JOINT
inner._move_mocap_to(np.array([*target_xy, inner.HOVER_Z]), inner.VERTICAL_QUAT)

result = inner.execute_grasp()
print(f'Grasp result: {result}')
assert result['success'] or result['reason'] in ['CUBE_NOT_FOUND', 'FINGER_CLOSED_EMPTY'], f'Unexpected failure: {result}'
print('PASS: Phase 6 executes without scoping error')
env.close()
"
```

**Expected**: No `NameError` on `release_target`. Either grasp succeeds or fails with a known reason.

---

## 5.4 — Fix: `_reset_sim` `sample_debug_freq` Reference

After Stage 2 removed `sample_debug_freq`, the `_reset_sim` method in task.py still contains:
```python
if self.sample_debug_freq is not None:
    self.debug = self.base_debug or (self.episode_number % self.sample_debug_freq == 0)
```

Since `self.sample_debug_freq` no longer exists, this will raise `AttributeError` on the first reset after Stage 2. This should have been caught in Stage 2 validation (Check 6 — reset cycle), but if it wasn't, fix it now:

Remove these lines from `_reset_sim`:
```python
# DELETE these 2 lines:
if self.sample_debug_freq is not None:
    self.debug = self.base_debug or (self.episode_number % self.sample_debug_freq == 0)
```

Replace with:
```python
# Stage 2 removed periodic debug sampling
self.debug = self.base_debug  # Assign self.base_debug at init: self.base_debug = debug
```

Also remove `self.base_debug` reference if it was only used for the sampling — since `debug` is now a simple bool, `_reset_sim` just keeps `self.debug = debug` (set once in `__init__` and never changed mid-run).

---

## 5.5 — Fix: `soft_reset` `self.goal` Assignment

In `soft_reset`, line 891:
```python
self.goal = self.goal_pos.copy()  # Critical for observation desync
```

After Stage 2, the parent class's `self.goal` attribute is still used internally by the Gymnasium-Robotics base class for `_is_success` checks. However, `_build_phase9_observation` (which used `self.goal` for the desired_goal dict key) has been removed. Verify that `self.goal` is still needed by searching:

```bash
grep -n "self\.goal" src/chess_env/task.py src/chess_env/simulation.py
```

If `self.goal` is only referenced in the deleted observation builder, remove the assignment from `soft_reset`. If it's referenced in other places (e.g., `_reset_sim`, `_sample_goal`), keep it.

The safe choice: keep the assignment. It's a small cost (one numpy copy) and prevents subtle bugs if any parent class method reads `self.goal`.

---

## 5.6 — Final Test Suite Run

Run the full pytest suite:

```bash
python -m pytest tests/ -v 2>&1 | tail -30
```

**Expected output:**
```
tests/chess_env/test_waypoints.py::test_validate_chain_valid        PASSED
tests/chess_env/test_waypoints.py::test_validate_chain_invalid      PASSED
tests/chess_env/test_waypoints.py::test_derive_goal_pos             PASSED  ← was FAIL
tests/chess_env/test_waypoints.py::test_derive_goal_pos_all_scenarios PASSED ← was FAIL
tests/chess_env/test_waypoints.py::test_validate_chain_invalid_transitions PASSED
tests/chess_env/test_waypoints.py::test_derive_goal_pos_error       PASSED
tests/chess_env/test_waypoints.py::test_exit_waypoint_error         PASSED
tests/chess_env/test_waypoints.py::test_exit_waypoint_all_scenarios PASSED  ← was FAIL
tests/chess_env/test_waypoints.py::test_chain_shortcuts_validity    PASSED

tests/chess_env/test_task_chaining.py::test_transition_validate     PASSED
tests/chess_env/test_task_chaining.py::test_soft_reset_flow         PASSED
tests/chess_env/test_task_chaining.py::test_soft_reset_finger_validation PASSED

12 passed
```

If `test_task_chaining.py` tests fail after Stage 2's `_get_obs` changes, update those tests to use the new obs dict keys:
- Old: `obs["observation"]` → New: `obs["grip_pos"]`
- Old: `obs["achieved_goal"]` → New: `obs["grip_pos"]` (same value)
- Old: `obs["desired_goal"]` → New: `obs["goal_pos"]`

---

## 5.7 — Final Physics Verification

```bash
python scripts/verify_physics.py
```

**Expected**: All 6 tests pass (5 original + 1 new table geometry test added in Stage 4).

---

## 5.8 — Final Comprehensive Evaluation

Run the full evaluation suite with realistic episode counts:

```bash
# All three stages, 50 episodes each
python scripts/eval_stages.py --stages transit,descend,ascend --n-episodes 50

# Full pick-and-place sequence, 20 episodes
python scripts/eval_sequence.py --chain full_move --n-episodes 20 \
    --src-xy "0.88 0.2641" --dst-xy "1.00 0.40"

# Stress test all corners, 10 episodes each
python scripts/eval_stress.py --chain pick --n-episodes 10

# Grid stress test (3×3 = 9 positions, 5 episodes each)
python scripts/eval_stress.py --chain vertical --n-episodes 5 --grid --grid-size 3
```

**Pass criteria:**

| Test | Minimum Pass |
|------|-------------|
| Transit success | ≥ 95% |
| Descend success | ≥ 90% |
| Ascend success | ≥ 90% |
| Full pick-and-place | ≥ 80% |
| All corners reachable | All 4 corners ≥ 70% |
| Grid coverage | ≥ 80% overall |

---

## 5.9 — Remove RL Dependencies from Remaining Files

Scan for any remaining `stable_baselines3` or `SAC` imports:

```bash
grep -rn "stable_baselines3\|from SAC\|import SAC\|SAC\.load" src/ scripts/ tests/
```

If any remain (e.g., in comments, archived test files), remove them.

Scan for `force_drift_limit`, `drift_curriculum_steps`, `fixed_drift` parameter usage:

```bash
grep -rn "force_drift_limit\|drift_curriculum_steps\|fixed_drift\|sample_debug_freq" src/ scripts/ tests/
```

These should all be gone after Stage 2. If any remain, remove them.

---

## 5.10 — Update Documentation

After Stage 5 is complete, update `docs/current_status/` files that refer to RL:

1. **`06_model_training.md`**: Add a note at the top: "This document describes the RL system that was removed in the scripted-only refactor. It is preserved for reference when RL is re-introduced."

2. **`01_system_architecture.md`**: Update the architecture diagram to show `ScriptedController` instead of SAC model.

These documentation updates are optional for Stage 5 completion but should be done before starting RL re-introduction.

---

## 5.11 — Summary Checklist

When all items below are checked, the refactor is complete:

- [ ] `src/training/` directory deleted (or archived)
- [ ] `configs/training.yaml` deleted (or archived)
- [ ] `scripts/train.py` deleted
- [ ] `scripts/record_move.py` deleted
- [ ] `scripts/eval.py` deleted
- [ ] `scripts/eval_grasp.py` deleted
- [ ] `scripts/test_corners.py` deleted
- [ ] `tests/chess_env/test_waypoints.py` — all 9 tests PASS
- [ ] `tests/chess_env/test_task_chaining.py` — all 3 tests PASS
- [ ] `scripts/verify_physics.py` — all 6 tests PASS
- [ ] `scripts/eval_stages.py` — ≥ 90% success for all 3 stages (50 episodes)
- [ ] `scripts/eval_sequence.py` — ≥ 80% full-move success (20 episodes)
- [ ] `scripts/eval_stress.py` — all corners reachable (≥ 70% each corner)
- [ ] `grep -rn "stable_baselines3\|SAC\.load"` returns no results in src/ scripts/ tests/
- [ ] `python scripts/visualize.py --scenario full_move --episodes 1` runs without error

**The refactor is complete when all items above are checked. The system is deterministic, scripted, fully testable, and ready for RL re-introduction.**
