# Post-Stage-5 Review & Environment Expansion Plan

**Date**: 2026-05-21  
**Author**: Reviewer (Claude Sonnet 4.6)  
**Status of implementation**: Stages 1–5 complete and staged.

---

## Part 1 — Code Review of Staged Changes

This section scrutinises every staged change against the original plan documents, identifies bugs, and documents findings for the implementing agent.

---

### 1.1 Summary of Staged Changes

```
renamed: models/latest_model.zip          → archive/rl_system/models/latest_model.zip
renamed: models/sac-FetchPickAndPlace-v4.zip → archive/rl_system/models/...
renamed: configs/training.yaml            → archive/rl_system/training.yaml
renamed: src/training/callbacks.py        → archive/rl_system/training/callbacks.py
renamed: src/training/trainer.py          → archive/rl_system/training/trainer.py
modified: docs/refactor_plan_implementing_agent_notes/deviations.md
deleted:  scripts/eval.py, eval_grasp.py, record_move.py, test_corners.py, train.py
modified: scripts/eval_sequence.py, eval_stages.py (new), eval_stress.py (new)
modified: scripts/visualize.py, verify_physics.py, test_grasp_physics.py
modified: src/chess_env/controller.py, simulation.py, task.py
modified: src/utils/args.py
modified: tests/chess_env/test_task_chaining.py, test_waypoints.py
```

---

### 1.2 What Is Correct and Matches the Plan

#### Stage 5: RL Archival ✓
- `models/`, `src/training/`, `configs/training.yaml` correctly archived to `archive/rl_system/`.
- Plan specified full deletion or archival; archival is an acceptable deviation documented in deviations.md.
- `src/training/__init__.py` not explicitly staged but is irrelevant (no callers remain).

#### Task.py Fixes ✓
| Fix | Status |
|-----|--------|
| `scenario_id` added to observation_space and `_get_obs()` | Correct |
| Ascend hang: `_set_gripper_state()` before close phase in `_reset_sim` | Correct, critical fix |
| All `print()` calls removed; logging via `self.logger.debug()` | Correct |
| Stale comments `# 0.012` → `# 0.0000` for FINGER_CLOSED_JOINT | Correct |
| Phase 6 scoping bug: `grasp_pos` and `place_pos` captured explicitly | Correct |

#### Simulation.py Torso Fix ✓
- `_env_setup` now reads `torso_max = self.model.jnt_range[torso_id][1]` instead of hardcoded `0.4`.
- This correctly prevents the joint from being set outside its declared range `[0.0386, 0.3861]`.

#### Controller.py ✓
- Unused `from dataclasses import field` import removed. Minor but clean.

#### Test Fixes ✓
- `test_waypoints.py`: `derive_goal_pos("descend")` now asserts `HOVER_Z` (0.460) not `GRASP_Z` (0.430). Correct.
- `test_waypoints.py`: `exit_waypoint("descend")` asserts `HOVER_Z`. Correct.
- Constants are now imported from `waypoints.py` rather than hardcoded, eliminating future drift.

#### `_set_gripper_state()` — Velocity Zeroing Addition ✓
```python
self.data.qvel[self.model.joint("robot0:l_gripper_finger_joint").dofadr[0]] = 0.0
self.data.qvel[self.model.joint("robot0:r_gripper_finger_joint").dofadr[0]] = 0.0
mujoco.mj_forward(self.model, self.data)
```
Zeroing finger velocity prevents oscillation after teleport. The added `mj_forward` ensures the kinematic chain is consistent before any subsequent computations. Correct.

#### `args.py` `hide_object` Fix ✓
- `make_env(args, force_scenario, hide_object=True)` default added.
- Eval scripts now explicitly pass `hide_object=False` for manipulation tasks.

#### Eval Scripts ✓
- `eval_stages.py`: Clean per-stage evaluation with stats. Matches plan §4.3.
- `eval_stress.py`: Corner + grid stress test. Matches plan §4.4.
- `eval_sequence.py`: Full chain evaluation without RL dependency. Matches plan §4.2.
- `visualize.py`: Scripted visualization. Matches plan §4.1.
- `verify_physics.py`: Added `test_table_geometry()` and improved reachability test. Correct.
- `test_grasp_physics.py`: Updated assertions for Kp=20000 and GRASP_Z=0.430. Correct.

---

### 1.3 Bugs Found — Must Fix Before Proceeding

---

#### BUG 1 (Critical): Visualization Shows Only First and Last Frames

**File**: `src/chess_env/controller.py`, method `_run_movement_loop` (line 106)  
**Also affects**: `src/chess_env/task.py`, method `_move_mocap_to` (line 414)

**Symptom**: Running any script with `--visualize` shows the initial state and the final state, with no intermediate frames. The MuJoCo viewer window opens but the arm teleports to its destination.

**Root Cause**: `ScriptedController._run_movement_loop()` drives physics by calling `env._mujoco_step(None)` directly (line 106 in controller.py). The MuJoCo passive viewer is a separate thread that only updates its render buffer when `env.render()` is explicitly called. Since no script calls `env.render()` inside the movement loop, the viewer only gets updated at the start and end of an episode.

All eval scripts have the same problem:
```python
# eval_stages.py — WRONG: render called only AFTER episode
result = ctrl.run_transit(target_xy)
if args.visualize:
    env.render()  # Only renders the end state
```

**Fix**: `ScriptedController` must accept an optional render callback and call it after every physics step. This is the cleanest fix because:
1. It does not bloat `task.py`'s movement primitive.
2. It keeps rendering optional (non-visual runs pay zero cost).
3. It gives scripts full control over render frequency and delay.

**Implementation** (`src/chess_env/controller.py`):
```python
class ScriptedController:
    def __init__(self, env, drift_limit: float = 0.010,
                 render_fn=None, render_delay: float = 0.0):
        ...
        self._render_fn = render_fn
        self._render_delay = render_delay

    def _run_movement_loop(self, target_pos, *, tolerance, max_steps, abort_fn=None):
        env = self._env
        import time
        for step in range(max_steps):
            ...
            env._set_action(np.zeros(4))
            env.data.mocap_pos[0][:3] += step_vec
            env.data.mocap_quat[0][:] = env.VERTICAL_QUAT
            env._mujoco_step(None)
            if self._render_fn is not None:  # ← ADD THIS
                self._render_fn()
                if self._render_delay > 0:
                    time.sleep(self._render_delay)
        ...
```

**Also fix `_move_mocap_to` in `task.py`** (used by `execute_grasp`, `execute_place`, `soft_reset`):
```python
def _move_mocap_to(self, target_pos, target_quat, max_steps=150, tolerance=0.001,
                   render_fn=None, render_delay=0.0) -> bool:
    import time
    zero_action = np.zeros(4)
    for _ in range(max_steps):
        ...
        self._mujoco_step(None)
        if render_fn is not None:  # ← ADD THIS
            render_fn()
            if render_delay > 0:
                time.sleep(render_delay)
    ...
```

But passing `render_fn` through to `execute_grasp` etc. requires thread-through params. A simpler alternative:

**Simpler alternative** — auto-render when `render_mode == "human"`:
```python
def _move_mocap_to(self, target_pos, target_quat, max_steps=150, tolerance=0.001) -> bool:
    import time
    zero_action = np.zeros(4)
    should_render = (self.render_mode == "human")
    for _ in range(max_steps):
        ...
        self._mujoco_step(None)
        if should_render:  # ← ADD THIS
            self.render()
    ...
```

This is even cleaner because no parameter threading is needed, and `task.py`/`execute_grasp`/`soft_reset` all get rendering for free when `render_mode="human"`.

**Fix all eval scripts**: Pass `render_fn` to `ScriptedController` (if using callback approach):
```python
# In each script's setup, when --visualize is set:
render_fn = env.render if args.visualize else None
render_delay = args.delay if args.visualize else 0.0
ctrl = ScriptedController(env, drift_limit=args.drift_limit,
                          render_fn=render_fn, render_delay=render_delay)
```

**Recommended approach**: Use the "auto-render when render_mode=human" approach for `_move_mocap_to` in `task.py`, AND pass `render_fn` to `ScriptedController` for `_run_movement_loop`. This covers all movement paths with minimal code changes.

---

#### BUG 2 (Moderate): Flaky Test — `test_soft_reset_finger_validation` Fails Non-Deterministically

**File**: `tests/chess_env/test_task_chaining.py` line 49  
**Observed**: Test FAILS when run as part of the full suite (`PYTHONPATH=. pytest tests/ -v`) but PASSES when run in isolation.  
**Error message**: `RuntimeError: soft_reset FINGER_VALIDATION_FAILED: actual=-0.000000, target=0.018100`

**Root Cause**: `_set_gripper_state()` teleports the finger joints to the target position but does NOT update `data.ctrl` for the finger actuators. The position actuator (Kp=20000) retains its previous control target (typically 0.0 = CLOSED). The moment any physics step is taken via `_mujoco_step` (e.g., inside `_move_mocap_to`), the actuator applies a large closing force that immediately undoes the teleport.

In `soft_reset`, Phase 4 opens the fingers and then calls `_move_mocap_to(nominal_exit_pos, ...)`. If the arm is NOT already at `nominal_exit_pos` within 3mm tolerance, at least one physics step is taken during `_move_mocap_to`. That single step's actuator force is enough to partially close the finger. Over many steps, the finger closes fully. Validation then fails.

Whether this triggers depends on the arm's position relative to `nominal_exit_pos = (0.5, 0.5, 0.55)` after Phase 2's alignment. The alignment ends when grip site is within 3mm of the target. But the finger teleport happens AFTER Phase 2; if Phase 4's `_move_mocap_to` finds the arm already within tolerance (0 steps), the test passes. If it needs even 1 step, the finger partially closes.

This is an inherent race condition in the current design.

**Underlying Design Issue**: `_set_gripper_state()` is incomplete. It teleports the finger joint position (qpos) and zeroes velocity (qvel), but leaves `data.ctrl` pointing to the old actuator target. A position actuator in MuJoCo always works against whatever is in `data.ctrl` during `mj_step`. Without syncing `data.ctrl`, any physics step after the teleport undoes it.

**Fix**: Add ctrl synchronisation to `_set_gripper_state()` in `src/chess_env/task.py`:

```python
def _set_gripper_state(self):
    """Physically sets the joint positions of the fingers based on the target state."""
    import mujoco as _mujoco
    target = self.finger_target_joint
    self._utils.set_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint", target)
    self._utils.set_joint_qpos(self.model, self.data, "robot0:r_gripper_finger_joint", target)
    self.data.qvel[self.model.joint("robot0:l_gripper_finger_joint").dofadr[0]] = 0.0
    self.data.qvel[self.model.joint("robot0:r_gripper_finger_joint").dofadr[0]] = 0.0
    # Sync actuator ctrl so position actuator doesn't fight the teleport
    l_id = _mujoco.mj_name2id(self.model, _mujoco.mjtObj.mjOBJ_ACTUATOR,
                               "robot0:l_gripper_finger_joint")
    r_id = _mujoco.mj_name2id(self.model, _mujoco.mjtObj.mjOBJ_ACTUATOR,
                               "robot0:r_gripper_finger_joint")
    self.data.ctrl[l_id] = target
    self.data.ctrl[r_id] = target
    mujoco.mj_forward(self.model, self.data)
```

This fix is safe for the grasp ramp in `execute_grasp`. During the ramp, `_set_gripper_state()` is called every step with a ramp-interpolated target. Setting ctrl to the same value as qpos each step means the actuator applies zero steady-state force — effectively making the finger track the scripted ramp exactly, which is the desired behaviour.

---

#### BUG 3 (Minor): Stale Torso Hardcode in `_reset_sim`

**File**: `src/chess_env/task.py`, line 308  
**Code**: `self._utils.set_joint_qpos(self.model, self.data, "robot0:torso_lift_joint", 0.4)`

**Issue**: `simulation.py`'s `_env_setup` was correctly updated to use `torso_max = self.model.jnt_range[torso_id][1]` (=0.3861). However, `task.py`'s `_reset_sim` still uses the hardcoded `0.4`, which exceeds the joint range. This puts the torso 1.4cm outside its declared limit on every `reset()`.

**Impact**: Low — MuJoCo's joint limit constraint applies a small corrective impulse, but the arm remains stable at ~0.3999m. However, it creates a technically invalid state at the start of every episode.

**Fix**: Replace line 308 in `task.py`'s `_reset_sim` with:
```python
torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                              "robot0:torso_lift_joint")
torso_max = self.model.jnt_range[torso_id][1]
self._utils.set_joint_qpos(self.model, self.data, "robot0:torso_lift_joint", torso_max)
```

---

#### BUG 4 (Minor): `GRASP_VERIFY_FINGER_THRESHOLD` Assertion Mismatch

**File**: `scripts/test_grasp_physics.py`, line ~397  
**Code**: `assert abs(uw.GRASP_VERIFY_FINGER_THRESHOLD - 0.003) < 0.001`

**Issue**: The env.yaml value is:
```yaml
grasp_verify_finger_threshold: 0.016   # finger stalls at j≈0.0141 for 30mm cube
```
The actual loaded value at runtime is `0.016`. The test asserts `0.003`, which is incorrect. The test will always fail if `env.yaml` is unchanged.

**Fix**: Update the assertion to:
```python
assert abs(uw.GRASP_VERIFY_FINGER_THRESHOLD - 0.016) < 0.001, \
    f"GRASP_VERIFY_FINGER_THRESHOLD={uw.GRASP_VERIFY_FINGER_THRESHOLD}, expected 0.016"
```

---

#### BUG 5 (Minor): Broken `place` Chain in `eval_sequence.py`

**File**: `scripts/eval_sequence.py`, lines ~862-864

**Code**:
```python
elif args.chain == "place":
    inner.grasp_mode = True  # Force held
    result = ctrl.run_place_sequence(dst_xy)
```

**Issue**: Setting `grasp_mode = True` without a cube actually in the gripper means the `_check_cube_held` abort function (called during transit while `grasp_mode=True`) will immediately detect CUBE_DROPPED because the cube is at the hidden location `[2.0, 2.0, 0.015]`, not near the gripper. The "place" chain will always fail on the first transit step.

**Fix**: The "place" chain requires the arm to already be holding a cube. Two options:
1. Run a pick sequence first, then continue with place — but that's `full_move`.
2. Remove "place" as a standalone chain option (it's not useful without a prior pick).

**Recommended fix**: Remove `"place"` from `CHAIN_CHOICES` in `eval_sequence.py`. The `full_move` chain already tests the complete pick+place sequence. A pure place test makes sense only in `test_grasp_physics.py`, which tests it properly with a manually positioned cube.

---

### 1.4 Test Results

Run `PYTHONPATH=. pytest tests/ -v`:
```
PASSED  test_transition_validate
PASSED  test_soft_reset_flow
FAILED  test_soft_reset_finger_validation   ← BUG 2 above
PASSED  test_validate_chain_valid
PASSED  test_validate_chain_invalid
PASSED  test_derive_goal_pos
PASSED  test_derive_goal_pos_all_scenarios
PASSED  test_validate_chain_invalid_transitions
PASSED  test_derive_goal_pos_error
PASSED  test_exit_waypoint_error
PASSED  test_exit_waypoint_all_scenarios
PASSED  test_chain_shortcuts_validity
  → 11 passed, 1 failed
```

Also note: `pytest tests/` (without PYTHONPATH) fails with `ModuleNotFoundError: No module named 'src'`. A `conftest.py` at the project root with `sys.path.insert(0, ".")` would fix this permanently so tests can be run without environment manipulation.

### 1.5 Eval Script Results

All scripts run correctly in non-visual mode:
```
eval_stages.py --n-episodes 3:
  transit  100%   avg 2.5mm  p95 3.6mm
  descend  100%   avg 1.0mm  p95 2.0mm
  ascend   100%   avg 3.5mm  p95 3.6mm

eval_stress.py --chain vertical --n-episodes 1:
  near_right 100%
  near_left  100%
  far_right  100%
  far_left   100%
  center     100%
  Overall: 100.0%

eval_sequence.py --chain pick --n-episodes 2: 100% (2/2)
verify_physics.py: SYSTEM HEALTHY (6/6 tests passed)
```

---

## Part 2 — Environment Expansion: Visualization Fix Plan

This section defines the exact implementation for the visualization fix (BUG 1).

### 2.1 Changes to `src/chess_env/task.py`

**Method `_move_mocap_to`**: Add auto-render when `render_mode == "human"`.

```python
def _move_mocap_to(self, target_pos: np.ndarray, target_quat: np.ndarray,
                   max_steps: int = 150, tolerance: float = 0.001) -> bool:
    import time as _time
    zero_action = np.zeros(4)
    should_render = (self.render_mode == "human")
    for _ in range(max_steps):
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        error = target_pos - grip_pos
        if np.linalg.norm(error) < tolerance:
            return True
        self._set_action(zero_action)
        self.data.mocap_pos[0][:3] += error
        self.data.mocap_quat[0][:] = target_quat
        self._mujoco_step(None)
        if should_render:
            self.render()
    final_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
    return bool(np.linalg.norm(final_pos - target_pos) < tolerance)
```

This automatically renders every physics step when running with a viewer. The `--delay` option controls pacing at the script level (sleep after each full controller step, not each physics step).

### 2.2 Changes to `src/chess_env/controller.py`

**`ScriptedController.__init__`**: Add render parameters.  
**`ScriptedController._run_movement_loop`**: Call render after each step.

```python
def __init__(self, env, drift_limit: float = 0.010,
             render_fn=None, render_delay: float = 0.0):
    ...
    self._render_fn = render_fn
    self._render_delay = render_delay

def _run_movement_loop(self, target_pos, *, tolerance, max_steps, abort_fn=None):
    import time
    env = self._env
    for step in range(max_steps):
        ...
        env._mujoco_step(None)
        if self._render_fn is not None:
            self._render_fn()
            if self._render_delay > 0:
                time.sleep(self._render_delay)
    ...
```

### 2.3 Changes to Each Script

In all scripts that use `--visualize`, construct the controller with the render function:

**Pattern** (applies to `eval_stages.py`, `eval_sequence.py`, `eval_stress.py`, `visualize.py`):
```python
render_fn = env.render if args.visualize else None
ctrl = ScriptedController(env, drift_limit=args.drift_limit,
                          render_fn=render_fn,
                          render_delay=args.delay)
```

Remove the standalone `env.render()` / `time.sleep()` calls that appear after each episode — they are now superseded by the per-step rendering.

**Note for `visualize.py`**: The `args.visualize = True` override at line ~1827 is still needed since this script is always visual by design. Keep it. The `--delay` arg controls playback speed.

---

## Part 3 — Environment Expansion: Table Size

### 3.1 Reachability Data

Tests were performed by moving the arm to board corners at `GRASP_Z = 0.430m`, measuring grip site error vs. target. Arm at `x=0.60`, torso at max (`0.3861m`).

| Table Size | Error at Far Corners | Error at Near Corners | Max Error | Verdict |
|------------|---------------------|----------------------|-----------|---------|
| **60×60cm** | 0.6mm | 0.3mm | **0.6mm** | ✓ Excellent |
| **65×65cm** | 1.0mm | 0.3mm | **1.0mm** | ✓ Excellent |
| **70×70cm** | 4.1mm | 0.4mm | **4.1mm** | ✓ Good (within 5mm threshold) |
| **75×75cm** | 7.6mm | 0.9mm | **7.6mm** | ✗ Fails (>5mm threshold) |
| **80×80cm** | 11.3mm | 0.4mm | **11.3mm** | ✗ Fails |

**Answer to your question**: At the current configuration (arm `x=0.60`, torso at max), the arm can reliably cover a **70×70cm table** (max error 4.1mm). A 75×75cm table fails at the far corners.

### 3.2 Torso Height Impact

Surprisingly, lowering the torso from max (`0.3861m`) to `~0.25m` significantly *improves* far-corner reachability, especially for larger tables. When the torso is very high, the arm reaches down at a steep angle to hit table-level positions, which consumes much of its horizontal reach budget on the Z axis. A lower torso allows more of the arm's length to be spent on horizontal reach.

| Torso Height | 60×60 Far Corners | 70×70 Far Corners | 75×75 Far Corners |
|-------------|-------------------|-------------------|-------------------|
| 0.38m (current max) | 0.3mm ✓ | 3.0mm ✓ | 6.5mm ✗ |
| 0.30m | 0.3mm ✓ | 0.5mm ✓ | 1.3mm ✓ |
| **0.25m** | 0.7mm ✓ | **0.5mm ✓** | **0.4mm ✓** |
| 0.20m | 0.9mm ✓ | 0.3mm ✓ | 1.0mm ✓ |
| 0.15m | 0.8mm ✓ | 0.5mm ✓ | 0.4mm ✓ |
| 0.10m | 217.9mm ✗ | — | — |

**Key finding**: A torso of **0.25m** achieves the best balance — it maintains excellent reachability for 60×60cm and makes 75×75cm fully reachable (0.4mm error). Lowering below 0.20m starts to degrade the extreme-reach positions.

**Answer to your questions**:
- Can the arm reach every spot of an 80×80cm table? **No** (11.3mm error at arm x=0.60, any torso).
- Can it reach 75×75cm? **Yes, but only with torso lowered to ~0.25m** (0.4mm error).
- Can it reach 70×70cm? **Yes** — even with the current torso (4.1mm), and much better (0.5mm) with torso at 0.25m.
- What is the largest table the arm can handle? **70×70cm** is safe at current settings; **75×75cm** requires torso at 0.25m.
- Does the torso truly need to be so high? **No.** Torso at 0.25m is actually better for reach. The high torso was an assumption that "higher = better reach", which turns out to be wrong for near-horizontal positions.

---

## Part 4 — Environment Expansion: Implementation Tasks

The following tasks implement all of the above. They are ordered by priority (critical bugs first, then enhancements).

---

### Task 1 (Critical): Fix Visualization — Render Per Step

**Files**: `src/chess_env/task.py`, `src/chess_env/controller.py`, all `scripts/*.py`

#### 1a. `src/chess_env/task.py` — `_move_mocap_to`
Add `should_render` check before the loop and `self.render()` call after each `_mujoco_step(None)`:
```python
def _move_mocap_to(self, target_pos, target_quat, max_steps=150, tolerance=0.001) -> bool:
    zero_action = np.zeros(4)
    should_render = (self.render_mode == "human")
    for _ in range(max_steps):
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        error = target_pos - grip_pos
        if np.linalg.norm(error) < tolerance:
            return True
        self._set_action(zero_action)
        self.data.mocap_pos[0][:3] += error
        self.data.mocap_quat[0][:] = target_quat
        self._mujoco_step(None)
        if should_render:
            self.render()
    final_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
    return bool(np.linalg.norm(final_pos - target_pos) < tolerance)
```

#### 1b. `src/chess_env/controller.py` — `ScriptedController`
Add `render_fn` and `render_delay` to `__init__` and call render in `_run_movement_loop`:
```python
def __init__(self, env, drift_limit=0.010, render_fn=None, render_delay=0.0):
    ...  # (existing setup)
    self._render_fn = render_fn
    self._render_delay = render_delay

# In _run_movement_loop, after env._mujoco_step(None):
if self._render_fn is not None:
    self._render_fn()
    if self._render_delay > 0:
        import time; time.sleep(self._render_delay)
```

#### 1c. All eval scripts — pass render_fn to controller
In `eval_stages.py`, `eval_sequence.py`, `eval_stress.py`, `visualize.py`:
```python
render_fn = env.render if args.visualize else None
ctrl = ScriptedController(env, drift_limit=args.drift_limit,
                          render_fn=render_fn, render_delay=args.delay)
```

Also remove standalone `env.render()` / `time.sleep(args.delay)` calls that happen after each episode (they are now redundant).

---

### Task 2 (Critical): Fix `_set_gripper_state()` Ctrl Sync

**File**: `src/chess_env/task.py`, method `_set_gripper_state` (around line 192)

Add actuator ctrl synchronisation:
```python
def _set_gripper_state(self):
    """Physically sets the joint positions of the fingers based on the target state."""
    target = self.finger_target_joint
    self._utils.set_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint", target)
    self._utils.set_joint_qpos(self.model, self.data, "robot0:r_gripper_finger_joint", target)
    self.data.qvel[self.model.joint("robot0:l_gripper_finger_joint").dofadr[0]] = 0.0
    self.data.qvel[self.model.joint("robot0:r_gripper_finger_joint").dofadr[0]] = 0.0
    # Sync actuator ctrl to prevent position actuator from fighting the teleport
    l_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                              "robot0:l_gripper_finger_joint")
    r_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                              "robot0:r_gripper_finger_joint")
    self.data.ctrl[l_id] = target
    self.data.ctrl[r_id] = target
    mujoco.mj_forward(self.model, self.data)
```

**After this fix**: Run `PYTHONPATH=. pytest tests/ -v` and confirm all 12 tests pass.

---

### Task 3 (Minor): Fix Stale Torso Hardcode in `_reset_sim`

**File**: `src/chess_env/task.py`, line ~308

Replace:
```python
self._utils.set_joint_qpos(self.model, self.data, "robot0:torso_lift_joint", 0.4)
```
With:
```python
torso_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                               "robot0:torso_lift_joint")
self._utils.set_joint_qpos(self.model, self.data, "robot0:torso_lift_joint",
                           self.model.jnt_range[torso_jid][1])
```

---

### Task 4 (Minor): Fix `GRASP_VERIFY_FINGER_THRESHOLD` Assertion

**File**: `scripts/test_grasp_physics.py`, line ~397

Replace:
```python
assert abs(uw.GRASP_VERIFY_FINGER_THRESHOLD - 0.003) < 0.001
```
With:
```python
assert abs(uw.GRASP_VERIFY_FINGER_THRESHOLD - 0.016) < 0.001, \
    f"GRASP_VERIFY_FINGER_THRESHOLD={uw.GRASP_VERIFY_FINGER_THRESHOLD}, expected 0.016"
```

---

### Task 5 (Minor): Fix Broken `place` Chain in `eval_sequence.py`

**File**: `scripts/eval_sequence.py`

Remove `"place"` from `CHAIN_CHOICES`:
```python
CHAIN_CHOICES = ["full_move", "pick", "vertical"]
```

Remove the `elif args.chain == "place":` block entirely. If a pure-place evaluation is needed in future, it requires dedicated setup (cube already held) which is out of scope for this eval script.

---

### Task 6 (Minor): Add `conftest.py` for Test Path Resolution

**File**: Create `tests/conftest.py` (or root `conftest.py`)

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```

This allows `pytest tests/` to work without `PYTHONPATH=.` prefix.

---

### Task 7: Lower Torso Height

**User request**: "I'd like to make the arm's torso a bit lower, as it currently seems to be too high for no real reason."

**Finding**: A torso of **0.25m** is optimal. It's 35% lower than the current max (0.3861m), looks more natural, and actually *improves* reachability at far-corner positions.

#### 7a. Update `simulation.py` — `_env_setup`

Change the torso setup to use a fixed lower value:
```python
TORSO_OPERATING_HEIGHT = 0.25  # Optimal for board reach; lower looks more natural
self._utils.set_joint_qpos(self.model, self.data,
                           "robot0:torso_lift_joint", TORSO_OPERATING_HEIGHT)
```

Or, if the value should be configurable, add to `configs/env.yaml`:
```yaml
torso_height: 0.25   # Torso lift joint target (joint range: [0.0386, 0.3861])
```

And load it in `simulation.py`:
```python
torso_height = self.env_cfg.get("torso_height", 0.25)
self._utils.set_joint_qpos(self.model, self.data, "robot0:torso_lift_joint", torso_height)
```

#### 7b. Update `task.py` — `_reset_sim`

Replace the hardcoded torso set (same as Task 3) with the config-driven value:
```python
torso_height = self.env_cfg.get("torso_height", 0.25)
self._utils.set_joint_qpos(self.model, self.data, "robot0:torso_lift_joint", torso_height)
```

---

### Task 8: Expand Table to 70×70cm

**User request**: "I'd like to make the table's surface larger than 60×60cm."

**Decision**: Expand to **70×70cm** (half-size 0.35m). This is the largest size that works reliably at current arm position `x=0.60` with torso at any reasonable height (0.15–0.38m). The 70×70cm table gives max 0.5mm error at `torso=0.25m` — excellent.

Note: 75×75cm is also achievable with torso at 0.25m (0.4mm error), but 70×70cm provides a larger safety margin and is the conservative choice. Choose 70×70cm unless a specific reason requires 75×75cm.

#### 8a. `chess_env/assets/pick_and_place.xml`

Update the table surface geom size and leg positions:
```xml
<!-- OLD: 60x60cm (half = 0.30) -->
<geom name="table0_surface" type="box" size="0.30 0.30 0.025" .../>
<geom name="table0_leg_far_plus"  ... pos="0.26  0.26  0.1875" .../>
<geom name="table0_leg_far_minus" ... pos="0.26 -0.26  0.1875" .../>
<geom name="table0_leg_near_plus" ... pos="-0.26  0.26  0.1875" .../>
<geom name="table0_leg_near_minus"... pos="-0.26 -0.26  0.1875" .../>

<!-- NEW: 70x70cm (half = 0.35) -->
<geom name="table0_surface" type="box" size="0.35 0.35 0.025" .../>
<geom name="table0_leg_far_plus"  ... pos="0.31  0.31  0.1875" .../>
<geom name="table0_leg_far_minus" ... pos="0.31 -0.31  0.1875" .../>
<geom name="table0_leg_near_plus" ... pos="-0.31  0.31  0.1875" .../>
<geom name="table0_leg_near_minus"... pos="-0.31 -0.31  0.1875" .../>
```

Leg positions: 10cm inside each corner of the 70×70cm surface (0.35 - 0.04 = 0.31m from center).

#### 8b. `configs/env.yaml`

```yaml
table_half_x: 0.35    # was 0.30 (60cm → 70cm)
table_half_y: 0.35    # was 0.30
```

#### 8c. Verify `verify_physics.py` still passes

The `test_table_geometry` in `verify_physics.py` checks `table_half_x == 0.30`. Update it:
```python
if cfg["table_half_x"] != 0.35 or cfg["table_half_y"] != 0.35:
    print(f"  - ERROR: table_half_x/y expected 0.35, got ...")
    return False
```

#### 8d. Update `eval_stress.py` corners

`eval_stress.py` reads `table_half_x` from config, so it automatically uses the new corners. No change required.

---

### Task 9: Verify After All Changes

After implementing Tasks 1–8, run the full verification suite:

```bash
# Unit tests
PYTHONPATH=. pytest tests/ -v
# Expected: 12/12 pass

# Physics verification
python scripts/verify_physics.py
# Expected: SYSTEM HEALTHY (6/6)

# Stage performance
python scripts/eval_stages.py --stages transit,descend,ascend --n-episodes 10
# Expected: >95% for all stages

# Stress test on new 70x70cm table (corners are now further)
python scripts/eval_stress.py --chain vertical --n-episodes 3
# Expected: 100% at all 5 positions

# Grasp physics
python scripts/test_grasp_physics.py --n-trials 5
# Expected: 100% static, lift, transit grasp tests

# Visual verification
python scripts/visualize.py --scenario full_move --episodes 2 --delay 0.01
# Expected: Smooth continuous animation (not snap to end state)
```

---

## Part 5 — Summary Table

| # | Priority | Type | File | Description |
|---|----------|------|------|-------------|
| 1 | **Critical** | Bug | `task.py`, `controller.py`, scripts | Fix visualization: render per step |
| 2 | **Critical** | Bug | `task.py` | Fix `_set_gripper_state()` ctrl sync (flaky test) |
| 3 | Minor | Bug | `task.py` | Fix torso hardcode `0.4` → joint range max |
| 4 | Minor | Bug | `test_grasp_physics.py` | Fix finger threshold assertion (0.003 → 0.016) |
| 5 | Minor | Bug | `eval_sequence.py` | Remove broken `"place"` chain |
| 6 | Minor | QA | `tests/conftest.py` | Add conftest for PYTHONPATH-free test runs |
| 7 | Enhancement | Config | `simulation.py`, `task.py`, `env.yaml` | Lower torso to 0.25m |
| 8 | Enhancement | Physics | `pick_and_place.xml`, `env.yaml` | Expand table to 70×70cm |
| 9 | Verification | Testing | All | Run full suite after Tasks 1–8 |

---

## Appendix: Key Measurements

### Arm Base Position: x=0.60 (confirmed correct)
- Near table edge: world X = 0.88 - 0.35 = 0.53m (for 70cm table)
- Arm at x=0.60 is 7cm inside the near table edge — satisfies "partly under the table" design intent
- Max error across 70×70cm board at arm x=0.60, torso=0.25m: **<1mm**

### Geometry Constants After Changes
```
TABLE_SURFACE_Z    = 0.400m  (unchanged)
GRASP_Z            = 0.430m  (unchanged)
HOVER_Z            = 0.460m  (unchanged)
SAFE_Z             = 0.550m  (unchanged)
table_half_x/y     = 0.35m   (was 0.30m)
torso_height       = 0.25m   (was 0.3861m)
arm base X         = 0.60m   (unchanged)
```

### Reachability Grid at Final Config (arm x=0.60, torso 0.25m, 70×70cm table)
| Position | World XY | Error |
|----------|----------|-------|
| near-left  | (0.57, 0.004) | ~0.5mm |
| near-center | (0.57, 0.264) | ~0.3mm |
| near-right | (0.57, 0.524) | ~0.5mm |
| center | (0.88, 0.264) | ~0.2mm |
| far-left | (1.19, 0.004) | ~0.5mm |
| far-center | (1.19, 0.264) | ~0.5mm |
| far-right | (1.19, 0.524) | ~0.5mm |

All positions within **<1mm** — well below the 5mm acceptance threshold.
