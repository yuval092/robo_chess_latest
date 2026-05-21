# Stage 2: Environment Interface Cleanup

## Goal

Remove all RL-specific code from `task.py` and `simulation.py`: the Phase 9 observation builder, the reward function, drift curriculum, and RL-oriented `step()` logic. Replace the 25D Dict observation with a minimal physics-state dict. Simplify the `__init__` signature and remove RL-only constructor params. The environment must load cleanly, reset correctly, and have all scripted movement methods intact and unchanged.

---

## Preconditions

- Stage 1 is complete and all 5 `verify_physics.py` tests pass.
- The current RL model is still present; this stage removes the Python-side RL interface but does not yet add the ScriptedController (that is Stage 3).

---

## 2.1 — What to Remove from `task.py`

### A. RL-Only Constructor Parameters

Current signature:
```python
def __init__(self, force_scenario=None, drift_curriculum_steps=None,
             force_drift_limit=None, hide_object=True, debug=False,
             fixed_drift=False, sample_debug_freq=None, **kwargs):
```

Remove: `drift_curriculum_steps`, `fixed_drift`, `force_drift_limit`, `sample_debug_freq`.

New signature:
```python
def __init__(self, force_scenario=None, hide_object=True, debug=False, **kwargs):
```

Rationale: `force_scenario` and `hide_object` remain because they control reset behavior (valid in scripted use). `debug` remains for logging. The removed params were purely for training curriculum management.

### B. RL-Only Constants (from `__init__`)

Remove these constant assignments:
```python
# Remove entirely:
self.DRIFT_CURRICULUM_STEPS = ...
self.FIXED_DRIFT = ...
self.force_drift_limit = ...
self.sample_debug_freq = ...

# Remove reward weights:
self.SUCCESS_BONUS = ...
self.CRASH_PENALTY = ...
self.Z_REWARD_WEIGHT = ...
self.XY_REWARD_WEIGHT = ...
self.JITTER_PENALTY_WEIGHT = ...
self.FLOOR_PENALTY = ...
self.BRAKING_REWARD_WEIGHT = ...
self.DIST_REWARD_WEIGHT = ...
self.BRAKING_DIST = ...
self.FLOOR_PROXIMITY_THRESHOLD = ...
self.STABILITY_VEL_THRESHOLD = ...
```

Keep:
```python
# These are needed for scripted safety checks:
self.DRIFT_LIMIT_END      # Used as the fixed drift limit for tube breach detection
self.DRIFT_LIMIT_START    # Keep for reference; no curriculum
self.FLOOR_LIMIT          # Used in transit crash detection
self.TABLE_SURFACE_Z      # Inherited from simulation.py, used in descend crash detection
self.SUCCESS_THRESHOLD    # Used to evaluate if arm has reached goal
```

### C. `_build_phase9_observation` Method

Remove the entire `_build_phase9_observation` method (lines 141–204 in current file). This method:
- Constructs the 25D observation vector with the "Holding Object" trick
- Maps object_pos → grip_pos for RL consumption
- Adds scenario one-hot encoding for RL policy conditioning

None of this is needed in the scripted system.

### D. `_get_obs` Override

Replace the current override:
```python
def _get_obs(self):
    return self._build_phase9_observation()
```

With a minimal implementation:
```python
def _get_obs(self):
    """Returns a minimal physics-state dict for status/debugging."""
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip").copy()
    l_finger = self._utils.get_joint_qpos(
        self.model, self.data, "robot0:l_gripper_finger_joint"
    ).item()
    return {
        "grip_pos": grip_pos,
        "grip_vel": grip_vel,
        "l_finger": l_finger,
        "scenario": self.current_scenario,
        "goal_pos": self.goal_pos.copy() if self.goal_pos is not None else np.zeros(3),
    }
```

This dict is used only by the ScriptedController (Stage 3) and eval scripts (Stage 4) for status reporting. It is not fed to any RL model.

### E. `compute_reward` Method

Remove the entire `compute_reward` method (lines 934–962 in current file). It contains:
- Negative L2 distance reward
- Z-error penalty
- XY-error penalty
- Braking penalty
- Velocity queries

None of this is needed in the scripted system.

### F. `step()` Method — Simplify

The current `step()` method (lines 964–1075) handles:
1. RL action clipping and application via `_set_action`
2. Physics advance via `_mujoco_step`
3. Observation building
4. Reward computation
5. Crash detection
6. Success detection
7. Episode termination

In the scripted system, the ScriptedController calls `_move_mocap_to`, `execute_grasp`, etc. directly — it does not call `step()` at all. However, `step()` must remain on the class to satisfy the Gymnasium interface (so `gym.make()` doesn't error). Replace with a minimal stub:

```python
def step(self, action):
    """
    Minimal physics step. In scripted mode the ScriptedController
    drives the arm directly via _move_mocap_to; this step() is a
    passthrough for compatibility and lightweight testing only.
    """
    action_copy = np.zeros(4)  # Ignore incoming action; all movement is scripted
    action_copy[3] = -1.0
    self._set_action(action_copy)
    self._mujoco_step(action_copy)
    obs = self._get_obs()
    return obs, 0.0, False, False, {"is_success": 0.0, "scenario": self.current_scenario}
```

**Important**: The crash detection logic (TUBE_BREACH, TABLE_HIT, FLOOR_HIT, FINGER_FAULT, cube drop) is **removed from step()** in this stub. Instead, all safety monitoring is moved into the ScriptedController (Stage 3), which monitors these conditions inside the scripted movement loops where it has full context about the movement intent.

### G. `_is_success` Method

Keep this method unchanged. It is used by `soft_reset` and the ScriptedController to determine when the arm has reached the target:
```python
def _is_success(self, achieved_goal, desired_goal):
    # ...distance threshold check...
```

### H. Drift Curriculum Logic in `step()`

The current step() contains:
```python
current_drift_limit = self.DRIFT_LIMIT_END
if not self.fixed_drift and self.force_drift_limit is None:
    dp = min(self.total_env_steps / self.DRIFT_CURRICULUM_STEPS, 1.0)
    current_drift_limit = self.DRIFT_LIMIT_START - ...
elif self.force_drift_limit is not None:
    current_drift_limit = self.force_drift_limit
```

After Stage 2, there is no curriculum. Replace with a single fixed drift limit:
```python
self.current_drift_limit = self.DRIFT_LIMIT_END  # Fixed constant, set in __init__
```

This is used by the ScriptedController for tube breach monitoring.

---

## 2.2 — What to Remove from `simulation.py`

The `simulation.py` file (the `ChessSimulationEnv` class) is largely kept intact. It provides the MuJoCo infrastructure, `_set_action`, `_env_setup`, `_sample_board_position`, `_sample_goal`, and `_reset_sim`. Only minor cleanup is needed:

### A. RL-Oriented `_sample_goal`

The current `_sample_goal` samples a goal position on the board. This method is called by the parent class `MujocoFetchPickAndPlaceEnv` during `reset()` to populate `self.goal`. After the refactor, `self.goal` is no longer fed to an RL model, but `_reset_sim` in `task.py` reads `self.goal_pos` which is derived from it.

Keep `_sample_goal` unchanged. The parent class calls it during reset and sets `self.goal`. `task.py`'s `_reset_sim` then reads `self.goal` and stores it as `self.goal_pos`. This chain remains valid in the scripted system.

### B. `_reset_sim` in `simulation.py`

The base `_reset_sim` in `simulation.py` places the cube on the board and does initial setup. Keep it unchanged. The override in `task.py` is the one that handles the multi-phase reset with `_settle_arm_to_start`, scenario selection, and finger transitions. Keep all of this.

---

## 2.3 — Gymnasium Registration Change

The current registration in `src/chess_env/__init__.py`:
```python
register(
    id="ChessFetchTask-v0",
    entry_point="src.chess_env.task:ChessTaskEnv",
    max_episode_steps=200,
)
```

Keep the registration unchanged. `max_episode_steps=200` still wraps the env in a `TimeLimit`. The ScriptedController will traverse the wrapper to reset `_elapsed_steps` between scenarios, exactly as the eval scripts do today.

---

## 2.4 — Updated `__init__` in Full

After Stage 2, the `ChessTaskEnv.__init__` signature and constant block should look like this (shown condensed for the plan; exact code must be edited in the file):

```python
def __init__(self, force_scenario=None, hide_object=True, debug=False, **kwargs):
    # Consume legacy params to avoid gym warnings
    kwargs.pop('total_curriculum_steps', None)
    kwargs.pop('num_envs', None)
    kwargs.pop('curriculum_progress_override', None)
    kwargs.pop('drift_curriculum_steps', None)
    kwargs.pop('force_drift_limit', None)
    kwargs.pop('fixed_drift', None)
    kwargs.pop('sample_debug_freq', None)
    
    self.env_cfg = load_config("env")
    self.physics_cfg = load_config("physics")

    self.force_scenario = force_scenario
    self.hide_object = hide_object
    self.debug = debug

    self.current_scenario = None
    self.tube_center_xy = None
    self.goal_pos = None
    self.episode_steps = 0
    self.total_env_steps = 0
    self.episode_number = 0

    super().__init__(debug=debug, **kwargs)

    # --- Task Constants (scripted-system subset) ---
    self.CUBE_HEIGHT = self.env_cfg["cube_height"]
    self.CUBE_Z = self.env_cfg["cube_z"]
    self.GRASP_Z = self.env_cfg["grasp_z"]
    self.HOVER_Z = self.env_cfg.get("hover_z", 0.460)
    self.SAFE_Z = self.env_cfg["safe_z"]
    self.SUCCESS_THRESHOLD = self.env_cfg["success_threshold"]
    self.FLOOR_LIMIT = self.env_cfg["floor_limit"]
    self.HIDDEN_OBJECT_POS = np.array(self.env_cfg["hidden_object_pos"])
    self.MAX_SETTLE_STEPS = self.physics_cfg["max_settle_steps"]
    self.SETTLE_TOLERANCE = self.physics_cfg["settle_tolerance"]
    self.SETTLE_GAIN = self.physics_cfg["settle_gain"]
    self.SETTLE_STEPS_FINAL = self.physics_cfg["settle_steps_final"]
    self.HALT_VEL_THRESHOLD = self.env_cfg.get("halt_vel_threshold", 0.0005)
    self.current_drift_limit = self.env_cfg["drift_limit_end"]  # Fixed; no curriculum

    # --- Actuator Enforcement ---
    self.FINGER_OPEN_JOINT = self.env_cfg["finger_open_joint"]
    self.FINGER_CLOSED_JOINT = self.env_cfg["finger_closed_joint"]
    self.FINGER_OUTER_OFFSET = self.env_cfg["finger_outer_offset"]
    self.finger_target_joint = self.FINGER_CLOSED_JOINT
    self.grasp_mode = False

    # --- Grasp Stage Constants ---
    self.GRASP_CONTACT_APPROACH_TOLERANCE = self.env_cfg.get("grasp_contact_approach_tolerance", 0.001)
    self.GRASP_CLOSE_STEPS = self.env_cfg.get("grasp_close_steps", 150)
    self.GRASP_HOLD_STEPS = self.env_cfg.get("grasp_hold_steps", 50)
    self.GRASP_VERIFY_XY_THRESHOLD = self.env_cfg.get("grasp_verify_xy_threshold", 0.015)
    self.GRASP_VERIFY_Z_THRESHOLD = self.env_cfg.get("grasp_verify_z_threshold", 0.020)
    self.CUBE_HELD_XY_LIMIT = self.env_cfg.get("cube_held_xy_limit", 0.030)
    self.CUBE_HELD_Z_LIMIT = self.env_cfg.get("cube_held_z_limit", 0.020)

    # Overrides for evaluation
    self.force_start_pos = None
    self.force_cube_pos = None

    home_xy = self.env_cfg.get("home_position_xy", [0.680, 0.2641])
    self.HOME_POS = np.array([home_xy[0], home_xy[1], self.SAFE_Z])

    # Logger
    # ... (unchanged)
```

Note that `kwargs.pop` lines now also consume the legacy RL params to avoid Gymnasium warnings in case any calling code still passes them.

---

## 2.5 — File Edit Summary

### `src/chess_env/task.py` Changes

| Section | Action |
|---------|--------|
| `__init__` signature | Remove 4 params: `drift_curriculum_steps`, `fixed_drift`, `force_drift_limit`, `sample_debug_freq` |
| `__init__` body | Add `kwargs.pop` for removed params; remove 12 reward-weight constants; add `self.current_drift_limit` |
| `_build_phase9_observation` | **DELETE entire method** |
| `_get_obs` | **REPLACE** with minimal 5-key dict |
| `compute_reward` | **DELETE entire method** |
| `step` | **REPLACE** with 6-line minimal stub (no reward, no safety checks) |
| Everything else | **KEEP UNCHANGED**: `_reset_sim`, `_settle_arm_to_start`, `transition_validate`, `_move_mocap_to`, `execute_grasp`, `execute_place`, `soft_reset`, `_set_gripper_state`, `_check_cube_held`, `get_cube_position`, `get_cube_quat`, `_is_success` |

### `src/chess_env/simulation.py` Changes

No changes in Stage 2.

### `src/chess_env/__init__.py` Changes

No changes in Stage 2.

---

## 2.6 — Editing Instructions

**Do not edit the file all at once.** Make these 5 targeted edits in order, running the smoke test after each one.

### Edit 1: `__init__` signature and legacy pop

Find:
```python
def __init__(self, force_scenario=None, drift_curriculum_steps=None, force_drift_limit=None, hide_object=True, debug=False, fixed_drift=False, sample_debug_freq=None, **kwargs):
```

Replace with:
```python
def __init__(self, force_scenario=None, hide_object=True, debug=False, **kwargs):
```

And add to the top of the method body (after the docstring):
```python
kwargs.pop('drift_curriculum_steps', None)
kwargs.pop('force_drift_limit', None)
kwargs.pop('fixed_drift', None)
kwargs.pop('sample_debug_freq', None)
```

### Edit 2: Remove reward constants from `__init__` body

Remove these 12 lines:
```python
self.DRIFT_LIMIT_START = self.env_cfg["drift_limit_start"]
...
self.DRIFT_CURRICULUM_STEPS = drift_curriculum_steps or ...
...
self.STABILITY_VEL_THRESHOLD = ...
self.BRAKING_DIST = ...
self.FLOOR_PROXIMITY_THRESHOLD = ...
self.Z_REWARD_WEIGHT = ...
self.XY_REWARD_WEIGHT = ...
self.JITTER_PENALTY_WEIGHT = ...
self.FLOOR_PENALTY = ...
self.BRAKING_REWARD_WEIGHT = ...
self.DIST_REWARD_WEIGHT = ...
```

Add this line where the drift constants were:
```python
self.current_drift_limit = self.env_cfg["drift_limit_end"]
```

Also remove:
```python
self.SUCCESS_BONUS = self.env_cfg["success_bonus"]
self.CRASH_PENALTY = self.env_cfg["crash_penalty"]
```

Keep:
```python
self.DRIFT_LIMIT_END = self.env_cfg["drift_limit_end"]  # Keep for reference
```

Also remove:
```python
self.fixed_drift = fixed_drift
self.force_drift_limit = force_drift_limit
self.sample_debug_freq = ...
```

### Edit 3: Delete `_build_phase9_observation`

Delete the entire method body from its `def` line through the closing `}` of the return statement (lines 141–204 in current file).

### Edit 4: Replace `_get_obs`

Find:
```python
def _get_obs(self):
    """Returns the current observation dictionary."""
    return self._build_phase9_observation()
```

Replace with:
```python
def _get_obs(self):
    """Returns a minimal physics-state dict for status/debugging."""
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
    grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip").copy()
    l_finger = self._utils.get_joint_qpos(
        self.model, self.data, "robot0:l_gripper_finger_joint"
    ).item()
    return {
        "grip_pos": grip_pos,
        "grip_vel": grip_vel,
        "l_finger": l_finger,
        "scenario": self.current_scenario,
        "goal_pos": self.goal_pos.copy() if self.goal_pos is not None else np.zeros(3),
    }
```

### Edit 5: Delete `compute_reward` and replace `step`

Delete the entire `compute_reward` method (lines 934–962).

Find the full `step` method (lines 964–1075). Replace the entire body with:
```python
def step(self, action):
    """
    Minimal physics step. The ScriptedController drives the arm via
    _move_mocap_to; this step() satisfies the Gymnasium interface only.
    """
    zero = np.zeros(4)
    zero[3] = -1.0
    self._set_action(zero)
    self._mujoco_step(zero)
    obs = self._get_obs()
    return obs, 0.0, False, False, {"is_success": 0.0, "scenario": self.current_scenario}
```

---

## 2.7 — Validation Steps

### Check 1: Environment Loads Without Error

```bash
python -c "
import gymnasium as gym
import src.chess_env
env = gym.make('ChessFetchTask-v0')
obs, info = env.reset()
print('Reset OK. Obs keys:', list(obs.keys()))
assert 'grip_pos' in obs, 'Expected grip_pos in obs'
assert 'scenario' in obs, 'Expected scenario in obs'
print('PASS: obs structure correct')
env.close()
"
```

**Expected**: Prints obs keys `['grip_pos', 'grip_vel', 'l_finger', 'scenario', 'goal_pos']` and `PASS`.

### Check 2: No RL Attribute Exists

```bash
python -c "
import gymnasium as gym
import src.chess_env
env = gym.make('ChessFetchTask-v0')
env.reset()
inner = env.unwrapped
assert not hasattr(inner, 'Z_REWARD_WEIGHT'), 'Z_REWARD_WEIGHT should be removed'
assert not hasattr(inner, 'CRASH_PENALTY'), 'CRASH_PENALTY should be removed'
assert not hasattr(inner, 'DRIFT_CURRICULUM_STEPS'), 'DRIFT_CURRICULUM_STEPS should be removed'
assert hasattr(inner, 'current_drift_limit'), 'current_drift_limit should exist'
assert hasattr(inner, 'execute_grasp'), 'execute_grasp should still exist'
assert hasattr(inner, 'soft_reset'), 'soft_reset should still exist'
print('PASS: RL attributes removed, scripted methods intact')
env.close()
"
```

**Expected**: `PASS`.

### Check 3: step() Returns Correct Structure

```bash
python -c "
import gymnasium as gym
import numpy as np
import src.chess_env
env = gym.make('ChessFetchTask-v0')
env.reset()
action = np.zeros(4)
obs, reward, terminated, truncated, info = env.step(action)
assert reward == 0.0, f'Expected 0.0 reward, got {reward}'
assert terminated == False
assert 'is_success' in info
assert 'grip_pos' in obs
print('PASS: step() returns correct structure')
env.close()
"
```

**Expected**: `PASS`.

### Check 4: No Import Errors from Old Attribute References

```bash
python -c "
import importlib
import src.chess_env.task
importlib.reload(src.chess_env.task)
print('PASS: task.py imports cleanly')
"
```

**Expected**: `PASS` with no AttributeError or NameError.

### Check 5: `verify_physics.py` Still Passes

```bash
python scripts/verify_physics.py
```

**Expected**: All 5 tests still pass. The verify_physics script does not use RL; it tests the physics directly.

### Check 6: Environment Reset Cycle Works

```bash
python -c "
import gymnasium as gym
import src.chess_env
env = gym.make('ChessFetchTask-v0')
for i in range(3):
    obs, info = env.reset()
    assert obs['grip_pos'].shape == (3,)
    assert obs['scenario'] in ['transit', 'descend', 'ascend', None]
    print(f'Reset {i+1}: scenario={obs[\"scenario\"]}, grip_z={obs[\"grip_pos\"][2]:.3f}')
print('PASS')
env.close()
"
```

**Expected**: 3 resets succeed, grip Z is near SAFE_Z (0.550) for transit scenario.

---

## 2.8 — What Should NOT Change

Verify after Stage 2 that the following methods still exist and are functionally identical to the pre-Stage-2 version. Do not edit these:

- `_reset_sim` (in task.py)
- `_settle_arm_to_start`
- `transition_validate`
- `_move_mocap_to`
- `execute_grasp`
- `execute_place`
- `soft_reset`
- `_set_gripper_state`
- `_check_cube_held`
- `get_cube_position`
- `get_cube_quat`
- `_is_success`

---

## 2.9 — Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `kwargs.pop` for legacy params breaks something | Low | The params are consumed silently; any code passing them will still work |
| `_get_obs` return format breaks test_task_chaining.py | Medium | Update test assertions from dict-key access to new key names if needed |
| Parent class calls `compute_reward` during reset | Low | The parent's `compute_reward` is a Gymnasium abstract method — we've overridden it. After removal of our override, the parent's stub raises NotImplementedError. Add a minimal stub: `def compute_reward(self, achieved_goal, desired_goal, info): return 0.0` |

**The compute_reward stub is required.** The Gymnasium-Robotics `GoalEnv` base class declares `compute_reward` as abstract. After removing the RL reward implementation, add this stub to prevent NotImplementedError:

```python
def compute_reward(self, achieved_goal, desired_goal, info):
    """Stub: scripted system does not use reward."""
    return 0.0
```

---

## 2.10 — Summary

| File | Lines Changed (approx) | Nature |
|------|----------------------|--------|
| `src/chess_env/task.py` | ~120 lines removed | Remove `_build_phase9_observation`, `compute_reward`; simplify `step`, `__init__` |

**Stage 2 is complete when all 6 validation checks pass, particularly `verify_physics.py` (all 5 tests) and the env reset cycle test.**
