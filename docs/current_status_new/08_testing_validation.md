# Testing and Validation

## pytest Test Suite (`tests/`)

### Running Tests

```bash
# Recommended (no PYTHONPATH needed — conftest.py handles it):
pytest tests/ -v

# With output capture:
pytest tests/ -v -s

# Single file:
pytest tests/chess_env/test_task_chaining.py -v
```

`tests/conftest.py` inserts the project root into `sys.path`, so tests can `import src.chess_env` without setting `PYTHONPATH`.

---

### `tests/chess_env/test_waypoints.py` — Pure Logic Tests

These tests import only `src/chess_env/waypoints.py`. No MuJoCo instantiation. Fast (< 0.1s total).

| Test | What It Checks |
|------|----------------|
| `test_validate_chain_valid` | `validate_chain(["transit", "descend", "ascend"])` passes |
| `test_validate_chain_invalid` | `["transit", "ascend"]` raises `ValueError` |
| `test_derive_goal_pos` | `derive_goal_pos("descend", xy)` returns `[xy, HOVER_Z]` |
| `test_derive_goal_pos_all_scenarios` | All three scenarios return correct Z levels |
| `test_validate_chain_invalid_transitions` | descend→transit, transit→ascend, descend→descend all fail |
| `test_derive_goal_pos_error` | Unknown scenario raises `ValueError` |
| `test_exit_waypoint_error` | Unknown scenario raises `ValueError` |
| `test_exit_waypoint_all_scenarios` | Correct exit Z for all scenarios |
| `test_chain_shortcuts_validity` | All entries in `CHAIN_SHORTCUTS` pass `validate_chain` |

Constants are imported from `waypoints.py` (not hardcoded), so the tests will not drift if Z-level constants change.

---

### `tests/chess_env/test_task_chaining.py` — Integration Tests

These tests create a real `ChessTaskEnv` instance. Each test takes 2–4 seconds (physics settlement).

#### `test_transition_validate`

```python
env = gym.make("ChessFetchTask-v0", render_mode=None)
env.reset()
diag = env.unwrapped.transition_validate()

assert "grip_speed_mm_s" in diag
assert "is_velocity_ok" in diag
assert diag["error_from_nominal_mm"] is None   # No nominal provided

nominal_pos = grip_pos + [0.001, 0, 0]   # 1mm offset
diag2 = env.unwrapped.transition_validate(nominal_exit_pos=nominal_pos)
assert abs(diag2["error_from_nominal_mm"] - 1.0) < 1e-3
```

Verifies the diagnostic API returns the correct structure and computes Euclidean distance correctly.

#### `test_soft_reset_flow`

```python
uw.current_scenario = "transit"
obs, info = uw.soft_reset(
    new_scenario="descend",
    new_goal_pos=[0.5, 0.5, HOVER_Z],
    nominal_exit_pos=[0.5, 0.5, SAFE_Z],
    nominal_xy=[0.5, 0.5]
)
assert isinstance(obs, dict)
assert "grip_pos" in obs
assert "observation" in obs
assert "halt_steps" in info
assert "align_steps" in info
assert uw.current_scenario == "descend"
assert np.allclose(uw.goal, next_goal)
assert uw.episode_steps == 0
```

Verifies the full soft_reset flow: state updates, observations returned, scenario switch, step counter reset.

#### `test_soft_reset_finger_validation`

```python
obs, info = uw.soft_reset(
    new_scenario="descend",
    new_goal_pos=[0.5, 0.5, HOVER_Z],
    nominal_exit_pos=[0.5, 0.5, SAFE_Z],
    nominal_xy=[0.5, 0.5]
)
assert isinstance(info, dict)
```

Previously flaky (failed in full-suite runs but passed in isolation). Root cause was `_set_gripper_state()` not syncing `data.ctrl`. Now fixed — passes reliably in all orderings because the actuator target is synced and cannot close the fingers after a teleport.

---

## Runtime Validation

### `_reset_sim` — Post-Reset Finger Validation

```python
l_pos = get_joint_qpos("robot0:l_gripper_finger_joint").item()
if abs(l_pos - self.finger_target_joint) > 0.0005:
    self.logger.error(f"Reset Failed: Finger at {l_pos}, target {self.finger_target_joint}")
    return False
```

If finger position deviates >0.5mm from target after reset settlement, `_reset_sim` returns `False`. The Gymnasium `TimeLimit` wrapper interprets this as a failed reset and automatically retries. In practice this should never trigger under normal operation.

### `soft_reset` — Three Failure Modes

| Exception | Cause |
|-----------|-------|
| `RuntimeError: HALT_FAILED` | Arm didn't stop in 100 steps (velocity never < 0.5mm/s) |
| `RuntimeError: ALIGN_FAILED` | Arm didn't reach `nominal_exit_pos` in 200 steps (> 3mm after 200 steps) |
| `RuntimeError: FINGER_VALIDATION_FAILED` | Finger not within 0.5mm of target after `_set_gripper_state` |

### `step()` — Per-Step Crash Detection

Every `env.step()` call checks:

| Condition | When Active | Crash Reason |
|-----------|-------------|--------------|
| `\|l_finger - finger_target\| > 0.003` | Always | Finger fault |
| `\|cube - grip\| > cube_held_limits` | `grasp_mode=True` | `CUBE_DROPPED_XY/Z` |
| `grip_z < FLOOR_LIMIT (0.400)` | Transit only | Floor hit |
| `\|grip_xy - tube_center_xy\| > drift_limit` | Descend/Ascend | `TUBE_BREACH` |
| `grip_z < TABLE_SURFACE_Z (0.400)` | Descend/Ascend | `TABLE_HIT` |

All crash conditions terminate the episode with `terminated=True` and `info["crash_reason"]` set.

---

## `scripts/verify_physics.py` — Full Physics Validation

Run this after any XML, config, or physics change:
```bash
python scripts/verify_physics.py
# Expected: SYSTEM HEALTHY (6/6 tests passed)
```

### Test 1: XML Integrity

Loads the model via `ChessTaskEnv` and checks:
- Model loads without errors
- `robot0:grip` site exists and is accessible

### Test 2: Table Geometry

```python
cfg = load_config("env")
assert cfg["table_half_x"] == 0.35
assert cfg["table_half_y"] == 0.35
# Checks 4 leg geoms exist
# Checks surface top Z = TABLE_SURFACE_Z (0.400m)
```

Validates that the config matches the expected 70×70cm table and that the XML surface position gives the correct Z.

### Test 3: Grasp XML Verification

Checks:
- Cube mass = 0.05 kg
- Actuator Kp = 20000
- Actuator ctrlrange max = 0.05
- `GRASP_Z = 0.430m`

### Test 4: Static Stability

Runs 100 `env.step(np.zeros(4))` steps and checks that the cube drifts < 1mm. Ensures gravity and contact physics are correctly configured and the cube rests stably.

### Test 5: Kinematic Reachability

Tests all **64 chess square centers** at both SAFE_Z and GRASP_Z using `_settle_arm_to_start`. Passes if max error ≤ 5mm across all 128 positions.

```python
# Computes 8×8 grid from env config
x0 = cx - hx + margin;  x1 = cx + hx - margin
y0 = cy - hy + margin;  y1 = cy + hy - margin
sq_x = (x1-x0)/8;  sq_y = (y1-y0)/8
squares = [(x0+(r+0.5)*sq_x, y0+(c+0.5)*sq_y) for r in 0..7 for c in 0..7]
# threshold: 5mm per position
```

**Current results** (arm_x=0.56, torso=0.3661, 70×70cm board):
- Max error: **2.9mm** (PASS)
- Worst position: row=0, col=3 (near-row center column), at SAFE_Z

**Why this works**: The absolute board edge at `(0.57, 0.2641)` is NOT a chess square center (chess squares start at x≈0.609). The old test incorrectly tested this unreachable edge point. The current test uses actual chess square centers, all of which are reachable with arm_x=0.56 and torso=0.3661.

### Test 6: Teleport Verification

Resets with `hide_object=True` and checks that the cube's world position is within 1mm of `HIDDEN_OBJECT_POS = [2.0, 2.0, 0.015]`. Validates the object-hiding mechanism used in transit/vertical training.

---

## Expected Test Results

Verified with arm_x=0.56, torso=0.3661, 70×70cm board:

```
pytest tests/ -v
→ 12/12 passed

python scripts/verify_physics.py
→ SYSTEM HEALTHY (6/6 tests passed)
→ Kinematic Reachability: 2.9mm max error (all 64 squares)

python scripts/eval_stages.py --stages transit,descend,ascend --n-episodes 5
→ transit 100%  descend 100%  ascend 100%

python scripts/eval_stress.py --chain pick --n-episodes 3
→ 5 positions × 3 episodes: 100% overall

python scripts/eval_sequence.py --chain full_move --n-episodes 5
→ 100% (5/5) full pick+place
```
