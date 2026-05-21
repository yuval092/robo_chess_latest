# Testing and Validation

## pytest Test Suite (`tests/`)

### `tests/chess_env/test_waypoints.py`

Unit tests for `src/chess_env/waypoints.py`. These test pure Python logic (no MuJoCo instantiation):

| Test | Status | Notes |
|------|--------|-------|
| `test_validate_chain_valid` | PASS | Validates `["transit", "descend", "ascend"]` |
| `test_validate_chain_invalid` | PASS | `["transit", "ascend"]` raises ValueError |
| `test_derive_goal_pos` | **FAIL** | Asserts `descend → z=0.430` but HOVER_Z=0.460 |
| `test_derive_goal_pos_all_scenarios` | **FAIL** | Same issue as above |
| `test_validate_chain_invalid_transitions` | PASS | Tests descend→transit, transit→ascend, descend→descend |
| `test_derive_goal_pos_error` | PASS | Invalid scenario raises ValueError |
| `test_exit_waypoint_error` | PASS | Invalid scenario raises ValueError |
| `test_exit_waypoint_all_scenarios` | **FAIL** | Asserts `descend → z=0.430` but HOVER_Z=0.460 |
| `test_chain_shortcuts_validity` | PASS | All CHAIN_SHORTCUTS pass validate_chain |

**Root cause of failures**: Tests were written when `hover_z` was 0.430. The config was later updated to 0.460 without updating the tests. Fix: update assertions from `0.430` to the current `HOVER_Z` value.

### `tests/chess_env/test_task_chaining.py`

Integration tests that instantiate the environment (MuJoCo required):

| Test | What It Tests |
|------|--------------|
| `test_transition_validate` | `transition_validate()` returns correct dict structure and computes distance to nominal position |
| `test_soft_reset_flow` | `soft_reset("descend", ...)` transitions state correctly, returns valid obs/info |
| `test_soft_reset_finger_validation` | `soft_reset` to descend successfully opens fingers and validates |

These tests work by:
1. Making the environment
2. Resetting (which runs a full _reset_sim including settle)
3. Calling the method under test
4. Asserting return values and state

**Limitation**: The tests don't validate that the arm actually moved to the correct position — only that the Python state variables are updated correctly. Positional accuracy requires visual inspection or the `verify_physics.py` scripts.

---

## Runtime Validation in `_reset_sim`

```python
# Phase 3: Final Validation
l_pos = get_joint_qpos("robot0:l_gripper_finger_joint").item()
if abs(l_pos - self.finger_target_joint) > 0.0005:
    self.logger.error(f"Reset Failed: ...")
    return False
```

If finger position deviates >0.5mm from target after the scripted transition, `_reset_sim` returns `False`. The Gymnasium wrapper treats this as a failed reset and retries. In practice, this validation almost never fails under normal operation; it serves as a safety catch for physics instability.

---

## Runtime Validation in `soft_reset`

Two runtime `RuntimeError` raises:
1. **HALT_FAILED**: Arm doesn't stop within 100 steps. This indicates the arm is in a physically impossible state (e.g., external force holding it).
2. **ALIGN_FAILED**: Arm doesn't converge to nominal_exit_pos within 200 steps. This indicates kinematic infeasibility (target outside workspace).
3. **FINGER_VALIDATION_FAILED**: Post-transition finger position outside 0.5mm. Indicates actuator failure or physics instability.

---

## Runtime Validation in `step`

Each `env.step()` call validates:
1. **Finger Fault**: `|l_finger - finger_target_joint| > 0.003` → crash. Ensures finger teleportation hasn't failed silently.
2. **Cube Drop** (when grasp_mode=True): `||cube_xy - grip_xy|| > 0.030` or `|cube_z - (grip_z - 0.015)| > 0.020` → crash.
3. **Floor Hit** (transit): `grip_z < FLOOR_LIMIT (0.400)` → crash.
4. **Tube Breach** (descend/ascend): Radial drift > `current_drift_limit` → crash.
5. **Table Hit** (descend/ascend): `grip_z < TABLE_SURFACE_Z (0.400)` → crash.

These form the **production certification** layer — any physics anomaly during a step results in a crash termination with a descriptive reason string.

---

## Logging Infrastructure

### Environment Logger (`ChessTaskEnv.logger`)
Per-process `logging.FileHandler` writing to `logs/env_debug/env_{pid}.log`:
- **DEBUG level** (when `debug=True` or sampled): Full step-by-step grip position, target, finger state, drift info.
- **INFO level**: Episode start/end summaries.
- **ERROR level**: Reset failures, finger faults.

### Training Progress Logger (`src/training/callbacks.py`)
```python
progress_logger = setup_logger("training_progress", "logs/training_progress_detailed.log")
```
Logs every 2000 training steps: rolling rewards, success rates, new best model saves.

### Debug Sampling
```python
sample_debug_freq: 50  # Enable debug every 50 episodes
```
Every 50th episode, the full debug log is enabled for that episode. This provides periodic high-fidelity diagnostics without overwhelming disk space.

---

## `verify_physics.py` — Runtime Sanity Checks

Runs without RL model, instantiates `ChessTaskEnv` directly:

```
test_xml_integrity      → Load model, check for "robot0:grip" site
test_grasp_xml_changes  → Verify cube mass=0.05, Kp=150000, ctrlrange max=0.05, GRASP_Z=0.425
test_static_stability   → 100 no-action steps; cube drift < 1mm
test_kinematic_reachability → _settle_arm_to_start at sampled positions; error < 5mm
test_teleport_verification  → hide_object=True; cube at hidden_object_pos ± 1mm
```

These tests are quick (run in <60 seconds) and cover the most critical physics invariants.
