# Arm Control

## ModelEmbeddedController

`src/chess_env/model_controller.py`

Orchestrates a full chess move by combining three SAC specialist models with scripted grasp and place transitions.

### Stage Pipeline

A full pick-and-place (`run_full_move`) runs eight stages:

```
pick sequence:
  transit  → descend  → grasp  → ascend

place sequence:
  transit  → descend  → place  → ascend
```

Each SAC stage uses `_run_model_stage`:

1. `_get_stage_model()` — retrieves the loaded SAC model, raises if missing.
2. `_setup_model_stage_env()` — sets `goal_pos`, `current_scenario`, `tube_center_xy`, `finger_target_joint` on the env.
3. `_check_finger_precondition()` — verifies the finger is at the expected position before inference (skipped in `grasp_mode`).
4. `_run_inference_loop()` — steps the model until success, crash, or `TIMEOUT`.
5. `_build_model_stage_result()` — wraps outcome in a `StageResult`.

### SAC Inference Loop

```python
for _ in range(max_steps):
    if _is_success(grip_pos, target_pos) and speed < stability_vel_threshold:
        return None   # success
    _execute_step(stage, model)
    crash = _check_crash(env, stage, grip_pos)
    if crash:
        return crash
return "TIMEOUT"
```

`_execute_step` overrides the gripper dimension: closed (`-1.0`) for transit/ascend, open (`+1.0`) for descend. Transit also clamps wrist orientation to `VERTICAL_QUAT` to match training conditions.

### Crash Checks (inference)

| Stage | Check |
|---|---|
| transit | `FLOOR_HIT` if `grip_z < FLOOR_LIMIT` |
| descend/ascend | `TUBE_BREACH` if XY drift > `eval_drift_limit`; `TABLE_HIT` if `grip_z < TABLE_SURFACE_Z` |
| transit/ascend | `PIECE_DROPPED_XY/Z` if `grasp_mode` and piece is out of range |

### Soft Reset Between Stages

`run_descend` and `run_ascend` call `_prepare_stage` before inference. This calls `env.soft_reset(...)` to switch scenario context and `reset_elapsed_steps` to clear the episode step counter.

---

## Scripted Grasp Pipeline

`ChessProductionEnv.execute_grasp()` — runs after DESCEND succeeds at `HOVER_Z`.

```
_halt()
_check_hover_preconditions()       ← speed, Z height, fingers open
_check_piece_yaw()                 ← abort if piece rotated >25°
_align_over_xy(piece_pos[:2])      ← XY alignment at hover height
_move_z(piece_xy, GRASP_Z, ...)    ← plunge down to grasp height
grasp_mode = True
_close_fingers()                   ← ramp fingers closed, detect empty grasp
_hold_and_verify()                 ← hold steady, verify XY/Z/finger
_move_z(grip_xy, HOVER_Z, ..., verify_held=True)  ← retract, watch for drop
```

### `_check_piece_yaw`

Extracts yaw (rotation around Z axis) from the piece quaternion. Uses 4-fold symmetry to compute the effective yaw: fold `[0,π]` into `[0,π/2]`, then take the distance to the nearest axis. Aborts with `PIECE_ROTATED` if `effective_yaw > 25°` (a piece rotated that far would have an effective width exceeding the maximum finger opening).

### `_close_fingers`

Ramps `finger_target_joint` from `FINGER_OPEN_JOINT` to `GRASP_RAMP_END` over `grasp_close_steps` sim steps, tracking live piece XY each step. Empty-grasp detection (checking if `finger_angle < EMPTY_GRASP_THRESHOLD`) starts at 65% of steps to avoid false positives before the fingers have had time to close.

### `_move_z`

Unified vertical movement used for both plunge and retract. Direction is inferred from sign of `(target_z - current_z)`. Returns `MOVE_Z_TIMEOUT` if the grip ends more than 8 mm from the target after all steps.

---

## Scripted Place Pipeline

`ChessProductionEnv.execute_place(dst_xy)` — runs after DESCEND succeeds at `HOVER_Z` over destination.

```
_halt()
_check_hover_preconditions()       ← speed, Z height (fingers expected closed — grasp_mode=True)
_align_over_xy(dst_xy)             ← XY alignment
_move_z(dst_xy, GRASP_Z, ...)      ← plunge
_open_fingers(place_pos)           ← ramp fingers open, settle
grasp_mode = False
_verify_placement(dst_xy)          ← check piece XY drift and Z height
_move_z(place_xy, HOVER_Z, ...)    ← retract (non-fatal timeout)
```

### Failure Reasons

| Reason | Source |
|---|---|
| `PRECONDITION_SPEED` | Grip moving too fast when pipeline starts |
| `PRECONDITION_Z` | Grip not within 25 mm of `HOVER_Z` |
| `PRECONDITION_FINGERS_NOT_OPEN` | Fingers not open before grasp |
| `PIECE_ROTATED` | Piece yaw exceeds 25° |
| `ALIGN_FAILED` | XY alignment didn't converge |
| `MOVE_Z_TIMEOUT` | Plunge/retract didn't reach target within 8 mm |
| `FINGER_CLOSED_EMPTY` | Fingers closed fully with no piece contact |
| `VERIFY_XY_FAILED` | Piece–grip XY error too large after hold |
| `VERIFY_Z_FAILED` | Piece–grip Z error too large after hold |
| `VERIFY_FINGERS_CLOSED_EMPTY` | Fingers fully closed in verify phase |
| `PIECE_DROPPED_DURING_RETRACT` | Piece Z deviated from expected offset during ascent |
| `PLACE_XY_FAILED` | Placed piece drifted too far from target |
| `PLACE_Z_FAILED` | Placed piece not flat on table |

---

## Home Posture

After each move the arm returns to `HOME_POS` via SAC transit, then `reset_arm_to_home_posture()` snaps redundant wrist/roll joints back to the exact reset-time configuration. This prevents configuration drift across moves.

The posture is captured once after the first episode reset (when the arm lands at `HOME_POS`) and reapplied via linear interpolation over 10 sim steps.
