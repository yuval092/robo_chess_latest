# Arm Control Algorithms

## Overview

The `ScriptedController` (`src/chess_env/controller.py`) is a purely deterministic arm driver. It replaces any RL policy and drives the Fetch arm through a fixed sequence of waypoint stages to execute a chess piece move. No neural network is involved at runtime.

---

## Movement Primitive: `_run_movement_loop`

Every stage (transit, descend, ascend) uses a single proportional-control loop:

```python
# controller.py:69
def _run_movement_loop(self, target_pos, *, tolerance, max_steps, abort_fn=None):
    for step in range(max_steps):
        grip_pos = get_site_xpos("robot0:grip")
        error = target_pos - grip_pos
        dist = norm(error)

        if dist < tolerance:
            return StageResult(success=True, ...)

        if abort_fn and abort_fn(grip_pos) → (True, reason):
            return StageResult(success=False, crash_reason=reason, ...)

        # Proportional step with clamping
        step_vec = STEP_GAIN * error        # STEP_GAIN = 1.0 (full error)
        if norm(step_vec) > MAX_STEP_SIZE:  # 24mm max
            step_vec = normalize(step_vec) * MAX_STEP_SIZE
        elif norm(step_vec) < MIN_STEP_SIZE: # 2mm floor
            step_vec = normalize(step_vec) * MIN_STEP_SIZE

        env._set_action(zeros(4))           # reset mocap to current body
        env.data.mocap_pos[0] += step_vec   # apply delta
        env.data.mocap_quat[0] = VERTICAL_QUAT  # enforce downward orientation
        env._mujoco_step(None)

    return StageResult(success=False, crash_reason="TIMEOUT", ...)
```

**Key constants** (all in `controller.py`):

| Constant | Value | Meaning |
|----------|-------|---------|
| `TRANSIT_TOLERANCE_M` | 4 mm | Target-reached threshold for horizontal moves |
| `VERTICAL_TOLERANCE_M` | 4 mm | Target-reached threshold for vertical moves |
| `MIN_STEP_SIZE_M` | 2 mm | Floor to prevent final-approach creep |
| `MAX_STEP_SIZE_M` | 24 mm | Cap to keep per-step physics stable |
| `TRANSIT_MAX_STEPS` | 300 | ~3.6m at 12mm/step — sufficient for any board diagonal |
| `VERTICAL_MAX_STEPS` | 200 | Sufficient for 50mm (SAFE_Z → HOVER_Z) |
| `FLOOR_LIMIT` | 0.400 m | Abort if grip Z drops below table surface |

---

## Stage 1: Transit (`run_transit`)

**Purpose:** Move the arm horizontally from its current position to the target XY at safe altitude (SAFE_Z = 0.530 m).

**Target:** `[target_xy[0], target_xy[1], SAFE_Z]`

**Abort conditions:**
- `FLOOR_HIT`: grip Z drops below 0.400 m
- `CUBE_DROPPED` (if `grasp_mode=True`): cube XY or Z deviates beyond hold limits

**Steps observed (e2→e4 debug log):** 8–10 steps, error ≤ 2 mm.

---

## Stage 2: Descend (`run_descend`)

**Purpose:** Move the arm vertically from SAFE_Z down to HOVER_Z (0.460 m) over the target XY.

**Target:** `[target_xy[0], target_xy[1], HOVER_Z]`

**Abort conditions:**
- `TUBE_BREACH`: XY drift from tube centre exceeds `drift_limit` (default 10 mm). This is the "tube constraint" — the arm must stay directly above the target square to avoid collisions during descent.
- `TABLE_HIT`: grip Z drops below TABLE_SURFACE_Z (0.400 m)

**Steps observed:** 4–8 steps, error ≤ 4 mm.

---

## Stage 3: Ascend (`run_ascend`)

**Purpose:** Move the arm vertically from HOVER_Z back up to SAFE_Z.

**Target:** `[target_xy[0], target_xy[1], SAFE_Z]`

**Abort conditions:**
- `TUBE_BREACH`: same tube constraint as descend
- `CUBE_DROPPED` (if `grasp_mode=True`): cube not following grip during lift

**Steps observed:** 5–6 steps, error ≤ 3 mm.

---

## Stage 4: Grasp Pipeline (`execute_grasp` in `task.py`)

The grasp pipeline is a multi-phase sequence that runs entirely within `ChessTaskEnv.execute_grasp()`. It runs after descend has placed the arm at HOVER_Z over the source piece.

### Phase 0 — Halt & Settle
- Zeroes `qvel`, `qacc`, `ctrl` to eliminate residual RL momentum.
- Precondition checks: arm speed ≤ 5 mm/s, grip Z within ±25 mm of HOVER_Z, fingers open.

### Phase 1+2 — Align & Verticalize
- Reads cube XY from `get_cube_position()`.
- Checks for dangerous cube rotation (yaw > 25°, which would make the 30mm cube too wide for the 38mm gripper opening).
- Calls `_move_mocap_to(align_target, VERTICAL_QUAT)` to translate the grip directly above the cube and enforce vertical quaternion.
- Max 100 alignment steps, tolerance 1 mm.

### Phase 3 — Plunge (HOVER_Z → GRASP_Z)
- Moves the grip downward in `GRASP_PLUNGE_STEP_M` (6 mm) increments to GRASP_Z (0.430 m).
- Tracks live cube XY so the arm follows any slight cube drift.
- Aborts if final Z deviates >8 mm from GRASP_Z.

### Phase 4 — Finger Close (Linear Ramp)
- Switches to `grasp_mode=True` (actuator-driven, contact physics enabled).
- Ramps `finger_target_joint` linearly from FINGER_OPEN_JOINT (0.0181 m) to GRASP_RAMP_END (0.010 m) over GRASP_CLOSE_STEPS (24 steps).
- **Empty-grasp detection:** starting at step 65% of the ramp, checks `l_finger < EMPTY_GRASP_THRESHOLD (0.011 m)`. If fingers fully closed with nothing to grip, aborts with `FINGER_CLOSED_EMPTY`.
- With a piece: fingers stall at approximately 0.0141 m due to contact resistance — above the empty threshold.

### Phase 5 — Hold & Verify
- Holds the finger ramp target for GRASP_HOLD_STEPS (2 steps) to let contact impulses settle.
- Verifies:
  - XY error between cube centre and grip ≤ `GRASP_VERIFY_XY_THRESHOLD` (15 mm)
  - Z error ≤ `GRASP_VERIFY_Z_THRESHOLD` (20 mm)
  - Finger position above `EMPTY_GRASP_THRESHOLD` (piece is held)

### Phase 6 — Retract (GRASP_Z → HOVER_Z)
- Lifts the grip back to HOVER_Z in GRASP_RETRACT_STEP_M (6 mm) increments.
- Monitors `CUBE_HELD_Z_LIMIT` (20 mm): if the cube Z deviates more than this from the expected "hanging below grip" position, aborts with `CUBE_DROPPED_DURING_RETRACT`.

**Returns:** dict with `success`, `reason`, `total_steps_used`, `final_finger_pos`, `post_grasp_cube_pos`.

---

## Stage 5: Place Pipeline (`execute_place` in `task.py`)

Runs after a second descend has placed the arm at HOVER_Z over the destination square.

### Phase 0 — Halt & Settle
- Same as grasp Phase 0.

### Phase 1+2 — Align Over Destination
- Calls `_move_mocap_to([dst_xy, HOVER_Z], VERTICAL_QUAT)` to centre the grip over the destination square.

### Phase 3 — Plunge (HOVER_Z → GRASP_Z)
- Same incremental descent as grasp plunge, now over the destination.
- Verifies plunge reached GRASP_Z within 8 mm.

### Phase 4 — Release (Linear Ramp Open)
- Ramps fingers from GRASP_RAMP_END (0.010) back to FINGER_OPEN_JOINT (0.0181) over RELEASE_RAMP_STEPS (4 steps).
- Holds fully open for RELEASE_SETTLE_STEPS (2 steps).
- Sets `grasp_mode = False`.

### Phase 5 — Verify Placement
- Reads cube position.
- Checks XY error from `dst_xy` ≤ 20 mm, Z error from expected table resting height ≤ 10 mm.

### Phase 6 — Retract (GRASP_Z → HOVER_Z)
- Same incremental ascent as grasp retract.

**Returns:** dict with `success`, `reason`, `total_steps_used`, `final_cube_pos`, `final_xy_error_mm`.

---

## Full Move Sequence (`run_full_move`)

```
run_transit(src_xy)
    ↓
run_descend(src_xy)
    ↓
execute_grasp()          ← runs phases 0–6, sets grasp_mode=True
    ↓
run_ascend(src_xy)       ← lifts piece to SAFE_Z
    ↓
run_transit(dst_xy)      ← carries piece horizontally to destination
    ↓
run_descend(dst_xy)
    ↓
execute_place(dst_xy)    ← runs phases 0–6, sets grasp_mode=False
    ↓
run_ascend(dst_xy)       ← arm returns to SAFE_Z
```

Each stage produces a `StageResult`. The overall `SequenceResult` is successful only if all stages succeed. The first failed stage's reason is recorded in `failed_at`.

---

## Post-Move Snap

After a successful `run_full_move`, `MovementExecutor.move_piece_xy` checks the piece's final XY and Z:
- XY error must be ≤ 20 mm, Z error ≤ 10 mm from expected resting height.
- If within tolerance, the piece is **teleported** to the exact square centre via `PieceTeleporter.teleport_piece_to_xyz`. This corrects any residual placement drift accumulated during the physical move.

---

## Return-to-Home

After every executed plan (human or computer move), `PhysicalPlanExecutor.return_to_home()` is called. It first executes `run_transit` to the home position (`env.yaml:home_position_xy = [0.88, 0.2641]` — the board centre). After that succeeds, `ChessTaskEnv.reset_arm_to_home_posture()` restores the exact reset-time torso/arm/wrist/finger joint posture and mocap pose. This reset is blocked while a piece is held, so it cannot move the fingers during carry. It prevents redundant roll/wrist joints from accumulating a different "twisted" configuration even when the gripper XYZ has returned home.

---

## Vertical Orientation Enforcement

Throughout all stages, the arm's wrist is locked to `VERTICAL_QUAT = [0.7071068, 0, 0.7071068, 0]` — a 90° rotation about the Y axis that points the gripper straight down. This is enforced after every `mocap_pos` update:

```python
env.data.mocap_quat[0][:] = env.VERTICAL_QUAT
```

The value is loaded from `configs/physics.yaml:vertical_quat` and normalised at load time.
