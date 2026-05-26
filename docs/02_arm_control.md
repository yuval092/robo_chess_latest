# Arm Control

## Overview

The arm is driven by one of two interchangeable controllers that both expose the same interface:

| Controller | Class | Description |
|---|---|---|
| Scripted | `ScriptedController` | Deterministic proportional control; no neural network |
| RL-embedded | `ModelEmbeddedController` | SAC model inference; scripted fallback for unloaded stages |

Both produce `StageResult` and `SequenceResult` objects and call the same underlying MuJoCo environment primitives. The caller (`MovementExecutor`) is entirely unaware of which controller is active.

---

## Shared Data Structures

### `StageResult`

```python
@dataclass
class StageResult:
    success:      bool
    steps:        int
    crash_reason: str | None  # e.g., "TUBE_BREACH (drift=12.4mm)"
    final_pos:    np.ndarray  # gripper XYZ at end of stage
    error_mm:     float       # distance from target in mm
```

### `SequenceResult`

```python
@dataclass
class SequenceResult:
    success:       bool
    stage_results: list[tuple[str, StageResult]]  # [("transit", sr), ...]
    failed_at:     str | None   # stage name where failure occurred
    grasp_quality: dict | None  # set after grasp
```

---

## `ScriptedController` (`src/chess_env/controller.py`)

### Purpose

Drives the arm through all movement stages using a pure proportional-control feedback loop. No learning, no stochasticity, and no uncertainty — if the physics are stable, the scripted controller always reaches the target.

### Configuration (from `env.yaml`)

| Parameter | Value | Description |
|---|---|---|
| `transit_tolerance_m` | 0.004 m | Target-reached threshold for horizontal stages |
| `vertical_tolerance_m` | 0.004 m | Target-reached threshold for vertical stages |
| `step_gain` | 1.0 | Full proportional gain (error = step size before clamping) |
| `min_step_size_m` | 0.002 m | Minimum step size; prevents creep near target |
| `max_step_size_m` | 0.024 m | Maximum step size; keeps per-step physics stable |
| `transit_max_steps` | 300 | Step budget for horizontal transit |
| `vertical_max_steps` | 200 | Step budget for vertical descend/ascend |
| `floor_limit` | 0.400 m | Abort if grip Z drops below this |
| `drift_limit` | configurable | Constructor parameter (production: `drift_limit_end = 0.008 m`) |

### `_run_movement_loop()` — Core Primitive

Every movement stage is implemented with this single loop:

```
for step in range(max_steps):
    grip_pos = get_site_xpos("robot0:grip")
    error = target_pos - grip_pos
    dist = norm(error)

    if dist < tolerance:
        return StageResult(success=True, steps=step, ...)

    if abort_fn is not None:
        abort, reason = abort_fn(grip_pos)
        if abort:
            return StageResult(success=False, crash_reason=reason, ...)

    # Proportional step with clamping
    step_vec = STEP_GAIN * error
    norm_sv = norm(step_vec)
    if norm_sv > MAX_STEP_SIZE_M:
        step_vec = step_vec / norm_sv * MAX_STEP_SIZE_M
    elif norm_sv < MIN_STEP_SIZE_M:
        step_vec = step_vec / norm_sv * MIN_STEP_SIZE_M

    env._set_action(np.zeros(4))        # reset mocap to current body
    env.data.mocap_pos[0][:3] += step_vec
    env.data.mocap_quat[0][:] = env.VERTICAL_QUAT
    env._mujoco_step(None)
    [render if requested]

return StageResult(success=False, crash_reason="TIMEOUT", ...)
```

**Why `_set_action(zeros)` before each update?**
`_set_action` calls `mocap_set_action` which resets the mocap body to the current physics body position before applying the action delta. Without this reset, the delta accumulates from the wrong base.

---

## Stage 1: Transit (`run_transit`)

**Purpose:** Move the arm horizontally at safe altitude from its current position to `[target_xy, SAFE_Z]`.

**Target position:** `[target_xy[0], target_xy[1], SAFE_Z (0.530 m)]`

**Tolerance:** `transit_tolerance_m = 4 mm`

**Max steps:** 300 (~3.6 m range at 12 mm/step — more than the full board diagonal of ~0.9 m)

**Abort conditions:**

| Code | Condition |
|---|---|
| `FLOOR_HIT` | `grip_z < FLOOR_LIMIT (0.400 m)` |
| `CUBE_DROPPED_XY` | `||piece_xy - grip_xy|| > 30 mm` (only when `grasp_mode=True`) |
| `CUBE_DROPPED_Z` | `|piece_z - (grip_z - 0.015)| > 20 mm` (only when `grasp_mode=True`) |

Transit is used both empty (before pickup) and loaded (carrying a piece to the destination square). When `grasp_mode=True`, the piece-drop checks are active.

---

## Stage 2: Descend (`run_descend`)

**Purpose:** Lower the arm vertically from SAFE_Z to HOVER_Z (70 mm descent) while staying directly above the target square.

**Target position:** `[target_xy[0], target_xy[1], HOVER_Z (0.460 m)]`

**Tolerance:** `vertical_tolerance_m = 4 mm`

**Max steps:** 200

**Abort conditions:**

| Code | Condition |
|---|---|
| `TUBE_BREACH` | `||grip_xy - tube_center_xy|| > drift_limit` |
| `TABLE_HIT` | `grip_z < TABLE_SURFACE_Z (0.400 m)` |

The "tube constraint" defines a vertical cylinder around the target square. The arm must stay within this cylinder during descent. This prevents the arm from clipping adjacent pieces and models the physical requirement that a robot arm descend vertically onto a chess piece.

**Finger state:** Open (0.0181 m) — fingers must be open to receive the piece.

---

## Stage 3: Ascend (`run_ascend`)

**Purpose:** Raise the arm vertically from HOVER_Z back up to SAFE_Z.

**Target position:** `[target_xy[0], target_xy[1], SAFE_Z (0.530 m)]`

**Tolerance/max steps:** Same as descend.

**Abort conditions:**

| Code | Condition |
|---|---|
| `TUBE_BREACH` | Same tube constraint as descend |
| `CUBE_DROPPED_XY/Z` | Only when `grasp_mode=True` |

**Finger state:** Closed (0.000 m) when empty; actuator-driven when `grasp_mode=True`.

---

## Stage 4: Grasp Pipeline (`execute_grasp` in `GraspPlaceMixin`)

The grasp pipeline runs entirely within the MuJoCo environment after descend has positioned the arm at HOVER_Z over the source piece. It is always scripted, regardless of which controller is active.

### Phase 0 — Halt and Settle

```
data.qvel[:] = 0
data.qacc[:] = 0
data.ctrl[:] = 0
_set_gripper_state()       ← restore finger ctrl after ctrl[:]=0
mj_forward()

# Precondition checks:
if ||grip_vel|| > 5 mm/s:         → PRECONDITION_SPEED
if |grip_z - HOVER_Z| > 25 mm:   → PRECONDITION_Z
if l_finger < FINGER_OPEN_JOINT - 3mm: → PRECONDITION_FINGERS_NOT_OPEN
```

Zeroing the full state (including `ctrl`) eliminates residual momentum from the RL model's final step.

### Phase 1+2 — Rotation Check and Alignment

1. Read piece position and quaternion from the free joint.
2. Compute effective yaw: `yaw_modulo = yaw % (π/2)`, then `effective_yaw = min(yaw_modulo, π/2 - yaw_modulo)`. This correctly identifies how far the piece is from a "flat" orientation regardless of absolute angle.
3. If `effective_yaw > 25°` → abort `CUBE_ROTATED`. A rotated 30mm cube has effective width 42.4mm, exceeding the maximum gripper opening of ~38mm.
4. Call `_move_mocap_to([piece_xy, HOVER_Z], VERTICAL_QUAT, max_steps=150, tol=1mm)` to align the gripper directly above the piece and enforce vertical orientation.

### Phase 3 — Plunge (HOVER_Z → GRASP_Z)

```
commanded_z = grip_z
for each step:
    if grip_z <= GRASP_Z + 1mm: break
    commanded_z = max(GRASP_Z, commanded_z - GRASP_PLUNGE_STEP_M)  # 6mm steps
    _step_locked_grip([piece_xy, commanded_z])
```

The arm follows the live piece XY at every step, compensating for any contact-induced lateral motion.

Abort if final position deviates more than 8 mm from `GRASP_Z = 0.430 m`.

### Phase 4 — Finger Close (Linear Ramp)

```
grasp_mode = True            ← enable contact physics for fingers
ramp_start = FINGER_OPEN_JOINT   (0.0181 m)
ramp_end   = GRASP_RAMP_END      (0.010 m)
ramp_delta = (ramp_start - ramp_end) / GRASP_CLOSE_STEPS  (24 steps)

for step in range(GRASP_CLOSE_STEPS):
    finger_target_joint = max(ramp_end, ramp_start - ramp_delta * step)
    _step_locked_grip([live_piece_xy, GRASP_Z])

    # Empty-grasp early abort (after 65% of ramp):
    if step >= 0.65 * GRASP_CLOSE_STEPS:
        if l_finger < EMPTY_GRASP_THRESHOLD (0.011 m):
            return FINGER_CLOSED_EMPTY
```

**Physics of grasp:** With a piece between the fingers, contact forces resist closure; fingers stall at ~0.0141 m. Without a piece, fingers reach `GRASP_RAMP_END (0.010 m)` — below the empty-grasp threshold (0.011 m) → abort.

The linear ramp prevents large contact impulses that would launch the piece.

### Phase 5 — Hold and Verify

```
hold GRASP_HOLD_STEPS (2 steps) tracking live piece XY

xy_error = ||piece_xy - grip_xy||  > GRASP_VERIFY_XY_THRESHOLD (15 mm) → VERIFY_XY_FAILED
z_error  = |piece_z - grip_z|      > GRASP_VERIFY_Z_THRESHOLD  (20 mm) → VERIFY_Z_FAILED
l_finger < EMPTY_GRASP_THRESHOLD                                        → VERIFY_FINGERS_CLOSED_EMPTY
```

### Phase 6 — Retract (GRASP_Z → HOVER_Z)

```
commanded_z = GRASP_Z
for each step:
    if grip_z >= HOVER_Z - 1mm: break
    commanded_z = min(HOVER_Z, commanded_z + GRASP_RETRACT_STEP_M)  # 6mm steps
    live_xy = get_cube_position()[:2]
    _step_locked_grip([live_xy, commanded_z])

    # Verify still held:
    if |piece_z - (grip_z - 0.015)| > CUBE_HELD_Z_LIMIT:
        return CUBE_DROPPED_DURING_RETRACT
```

---

## Stage 5: Place Pipeline (`execute_place`)

Runs after a second descend has placed the arm at HOVER_Z over the destination square, with `grasp_mode=True`.

### Phase 0 — Halt and Settle

Same as grasp Phase 0. Zeroes full sim state.

### Phase 1+2 — Align Over Destination

```
_move_mocap_to([dst_xy, HOVER_Z], VERTICAL_QUAT, max_steps=150, tol=1mm)
```

### Phase 3 — Plunge (HOVER_Z → GRASP_Z)

Same as grasp plunge, now over the destination square. Verifies plunge reached `GRASP_Z` within 8 mm.

### Phase 4 — Release (Linear Ramp Open)

```
ramp_start = GRASP_RAMP_END       (0.010 m)  ← actual finger position during grip
ramp_end   = FINGER_OPEN_JOINT    (0.0181 m)
ramp_delta = (ramp_end - ramp_start) / RELEASE_RAMP_STEPS  (4 steps)

for step in range(RELEASE_RAMP_STEPS):
    finger_target_joint = min(ramp_end, ramp_start + ramp_delta * step)
    _step_locked_grip(release_target)

# Hold fully open RELEASE_SETTLE_STEPS (2) more steps
finger_target_joint = FINGER_OPEN_JOINT
grasp_mode = False
```

Starting ramp from `GRASP_RAMP_END` (not 0.000) prevents squeezing the piece for the first 29 steps before opening.

### Phase 5 — Verify Placement

```
piece_pos = get_cube_position()
xy_error = ||piece_xy - dst_xy||  > 20 mm → PLACE_XY_FAILED
z_error  = |piece_z - (TABLE_Z + CUBE_HEIGHT/2)| > 10 mm → PLACE_Z_FAILED
```

### Phase 6 — Retract (GRASP_Z → HOVER_Z)

Same as grasp retract, but without cube-held monitoring (gripper is now open).

---

## Full Pick-and-Place Sequence

```
run_pick_sequence(src_xy):
  1. run_transit(src_xy)
  2. soft_reset / transition → "descend"
  3. run_descend(src_xy)
  4. run_grasp()           ← grasp_mode=True after this
  5. soft_reset / transition → "ascend"
  6. run_ascend(src_xy)

run_place_sequence(dst_xy):
  7. run_transit(dst_xy)   ← grasp_mode=True throughout
  8. soft_reset / transition → "descend"
  9. run_descend(dst_xy)
 10. run_place(dst_xy)     ← grasp_mode=False after this
 11. soft_reset / transition → "ascend"
 12. run_ascend(dst_xy)
```

Between each stage pair, `transition()` (scripted) or `soft_reset()` (RL) is called to:
1. Halt the arm
2. Align to the nominal exit position
3. Update `current_scenario`, `goal_pos`, `tube_center_xy`
4. Perform scripted finger open/close

---

## Post-Move Snap

After `run_full_move` succeeds, `MovementExecutor` performs a post-placement reconciliation:

```python
piece_pos = env.get_active_piece_position()
expected_z = TABLE_Z + CUBE_HEIGHT / 2
xy_error = ||piece_xy - dst_xy||
z_error  = |piece_z - expected_z|

if xy_error > reconcile_xy_tolerance_m (20 mm): → XY_RECONCILE_FAILED
if z_error  > reconcile_z_tolerance_m  (10 mm): → Z_RECONCILE_FAILED

# Snap piece to exact square centre:
placed_xyz = board_mapper.square_to_piece_xyz(dst_square)
teleporter.teleport_piece_to_xyz(piece_id, placed_xyz, quat=IDENTITY_QUAT)
occupancy.set_piece_square(piece_id, dst_square)
```

The snap corrects any residual placement drift accumulated during the physical move.

---

## Return to Home

After every committed move (human or computer), `PhysicalPlanExecutor.return_to_home()` is called:

```python
home_xy = load_config("env")["home_position_xy"]  # [0.88, 0.2641] = board centre
result = controller.run_transit(home_xy)

if success:
    env.reset_arm_to_home_posture()  # normalise redundant joints
```

`reset_arm_to_home_posture()` interpolates all arm joints, mocap position, and mocap quaternion back to the captured reset-time posture over 10 steps, then snaps them exactly. This prevents redundant wrist/roll joints from accumulating different configurations across many moves. It is blocked if `grasp_mode=True` (piece still held).

---

## `ModelEmbeddedController` (`src/chess_env/model_controller.py`)

### Architecture

`ModelEmbeddedController` wraps `ScriptedController` and `ModelRegistry`. For each of the three movement stages, it either runs the specialist SAC model (if loaded) or falls back to the scripted controller.

```python
def _run_stage(self, stage: str, target_pos: np.ndarray) -> StageResult:
    model = self._registry.get(stage)
    if model is None:
        return self.scripted_controller.run_transit/descend/ascend(...)

    # Model inference loop
    env._use_transfer_obs = True
    for step in range(max_steps):
        grip_pos = get_site_xpos("robot0:grip")
        grip_vel = get_site_xvelp("robot0:grip")

        if is_near(grip_pos, target_pos) and ||grip_vel|| < stability_threshold:
            success = True; break

        obs = env._get_obs()   # 25-D transfer observation
        action, _ = model.predict(obs, deterministic=True)
        action[3] = -1.0 if stage in {"transit","ascend"} else 1.0

        env._set_action(action)
        if stage == "transit":
            env.data.mocap_quat[0][:] = env.VERTICAL_QUAT  # enforce only for transit
        env._mujoco_step(action)

        crash = _check_crash(env, stage, grip_pos)
        if crash: break
    env._use_transfer_obs = False
```

### Gripper Action Override

The SAC model's gripper dimension (action[3]) is overridden deterministically:
- `transit` and `ascend`: `-1.0` → closed (fingers held; piece is not dropped)
- `descend`: `+1.0` → open (fingers must be open to receive the piece)

This prevents the model from accidentally toggling the gripper mid-stage.

### VERTICAL_QUAT Enforcement

Only applied for `transit` because the transit model was trained with `VERTICAL_QUAT` enforced in every step. The descend and ascend models were trained without it (the `step()` method enforces it there, but the model sees free wrist actions). Applying it to all stages would break the descend/ascend policies.

### Crash Checks

| Stage | Crash |
|---|---|
| transit | `FLOOR_HIT`, `CUBE_DROPPED_*` (if `grasp_mode=True`) |
| descend/ascend | `TUBE_BREACH (eval_drift_limit=10mm)`, `TABLE_HIT`, `CUBE_DROPPED_*` (ascend only, `grasp_mode`) |

### Per-Stage Finger Precondition Check

Before running a model stage, the finger joint is verified:
- Descend: must be open (0.0181 m ± 3 mm)
- Transit/Ascend: must be closed (0.000 m ± 3 mm)
- Exception: `grasp_mode=True` skips this check (fingers are actuator-driven and physically blocked by the piece)

### `load_all()` / `load_available()`

Models are loaded through `ModelRegistry` which uses the `transfer_obs_enabled` context manager to temporarily switch the observation space before calling `SAC.load(path, env=...)`. This ensures stable-baselines3 validates the loaded model against the correct observation space.

---

## Waypoints and Transitions (`src/chess_env/waypoints.py`)

### Z-Level Map

```python
SCENARIO_EXIT_Z  = {"transit": SAFE_Z, "descend": HOVER_Z, "ascend": SAFE_Z}
SCENARIO_ENTRY_Z = {"transit": SAFE_Z, "descend": SAFE_Z,  "ascend": HOVER_Z}
```

### Valid Transitions

```python
VALID_TRANSITIONS = {
    ("transit", "transit"): True,   # same-height transit
    ("transit", "descend"): True,   # horizontal then down
    ("descend", "ascend"):  True,   # down then up (grasp/place between)
    ("ascend",  "transit"): True,   # up then horizontal
    ("ascend",  "descend"): True,   # up then down (second board square)
}
```

Any other transition (e.g., descend→transit, descend→descend) raises `ValueError`.

### Named Chains

```python
CHAIN_SHORTCUTS = {
    "full_move": ["transit","descend","ascend","transit","descend","ascend"],
    "pick":      ["transit","descend","ascend"],
    "place":     ["transit","descend","ascend"],
    "vertical":  ["descend","ascend"],
}
```

These shortcuts are used by `eval_sequence.py` to specify evaluation scenarios.
