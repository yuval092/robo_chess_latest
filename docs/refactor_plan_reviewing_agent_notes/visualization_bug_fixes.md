# Visualization Bug Fixes

**Date**: 2026-05-21  
**Scope**: Three visual/behavioral issues observed during `eval_sequence.py --visualize --delay`

---

## Bug 1: "Teleport" Between Descend and Grasp/Place

### What You See

The arm descends smoothly toward HOVER_Z (0.460m), stops slightly below it, then appears to jump back upward a few millimeters, and then descends again into the grasp/place plunge. The same pattern repeats on the place side. It looks like a teleport but is actually rendered motion — it just feels wrong because the arm reverses direction mid-operation.

### Root Cause

**Stage 1 — `run_descend` overshoots slightly:**  
`ScriptedController._run_movement_loop` uses a proportional controller (step = error, capped at 8mm/step) with `VERTICAL_TOLERANCE_M = 0.004` (4mm). Descent is approaching from above. Due to downward inertia and weld-constraint compliance (~1mm lag between `mocap_pos` and actual `grip_pos`), the arm sometimes passes through HOVER_Z by 2–4mm before the tolerance check fires and returns success. The arm exits `run_descend` at, say, Z = 0.456 (4mm below HOVER_Z).

**Stage 2 — `execute_grasp` Phase 2 moves the arm back UP:**  
`execute_grasp` Phase 2 (and equivalently `execute_place` Phase 2) performs a "perfect align" call:

```python
# task.py execute_grasp(), Phase 2 — line ~522
align_target = np.array([cube_pos[0], cube_pos[1], self.HOVER_Z])
if not self._move_mocap_to(align_target, self.VERTICAL_QUAT, max_steps=100, tolerance=0.001):
    ...
```

`align_target` is always at exactly `HOVER_Z`. If the arm is at 0.456 after descent, `_move_mocap_to` moves it UP by ~4mm back to 0.460. This is fully rendered — not a teleport — but the motion reversal (downward descent → small upward correction → downward plunge) is visually jarring.

**Stage 3 — Phase 3 plunges down again:**  
After Phase 2 forces the arm to HOVER_Z, Phase 3 plunges from HOVER_Z to GRASP_Z (0.460 → 0.430, 30mm). So the arm goes: descend → slightly past HOVER_Z → snap back to HOVER_Z → plunge. Two downward motions with an upward reversal in between.

`execute_place` has the exact same structure and the same bug (Phase 2 line ~710, Phase 3 line ~716).

### Fix

In both `execute_grasp` and `execute_place`, Phase 2 should **not move the arm upward** if it already overshot below HOVER_Z. The purpose of Phase 2 is XY alignment (center over the cube/destination), not Z correction. The plunge (Phase 3) should start from wherever the arm actually is, not from a hardcoded HOVER_Z.

**Changes required in `src/chess_env/task.py`:**

#### In `execute_grasp()`, replace Phase 2 + Phase 3 start:

```python
# ── Phase 1+2: Rotation Abort Check + Perfect Align & Verticalize ────────
# ... (rotation check unchanged) ...

# Get actual current grip Z after descent (may be slightly below HOVER_Z due to overshoot)
grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
# Align XY only — never move UP to HOVER_Z; clamp to current Z if already below
align_z = min(grip_pos[2], self.HOVER_Z)
align_target = np.array([cube_pos[0], cube_pos[1], align_z])
if not self._move_mocap_to(align_target, self.VERTICAL_QUAT, max_steps=100, tolerance=0.001):
    result["reason"] = "ROTATION_FAILED (kinematic limit — arm cannot reach vertical at this position)"
    return result

# ── Phase 3: Plunge (current_Z → GRASP_Z) ────────────────────────────────
place_z = self.GRASP_Z
# Start from ACTUAL current grip Z, not hardcoded HOVER_Z
grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
target_z = grip_pos[2]
for _ in range(int(round((target_z - place_z) / 0.001)) + 15):
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
    if grip_pos[2] <= place_z + 0.001:
        break
    target_z = max(place_z, target_z - 0.001)
    plunge_target = np.array([cube_pos[0], cube_pos[1], target_z])
    # ... rest of loop unchanged ...
```

#### Apply the same change in `execute_place()`, Phase 2 + Phase 3:

```python
# Align XY only — clamp to current Z, never move up
grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
align_z = min(grip_pos[2], self.HOVER_Z)
align_target = np.array([dst_xy[0], dst_xy[1], align_z])
if not self._move_mocap_to(align_target, self.VERTICAL_QUAT, max_steps=100, tolerance=0.001):
    result["reason"] = "ALIGN_FAILED"
    return result

# Phase 3: Plunge from actual current Z
grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
target_z = grip_pos[2]
for _ in range(int(round((target_z - place_z) / 0.001)) + 15):
    # ... rest unchanged ...
```

**Result**: The arm descends → stops at or slightly below HOVER_Z → Phase 2 only corrects XY (no upward motion) → Phase 3 plunges directly from wherever it stopped. Smooth single descent into the grasp/place.

---

## Bug 2: Gripper Twitching During Grasp (Not During Place)

### What You See

When the fingers are closing on the cube, the cube (and arm) twitch or oscillate in place. This only happens during `execute_grasp`, not during `execute_place`.

### Root Cause

**Why grasp twitches and place does not:**  
`execute_place` opens the fingers (ramps from FINGER_CLOSED_JOINT = 0.0000 toward FINGER_OPEN_JOINT = 0.0181). As fingers open, they gradually lose contact with the cube and forces release smoothly. No impulse.

`execute_grasp` closes the fingers (ramps from 0.0181 → 0.0000). When the fingers first make contact with the cube (at approximately joint position 0.014 = 14mm gap), the cube starts being squeezed. The ramp continues all the way to the target `FINGER_CLOSED_JOINT = 0.0000`, but the cube physically prevents closure past ~14mm. The actuator with Kp=20000 exerts a residual squeezing force of:

```
F = Kp × (target - actual) = 20000 × (0.000 - 0.014) = 280N per finger
```

This 280N squeezing force on a 50g cube generates large contact impulses. Due to minor physics asymmetries (floating-point), these impulses are not perfectly balanced, producing a net lateral force on the cube each step → twitching.

**Why the position correction loop makes it worse:**  
Phase 4 re-asserts the arm position every step via:
```python
error = close_target - self._utils.get_site_xpos(...)
self.data.mocap_pos[0][:3] += error
```
`close_target` is the PRE-CLOSE arm position (a snapshot taken before finger closure). As the cube jiggles under the squeezing force, the arm's position error grows slightly (the arm gets pushed by the cube reaction), and the correction tries to snap the arm back to its pre-jiggle position — which exerts more force on the cube, amplifying the oscillation.

### Fix

Two complementary changes:

**Change 1: Track the live cube position during closure (instead of the pre-close snapshot)**

Replace the fixed `close_target` with a live cube-following target. When the cube jiggles, the arm follows it rather than fighting it:

```python
# execute_grasp(), Phase 4 loop — replace the error computation
for step in range(self.GRASP_CLOSE_STEPS):
    self.finger_target_joint = max(ramp_end, ramp_start - ramp_delta * step)
    self._set_action(zero_action)
    # Track live cube XY so arm follows cube instead of fighting contact forces
    live_cube = self.get_cube_position()
    live_target = np.array([live_cube[0], live_cube[1], self.GRASP_Z])
    error = live_target - self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
    self.data.mocap_pos[0][:3] += error
    self.data.mocap_quat[0][:] = self.VERTICAL_QUAT
    self._mujoco_step(None)
    if should_render:
        self.render()
    # ... early abort check unchanged ...
```

Also update Phase 5 (hold) and Phase 6 (retract) to use live cube position similarly:
- In Phase 5: replace `close_target` with `live_target` computed from current cube pos
- In Phase 6: `target_vec = np.array([self.get_cube_position()[0], self.get_cube_position()[1], target_z])`

**Change 2: Reduce the over-squeezing force by lowering the close ramp target**

Instead of ramping all the way to `FINGER_CLOSED_JOINT = 0.0000`, ramp to a "secure grip" value like `0.010` (10mm). With the cube physically stalling fingers at ~14mm, the residual force becomes:

```
Old: F = 20000 × (0.000 - 0.014) = 280N   ← excessive
New: F = 20000 × (0.010 - 0.014) = 80N    ← secure but gentler
```

**IMPORTANT: adjust the empty-gripper detection threshold accordingly.**  
The early abort check at `step > 30` currently fires when `l_now < 0.003` (fingers fully closed = no cube). If the ramp target changes to `0.010`, an empty gripper will settle at `0.010`, not `0.000`, and the `< 0.003` check will never fire. Update the threshold:

```python
# In execute_grasp(), before Phase 4:
GRASP_RAMP_END = 0.010    # "secure grip" target, gentler than fully-closed
EMPTY_DETECT_THRESHOLD = GRASP_RAMP_END - 0.003  # 0.007: fires if no cube stops fingers

ramp_end   = GRASP_RAMP_END
ramp_delta = (ramp_start - ramp_end) / self.GRASP_CLOSE_STEPS

# In the early abort check inside the loop:
if step > 30:
    l_now = ...
    if l_now < EMPTY_DETECT_THRESHOLD:  # changed from 0.003
        result["reason"] = f"FINGER_CLOSED_EMPTY (j={l_now:.4f} at step {step})"
        return result
```

Optionally, add `grasp_ramp_end: 0.010` to `configs/env.yaml` and read it in `__init__` like other thresholds, so it can be tuned without changing source.

> **Note on Phase 5 hold:** After Phase 4 ramps to 0.010, during Phase 5 (50 hold steps), `finger_target_joint` stays at 0.010. The residual force is 80N instead of 280N, which should stop the twitching entirely during the hold phase.

---

## Bug 3: Red Dot Floating Around the Scene

### What You See

A red sphere appears in various places in the 3D scene — above the table, at different heights — and moves around as the sequence progresses. This is a leftover visual from the original FetchPickAndPlace environment.

### Root Cause

**Two sources in `chess_env/assets/pick_and_place.xml`:**

```xml
<!-- Source 1: target0 — the Fetch env's goal marker -->
<body name="floor0" pos="0.88 0.2641 0">
    <site name="target0" pos="0 0 0.5" size="0.02 0.02 0.02" rgba="1 0 0 1" type="sphere"/>
</body>

<!-- Source 2: object0 — a visual marker overlaid on the cube -->
<body name="object0" pos="0.01 0.01 0.01">
    ...
    <site name="object0" pos="0 0 0" size="0.015 0.015 0.015" rgba="1 0 0 1" type="sphere"/>
</body>
```

**Why it moves:** The parent class `MujocoFetchPickAndPlaceEnv` defines `_render_callback()` which is called every time `render()` is invoked. It repositions `target0` to follow `self.goal`:

```python
# gymnasium_robotics/envs/fetch/fetch_env.py
def _render_callback(self):
    sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
    site_id = self._mujoco.mj_name2id(self.model, ..., "target0")
    self.model.site_pos[site_id] = self.goal - sites_offset[0]
    self._mujoco.mj_forward(self.model, self.data)
```

Since `ChessTaskEnv` sets `self.goal = self.goal_pos` (the current movement target, which changes between SAFE_Z, HOVER_Z, etc.), the red dot teleports to each successive target position during the sequence.

The `object0` site is a fixed red sphere centered on the cube body — it always appears as a red dot on top of the cube.

### Fix

**Two changes required:**

**Change 1: Override `_render_callback` in `ChessSimulationEnv` (or `ChessTaskEnv`) to suppress the target visualization:**

```python
# src/chess_env/simulation.py — add to ChessSimulationEnv class
def _render_callback(self):
    pass  # Suppress Fetch env's target0 marker — not used in chess pipeline
```

This prevents the red dot from moving around. The site still exists in the model but stays at its XML position and is never updated.

**Change 2: Make both red sites invisible in `chess_env/assets/pick_and_place.xml`:**

```xml
<!-- Set alpha=0 to make invisible -->
<site name="target0" pos="0 0 0.5" size="0.02 0.02 0.02" rgba="1 0 0 0" type="sphere"/>
```

```xml
<site name="object0" pos="0 0 0" size="0.015 0.015 0.015" rgba="1 0 0 0" type="sphere"/>
```

Both changes together: the site is invisible AND never repositioned. Either change alone is sufficient to eliminate the visible red dot, but both together is cleaner.

> **Why not remove the sites from the XML entirely?** The parent class `_render_callback` looks up `target0` by name. If the site is missing from the XML and `_render_callback` is NOT overridden, it would raise a MuJoCo error on every render call. If `_render_callback` IS overridden (Change 1), the site can safely be removed. For safety, keep the site but with alpha=0, so the XML remains valid whether or not `_render_callback` is overridden.

---

## Summary of All Changes

| Issue | File(s) | Change |
|-------|---------|--------|
| Arm reverses direction before plunge | `src/chess_env/task.py` | Phase 2 of `execute_grasp` and `execute_place`: use `min(grip_z, HOVER_Z)` for align_z; Phase 3: start `target_z = grip_pos[2]` |
| Cube twitches during grasp closure | `src/chess_env/task.py` | Phase 4: track live cube XY instead of pre-close snapshot; ramp to `0.010` instead of `0.0000`; adjust empty detection threshold to `GRASP_RAMP_END - 0.003` |
| Red dot floating in scene | `chess_env/assets/pick_and_place.xml` + `src/chess_env/simulation.py` | Set `rgba` alpha=0 on `target0` and `object0` sites; add `_render_callback(self): pass` override |
