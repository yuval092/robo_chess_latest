# Arm Movement Improvement Plan

**Generated from:** `logs/debug_move_20260522_143153.jsonl`  
**Move analyzed:** `white_pawn_e`: e2 → e3 (simple pawn push, no captures)  
**Total physics steps:** 378 (0.756s simulated, 11s wall-clock headless)

---

## Summary of Findings

The full pick-and-place move completes successfully with all stages passing. The
problems are not correctness bugs but efficiency and timing issues. The five issues
below, sorted by impact, account for the bulk of the wasted steps.

---

## Issue 1 — `soft_reset` halt loop is the biggest single bottleneck

**Data:** `softreset_p1_halt` = 120 steps total (4 calls × 30 steps each)  
**Sim time wasted:** 120 × 0.002s = **0.240s** (31% of total move time)

### Root cause

`soft_reset` Phase 1 runs a 30-step halt loop before every stage transition:

```
transit → soft_reset → descend
grasp   → soft_reset → ascend
transit → soft_reset → descend
place   → soft_reset → ascend
```

Every single call exhausts all 30 steps without breaking early. The arm exits
`run_transit` / `execute_grasp` / `execute_place` with residual velocity that takes
the full 30 steps to damp out, even though the arm is already near-stationary.

### Evidence

- Grip Z drops only 2.4mm during each halt → arm is barely moving  
- All four calls hit the max 30-step limit → the velocity threshold is never met before the loop ends  
- `softreset_p2_align` has 0 logged steps → arm is already within 4mm tolerance after halt, so alignment converges in 0 steps. The only wasted time is in Phase 1.

### Fix

**Option A (recommended):** Zero velocity actively at the very start of soft_reset,
then check: if speed is already below threshold, skip the loop entirely.

```python
self.data.qvel[:ROBOT_DOF] = 0.0
self.data.qacc[:ROBOT_DOF] = 0.0
mujoco.mj_forward(self.model, self.data)
grip_vel = self._utils.get_site_xvelp(...)
if float(np.linalg.norm(grip_vel)) >= self.HALT_VEL_THRESHOLD:
    # Only run the loop if still moving after active zeroing
    for _ in range(HALT_HOLD_MAX_STEPS):
        ...
```

This eliminates ~30 steps per call when the arm has already converged, saving
~90 steps (0.180s sim) per full move.

**Option B:** Reduce `HALT_HOLD_MAX_STEPS` from 30 → 5. The active qvel zeroing
that follows the loop already stops the arm; the loop is defensive padding.

---

## Issue 2 — Proportional controller "creep" at end of every movement stage

**Data:** 8 stalls detected, most in the final approach of each stage  

| Stall | Phase | Steps | Sim time |
|-------|-------|-------|---------|
| 1 | descend (arrival) | 29 | 0.058s |
| 2 | grasp_p3_plunge (arrival) | 13 | 0.026s |
| 3 | grasp_p5_hold (intentional) | 51 | 0.102s |
| 4 | ascend (arrival) | 30 | 0.060s |
| 5 | descend (arrival) | 30 | 0.060s |
| 6 | place_p3_plunge (arrival) | 13 | 0.026s |
| 7 | place_p6_retract (arrival) | 35 | 0.070s |
| 8 | ascend (arrival) | 30 | 0.060s |

Items 3 is intentional (hold phase). Items 1, 4, 5, 8 are the proportional
controller creeping in the last millimetres of each stage.

### Root cause

The proportional controller in `_run_movement_loop` applies:

```python
step_vec = STEP_GAIN * error          # proportional
if norm(step_vec) > MAX_STEP_SIZE_M:  # cap at 12mm
    step_vec = step_vec / norm * MAX_STEP_SIZE_M
```

As the arm nears the target the step vector shrinks to sub-millimetre values. With
`MAX_STEP_SIZE_M = 0.012` and `TRANSIT_TOLERANCE_M = 0.004`, the arm spends many
steps inching forward at <0.5mm/step before the tolerance check fires.

The same pattern appears in `execute_grasp` retract and `execute_place` retract:
these use fixed `0.001m/step` increments and always overshoot the loop budget by
~15 padding steps even after reaching HOVER_Z.

### Fix

**For `_run_movement_loop`:** Add a minimum step floor to prevent near-zero steps:

```python
step_norm = np.linalg.norm(step_vec)
if step_norm < MIN_STEP_SIZE_M:       # e.g. MIN_STEP_SIZE_M = 0.002
    step_vec = step_vec / step_norm * MIN_STEP_SIZE_M
```

This keeps the arm moving at ≥ 2mm/step all the way to the tolerance sphere,
eliminating the 29–35 step creep at stage ends.

**For retract loops:** Replace the padding count `+ 15` with an early-exit check
(already present but triggered only at loop start). The retract loop overshoots
because it uses `grip_pos[2] >= HOVER_Z - 0.001` as the break condition and the
padding keeps running after this fires. Reduce `+ 15` → `+ 3`.

---

## Issue 3 — Arm Z height at grasp plunge is 17mm below HOVER_Z

**Data:**  
- `descend` exits at **461mm**  
- `grasp_p0_halt` enters at **443mm** (−18mm)  
- `grasp_p3_plunge` starts at **439mm** (−21mm below HOVER_Z=460mm)

### Root cause

Between `run_descend` ending at 461mm and `execute_grasp Phase 3` starting at
439mm, two unlogged phases move the arm downward:

1. **Phase 0 halt settle loop**: captures `settle_pos` at the current grip position
   (~461mm) but `_set_action(zero_action)` may apply joint-space ctrl signals that
   the weld constraint cannot fully counteract, allowing slight downward drift under
   gravity. Over 15 settle steps this accumulates to ~18mm below the captured position.

2. **Phase 1+2 align** (`_move_mocap_to`): targets `min(grip_pos[2], HOVER_Z)`.
   Since `grip_pos[2]` is already below HOVER_Z at this point, it targets the current
   low position rather than correcting back up to HOVER_Z.

### Impact

The arm descends an extra 21mm unnecessarily before plunging the final 9mm to
GRASP_Z. Grasping still succeeds because GRASP_Z is the same regardless, but:
- The approach is 21mm lower than intended — if a piece is displaced or tall, this
  risks collision.
- Phase 3 plunge only needs to travel 9mm instead of the designed 30mm (HOVER_Z −
  GRASP_Z = 460 − 430 = 30mm), meaning the arm sometimes arrives at GRASP_Z without
  full plunge engagement.

### Fix

In `execute_grasp` Phase 0, capture `settle_pos` BEFORE calling `_set_action` and
ensure the settle loop holds mocap at HOVER_Z rather than at the current drifted
position:

```python
# Phase 0: Halt at HOVER_Z
self._debug_current_phase = "grasp_p0_halt"
self.data.qvel[:] = 0.0
self.data.qacc[:] = 0.0
self.data.ctrl[:] = 0.0
self._set_gripper_state()
# Anchor at HOVER_Z directly — do not drift with the arm's settled position
settle_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
settle_pos[2] = self.HOVER_Z   # force Z to nominal
mujoco.mj_forward(self.model, self.data)
```

Apply the same fix to `execute_place` Phase 0.

---

## Issue 4 — Finger actuator lag throughout close and open ramps

**Data:** 216 steps where `|l_finger_actual − target| > 0.5mm`, all during
`grasp_p4_close` and `place_p4_release`.

Peak lag: **0.91mm** at the steepest part of the close ramp.

### Root cause

The linear ramp steps `finger_target_joint` by `(ramp_end − ramp_start) / 20` per
step. The finger joint actuator is a position servo with damping and inertia. When
the ramp moves the target by `~0.43mm/step`, the actuator cannot fully close the
gap within 1 step, so lag accumulates mid-ramp and recovers at the plateau.

### Impact

- During grasp close: fingers lag behind target → cube may be contacted slightly
  later (potentially ~5 steps) than the ramp implies. Not a grasping failure risk
  at current settings.
- During place release: lag means fingers open slightly *later* than target — this
  is actually **beneficial** (cube is held slightly longer as it touches the table).

### Fix (minor)

The current ramp over 20 steps works. If future changes increase piece mass or
grasp force, the lag could be reduced by:
1. Increasing `GRASP_CLOSE_STEPS` from 50 → 60 (more time for full engagement)
2. Or applying `_set_gripper_state()` twice per step during the ramp

This is low priority — the lag does not cause failures.

---

## Issue 5 — Vibration metric fires on normal piece transport

**Data:** 171 "vibration" events, all from step 207 onward (during transit with
piece held), showing 10–80mm drift from the e2 start position.

### Root cause

This is not vibration — it is normal piece carriage. Once grasped, the piece moves
with the arm from e2 to e3. The baseline was set at e2, so any position during
transit triggers the metric.

The actual physics vibration (piece jitter while on the table) cannot be measured
this way.

### Fix for analysis script

Reset the vibration baseline at the start of each retract phase (after grasping or
after placing), so only unexpected motion during stationary hold phases is flagged:

```python
if rec["phase"] in {"grasp_p6_retract", "place_p0_halt"}:
    baseline = None  # reset — piece should be stable relative to arm
```

Alternatively, measure piece velocity instead of position drift — a vibrating piece
will have non-zero `src_piece_vel` while the arm is stationary.

---

## Implementation Priority

| # | Fix | Sim time saved | Complexity |
|---|-----|---------------|-----------|
| 1 | soft_reset: skip halt loop if already slow | ~0.18s / move | Low |
| 2 | Movement loop: add minimum step floor (2mm) | ~0.12s / move | Low |
| 3 | Grasp/place Phase 0: anchor mocap at HOVER_Z | prevents Z drift | Low |
| 4 | Retract loops: reduce padding from +15 → +3 | ~0.03s / move | Trivial |
| 5 | Finger lag: increase close steps to 60 | grasping robustness | Low |

Combined, fixes 1–4 should reduce total move steps from **378 → ~200** (~47%
reduction) without changing any success criteria.

---

## Appendix: Phase Step Counts (Raw)

| Phase | Steps | Sim time (s) | Notes |
|-------|-------|-------------|-------|
| transit (×2) | 27 | 0.054 | Good — 12mm/step |
| softreset_p1_halt (×4) | 120 | 0.240 | Main bottleneck |
| descend (×2) | 16 | 0.032 | Good |
| grasp_p0_halt | 15 | 0.030 | OK |
| grasp_p3_plunge | 6 | 0.012 | Fast |
| grasp_p4_close | 50 | 0.100 | By design |
| grasp_p5_hold | 10 | 0.020 | By design |
| grasp_p6_retract | 30 | 0.060 | Retract creep |
| ascend (×2) | 18 | 0.036 | Good |
| place_p0_halt | 15 | 0.030 | OK |
| place_p3_plunge | 6 | 0.012 | Fast |
| place_p4_release | 35 | 0.070 | By design |
| place_p6_retract | 30 | 0.060 | Retract creep |
| **Total** | **378** | **0.756** | |
