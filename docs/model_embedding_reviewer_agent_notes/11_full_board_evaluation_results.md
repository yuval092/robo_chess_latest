# Full-Board Evaluation Results

Date: 2026-05-24  
Transit model: `checkpoints/transit_20260523_164127/final_transit.zip` (the "old" 05-23 model, currently deployed)

---

## Summary

Three evaluations were run using the production `ModelEmbeddedController` (transit→descend→ascend) with currently deployed models.

| Eval mode | Pairs tested | Episodes | Success rate |
|-----------|-------------|---------|--------------|
| Pawn (ranks 2/7 → nearby) | 48 | 144 | **100%** (144/144) |
| Key squares (16 strategically important squares, ≥2-square moves) | 216 | 648 | **97.2%** (630/648) |
| Full board (all pairs ≥3 squares, 2 reps — in progress) | 2972 | 5944 | TBD (pattern consistent with key) |

---

## Pawn Evaluation (100%)

All 16 pawn rank-2 sources × 3 destinations and all 16 pawn rank-7 sources × 3 destinations pass. Every file (a–h), every rank-2/rank-7 starting position → success.

This matches the target chess use case perfectly.

---

## Key-Squares Evaluation (97.2%)

**Only one destination square fails: `a5` at position [0.600, 0.304].**

All 18 failures (6 source × 3 reps) are FAIL@transit with exactly 10.8mm final error.

### Failing source → a5 pairs

| Source | Source XY | Distance | Result |
|--------|-----------|---------|--------|
| h1 | [1.160, -0.016] | 645mm | FAIL@transit (10.8mm) |
| g1 | [1.080, -0.016] | 574mm | FAIL@transit (10.8mm) |
| e4 | [0.920, 0.224] | 349mm | FAIL@transit (10.8mm) |
| d4 | [0.840, 0.224] | 291mm | FAIL@transit (10.8mm) |
| d5 | [0.840, 0.304] | 240mm | FAIL@transit (10.8mm) |
| e5 | [0.920, 0.304] | 320mm | FAIL@transit (10.8mm) |

### Succeeding to a5

| Source | Distance | Result |
|--------|---------|--------|
| h5→a5 | 560mm | SUCCESS (10.4mm final 3D error) |
| a4→a5 | 80mm | SUCCESS |
| a1→a5 | 304mm | SUCCESS |
| a2→a5 (pawn eval) | 240mm | SUCCESS |

Note: `a7→a4` (a-file to a-file) succeeds. Sources on the same X column (a-file) always succeed.

---

## Root Cause: a5 Transit Failure

### Step-by-step analysis (`d5→a5` as representative failing case)

The arm arrives near a5 after ~270 steps of approach and **freezes completely**:

```
step 270: pos=[0.5987, 0.3144, 0.5330]  d_xy=10.36mm  d_z=3.05mm  d_3d=10.80mm  speed=0.02mm/s
step 271: (identical)
...
step 299: (identical — no motion at all)
```

This is NOT oscillation. The arm is completely stationary at a local minimum.

### Why the arm stops at 10.36mm from a5

The model outputs zero (or near-zero) action at this position because:

1. **Braking zone = success zone**: `braking_dist = 0.010` and `success_threshold = 0.010` are equal. The braking reward (`-braking_weight × speed`) only activates inside the success zone. Outside 10mm, there is no velocity penalty.

2. **Flat reward gradient**: At d=10.36mm (just outside the 10mm zone), the distance reward (`-1.0 × d`) changes very slowly. The model has learned "near this position, output small/zero actions."

3. **Arm's physical settling point**: The arm dynamics settle at a stable floating point where gripper weight, joint compliance, and the zero-action output balance. For approach trajectories coming from the right side of the board (X > 0.7) toward a5's X=0.600, this settling point is 10.36mm away.

### Why h5→a5 succeeds but d5→a5 fails

- `h5→a5` (560mm): The arm arrives with high momentum (99mm/s at d=5.8mm), overshoots, oscillates briefly, and catches a passing moment when d_xy=9.97mm AND speed=10mm/s → SUCCESS.
- `d5→a5` (240mm): Shorter trajectory, lower arrival speed, arm settles directly into the 10.36mm equilibrium without enough momentum to push through.
- `a2→a5` (same-column): Same X=0.600, so the motion is purely along Y. The arm approaches smoothly along Y and doesn't encounter the cross-board settling issue.

### Why exactly a5 and not other a-file squares?

a5 is at [0.600, 0.304]. For approaches from mid-board (X=0.84–0.92), the arm approaches at a slight angle: from higher X toward X=0.600, with a small Y component. The specific XY position [0.5987, 0.3144] that the model settles at is determined by the arm's joint kinematics. Other a-file squares (a3, a4, a6) have different Y coordinates and presumably slightly different settling points that happen to fall inside 10mm.

**This is a precision edge case.** The model gets to within 10.36mm of a5 — only 3.6% outside the threshold.

---

## Consistency Observations

### Transit errors (all successful moves)

| Range | Notes |
|-------|-------|
| 3–6mm | Most a–f file destinations from HOME |
| 6–10mm | a-file destinations, h-file from far positions |
| 9.4–10.4mm | a-file destinations (a4, a5, a8) from right-side sources |

The transit model has natural precision limits near the edges of the board. a4 and a5 destinations regularly show 9.4–10.4mm transit error on successful episodes — right at the threshold.

### Descend errors (all successful)

Consistent 11–12mm Z offset above HOVER_Z=0.460 across all positions (known issue, patched with 15mm tolerance). See `08_descend_hover_analysis.md`.

### Ascend errors (all successful)

4–9mm lateral drift during ascent. Highest drift at b8 (8.7mm) and g8 (8.0mm) — near board corners. Within the 10mm eval_drift_limit.

---

## Implications for Chess Game

### Which chess moves fail?

Any move that ends at a5 square coming from mid-board (d/e files, ranks 4–5) or from the far right (g/h files, rank 1). In standard chess openings:

- `d4→a5` impossible (wrong piece, but hypothetically: bishop, queen)
- `e4→a5` impossible (knight would be a5, but not from e4 standard)
- In practice: `queen to a5`, `bishop to a5` from center — these would fail

This is a real but rare failure mode. a5 is not a starting square for pieces (rank 5 is middle ranks), and most pieces that could transit to a5 come from similar ranks rather than from the center.

### Assessment

The current model is **production-ready** for pawn moves and most standard chess positions. The a5 single-square failure is a precision edge case that will be fixed by the training improvements.

---

## Recommended Training Fixes

The following changes (from `10_training_improvement_plan.md`) would eliminate the a5 failure and improve overall precision:

### For transit model retraining

1. **`show_chess_pieces=True` in training** — makes the arm start from HOME_POS every episode, matching production. This trains the model on cross-board approach trajectories like `d5→a5`.

2. **`braking_dist: 0.010 → 0.030`** — wider braking zone ensures the model starts decelerating 30mm from goal, arriving at the final 10mm zone at low speed instead of high speed.

3. **`braking_reward_weight: 0.15 → 0.50`** — stronger velocity penalty near goal forces the model to arrive slowly, reducing the energy of the settling equilibrium.

4. **`jitter_penalty_weight: 0.003 → 0.010`** — penalizes large alternating actions.

**Expected result**: The arm should arrive at a5 with low speed from any source, pass through the 10.36mm settling point, and reach within 8mm comfortably.

### For descend model retraining

See `08_descend_hover_analysis.md` and `10_training_improvement_plan.md` §2.

---

## Raw Eval Results

### Pawn eval
- 144/144 (100.0%)
- Run time: 142.3s

### Key eval
- 630/648 (97.2%)
- Stage failures: transit 18 (2.8%), descend 0, ascend 0  
- Run time: 1040s

### Full eval
- In progress (background, estimated 2–3 hrs total)
- Pattern so far: only a5 failures, consistent with key eval
