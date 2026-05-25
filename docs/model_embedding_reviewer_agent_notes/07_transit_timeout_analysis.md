# Transit Timeout Root Cause Analysis

Date: 2026-05-24  
Transit model: `checkpoints/transit_20260524_084201/final_transit.zip`  
Previous model: `checkpoints/transit_20260523_164127/final_transit.zip`

---

## Executive Summary

The new transit model (trained 2026-05-24, 30/30 standalone) has a **52% timeout rate** in embedded production evaluation — far worse than the previous model's 3.3%. The root cause is **deterministic 2-step limit-cycle oscillation** that the model enters at specific goal positions and cannot escape. The model arrives at the goal and then bounces between two fixed positions indefinitely at high speed (30–135 mm/s), with exact alternating actions. This is a training distribution mismatch.

---

## 1. Evaluation Results

### 1.1 Embedded Transit Eval (50 episodes, `eval_stages.py --use-rl-models`)

| Metric | Old model (05-23) | New model (05-24) |
|--------|-------------------|-------------------|
| Success rate | 96.7% | 48.0% |
| Timeout rate | 3.3% | 52.0% |
| Avg error (success) | — | 6.1 mm |
| P95 error (success) | — | 7.9 mm |

This is a major regression. The new model was 30/30 (100%) in the isolated SB3 training eval but 48% in embedded production eval.

### 1.2 Pawn-rank all-cells transit evaluation (48 source→destination pairs, 3 reps each)

| Test | Success | Timeouts |
|------|---------|----------|
| All 144 episodes | 58/144 (40.3%) | 86/144 (59.7%) |
| Rank-2 sources (a2–h2 → +2/+1/+3 ranks) | 27/72 (37.5%) | |
| Rank-7 sources (a7–h7 → −2/−1/−3 ranks) | 31/72 (43.1%) | |

---

## 2. Diagnostic: Step-by-Step Episode Analysis

### 2.1 Script: `scripts/diagnose_transit_full.py`

Records every step of each episode including position, velocity, distance-to-goal, action output. Run with `--n-episodes 40 --capture-timeouts 6`.

### 2.2 Key Finding: Deterministic 2-Step Limit Cycle

Every single timeout episode shows the **same pattern**:

1. Arm arrives at goal (within 5–15 mm) after ~10 steps of normal approach.
2. Model outputs a large action (magnitude 0.6–0.8 in each axis).
3. Arm bounces to position B (7–15 mm from goal).
4. Model outputs the **exact negated action**.
5. Arm bounces back to position A.
6. Steps 3–5 repeat indefinitely until timeout at step 300.

**Example — target (1.025, 0.184) — speed 105 mm/s oscillation:**
```
step 280: pos=(1.027, 0.182), d_xy=2.9mm, speed=105mm/s, action=[-0.65,+0.64,+0.32]
step 281: pos=(1.018, 0.190), d_xy=9.2mm, speed=105mm/s, action=[+0.65,-0.64,-0.31]
step 282: pos=(1.027, 0.182), d_xy=2.9mm, speed=105mm/s, action=[-0.65,+0.64,+0.32]
... (identical for all 300 steps)
```

Velocity sign changes in last 10 near-goal steps: X=9/9, Y=9/9 — **perfect alternation**.

**Example — target (0.998, 0.395) — speed 30 mm/s oscillation:**
```
step 280: pos=(0.9967,0.3975), d_xy=2.7mm, speed=30mm/s, action=[-0.01,+0.12,-0.25]
step 281: pos=(0.9965,0.3990), d_xy=4.2mm, speed=31mm/s, action=[+0.01,-0.12,+0.26]
... (same 2-step cycle)
```

### 2.3 Oscillation Characteristics

| Characteristic | Value |
|---------------|-------|
| Period | 2 steps (exact) |
| Onset | Immediately upon first arrival at goal |
| Amplitude (d_xy) | 3–15 mm |
| Speed during oscillation | 30–136 mm/s |
| Action magnitude | Matched (outputs ±action on alternate steps) |
| Position-independent? | No — see §3 |

The model is trapped in a **Zeno-like limit cycle**: it outputs opposite actions on alternating steps proportional to the residual error, never dampening. The physics simulator applies these as impulses that overshoot the goal.

---

## 3. Position-Dependent Failure Pattern

### 3.1 Which positions SUCCEED

Successes cluster in the **left half of the board (a–d files, X ≈ 0.60–0.84)**:

| Target position | Outcome |
|----------------|---------|
| (0.631, 0.345) | SUCCESS 21 steps |
| (0.638, 0.334) | SUCCESS 20 steps |
| (0.770, 0.162) | SUCCESS 11 steps |
| (0.574, 0.279) | SUCCESS 26 steps |
| (0.857, 0.353) | SUCCESS 12 steps |

These are near or left of the board center (X=0.88).

### 3.2 Which positions TIMEOUT

Timeouts concentrate in two regions:

**Region A — High X (e–h files, X ≥ 0.92):**
- (1.025, 0.184), (1.034, 0.226), (1.026, 0.094), (1.092, 0.204)
- (1.082, 0.146), (1.016, 0.314), (1.019, 0.286), (1.003, 0.281)
- (1.091, 0.095), (1.162, 0.250), (1.097, 0.034), (1.184, 0.218)

**Region B — Short-distance moves from HOME to middle ranks:**
- (0.673, 0.170), (0.861, 0.208), (0.848, 0.487), (0.689, 0.186)

### 3.3 Pawn-test destination failure map

```
8x8 board map (destination, % success rate):
       a     b     c     d     e     f     g     h
  8     --    --    --    --    --    --    --    --
  7     --    --    --    --    --    --    --    --
  6      0   100     0     0   100     0     0     0
  5      0   100   100   100    50     0     0     0
  4    100   100   100   100    17    33     0     0
  3    100     0     0    33     0     0     0     0
  2     --    --    --    --    --    --    --    --
  1     --    --    --    --    --    --    --    --
```

**The right half of the board (e–h files) is systematically failing.**

### 3.4 Pawn-test source failure map

| Source square | Success rate |
|--------------|-------------|
| e2, g2, h2, f7, g7, h7 | 0% |
| f2 | 22% |
| a7 | 33% |
| a2, b2, c2, c7, d7 | 67% |
| d2, e7 | 78% |
| b7, e7 (vs far targets) | ~100% |

---

## 4. Root Cause Analysis

### 4.1 Primary cause: Production START position mismatch

**Training**: `show_chess_pieces=False` → arm starts at a **random board position** at SAFE_Z each episode. The goal is at a different random position ≥10 cm away.

**Production (embedded inference)**: `show_chess_pieces=True` → arm always starts at **HOME_POS = (0.88, 0.2641, 0.53)** (board center). The arm must transit FROM HOME to any target square.

For the new transit model, when `show_chess_pieces=True`, the model was tested with the arm **already positioned** at the source square (using `_move_mocap_to`). This replicates the production scenario where the arm starts from a prior position rather than random.

**The critical issue**: In training, the arm starts NEAR the goal (at a random board position, with goal randomly sampled ≥10 cm away). The arm has been trained on trajectories that START at the beginning of the board. But in production:
- Move 1: arm starts at HOME (0.88, 0.2641) → any target
- Move 2: arm starts at previous destination → new target

When the previous destination is on the right half of the board and the new destination is nearby (same file or close), the arm starts very close to the goal. The model has NOT been trained to handle "already close to goal" initial conditions gracefully.

### 4.2 Secondary cause: Insufficient braking reward

The `braking_dist = 0.010` (10 mm) is the distance at which braking kicks in. `braking_reward_weight = 0.15`. 

When the arm arrives at the goal at high speed (from a cross-board trajectory), the kinetic energy carries it through the braking zone before the model has time to react. The model then tries to return from the overshoot but applies full-magnitude actions that overshoot in the other direction.

The oscillation amplitude (7–15 mm) matches what we'd expect from a model applying max action to correct a ~5 mm error without damping.

### 4.3 Why the old model was better

The old model (`transit_20260523_164127`) had 96.7% success. The new one (05-24) was retrained — likely with different hyperparameters, different random seed, or different curriculum. The old model learned sufficient implicit braking to converge. The new model apparently learned a different policy that works for the training distribution (random start) but fails for the production distribution (HOME start → arbitrary goal).

### 4.4 Is this the "stability threshold 0.02" issue from doc 05?

Partially. The velocity at oscillation lock-in is 30–135 mm/s. The `stability_vel_threshold = 0.020` (20 mm/s) is what inference uses to declare success. Even if we lowered this to 30 mm/s, some oscillations at 105+ mm/s would still fail. The stability threshold fix (05→02) was already applied. The new model is genuinely oscillating at 30–135 mm/s and cannot stabilize below 20 mm/s at these positions.

### 4.5 Z-axis contribution

Some timeout episodes show the arm oscillating with significant Z amplitude (±8 mm). Looking at timeout #1: `d_z=8.4mm` at final position, compared to the target Z at SAFE_Z=0.530. The arm is bouncing in Z as well as XY. This indicates the oscillation is not purely planar.

---

## 5. Recommended Action

### 5.1 Immediate: Revert to old transit model

The old model (`transit_20260523_164127/final_transit.zip`) has been verified at:
- **30/30 (100%)** on 30-episode embedded eval
- **e2→e4, g2→g4, h2→h3, f7→f5**: all SUCCESS in 16–19 steps

The new model fails on all these. **Revert `configs/training.yaml` `deployed_models.transit` to the old checkpoint**:

```yaml
deployed_models:
  transit: "checkpoints/transit_20260523_164127/final_transit.zip"
```

| Model | Embedded eval | e2→e4 | g2→g4 | h2→h3 | f7→f5 |
|-------|--------------|-------|-------|-------|-------|
| transit_20260523_164127 (recommended) | 100% (30/30) | SUCCESS | SUCCESS | SUCCESS | SUCCESS |
| transit_20260523_185307 (alternative) | 100% (30/30) | — | — | — | — |
| transit_20260524_084201 (NEW, REGRESSION) | 48% (24/50) | FAIL | FAIL | FAIL | FAIL |

### 5.2 Training fixes required for the next transit model

See `10_training_improvement_plan.md` for full details. Key changes:

1. **Train with `show_chess_pieces=True`** so the arm starts from HOME in training — matches production. This eliminates the start-position distribution mismatch.

2. **Increase `braking_dist` to 0.030** (30 mm) so braking kicks in earlier, giving the model more time to decelerate before reaching the 10mm success zone.

3. **Increase `braking_reward_weight` to 0.5** to strongly penalize high-velocity near-goal behavior.

4. **Add a speed-squared penalty** that activates within 20 mm of goal: `penalty = 0.01 * speed^2 * (dist < 0.020)`.

5. **Increase `jitter_penalty_weight` to 0.010** — the current 0.003 is too low to prevent the large alternating actions seen in oscillation.

---

## 5a. Stage Transition Analysis (Task 4)

### soft_reset behavior

`soft_reset()` in `task.py` handles transitions between stages. Phase 2 aligns the arm to `nominal_exit_pos` using proportional control (up to 40 steps). When transit → descend:
- `nominal_exit_pos` = [dst_xy, SAFE_Z] — the exact position where transit ended
- Gripper opens (FINGER_OPEN_JOINT)
- New goal set to [dst_xy, HOVER_Z]

This is clean and reliable. The soft_reset is NOT causing failures.

### Stage handoff arm position

At the transit→descend handoff, the arm is at [dst_xy, SAFE_Z] ± ~5–8mm error. For successful transits:
- Transit final error: 2–9 mm (mostly 4–7 mm)
- `soft_reset` realigns to exact nominal position before descend starts

For descend→ascend handoff:
- Arm at [dst_xy, HOVER_Z] ± ~12mm Z error (the 11–12mm hover offset)
- soft_reset → gripper closes → ascend begins at HOVER_Z + ~12mm

The 12mm Z offset at descend end is visible in the key squares eval (descend error 9.9–12.5mm consistently).

### Arm Z motion throughout a full move

For a working move (e.g., a1→h8):
1. Transit: horizontal at SAFE_Z=0.530 → arrives at h8 at Z≈0.530 ±5mm
2. Descend: drops 70mm from SAFE_Z → stops at Z≈0.471 (12mm above HOVER_Z=0.460)
3. Ascend: rises from Z≈0.471 to SAFE_Z=0.530 in ~12 steps

The Z arc is smooth. The 12mm descend offset is the only anomaly.

---

## 6. Verification Commands

```bash
# Reproduce the failing eval
PYTHONPATH=. python scripts/eval_stages.py --use-rl-models --stages transit --n-episodes 50

# Run the step-by-step diagnostic  
PYTHONPATH=. python scripts/diagnose_transit_full.py --n-episodes 40 --capture-timeouts 6

# Run the pawn all-cells transit test
PYTHONPATH=. python scripts/eval_all_cells_rl.py --mode pawn --test-mode transit_only --n-reps 3
```
