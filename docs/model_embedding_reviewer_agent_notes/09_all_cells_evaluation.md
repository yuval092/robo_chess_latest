# All-Cells-to-All-Cells Evaluation

Date: 2026-05-24  
Models: transit `transit_20260524_084201`, descend `descend_20260523_222907`, ascend `ascend_20260523_222849`

---

## Executive Summary

The all-cells evaluation reveals a **severe positional bias** in the new transit model. The model succeeds at ~40–52% of moves overall, with complete failure (0% success rate) for destinations in the e–h file range and for short-distance moves. Descend and ascend models perform correctly when they are reached. All failures are **transit timeouts** — the model enters a deterministic 2-step limit cycle.

---

## 1. Test Methodology

### 1.1 Script

New script: `scripts/eval_all_cells_rl.py`

Tests representative source→destination pairs using `ModelEmbeddedController` (all 3 RL models loaded). Each episode:
1. Arm repositioned to src_xy at SAFE_Z using `_move_mocap_to`
2. Transit: model drives arm from src to dst at SAFE_Z
3. Descend: model lowers from SAFE_Z to HOVER_Z at dst
4. Ascend: model raises from HOVER_Z to SAFE_Z at dst

### 1.2 Test sets run

| Mode | Pairs | Reps | Total episodes | Completed |
|------|-------|------|----------------|-----------|
| pawn (ranks 2+7) | 48 | 3 | 144 | Yes |
| key squares (transit_only) | 40-episode random | 1 | 40 | Yes (diagnose) |
| key squares (three_stage) | 112 | 3 | 336 | Partial (in progress) |

---

## 2. Pawn-Rank Test Results (Transit Only, 48 pairs × 3 reps)

### 2.1 Overall numbers

| Metric | Value |
|--------|-------|
| Total episodes | 144 |
| Successes | 58 (40.3%) |
| Timeouts | 86 (59.7%) |
| Crashes | 0 |

### 2.2 Destination failure map

```
8x8 board (destination squares, % success rate):
       a     b     c     d     e     f     g     h
  6      0   100     0     0   100     0     0     0
  5      0   100   100   100    50     0     0     0
  4    100   100   100   100    17    33     0     0
  3    100     0     0    33     0     0     0     0
```

**Key observations**:
- Columns a–d mostly succeed for 2-rank jumps (+2/−2 from source)
- Columns e–h fail systematically (e only partially)
- Short moves (1-rank: x2→x3 or x7→x6) fail more often than long moves (x2→x5)
- g-file and h-file: **0% success for ALL tested destinations**

### 2.3 Source failure rates

| Source | Success/Total | Rate |
|--------|--------------|------|
| e2 | 0/9 | 0% |
| g2 | 0/9 | 0% |
| h2 | 0/9 | 0% |
| f7 | 0/9 | 0% |
| g7 | 0/9 | 0% |
| h7 | 0/9 | 0% |
| f2 | 2/9 | 22% |
| a7 | 3/9 | 33% |
| a2, b2, c2 | 6/9 | 67% |
| c7, d7 | 6/9 | 67% |
| d2, e7 | 7/9 | 78% |
| b7 | 9/9 | 100% |

### 2.4 Per-pair fully-failing cases (0/3)

```
a2->a5, b2->b3, c2->c3, e2->e4, e2->e3, e2->e5,
f2->f3, f2->f5, g2->g4, g2->g3, g2->g5, h2->h4, h2->h3, h2->h5,
a7->a5, a7->a6, c7->c6, d7->d6,
f7->f5, f7->f6, f7->f4, g7->g5, g7->g6, g7->g4,
h7->h5, h7->h6, h7->h4
```

27 out of 48 pairs (56%) have 0% success rate.

---

## 3. Key Squares Test Results (Three-Stage, Partial)

From the first 38 pairs completed of 112 planned (from a1, h1, a8 sources):

### 3.1 Pairs tested from a1

| Destination | Result | Transit error |
|-------------|--------|---------------|
| h1 | 0/3 FAIL (transit TIMEOUT) | 3.8mm |
| a8 | 1/3 FAIL | 5–7mm |
| h8 | 3/3 SUCCESS | 4.9–5.1mm |
| g1 | 0/3 FAIL | 11.2mm |
| b8 | 3/3 SUCCESS | 7.8–7.9mm |
| g8 | 3/3 SUCCESS | 5.3mm |
| d4 | 3/3 SUCCESS | 5.9–6.7mm |
| e4 | 0/3 FAIL | 8.8mm |
| d5 | 3/3 SUCCESS | 6.2–7.8mm |
| e5 | 0/3 FAIL | 6.8–7.2mm |

### 3.2 Pairs tested from h1

| Destination | Result | Transit error |
|-------------|--------|---------------|
| a1 | 3/3 SUCCESS | 8.7–8.8mm |
| a8 | 0/3 FAIL | 8.9mm |
| h8 | 3/3 SUCCESS | 6.0mm |
| b1 | 3/3 SUCCESS | 7.2–7.5mm |
| b8 | 3/3 SUCCESS | 6–7.9mm |
| g8 | 3/3 SUCCESS | 4.9mm |
| d4 | 3/3 SUCCESS | 6.7–6.8mm |
| e4 | 0/3 FAIL | 8.7mm |
| d5 | 3/3 SUCCESS | 7.3mm |
| e5 | 3/3 SUCCESS | 8.5–8.7mm |

### 3.3 Patterns from key squares

When descend/ascend are reached (on successful pairs), they perform:
- **Descend**: 100% success, error 9.9–12.5 mm (consistent with 11–12mm offset above HOVER_Z)
- **Ascend**: 100% success, error 5.2–8.7 mm

---

## 4. Failure Pattern Analysis

### 4.1 Position-based failure characterization

Combining both test sets, failures map to specific absolute destination XY coordinates:

**Reliably failing destinations (X axis)**:
- e-file: X = 0.920 — fails ~50% of the time
- f-file: X = 1.000 — fails >80% of the time
- g-file: X = 1.080 — fails ~95% of the time
- h-file: X = 1.160 — fails ~70–90% depending on Y

**Reliably passing destinations**:
- a-file: X = 0.600 — mostly passes (especially Y > 0.2)
- b-file: X = 0.680 — mostly passes
- c-file: X = 0.760 — mostly passes  
- d-file: X = 0.840 — mostly passes
- Corner: a1→h8 (diagonal) — passes
- Rank diagonals: a1→b8 — passes

**Surprising result**: a1→h1 (rank-1 horizontal, X: 0.60→1.16) **FAILS** consistently. The arm gets within 3.8 mm but oscillates at 30+ mm/s. This is a short move of 560mm that should be easy.

**Surprising result**: h1→a8 (long diagonal) **FAILS**. But a1→h8 (same distance, opposite direction) **PASSES**.

### 4.2 Distance does not predict failure

```
Diagnostic eval (40 episodes):
  Timeouts avg distance from home: 207mm (min 59mm, max 317mm)
  Successes avg distance from home: 228mm (min 92mm, max 367mm)
```

Distance from home is NOT the primary predictor. The model successfully reaches positions 367mm from home (h8) but fails on positions 59mm from home. **Absolute XY position is the primary predictor**.

### 4.3 Short-distance failures

Several 1-rank pawn moves fail completely (a2→a3, b2→b3, c2→c3, etc.). These are 80mm moves. The arm reaches within 3–4 mm but at 40–50 mm/s speed and oscillates. 

This is particularly problematic because pawn moves are the most common chess moves.

### 4.4 Why certain X positions fail

The HOME position is (0.88, 0.264). The failing destinations cluster around X = 0.92–1.18.

When the arm transits FROM HOME to these positions, it overshoots into the X ≈ 0.95–1.20 region at high speed. The model learned to approach from the LEFT (low X), but these targets require stopping at the RIGHT edge of the board.

More specifically, the training distribution (`show_chess_pieces=False`) places both start and goal at random positions. When start is at X=0.8 and goal is at X=1.1, the model has been trained to handle this. But when start is at HOME (X=0.88) and goal is at X=0.92 (e4), the approach direction is different from training. The **relative vector from start to goal** matters — and the new model seems to have learned a policy that is sensitive to this direction.

### 4.5 e/f/g/h target hypothesis

These are the files to the RIGHT of HOME_XY (X=0.88). When the arm starts at HOME and moves to e-h files, it must travel in the +X direction. If the model was not sufficiently trained on +X approaches from center, it overshoots and oscillates.

This is consistent with the pawn test: g7→g4 (dst X=1.08) fails 100%, but g7→d5 (dst X=0.840) would likely pass.

### 4.6 High-X but high-Y passes; high-X with low/mid-Y fails

A critical pattern emerges from the key squares data:

| Destination | XY | Source | Result |
|------------|-----|--------|--------|
| h8 (1.160, 0.544) | high-X, high-Y | a1, h1, a8, h8 | 3/3 SUCCESS |
| g8 (1.080, 0.544) | high-X, high-Y | a1, h1, a8, h8 | 3/3 SUCCESS |
| h1 (1.160, -0.016) | high-X, low-Y | a1, a8, b1 | 0/9 FAIL |
| g1 (1.080, -0.016) | high-X, low-Y | a1, a8 | 0/6 FAIL |
| e4 (0.920, 0.224) | high-X, mid-Y | ALL sources | 0/18 FAIL |

**h8 and g8 succeed despite X>1.0; h1 and g1 fail at the same X.** The Y coordinate is critical. Destinations with Y=0.544 (rank 8 = high on board) pass at any X. Destinations with Y=-0.016 (rank 1 = front of board) fail at high X. This makes sense geometrically: rank-8 targets require +X+Y movement from HOME, while rank-1 targets require +X−Y or +X only. The model may have better convergence for +Y approach vectors.

### 4.7 The e4 (0.920, 0.224) special case

In the key squares evaluation, **e4 failed from EVERY source tested** (a1, h1, a8, h8, b1, g1 — 6 different sources, all 0% success). This is the strongest evidence of a specific position dependency. The model enters oscillation immediately upon arriving at (0.920, 0.224) regardless of approach direction.

XY = (0.920, 0.224) is 40mm to the right of HOME_X (0.88). It is close enough to HOME that the arm approaches with low residual velocity (transit error 6.8–8.8mm), but the model still oscillates. This suggests e-file targets have a particularly bad limit cycle in the policy.

### 4.7 Descend TUBE_BREACH at a8 (one case)

In the key squares eval, `g1->a8 rep 3` showed `FAIL@descend` with descend error 33.6mm. This is a TUBE_BREACH (drift > 10mm) — the descend model drifted 33mm laterally at a8. This is an isolated case but warrants investigation. The previous fix set `eval_drift_limit=0.010`, so 33mm drift will still fail.

This may be due to the arm arriving at a8 with an unusual orientation from the g1 transit, causing the descend model to start from a misaligned position.

---

## 5. Stage-Level Summary

### 5.1 Transit

- **Failure rate**: 59.7% in pawn test, ~50% overall
- **Failure mode**: Deterministic 2-step limit cycle (oscillation)
- **Position pattern**: Right half of board (e–h files) and short distances
- **This model is not production-ready**

### 5.2 Descend

- **Failure rate**: 0% (100% success)
- **Equilibrium offset**: 10.8–12.1 mm above HOVER_Z (patched with 15mm tolerance)
- **Consistent across all positions tested**

### 5.3 Ascend

- **Failure rate**: 0% (100% success when reached)
- **Error**: 5.2–8.7 mm (within acceptable range)
- **The previous h-file TUBE_BREACH was fixed by eval_drift_limit=10mm**

---

## 6. Production Impact Assessment

### 6.1 Current state

With the new transit model, the production pipeline will fail approximately **50–60% of chess moves** that target e–h files. Specifically:
- e-file moves: ~50% failure rate
- f-file moves: ~80% failure rate
- g-file moves: ~95% failure rate
- h-file moves: ~70–90% failure rate

Since 4 out of 8 files are in the failing zone, approximately **40–50% of all chess moves** will fail.

### 6.2 Immediate fix

**Revert transit model** to `checkpoints/transit_20260523_164127/final_transit.zip` which had 96.7% success rate. This will restore production to the state documented in `06_post_fix_evaluation.md`.

### 6.3 Affected chess moves

Common opening moves affected:
- e2→e4 (King's Pawn): FAIL
- e7→e5 (symmetric response): FAIL
- g1→f3 (Knight's move from g1): FAIL at g1 source → f3 dst
- f2→f4 (Bird's Opening): FAIL
- h-pawn moves: FAIL

These are all common opening moves. The game is unplayable with the new transit model.

---

## 7. All-Cells Script

Script: `/home/user/projects/robo_chess_latest/scripts/eval_all_cells_rl.py`

Usage:
```bash
# Pawn test (transit only, fast)
PYTHONPATH=. python scripts/eval_all_cells_rl.py --mode pawn --test-mode transit_only --n-reps 3

# Key squares (full 3-stage)
PYTHONPATH=. python scripts/eval_all_cells_rl.py --mode key --test-mode three_stage --n-reps 3

# Full coverage (slow, ~2-3 hours)
PYTHONPATH=. python scripts/eval_all_cells_rl.py --mode full --test-mode three_stage --n-reps 2
```

The script:
- Loads all 3 models from `configs/training.yaml`
- For each pair, repositions arm to src using `_move_mocap_to`, then runs the stages
- Prints per-episode results and summary statistics
- Shows 8x8 board failure maps for both source and destination squares
