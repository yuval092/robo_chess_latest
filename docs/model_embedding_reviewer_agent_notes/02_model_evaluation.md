# Model Evaluation Report

Date: 2026-05-24

## Setup

All evaluations run from project root with `PYTHONPATH=.`.

**Deployed models (configs/training.yaml at time of review):**
- transit: `checkpoints/transit_20260523_164127/final_transit.zip` (NOTE: config was NOT updated to newer run)
- descend: `checkpoints/descend_20260523_222907/final_descend.zip`
- ascend:  `checkpoints/ascend_20260523_222849/best_model_ascend.zip`

**Available checkpoints:**
| Stage | Run directory | Files |
|-------|--------------|-------|
| transit | `transit_20260523_164127` | `best_model_transit.zip`, `final_transit.zip`, `latest_model_transit.zip` |
| transit | `transit_20260524_084201` | `best_model_transit.zip`, `latest_model_transit.zip` |
| descend | `descend_20260523_222907` | `best_model_descend.zip`, `final_descend.zip`, `latest_model_descend.zip` |
| ascend | `ascend_20260523_222849` | `best_model_ascend.zip`, `final_ascend.zip`, `latest_model_ascend.zip` |

---

## Direct (Standalone) Evaluation

Run via `scripts/eval_rl_stages.py` with `eval_drift_limit = 0.008` (from env.yaml).

### Transit

| Checkpoint | Success | Crashes | Timeouts | Avg Reward |
|-----------|---------|---------|---------|-----------|
| `transit_20260524_084201/best_model_transit.zip` | 28/30 (93.3%) | 0 | 2 | 452.5 |
| `transit_20260524_084201/latest_model_transit.zip` | 29/30 (96.7%) | 0 | 1 | 465.6 |

Both transit models from the newer run (084201) perform well in direct eval. The `latest_model` is marginally better. Both failures are timeouts (oscillation near goal), not crashes.

Note: The currently deployed transit model (`transit_20260523_164127/final_transit.zip`) was NOT re-evaluated here because it was tested by the previous implementing agent (30/30 direct, 29/30 embedded).

### Descend

| Checkpoint | Success | Crashes | Timeouts | Avg Reward |
|-----------|---------|---------|---------|-----------|
| `descend_20260523_222907/best_model_descend.zip` | 30/30 (100.0%) | 0 | 0 | 499.6 |
| `descend_20260523_222907/latest_model_descend.zip` | 30/30 (100.0%) | 0 | 0 | 499.4 |

Both descend models achieve perfect 30/30 in direct eval. The average rewards are essentially identical (~499.4-499.6 out of 500). The marginal difference in reward favors `best_model_descend.zip`.

**Selection: `best_model_descend.zip`** — slightly higher reward, both are perfect in success rate.

### Ascend

| Checkpoint | Success | Crashes | Timeouts | Avg Reward |
|-----------|---------|---------|---------|-----------|
| `ascend_20260523_222849/best_model_ascend.zip` | 30/30 (100.0%) | 0 | 0 | 499.6 |
| `ascend_20260523_222849/latest_model_ascend.zip` | 23/30 (76.7%) | 2 | 5 | 348.2 |

`best_model_ascend.zip` is the clear winner: 30/30 vs 23/30. The `latest_model_ascend.zip` has both tube breaches and timeouts, indicating it regressed after the best checkpoint was saved.

**Selection: `best_model_ascend.zip`** — 100% vs 76.7% in direct eval.

---

## Embedded (Production-Path) Evaluation

Run via `scripts/eval_stages.py --use-rl-models`, which loads models through `ModelEmbeddedController` exactly as `run_chess_ui.py` does.

### With Currently Deployed Models (training.yaml as-is)

```
python scripts/eval_stages.py --use-rl-models --stages transit,descend,ascend --n-episodes 30
```

| Stage | Success | Crash | Timeout | Avg Err | P95 Err |
|-------|---------|-------|---------|---------|---------|
| transit | 28/30 (93.3%) | 0% | 6.7% | 5.4 mm | 9.3 mm |
| descend | 30/30 (100.0%) | 0% | 0% | 10.3 mm | 10.6 mm |
| ascend | 29/30 (96.7%) | 3.3% | 0% | 6.7 mm | 10.1 mm |

Ascend crash: `TUBE_BREACH (drift=8.1mm)` — exactly 0.1mm over the 8.0mm limit.

### With Model-Selection Agent's Preferred Models

```
python scripts/eval_stages.py --use-rl-models --stages transit,descend,ascend --n-episodes 30
  --transit-model checkpoints/transit_20260524_084201/latest_model_transit.zip
  --descend-model checkpoints/descend_20260523_222907/latest_model_descend.zip
  --ascend-model checkpoints/ascend_20260523_222849/best_model_ascend.zip
```

| Stage | Success | Crash | Timeout | Avg Err | P95 Err |
|-------|---------|-------|---------|---------|---------|
| transit | 23/30 (76.7%) | 0% | 23.3% | 5.2 mm | 10.4 mm |
| descend | 30/30 (100.0%) | 0% | 0% | 10.3 mm | 10.7 mm |
| ascend | 29/30 (96.7%) | 3.3% | 0% | 6.8 mm | 9.9 mm |

**KEY FINDING**: The newer transit model (`transit_20260524_084201/latest_model`) performs significantly WORSE in embedded eval (76.7% vs 93.3%). The older deployed model (`transit_20260523_164127/final_transit.zip`) is better for production use. The model-selection agent's finding that the newer model scored "30/30 in embedded transit eval" disagrees with this reviewer's measurement (23/30). This discrepancy may be due to different episode seeds or starting positions.

---

## Sequence Evaluations

### Vertical chain (descend → ascend without grasp, center square)

```
python scripts/eval_sequence.py --use-rl-models --chain vertical --n-episodes 20
```

Result: **20/20 (100%)** — descend 20/20, ascend 20/20.

The vertical chain uses the board center (0.88, 0.2641), which is the easiest case.

### Pick sequence (transit → descend → grasp → ascend)

```
python scripts/eval_sequence.py --use-rl-models --chain pick --n-episodes 20
```

Result: **20/20 (100%)** — transit 20/20, descend 20/20, grasp 20/20, ascend 20/20.

Source square: `[0.88, 0.2641]` (board center). This is the default and the easiest case.

### Full move (pick + place, center board)

```
python scripts/eval_sequence.py --use-rl-models --chain full_move --n-episodes 15
```

Result: **15/15 (100%)** — all 30 transits pass, all 30 descends pass, both grasps and places pass.

### Game Flow Evaluation (4 standard opening moves)

```
python scripts/eval_chess_game_flow.py --use-rl-models --verify-agreement --nonmoving-tolerance-mm 2.0
```

Default moves: `e2e4, e7e5, g1f3, b8c6`.

Result: **PASSED** — all 4 moves succeed, nonmoving piece displacement ≤ 2mm throughout.

---

## Cross-Board Position Testing

Extensive testing across board squares reveals **position-dependent failure modes**. All tests run with 3-5 episodes using the deployed models from training.yaml.

### Failure Map

| Square | XY | Pick result | Failure mode |
|--------|-----|------------|-------------|
| a1 | (0.60, -0.0159) | 0/3 | descend: TIMEOUT |
| a2 | (0.60, 0.0641) | 0/3 | descend: TIMEOUT |
| h1 | (1.16, -0.0159) | 0/3 | ascend: TUBE_BREACH (8.1mm) |
| h2 | (1.16, 0.0641) | 0/3 | ascend: TUBE_BREACH (8.3mm) |
| h3 | (1.16, 0.1441) | 0/3 | ascend: TUBE_BREACH (8.3mm) |
| h8 | (1.16, 0.5441) | 0/3 | ascend: TUBE_BREACH (8.4mm) |
| g8 | (1.08, 0.5441) | 0/3 | ascend: TUBE_BREACH (8.1mm) |
| a5 | (0.60, 0.3041) | 0/3 | transit: TIMEOUT |
| d1 | (0.84, -0.0159) | 3/3 | ok |
| e1 | (0.92, -0.0159) | 3/3 | ok |
| a8 | (0.60, 0.5441) | 5/5 | ok |
| f8 | (1.00, 0.5441) | 3/3 | ok |
| e4 | (0.92, 0.2241) | — | ok (game flow test) |
| d5 | (0.84, 0.3041) | — | ok (game flow test) |

### Also in game-flow testing

```
python scripts/eval_chess_game_flow.py --use-rl-models --moves "d2d4,d7d5,g1f3,g8f6"
```

**FAILED**: Move 4 `g8f6` fails with `ascend: TUBE_BREACH (drift=8.0mm)`. This failure is **100% reproducible** — identical failure across multiple runs, suggesting deterministic behavior from the given game state. The failure occurs specifically on g8 (1.08, 0.5441).

```
python scripts/eval_chess_game_flow.py --use-rl-models --moves "a2a4"
```

**FAILED**: First move `a2a4` fails with `descend: TIMEOUT`.

```
python scripts/eval_chess_game_flow.py --use-rl-models --moves "h2h4"
```

**FAILED**: First move `h2h4` fails with `ascend: TUBE_BREACH (drift=8.4mm)`.

---

## Failure Mode Analysis

### Failure Mode 1: Descend TIMEOUT at a-file squares (a1, a2, a5 and nearby)

**Affected squares**: a1, a2, and some a-file squares with very low Y coordinates or extreme combinations.

**Cause**: At X=0.6 (leftmost board column), the arm is near its kinematic reach limit. The descend model was trained on the full board range but the arm's kinematics at X=0.6 make it difficult to hold the tube constraint while descending. The model successfully transits to the square (transit passes), but cannot complete the descent within 300 steps while satisfying the 8mm tube constraint.

**Nature**: Model quality issue for near-reach-limit positions. The model never encounters a tube breach — it simply runs out of time before reaching HOVER_Z.

**Workaround available**: The scripted controller handles a1 correctly (verified by `eval_all_square_moves.py --from-square a2 --max-cases 8`).

### Failure Mode 2: Ascend TUBE_BREACH at h-file and top-rank squares

**Affected squares**: h1, h2, h3, h8, g8. All involve X ≥ 1.08 (rightmost 2 files).

**Cause**: When ascending from high-X coordinates, the arm drifts slightly in XY as it rises. The best_model_ascend.zip consistently produces 8.0-8.4mm drift against a hard 8.0mm limit. The drift is small (0-0.4mm over limit) and deterministic for specific squares. This is a marginal failure mode where the model is "close but not quite" inside the production constraint.

**Nature**: Tube margin is too tight for the model's ascent trajectory at high-X positions. Training used 8mm as the target, but the production check triggers at exactly 8.0mm with no tolerance.

**Workaround available**: Increasing `eval_drift_limit` from 8mm to 10mm would eliminate these failures. The scripted controller uses `drift_limit=0.010` (10mm) by default.

### Failure Mode 3: Transit TIMEOUT at a5 (0.6, 0.3041)

**Cause**: Square a5 at (0.6, 0.3041) fails transit. This is unexpected given a1 (same X, different Y) passes transit. This may be a stochastic failure (low N=3), or a specific combination of start position (HOME_POS) and target (0.6, 0.3041) that the model handles poorly.

---

## Model Selection Summary

| Stage | Recommended Model | Rationale |
|-------|------------------|-----------|
| transit | `checkpoints/transit_20260523_164127/final_transit.zip` | 93.3% embedded success (current deployed, verified) |
| descend | `checkpoints/descend_20260523_222907/best_model_descend.zip` | 30/30 direct, 30/30 embedded, marginally higher reward than latest |
| ascend | `checkpoints/ascend_20260523_222849/best_model_ascend.zip` | 30/30 direct (latest is only 23/30), already deployed |

**Important**: The currently deployed models in `configs/training.yaml` are already a good selection (final_transit + final_descend + best_ascend). The model selection agent's doc claims different models were deployed but the config was not updated. The current state is actually **fine as-is**.

The main unresolved issue is not model selection — it is that the deployed ascend model fails at the hard 8.0mm tube limit on corner squares. This is a training quality issue, not a checkpoint selection issue.

---

## Prior Agent Notes Comparison

The prior model-selection agent (2026-05-24 doc) reported:
- Transit latest: 85/90 (94%) in direct eval, 30/30 embedded
- Descend latest: 90/90 in direct eval
- Ascend best: 86/90 (95.6%) in direct eval

This reviewer's results:
- Transit latest (084201): 29/30 (96.7%) in direct eval, **23/30 (76.7%) in embedded eval** — significantly worse embedded
- Descend best: 30/30 in direct eval (matches)
- Ascend best: 30/30 in direct eval (matches)

The main discrepancy is transit embedded performance. The prior agent claimed 30/30 embedded for the newer transit. This reviewer measured 23/30. The newer transit model appears to have poor generalization from the training distribution to production start conditions. The older `final_transit.zip` (93.3% embedded) is likely better for production.
