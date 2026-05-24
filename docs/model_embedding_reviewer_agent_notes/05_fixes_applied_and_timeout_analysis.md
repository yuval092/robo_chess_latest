# Fixes Applied and Timeout Root Cause Analysis

Date: 2026-05-24

## Fixes Applied (3 total)

### Fix 1: Stability threshold mismatch (model_controller.py)

**File**: `src/chess_env/model_controller.py`, line 322  
**Change**: Default `stability_vel_threshold` fallback from `0.05` → `0.02`

Training declares success when `speed < 20mm/s`. Inference was using `50mm/s` as the default.
Now both match. The arm must be well-stopped before a stage is declared successful, preventing
jerky transitions.

### Fix 2: eval_drift_limit too tight for ascend at h-file (env.yaml)

**File**: `configs/env.yaml`  
**Change**: `eval_drift_limit: 0.008` → `eval_drift_limit: 0.010`

The ascend model consistently drifts 8.0–8.4mm at h-file (X≥1.08) and g8 positions.
With 8mm limit these were hard TUBE_BREACH failures. The scripted controller uses 10mm.
Changing to 10mm fixes all h-file and g8 ascend failures without retraining.

### Fix 3: Descend Z tolerance too tight for edge positions (new finding)

**Files**:
- `configs/env.yaml`: Added `descend_success_z_tolerance: 0.015`
- `src/chess_env/model_controller.py`: Per-stage Z check in `_run_stage()`

**Root cause** (found via step-by-step diagnostic at `scripts/diagnose_descend_afile.py`):

The descend model converges to a **fixed equilibrium ~10–13mm above HOVER_Z** and freezes
there (outputting near-zero actions, speed < 0.1mm/s). This is the model's natural stable
point — the pretrained FetchPickAndPlace-v4 weights learned to hover just above the target
rather than touching it exactly.

At center positions (X=0.88): equilibrium at 9.1mm above HOVER_Z → d_z < 10mm → passes.  
At a-file positions (X=0.60): equilibrium at 10.4mm above HOVER_Z → d_z > 10mm → TIMEOUT.

The difference is **1.3mm**. Both the arm position AND absolute XY coordinate affect the
policy's equilibrium point (absolute grip position is part of the Phase-9 obs).

| Position | Equilibrium Z error | Passes 10mm check? |
|----------|--------------------|--------------------|
| Center   | 9.1mm              | YES (barely)       |
| a2       | 10.0mm             | NO (barely)        |
| a1       | 10.4mm             | NO                 |

**The fix**: Use a wider Z tolerance (15mm) for descend-only in `_run_stage()`. The grasp
precondition accepts ±25mm from HOVER_Z, so stopping 12mm above is safe. XY tolerance
remains at the standard 10mm.

```python
# In _run_stage(), for descend only:
z_tol = env.env_cfg.get("descend_success_z_tolerance", 0.015)
is_near = d_xy < env.SUCCESS_THRESHOLD and d_z < z_tol
```

After this fix, all tested positions succeed in 7–9 steps instead of timing out.

---

## Why a-file Descend Timeout Was Not Seen in Training Eval

The training eval (`eval_rl_stages.py`) uses **random positions** from the full board.
With uniform sampling, the probability of hitting X=0.60, Y∈[-0.02, 0.07] in any single
episode is ~2%. With 30 eval episodes, the expected number of a-file-low-Y samples is <1.
The training eval never reliably hits this failing region.

Production sequences always start at specific chess squares. a1/a2 are real pawn positions
that appear in every real game.

---

## What Could Be Done to Fix This at Training Level (TRAINING REQUIRED — user approval needed)

The descend model's equilibrium offset is position-dependent. To make it more consistent
across the board:

1. **Reward the model for reaching HOVER_Z exactly**: Add a `final_z_bonus` reward when
   the arm terminates within 5mm of HOVER_Z (currently the braking reward is at 10mm).

2. **Use a tighter success criterion during training**: Train with `success_threshold: 0.008`
   (8mm) instead of 10mm. This forces the model to get closer, which would reduce the
   equilibrium offset.

3. **More training steps**: A longer training run (1M+ steps) would allow the policy to
   converge more precisely at all positions.

However, the config fix (15mm Z tolerance in inference) is sufficient for production use
and doesn't require retraining.

---

## Summary of All Fixes

| Fix | File | Change | Impact |
|-----|------|--------|--------|
| 1. Stability threshold | model_controller.py | 0.05 → 0.02 default | Cleaner stage transitions |
| 2. Drift limit | env.yaml | 0.008 → 0.010 | Fixes h-file/g8 TUBE_BREACH |
| 3. Descend Z tolerance | model_controller.py + env.yaml | New 15mm descend Z | Fixes a1/a2 TIMEOUT |

All 84 unit tests pass after applying these fixes.
