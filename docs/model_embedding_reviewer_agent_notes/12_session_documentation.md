# Session Documentation: Model Embedding, Evaluation & Training

Date: 2026-05-24  
Author: Claude (AI pair programmer)

---

## Training Status (as of ~98K / 600K steps)

| Model | Steps | Success Rate | ep_rew_mean | ep_len_mean | Notes |
|-------|-------|-------------|-------------|-------------|-------|
| Transit | 98,412 (16%) | **100%** | 485 | 24.8 steps | Excellent — already converged |
| Descend | 95,956 (16%) | **77%** | 380 | 55.5 steps | Good progress, still learning |

Checkpoints saved at:
- `checkpoints/transit_20260524_142519/` → `best_model_transit.zip`, `latest_model_transit.zip`
- `checkpoints/descend_20260524_142540/` → `best_model_descend.zip`, `latest_model_descend.zip`

---

## Overview of the Full Session

This document covers three connected sessions:

1. **Session 1**: Initial review of model-embedding commits and evaluation
2. **Session 2**: Bug fixing and post-fix evaluation  
3. **Session 3**: Root-cause analysis, full-board evaluation, training parameter design, and retraining

---

## What Was Found (Bugs & Root Causes)

### Bug 1: Stability Threshold Mismatch (Fixed)

**File**: `src/chess_env/model_controller.py`, line ~322  
**Problem**: The inference loop used `stability_vel_threshold` default of `0.05` (50 mm/s) but training used `0.02` (20 mm/s). A model trained to be stable at 20 mm/s was being declared successful at 50 mm/s — accepting sloppy arrivals.  
**Fix**: Changed default to `0.02`.

### Bug 2: h-file TUBE_BREACH at 0% Success (Fixed)

**File**: `configs/env.yaml`, `eval_drift_limit`  
**Problem**: The ascend model drifts 8.0–8.4 mm laterally at h-file positions (X ≥ 1.08). The inference crash check used `eval_drift_limit: 0.008` (8 mm). Since drift ≈ limit, h-file ascend always triggered TUBE_BREACH.  
**Fix**: Changed `eval_drift_limit: 0.008 → 0.010`. This keeps training strict (8 mm via `drift_limit_end`) while giving 2 mm of headroom in production — exactly the principle the user later confirmed in Point 6.  
**Why the drift happens**: The ascend model was trained with random start positions including h-file; lateral arm dynamics at that extreme reach cause slight drift. The model is functional (drift < 10 mm is acceptable for grasp pipeline).

### Bug 3: a-file TIMEOUT at 0% Success (Fixed via patch, now fixed properly via retraining)

**File**: `configs/env.yaml`, `src/chess_env/model_controller.py`  
**Problem**: The descend model consistently converges to 11–12 mm above HOVER_Z = 0.460. This is a training artifact: `braking_dist = success_threshold = 10 mm`, meaning the velocity penalty only activates *inside* the success zone. The model learns a stable hover point at ~11–12 mm (just outside the zone) because there is no reward signal incentivizing it to push further.  
**Patch (now removed)**: Added `descend_success_z_tolerance: 0.015` in `env.yaml` and a per-stage Z check in `model_controller.py` so inference accepted up to 15 mm Z error for descend.  
**Proper fix**: Retrain descend with wider braking zone + tighter training success criterion (see Training Changes below).

### Bug 4: New Transit Model Regression (Identified, Reverted)

**File**: `configs/training.yaml`  
**Problem**: A new transit model (`transit_20260524_084201`) trained during the session achieved 30/30 (100%) in standalone SB3 eval but only 48% in embedded production eval. Root cause: deterministic 2-step limit-cycle oscillation at specific goal positions (especially e–h files, X ≥ 0.92). The model outputs perfectly alternating actions (e.g. `[-0.65, +0.64, +0.32]` then `[+0.65, -0.64, -0.31]`) and never escapes.  
**Root cause**: That model was trained with `show_chess_pieces=False`, meaning random start positions. The model learned a policy that works for random → random transits but produces oscillation when the approach comes from predictable cross-board trajectories.  
**Action**: Immediately reverted `deployed_models.transit` back to `transit_20260523_164127/final_transit.zip` which had 96.7% embedded success rate.

---

## Full-Board Evaluation Results

Three evaluations were run against all currently deployed models:

### Deployed models tested

```yaml
transit: "checkpoints/transit_20260523_164127/final_transit.zip"
descend: "checkpoints/descend_20260523_222907/final_descend.zip"
ascend:  "checkpoints/ascend_20260523_222849/best_model_ascend.zip"
```

### Results

| Mode | Episodes | Success |
|------|----------|---------|
| Pawn moves (rank 2/7) | 144 | **100%** |
| Key squares (corners, edges, center) | 648 | **97.2%** (630/648) |

### The Only Failure: a5 as destination

All 18 key-eval failures (6 source positions × 3 reps) are FAIL@transit to `a5 = [0.600, 0.304]`, with exactly 10.8 mm final 3D error in every case.

**Step-by-step diagnostic (see `scripts/diagnose_transit_full.py`)**:

The arm freezes at a **stationary equilibrium** at [0.5987, 0.3144, 0.5330]:
- XY error: 10.36 mm (just 0.36 mm outside the 10 mm threshold)
- Z error: 3.05 mm
- Speed: 0.02 mm/s — **completely stopped, not oscillating**

This is a local minimum: the model outputs zero action because it receives no braking penalty at 10.36 mm (outside the 10 mm braking zone) and the flat distance gradient doesn't pull it further.

**Why `h5→a5` succeeds but `d5→a5` fails**: h5 is 560 mm away; the arm arrives with 99 mm/s speed, overshoots to 5.8 mm, bounces back, and catches a moment (step 46) at d_xy = 9.97 mm with speed 10 mm/s → SUCCESS. d5 is only 240 mm; the arm arrives more gently and settles directly into the 10.36 mm equilibrium.

**Same root cause as the descend Z offset**: `braking_dist = success_threshold = 10 mm`. No velocity penalty outside the success zone means the model can find stable non-success equilibria.

---

## What Was Changed and Why

### 1. Scenario-Specific Braking Distances (`configs/env.yaml`)

```yaml
# Before:
braking_dist: 0.010
braking_reward_weight: 0.15

# After (scenario-specific, ascend unchanged as fallback):
braking_dist: 0.010                     # fallback — ascend still uses this
transit_braking_dist: 0.030             # 30 mm braking zone for transit
transit_braking_reward_weight: 0.50     # strong velocity penalty near transit goal
descend_braking_dist: 0.025             # 25 mm braking zone for descend
descend_braking_reward_weight: 0.30     # moderate velocity penalty near HOVER_Z
```

**Why wider braking zones**: The core issue is that `braking_dist = success_threshold = 10 mm` creates a blind spot — the model learns to park just outside 10 mm where there's no velocity penalty. With 30 mm (transit) and 25 mm (descend), the velocity penalty activates *before* the success zone, so the model must arrive slowly. A slowly-arriving arm has less energy to settle at a wrong equilibrium and can reach the goal reliably.

**Why scenario-specific**: The ascend model is already trained and working (100%). Changing global `braking_dist` would affect ascend if it were ever retrained. Making it scenario-specific lets each stage have its own tuned value without breaking the others.

**Why NOT setting `show_chess_pieces=True` for transit training**: Forcing the arm to always start from HOME_POS would train a model that only handles HOME→square transits. In the actual chess game, most transits are square→square (carrying a piece). The current random-start training already handles most cases at 97.2%. The braking zone fix addresses the precision issue without distribution distortion.

### 2. Descend Success Criterion (`configs/env.yaml`, `src/chess_env/task.py`)

```yaml
# Added:
descend_success_threshold: 0.008   # d_z < 8 mm required (d_xy still < 10 mm)
```

```python
# task.py step() — now uses tighter Z criterion for descend training:
if self.current_scenario == "descend":
    d_xy = float(np.linalg.norm(grip_pos[:2] - self.goal_pos[:2]))
    d_z  = float(abs(grip_pos[2] - self.goal_pos[2]))
    descend_thresh = self.env_cfg.get("descend_success_threshold", self.SUCCESS_THRESHOLD)
    is_near = bool(d_xy < self.SUCCESS_THRESHOLD and d_z < descend_thresh)
else:
    is_near = bool(self._is_success(grip_pos, self.goal_pos))
```

**Why**: The old descend model settled at 11–12 mm above HOVER_Z — just outside the 10 mm training threshold. Tightening to 8 mm forces the model to genuinely reach within 8 mm of HOVER_Z. This eliminates the "hover at boundary" equilibrium.

**Why 8 mm (not 6 mm as originally proposed)**: 8 mm gives a 2 mm margin above the critical 10 mm threshold and below the 15 mm patch we just removed. It is achievable given the wider braking zone (arm decelerates 25 mm before HOVER_Z), and avoids the training instability that could arise from being too tight.

**Why NOT `final_approach_bonus`**: A per-step bonus within 5 mm of HOVER_Z would create a "hover at 4.9 mm" local maximum — the model could collect 5.0 reward/step indefinitely vs the 500-point one-time success bonus. At ~100 steps in the 4.9 mm zone, reward equals success. We dropped this idea and rely solely on braking + tighter criterion.

### 3. Jitter Penalty Increase (`configs/env.yaml`)

```yaml
# Before:
jitter_penalty_weight: 0.003

# After:
jitter_penalty_weight: 0.010
```

**Why**: The 05-24 transit regression showed large alternating actions (magnitude 0.6–0.8/axis) that the 0.003 penalty couldn't suppress. At magnitude 0.8 per axis: `0.003 × (0.8² + 0.8²) = 0.0039` per step — negligible vs the 500 success bonus. At 0.010: `0.0124` per step — still small but combined with the wider braking zone makes high-frequency oscillation less profitable.

**Risk**: This applies to all stages. For descend (which has slower, smoother approach), the higher jitter penalty discourages micro-adjustments near HOVER_Z. Since we want the arm to arrive smoothly, this is intentional.

### 4. Removed Descend Inference Patch (`src/chess_env/model_controller.py`, `configs/env.yaml`)

```python
# Removed from model_controller.py _run_stage():
if stage == "descend":
    d_xy = float(np.linalg.norm(grip_pos[:2] - target_pos[:2]))
    d_z  = float(abs(grip_pos[2] - target_pos[2]))
    z_tol = env.env_cfg.get("descend_success_z_tolerance", 0.015)
    is_near = d_xy < env.SUCCESS_THRESHOLD and d_z < z_tol
else:
    is_near = bool(env._is_success(grip_pos, target_pos))

# Replaced with clean uniform check:
is_near = bool(env._is_success(grip_pos, target_pos))
```

```yaml
# Removed from env.yaml:
# descend_success_z_tolerance: 0.015
```

**Why**: With the descend model being retrained to achieve genuine 8 mm precision, the 15 mm inference patch is no longer needed. Removing it allows the new model to be evaluated fairly and keeps the codebase clean. The patch was a workaround for a training deficiency that is now being fixed at the source.

**Consequence**: The *old* descend model (pre-retrain) will now fail production tests because it converges to 11–12 mm above HOVER_Z but the standard 10 mm threshold requires ≤ 10 mm. This is intentional — we are mid-retrain and the old model is being replaced.

### 5. Drift Strictness Confirmation

The user asked to verify: training 8 mm strict, production 10 mm loose.

```yaml
drift_limit_end: 0.008    # Training: arm crash if drift > 8 mm from tube center
eval_drift_limit: 0.010   # Production: inference failure if drift > 10 mm
```

These were **already correct** from the Fix 2 applied earlier in the session. The 2 mm gap means the model trains to stay comfortably within 8 mm; in production, occasional 8–10 mm drift (e.g. h-file ascend) doesn't cause a false failure. No code change needed — just confirmed and documented.

---

## What Was NOT Changed and Why

### Start position for transit training (`training/envs/__init__.py`)

The original proposal was to set `show_chess_pieces=True` during transit training so the arm always starts from HOME_POS. This was rejected because:

1. In a chess game, **only the very first transit per move** starts from HOME_POS. All subsequent transits (carrying piece, returning) start from arbitrary board positions.
2. Training exclusively from HOME_POS would create a new distribution mismatch for the majority of in-game transits.
3. The a5 precision failure is not caused by a start-position distribution problem — it is caused by the flat reward gradient at the edge of the braking zone. The wider braking zone fixes it without touching start positions.

### Ascend model

The ascend model is at 100% success rate with `eval_drift_limit: 0.010`. It is not being retrained. The drift strictness (8 mm training, 10 mm production) is already in place for it.

### Success threshold for transit inference

The transit threshold remains `success_threshold: 0.010` (10 mm). Tightening the *inference* threshold would produce more failures without fixing the underlying policy. The right lever is the *training* braking zone, not the inference criterion.

---

## Training Commands Used

```bash
PYTHONPATH=. python3 scripts/train_rl.py --stage transit --envs 4 --timesteps 600000
PYTHONPATH=. python3 scripts/train_rl.py --stage descend --envs 4 --timesteps 600000
```

Both start from `archive/rl_system/models/sac-FetchPickAndPlace-v4.zip` (FetchPickAndPlace pretrained weights — the base model for all specialists).

---

## Full-Board Evaluation Results (Old Deployed Models, 2026-05-24)

A full 64×63 evaluation (5944 episodes, 2 reps per pair) was run with the old deployed models (before retraining completes). Results:

**Overall: 5850/5944 (98.4%) success**

All 94 failures are transit failures. Two destination squares fail consistently:

| Destination | Success Rate | Failing source count | Root cause |
|---|---|---|---|
| **c1** `[0.760, -0.0159]` | 36/98 (37%) | 25 sources | Same as a5 |
| **a5** `[0.600, 0.304]` | 68/100 (68%) | 15 sources | Known braking equilibrium |

Both failures are the same braking equilibrium root cause: `braking_dist = success_threshold = 10 mm`. The arm finds a stationary local minimum just outside the 10 mm threshold with no velocity penalty to pull it further.

**c1 was not caught by the earlier key-square evaluation** — it is a new failure confirmed only by the full-board run.

The 30 mm transit braking zone fix (currently training) eliminates the reward blind spot that creates both equilibria. Expected outcome after retraining: both a5 and c1 clear >95%.

8×8 destination success map:
```
       a     b     c     d     e     f     g     h
  1   100   100    37   100   100   100   100   100  
  5    68   100   100   100   100   100   100   100  
  (all others: 100%)
```

---

## Expected Outcomes After Retraining

| Metric | Before (unpatched old models) | Expected (new models) |
|--------|------------------------------|-----------------------|
| Pawn eval success | 100% | 100% |
| Key eval success | 97.2% | >99% |
| Full-board success | 98.4% | >99.5% |
| Transit to a5 from mid-board | 0% (15 sources fail) | >95% |
| Transit to c1 from far sources | 37% (25 sources fail) | >95% |
| Descend Z offset | 11–12 mm (patch removed) | <8 mm (no patch needed) |
| Inference code complexity | Per-stage Z check (removed) | Uniform check for all stages |

---

## Scripts Created This Session

| Script | Purpose |
|--------|---------|
| `scripts/eval_all_cells_rl.py` | Full-board evaluation using `ModelEmbeddedController` (pawn/key/full modes) |
| `scripts/diagnose_transit_full.py` | Step-by-step transit episode recorder (captures limit cycles and equilibria) |
| `scripts/diagnose_descend_afile.py` | Step-by-step descend diagnosis at specific board positions |

---

## Documents Written This Session

| Document | Contents |
|----------|----------|
| `01_code_review.md` | Review of 4 model-embedding commits |
| `02_model_evaluation.md` | Standalone model performance comparison |
| `03_production_integration.md` | Cross-board position tests with ModelEmbeddedController |
| `04_implementation_plan.md` | Original fix plan |
| `05_fixes_applied_and_timeout_analysis.md` | Three bugs fixed, initial timeout analysis |
| `06_post_fix_evaluation.md` | Verified all three fixes working |
| `07_transit_timeout_analysis.md` | Deep dive: 05-24 model regression, deterministic limit cycles, HOME_POS mismatch |
| `08_descend_hover_analysis.md` | Root cause of 11–12 mm Z offset, reward structure analysis |
| `09_all_cells_evaluation.md` | ⚠ INVALID — used regressed 05-24 transit model; do not use |
| `10_training_improvement_plan.md` | Full prioritized training parameter plan |
| `11_full_board_evaluation_results.md` | Full-board eval with correct deployed models (97.2% key, 100% pawn) |
| `12_session_documentation.md` | This document |
