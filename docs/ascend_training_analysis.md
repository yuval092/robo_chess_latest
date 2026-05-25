# Ascend RL Training Analysis — RoboChess

**Date:** 2026-05-25  
**Purpose:** Full account of every ascend training attempt, evaluation results, failure modes, and open hypotheses. Written for a fresh AI agent with no prior context.

---

## 1. Project Context

The robot arm moves chess pieces using a 6-stage pipeline:

```
transit → descend → grasp → ascend → place → transit-home
```

- **Transit:** Move arm above the target square (XY only, at SAFE_Z height)
- **Descend:** Lower from SAFE_Z → HOVER_Z → GRASP_Z to pick up piece
- **Grasp:** Close gripper on piece (fully scripted, not RL)
- **Ascend:** Lift piece from HOVER_Z up to SAFE_Z
- **Place:** Lower piece onto destination square (scripted)
- **Transit-home:** Return arm to board center (RL)

Transit and descend RL models are **solved and deployed at 100%**. All current work is on **ascend only**.

### Key Heights (from `configs/env.yaml`)

| Name | Z value | Meaning |
|------|---------|---------|
| `table_surface_z` | 0.400 m | Table top |
| `grasp_z` | 0.430 m | Gripper closes here |
| `hover_z` | 0.460 m | Safe stop before grasp; ascend starts here |
| `safe_z` | 0.530 m | Target height for ascend; arm travels between moves at this height |

**Ascend task:** Lift from `hover_z` (0.460 m) to `safe_z` (0.530 m) = **70mm vertical travel**, while keeping XY within 10mm of the starting column at all times.

---

## 2. The RL Approach

### Algorithm: SAC (Soft Actor-Critic) with transfer learning

All models start from pretrained weights:  
`archive/rl_system/models/sac-FetchPickAndPlace-v4.zip`

This is a MuJoCo FetchPickAndPlace model. Transfer works because both tasks share a similar observation space (the "Phase-9 trick": in the 25D FetchPickAndPlace observation, we set `object_pos = grip_pos` to simulate a held object).

### Training Environment (`training/envs/ascend_env.py`)

- Inherits from the base task (`src/chess_env/task.py`)
- At reset: arm placed at a random board square at `hover_z`, goal set to `safe_z` directly above
- At each step: arm applies a velocity/force action; environment checks drift and success
- **Drift tube:** A cylindrical constraint around the starting XY. If the arm drifts beyond `drift_limit` in XY, the episode crashes with `TUBE_BREACH` (reward = -500)
- **Success:** Arm reaches within `success_threshold` (10mm) of `safe_z` in both XY and Z

### Drift Curriculum (`configs/env.yaml`)

```yaml
drift_limit_start: 0.100        # 100mm at episode 0 — very wide tube
drift_limit_end: 0.008          # 8mm at curriculum end — stricter than eval's 10mm
drift_curriculum_steps: 30000   # Per-worker steps to reach drift_limit_end
```

With 10 parallel training envs: total curriculum steps = 30,000 × 10 = **300,000 total steps** (halfway through 600K training). After that, all remaining training uses the strict 8mm tube.

**Assumption:** Tightening the tube gradually forces the model to learn XY precision without crashing early in training when it still needs to explore.

### VERTICAL_QUAT Enforcement

`VERTICAL_QUAT` is the wrist orientation quaternion for a perfectly vertical arm. MuJoCo's `reset_mocap2body_xpos()` (called inside `_set_action()`) copies the *physical* wrist orientation back into `mocap_quat`, which causes wrist drift to accumulate across steps.

Fix: after every `_set_action()` call, override:
```python
env.data.mocap_quat[0][:] = env.VERTICAL_QUAT
```

This must be done in **both training** (`task.py` step, line ~1273) **and inference** (`model_controller.py` `_run_stage()`, line ~340).

**If training enforces VQ but inference does not (or vice versa), the policy experiences different dynamics between training and deployment, which can cause unexpected drift at certain board positions.**

---

## 3. Evaluation Tools — Critical Differences

This is extremely important. The three evaluators give different numbers for the same model. Understanding why requires knowing what each one actually does.

### 3a. Training Eval Callback (built into SB3 EvalCallback)

- Runs every 50,000 training steps
- 20 episodes using the `make_eval_env()` environment
- Uses `force_drift_limit = eval_drift_limit = 0.010m` (strict 10mm tube)
- **Known to be unreliable:** Episodes terminate in 2–17 steps in our runs. Physically, lifting 70mm in 5 steps is implausible under realistic arm dynamics. The cause is likely a distribution mismatch between the eval env's reset position and production (exact HOVER_Z above source square). Numbers from this tool should be treated as qualitative trend indicators only, not ground truth.

### 3b. `scripts/eval_stages.py`

- Runs N independent episodes of a single stage (e.g., ascend only)
- Uses `ModelEmbeddedController._run_stage()` — the **production inference path**
- Random starting positions sampled from board squares
- More reliable than the training callback
- **Limitation:** Random positions may over-sample easy center squares. Corner squares may be under-represented, causing the reported success rate to be optimistic relative to real chess games (which frequently involve corner squares).

### 3c. `scripts/eval_all_cells_rl.py --mode key`

- Runs the **full 3-stage pipeline** (transit → descend → ascend) for 216 specific chess move pairs
- 5 repetitions per pair = **1,080 total episodes**
- The 216 pairs are chosen to cover all key board positions (corners, edges, center)
- Ascend starts from wherever descend actually left the arm — this captures real-world compounding errors
- **This is the ground truth evaluator.** All deployment decisions are based on this.

---

## 4. Version History

### Version 0 — Scripted-only Baseline

Before any RL was involved, the scripted arm (no RL for any stage) achieved **100% on all moves** including VERTICAL_QUAT enforced. This is the reference point.

---

### Version 1 / v2 — Original Ascend Model

**Checkpoint:** `checkpoints/ascend_20260523_222849/best_model_ascend.zip`

**Training conditions:**
- VERTICAL_QUAT was **NOT enforced** in `task.py` during training (the code didn't have it yet)
- Braking distance: default/untuned
- Number of envs: unknown (earlier training run)
- Transfer from FetchPickAndPlace weights

**Results:**

| Evaluator | Result | Notes |
|-----------|--------|-------|
| Full board eval, 8,916 eps, **without** VQ at inference | **100%** | VQ not enforced in model_controller.py yet |
| Key board eval, 1,080 eps, **with** VQ at inference | **97.2%** | After VQ enforcement was added |

**Failure pattern with VQ enforcement:**
- a4 and a5 as **destination squares** fail 0/5 (deterministic)
- Failure mode: TUBE_BREACH at 11–13mm drift during ascend
- All other squares pass

**Root cause hypothesis:**
The model was trained without VQ, meaning the wrist was free to rotate slightly during training. The policy may have implicitly relied on small wrist micro-adjustments as part of its ascent strategy. When VQ is clamped at inference (wrist locked vertical every step), this conflicts specifically at the a4/a5 positions — possibly because the arm geometry at those board coordinates creates a torque that the wrist rotation was silently counteracting during training.

**Why only a4/a5 specifically:** This is still an open question. One hypothesis is that these squares sit at a particular combination of reach distance and angle from the robot's base where the wrist torque is highest. Another hypothesis is that the specific random seed during training happened to underrepresent those squares, leaving the policy less robust there.

**Current status of v2:** Still the best-performing model in production (97.2% on key eval). Currently **not deployed** because training.yaml points to v4 final.

---

### Version 3 (v3)

**Checkpoint:** `checkpoints/ascend_20260525_103829/final_ascend.zip`

**Why we made v3:** To train WITH VERTICAL_QUAT in task.py so that training matches inference. Also to add proper braking distance tuning.

**Training config changes from v2:**
```yaml
ascend_braking_dist: 0.030       # 30mm braking zone (NEW — was untuned)
ascend_braking_reward_weight: 0.20
jitter_penalty_weight: 0.005
drift_curriculum_steps: 30000    # per worker × 10 envs = 300K total
```

**VERTICAL_QUAT in task.py:** YES (enforced in training)  
**VERTICAL_QUAT in model_controller.py at inference:** YES

**Training eval callback (unreliable):**
- At some point showed ~100% in 5.1 steps — same short-episode artifact. Ignored.

**eval_stages.py (100 episodes):**

| Checkpoint | Success | Crash (TUBE_BREACH) | Timeout |
|---|---|---|---|
| best_model (~300K steps) | 14% | 33% | 53% |
| latest (~550K steps) | 72% | 24% | 4% |
| **final (600K steps)** | **80%** | 19% | 1% |

**Key board eval, 1,080 eps (final model): 66.7% (720/1,080)**
- All 360 failures: ascend stage
- Failure pattern: many (source → destination) pairs fail deterministically 0/5
- Arm reaches 14–19mm from SAFE_Z and times out (does not reach the goal, doesn't crash — just never closes the final gap)

**Root cause: Braking Equilibrium at 19mm**

The `ascend_braking_reward_weight` penalizes velocity when the arm is within `ascend_braking_dist` (30mm) of SAFE_Z. The intent is to prevent overshoot. However, the RL agent found an unintended equilibrium:

Inside the 30mm braking zone, the velocity penalty gradient and the distance-reward gradient cancel out at approximately 19mm from SAFE_Z. The arm learned to decelerate into this equilibrium point and park there — never reaching the 10mm success zone.

This is not a timeout caused by slowness. It is a convergence to a wrong equilibrium point. The arm is "done" from its perspective, but 19mm short of the actual goal.

**Evidence:** TUBE_BREACH crashes in eval_stages.py showed drift of 10–13mm (just over 10mm threshold). The arm was ascending but its XY column wandered 1–3mm past the limit while trying to reach the parked equilibrium.

**Why didn't this appear in the training eval callback?** The training eval callback uses a different reset distribution where the arm may already start closer to SAFE_Z, making the 19mm parking position fall inside the success zone by accident. This is the core artifact that makes the training eval unreliable.

---

### Version 4 (v4) — Current

**Checkpoint:** `checkpoints/ascend_20260525_133712/final_ascend.zip`  
**Currently deployed in `configs/training.yaml`**

**Why we made v4:** Fix the braking equilibrium by narrowing the braking zone to 12mm (just above the 10mm success threshold). This means any equilibrium point the model finds inside the braking zone is automatically inside the success zone, so it gets counted as a success.

**Training config changes from v3:**
```yaml
ascend_braking_dist: 0.012       # was 0.030 — narrowed to just above success threshold
ascend_braking_reward_weight: 0.10  # was 0.20 — gentled to avoid creating new equilibrium
jitter_penalty_weight: 0.005     # same
drift_curriculum_steps: 30000    # same
```

**VERTICAL_QUAT in task.py:** YES  
**VERTICAL_QUAT in model_controller.py at inference:** YES

**Training eval callback (observed pattern, unreliable for ground truth):**

| Timestep | Eval success (n=20) | Rollout SR | Notes |
|---|---|---|---|
| 50K | 25% | ~0.55 | Early phase |
| 100K–200K | 0% | 0.12–0.22 | Curriculum tightening — expected |
| 250K | 65% | 0.83–0.88 | Policy snaps into shape as curriculum nears end |
| 300K | **100%** | ~0.70 | Best model saved here |
| 350K–400K | 100% | ~0.70 | Stable |
| 450K | 90% | 0.62 | Beginning of decline |
| 500K | 95% | ~0.65 | Slight recovery |
| 550K | 80% | 0.62 | Continued decline |
| 600K | 80% | 0.62 | Training ends |

**eval_stages.py (100 episodes each):**

| Checkpoint | Success | Crash (TUBE_BREACH) | Timeout | AvgErr | P95Err |
|---|---|---|---|---|---|
| best_model (300K) — run 1 | 27% | 36% | 37% | 5.8mm | 10.3mm |
| best_model (300K) — run 2 | 43% | 35% | 22% | 6.7mm | 10.7mm |
| latest (~400K) | 39% | 16% | 45% | 7.8mm | 11.2mm |
| **final (600K)** | **53%** | 44% | 3% | 10.4mm | 13.2mm |

Note: Two runs of best_model gave 27% and 43% — this is sampling noise at n=100 (true rate likely ~35%). All TUBE_BREACH crashes occur at 10–14mm (0–4mm over the threshold).

**Key board eval, 1,080 eps (final model): 34.3% (370/1,080)**

All 710 failures are in ascend. Failure pattern by destination square:

```
Destination success map:
       a     b     c     d     e     f     g     h
  8     0%    0%   --   --   --   --    0%    0%
  5    71%   --   --  100%    0%   --   --    0%
  4     0%   --   --  100%  100%   --   --    0%
  1   100%  100%  --   --   --   --    0%    0%
```

- Corners and edges: 0% success as destination
- Center (d4, d5, e4, e5 partial): 100% success
- Worst source squares: a1, b1 (21%); d4, e4, d5, e5 (25%)

---

## 5. The Central Mystery: Why Is v4 Worse Than v2?

This is the most important open question. Here is a rigorous breakdown of hypotheses.

### v2 vs v4 comparison

| Property | v2 (original) | v4 (retrain) |
|---|---|---|
| Trained with VERTICAL_QUAT | No | Yes |
| Inference with VERTICAL_QUAT | Yes | Yes |
| Braking zone | Untuned | 12mm |
| Key board eval | **97.2%** | **34.3%** |

Training v4 WITH VQ should have been better than v2 (which was trained without VQ). Instead it is dramatically worse. This is counterintuitive and suggests something unexpected.

### Hypothesis A: Training Distribution Mismatch (most likely)

The training eval callback consistently shows 2–17 step episodes. This is physically implausible for a 70mm ascent under real arm dynamics. It strongly suggests the training environment's reset function places the arm much closer to SAFE_Z than production does.

If the model learned to ascend from a distribution of starting positions that is shifted upward (closer to SAFE_Z), it would appear to work perfectly in training but fail in production where it always starts exactly at HOVER_Z (70mm below SAFE_Z).

This mismatch is the most likely explanation for the large gap between training eval (100%) and production eval (34–53%).

**Why didn't this affect v2?** Possibly because v2 used a different training setup or fewer envs, and the policy happened to be more general. Or the pre-trained FetchPickAndPlace weights provide a strong prior for lifting that covers the full 70mm range regardless of the distribution mismatch.

**What to investigate:** Print the actual reset position in the ascend training env and compare it to `hover_z` (0.460m). If they differ, fix the reset function.

### Hypothesis B: 12mm Braking Zone Is Too Weak

V3's 30mm braking zone caused equilibrium at 19mm. V4's 12mm zone was supposed to fix this by making any equilibrium fall inside the success zone.

However, the v4 final model shows 44% TUBE_BREACH in eval_stages.py — all at 10–12mm. This means the arm is drifting XY while ascending, not stalling in Z. The braking zone only affects the velocity penalty in Z, not XY drift. So the braking zone change may have been solving the wrong problem.

The XY drift problem (arm tilting sideways during ascent) may require a stronger `xy_reward_weight` or a different approach entirely.

### Hypothesis C: VQ Enforcement Constrains the Policy Too Tightly

V2 was trained without VQ (wrist free) and performed at 97.2% even when VQ is enforced at inference. V4 was trained with VQ enforced every step and performs at 34.3%.

One possibility: with VQ enforced every step during training, the optimization landscape is harder — the arm can't use any wrist rotation to compensate for arm dynamics, so the optimizer finds a narrower, more fragile policy. V2's training allowed the wrist to move freely, producing a policy with more degrees of freedom that happened to generalize better.

**This is the most surprising hypothesis.** It would mean "train with the constraint you'll use at inference" is not always better — sometimes removing constraints during training produces more robust policies.

### Hypothesis D: Eval_stages.py Over-Estimates Performance

The 53% figure from eval_stages.py samples random board positions. The key board eval uses a predetermined set of 216 pairs heavily weighted toward corners and edges (because those are the most common hard moves in chess). If the v4 model genuinely achieves 80%+ on center squares but 0% on corners, the random sampling in eval_stages.py would report an average that looks better than real-game performance.

This doesn't explain WHY corners fail, but it explains the 53% → 34.3% gap between the two evaluators.

---

## 6. Comparison Table: All Ascend Versions

| Metric | v2 (original) | v3 (30mm braking) | v4 (12mm braking) |
|---|---|---|---|
| Trained with VQ | No | Yes | Yes |
| Inference with VQ | Yes | Yes | Yes |
| `ascend_braking_dist` | untuned | 30mm | 12mm |
| `ascend_braking_reward_weight` | untuned | 0.20 | 0.10 |
| `jitter_penalty_weight` | unknown | 0.005 | 0.005 |
| num_envs | unknown | 10 | 10 |
| eval_stages.py (final) | not run | 80% | 53% |
| **Key board eval (final)** | **97.2%** | **66.7%** | **34.3%** |
| Failure mode | a4/a5 deterministic 0/5 | Arms parks at 19mm (timeout) | Corner squares 0%, center 100% |
| Root cause | VQ train/inference mismatch | Braking equilibrium at 19mm | Unknown — possibly reset distribution |

---

## 7. Current State of Deployed Models

```yaml
# configs/training.yaml
deployed_models:
  transit: "checkpoints/transit_20260524_142519/best_model_transit.zip"
  # NOTE: best_model, NOT final — transit final collapsed 100%→30% at 600K steps
  
  descend: "checkpoints/descend_20260524_142540/final_descend.zip"
  # NOTE: final, NOT best_model — descend best_model (450K) oscillated; final converged
  
  ascend: "checkpoints/ascend_20260525_133712/final_ascend.zip"
  # v4 final — 34.3% key board eval. SHOULD LIKELY BE REVERTED to v2 (97.2%)
```

### Recommendation: Revert Ascend to v2

`checkpoints/ascend_20260523_222849/best_model_ascend.zip`

- v2 achieves 97.2% on key board eval — by far the best of all versions
- The only failures are a4 and a5 as destination squares (deterministic 0/5)
- These 2 squares are 30/1080 episodes (2.8% failure rate)
- Before attempting another full retrain, it is worth investigating whether the a4/a5 failures can be fixed with a targeted approach (see Section 8)

---

## 8. Open Problems and Suggested Next Steps

### 8a. Investigate the Reset Distribution Bug (High Priority)

The training eval callback showing 2–17 step episodes is a red flag. The ascend task should require ~50–150 steps to lift 70mm under realistic physics. If the eval and/or training envs are resetting the arm at a position closer to SAFE_Z than HOVER_Z, the model is training on the wrong distribution.

**Action:** In `training/envs/ascend_env.py` (or wherever `reset()` is defined for ascend training), print the actual reset Z position and compare to `hover_z = 0.460`. If different, this is the root cause of v4's production failure.

### 8b. Understand the a4/a5 Failure in v2 (High Priority)

V2 fails deterministically at a4 and a5 as destination squares (TUBE_BREACH at 11–13mm). Before retraining again, understand exactly what is different about these squares:

- What is the XY position of a4 and a5 in world coordinates?
- What is the arm configuration (joint angles) when ascending from a4/a5?
- Does the failure occur at a specific point in the ascent (beginning, middle, top)?
- Is there a wrist torque or joint limit interaction at these positions?

**Action:** Run eval with debug logging enabled for a4→X and X→a4 pairs to see the exact trajectory of drift as a function of ascent height.

### 8c. Consider Retraining with Corrected Reset Distribution

If Hypothesis A (reset distribution mismatch) is confirmed, retrain with:
1. Fixed reset: always start exactly at `hover_z` (not a sampled position)
2. VQ enforced in training (keep this — it's correct in principle)
3. Braking: keep 12mm (fixes v3's equilibrium bug)
4. Increase `xy_reward_weight` to penalize lateral drift more aggressively

### 8d. Consider Removing VQ Enforcement from Ascend Training Only

If Hypothesis C is confirmed (VQ constraint degrades policy quality), consider whether VQ should be enforced during ascend training. The evidence from v2 (trained without VQ, 97.2% with VQ at inference) suggests the policy is more robust without the training constraint.

**Risk:** This was the original state that caused the a4/a5 failures. However, 97.2% is still much better than v4's 34.3%.

### 8e. Do Not Retrain Transit or Descend

Both are performing at 100% in production. Any retrain risks regression. Leave them.

---

## 9. Key Config Parameters (Current State)

All values from `configs/env.yaml` as of 2026-05-25:

```yaml
# Heights
hover_z: 0.460          # Ascend start
safe_z: 0.530           # Ascend goal (70mm travel)

# Success threshold (shared by all stages)
success_threshold: 0.010  # 10mm in XY and Z

# Drift tube (training)
drift_limit_start: 0.100  # 100mm wide at start of curriculum
drift_limit_end: 0.008    # 8mm strict at end (stricter than eval's 10mm)
drift_curriculum_steps: 30000  # per worker; 30K × 10 envs = 300K total steps

# Drift limit (evaluation / production)
eval_drift_limit: 0.010   # 10mm; TUBE_BREACH triggered at this in model_controller.py

# Ascend braking (v4 config — in use)
ascend_braking_dist: 0.012           # 12mm zone
ascend_braking_reward_weight: 0.10   # gentle velocity penalty

# Reward shaping
xy_reward_weight: 2.0        # lateral drift penalty multiplier
z_reward_weight: 1.5         # vertical accuracy reward multiplier
jitter_penalty_weight: 0.005 # smoothness penalty
dist_reward_weight: 1.0      # distance-to-goal reward multiplier
crash_penalty: -500.0        # TUBE_BREACH or floor crash
success_bonus: 500.0         # success reward
```
