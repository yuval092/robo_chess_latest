# Training Improvement Plan

Date: 2026-05-24  
Based on findings in documents 07, 08, 09.

---

## Summary of Issues Found

| Issue | Severity | Root Cause | Impact |
|-------|----------|------------|--------|
| Transit oscillation at goal | CRITICAL | HOME_POS mismatch + insufficient braking | 52% timeout rate on new model |
| Transit positional bias (e–h files) | CRITICAL | Training distribution doesn't cover +X approaches | 80–100% failure in right half |
| Descend 11–12mm Z offset | LOW | Braking zone = success zone, flat reward gradient | Patched with 15mm tolerance |
| Ascend lateral drift at h-file | LOW (fixed) | Fixed by eval_drift_limit=0.010 | Was TUBE_BREACH, now resolved |

---

## Priority 1: Transit Model Retraining

The new transit model is a regression. It **must be retrained** before use in production.

### 1.1 Immediate action

Revert `configs/training.yaml`:
```yaml
deployed_models:
  transit: "checkpoints/transit_20260523_164127/final_transit.zip"
```

The 05-23 model had 96.7% embedded success rate. The 05-24 model has 48%.

### 1.2 Root cause of the regression

The 05-24 model was trained with `show_chess_pieces=False`, meaning the arm starts at a random board position each episode. In production (`show_chess_pieces=True`), the arm starts at HOME_POS = (0.88, 0.264) for every transit.

When HOME_POS starts are used, the model is evaluated on a distribution it was NOT trained on: all start positions are identical (HOME) while targets are uniformly distributed. This changes:
- The relative approach direction (always from center → target, not random → random)
- The effective start distance distribution

### 1.3 Training fix: HOME_POS start position

**Change**: Add a `force_start_pos` training mode or set `show_chess_pieces=True` during transit training.

**Option A (recommended)**: Use `show_chess_pieces=True` during transit training. This will force the arm to start from HOME_POS every episode, matching production.

In `task.py _reset_sim()`, the relevant code is:
```python
if self.force_start_pos is not None:
    arm_start_pos = self.force_start_pos.copy()
elif self.show_chess_pieces:
    arm_start_pos = self.HOME_POS.copy()   # <-- USE THIS IN TRAINING
else:
    arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
```

In `training/envs/__init__.py`, the `make_train_env` function creates the base env without `show_chess_pieces=True`:
```python
# Current (MISSING show_chess_pieces):
base = gym.make(
    "ChessFetchTask-v0",
    force_scenario=stage,
    drift_curriculum_steps=drift_curriculum_steps,
    ...
)
```

**Fix** — add `show_chess_pieces=True` for transit stage:
```python
def make_train_env(stage, drift_curriculum_steps, debug=False, fixed_drift=False):
    def _init():
        base = gym.make(
            "ChessFetchTask-v0",
            force_scenario=stage,
            drift_curriculum_steps=drift_curriculum_steps,
            debug=debug,
            fixed_drift=fixed_drift,
            # For transit: start from HOME_POS to match production
            show_chess_pieces=(stage == "transit"),
        )
        ...
```

**Option B**: Add a `home_start_prob` parameter — with probability p=0.5, start from HOME; otherwise start from random. This exposes the model to both distributions and improves generalization.

### 1.4 Training fix: Wider braking zone

**Current**: `braking_dist: 0.010` — braking activates at 10mm  
**Proposed**: `braking_dist: 0.030` — braking activates at 30mm

**Why**: The arm arrives at goal positions traveling at 90–135 mm/s (from a 200–400mm transit). The current 10mm braking zone gives only 0.1 steps of braking at 100mm/s. With 30mm braking zone, the model has 3 steps to decelerate from max speed to zero.

**Expected impact**: Models learn to decelerate earlier, arriving at the 10mm success zone with speed < 20mm/s.

### 1.5 Training fix: Stronger braking reward

**Current**: `braking_reward_weight: 0.15`  
**Proposed**: `braking_reward_weight: 0.50`

**Why**: With the wider braking zone (30mm), the velocity penalty needs to dominate the distance reward near goal. At 30mm with current 0.15 weight:
- Distance reward: `-1.0 × 0.030 = -0.030`
- Braking penalty: `-0.15 × 0.100 = -0.015` (at 100mm/s)
- Total: -0.045

The distance reward is still the dominant term. With 0.50 weight:
- Braking penalty: `-0.50 × 0.100 = -0.050` (at 100mm/s)
- This makes velocity the dominant concern within 30mm.

### 1.6 Training fix: Anti-jitter penalty increase

**Current**: `jitter_penalty_weight: 0.003`  
**Proposed**: `jitter_penalty_weight: 0.010`

**Why**: The oscillating model outputs large actions (magnitude 0.6–0.8 per axis) on alternating steps. These are large jitter actions that should be heavily penalized. The current penalty at magnitude=0.8:
- `0.003 × (0.8² + 0.8² + 0.1²) = 0.003 × 1.29 = 0.0039`

This is too small to discourage the oscillation (which saves 0.045/step in braking penalty). With 0.010:
- `0.010 × 1.29 = 0.013`

Still not enough to break the cycle, but combined with the wider braking zone it should prevent the oscillation from being a stable strategy.

### 1.7 Additional fix: Goal-position diversity in training

The training env samples random board positions for both start and goal. The new model may have learned to avoid certain XY coordinates. 

**Change**: Verify that `_sample_board_position()` gives uniform coverage over the full board including h-file (X=1.16), and increase `num_envs` to ensure sufficient coverage.

Current `_sample_board_position()` samples within `edge_margin=0.04` of the board. This should cover h-file (X=1.16 is within bounds). No change needed here.

### 1.8 Training fix summary (transit)

| Parameter | Current | Proposed | Location |
|-----------|---------|----------|----------|
| `show_chess_pieces` in training | False | True | `train_rl.py` or `transit_env.py` |
| `braking_dist` | 0.010 | 0.030 | `configs/env.yaml` |
| `braking_reward_weight` | 0.15 | 0.50 | `configs/env.yaml` |
| `jitter_penalty_weight` | 0.003 | 0.010 | `configs/env.yaml` |
| `stability_vel_threshold` | 0.020 | 0.020 (no change) | — |
| Training steps | 600K | 600K (no change needed) | — |

**Expected improvement**: With HOME_POS starts and stronger braking, the model should achieve >95% embedded success rate across all 64 squares.

---

## Priority 2: Descend Model Retraining (Low Priority)

The current descend model has a 12mm Z offset above HOVER_Z. It is **functionally correct** with the 15mm tolerance patch. Retraining is not urgent but should be done to eliminate the inference patch.

### 2.1 Training fix: Decouple braking distance

**Current**: `braking_dist: 0.010` (same as success_threshold)  
**Proposed**: `braking_dist: 0.025` for descend (separate parameter)

**Implementation**: Add `descend_braking_dist: 0.025` to `env.yaml` and use it in `task.py step()` for descend scenario only.

### 2.2 Training fix: Final approach bonus

Add a small bonus reward when Z is within 5mm of HOVER_Z:
```python
# In task.py step(), inside else (not crashed):
if self.current_scenario == "descend":
    hover_z = self.env_cfg.get("hover_z", 0.460)
    if abs(grip_pos[2] - hover_z) < 0.005:
        reward += self.env_cfg.get("final_approach_bonus", 5.0)
```

Add to `env.yaml`:
```yaml
final_approach_bonus: 5.0
```

**Why 5.0**: Success bonus is 500. Final approach is 1% of success bonus. This is small enough not to create a "hover just above" equilibrium but large enough to pull the model to exactly HOVER_Z.

### 2.3 Training fix: Tighter descend success criterion

**Change**: Add `descend_success_threshold: 0.006` to `env.yaml`.

In `task.py step()`, for descend:
```python
if self.current_scenario == "descend":
    thresh = self.env_cfg.get("descend_success_threshold", 
                               self.env_cfg.get("success_threshold", 0.010))
    is_near = bool(d_xy < thresh and d_z < thresh)
else:
    is_near = bool(self._is_success(grip_pos, self.goal_pos))
```

**Training curriculum**: Start with `descend_success_threshold: 0.010`, anneal to 0.006 after 400K steps.

### 2.4 Descend retraining summary

| Parameter | Current | Proposed | Location |
|-----------|---------|----------|----------|
| `descend_braking_dist` | (new) | 0.025 | `configs/env.yaml` + `task.py` |
| `braking_reward_weight` | 0.15 | 0.30 | `configs/env.yaml` |
| `final_approach_bonus` | (new) | 5.0 | `configs/env.yaml` + `task.py` |
| `descend_success_threshold` | (new) | 0.006 | `configs/env.yaml` + `task.py` |

**Keep inference patch** until retrained model is deployed and validated.

---

## Priority 3: Ascend Model Improvements (Very Low Priority)

The ascend model is currently working correctly (100% success rate, eval_drift_limit=10mm fixes all h-file issues). No retraining needed.

If future retraining is done, consider:
- Adding an explicit lateral containment reward: `-xy_drift_weight × drift` where drift = XY error from tube center
- This would reduce the 8mm h-file drift without requiring the 10mm inference patch

### 3.1 Training fix for lateral drift (optional, for next ascend retrain)

**Change**: Add lateral drift penalty during ascend:
```python
# In task.py step(), for ascend:
if self.current_scenario == "ascend" and self.tube_center_xy is not None:
    drift = float(np.linalg.norm(grip_pos[:2] - self.tube_center_xy))
    if drift > self.env_cfg.get("drift_limit_end", 0.008):
        reward -= self.env_cfg.get("ascend_drift_penalty_weight", 2.0) * drift
```

**Expected impact**: The model learns to stay vertically above the starting XY during ascent, reducing h-file drift from 8mm to <5mm.

---

## Priority 4: Cross-Stage Reward Consistency

When all three models are retrained, verify that these parameters are shared/consistent:

| Parameter | Transit | Descend | Ascend |
|-----------|---------|---------|--------|
| `success_threshold` | 0.010 | 0.006 (new) | 0.010 |
| `stability_vel_threshold` | 0.020 | 0.020 | 0.020 |
| `braking_dist` | 0.030 (new) | 0.025 (new) | 0.015 (new, optional) |
| `braking_reward_weight` | 0.50 (new) | 0.30 (new) | 0.15 (no change) |
| `jitter_penalty_weight` | 0.010 (new) | 0.003 (no change) | 0.003 (no change) |

---

## Implementation Checklist

### Phase 1: Immediate (today)
- [x] Revert `configs/training.yaml` transit model to `transit_20260523_164127`
- [ ] Verify production pipeline works with reverted model

### Phase 2: Parameter changes for next transit training
1. `configs/env.yaml`: Change `braking_dist: 0.010` → `braking_dist: 0.030`
2. `configs/env.yaml`: Change `braking_reward_weight: 0.15` → `braking_reward_weight: 0.50`
3. `configs/env.yaml`: Change `jitter_penalty_weight: 0.003` → `jitter_penalty_weight: 0.010`
4. `scripts/train_rl.py` or `training/envs/transit_env.py`: Add `show_chess_pieces=True` for transit training

### Phase 3: Train new transit model
```bash
PYTHONPATH=. python scripts/train_rl.py --stage transit --total-timesteps 600000
```
Validate with:
```bash
PYTHONPATH=. python scripts/eval_stages.py --use-rl-models --stages transit --n-episodes 50
PYTHONPATH=. python scripts/eval_all_cells_rl.py --mode pawn --test-mode transit_only --n-reps 3
```
Pass criteria: >95% pawn-test success across all 48 pairs.

### Phase 4: Optional descend retrain
1. Add `descend_braking_dist: 0.025` to `env.yaml`
2. Add `final_approach_bonus: 5.0` to `env.yaml`
3. Implement in `task.py step()` (see §2 above)
4. Retrain descend model
5. If equilibrium offset < 6mm: remove `descend_success_z_tolerance` patch

### Phase 5: Full regression suite
```bash
PYTHONPATH=. python scripts/eval_stages.py --use-rl-models --stages transit,descend,ascend --n-episodes 50
PYTHONPATH=. python scripts/eval_all_cells_rl.py --mode key --test-mode three_stage --n-reps 3
PYTHONPATH=. python scripts/eval_chess_game_flow.py --use-rl-models --verify-agreement
```

---

## Expected Outcomes After Retraining

| Metric | Before (05-24 model) | After (improved training) |
|--------|----------------------|---------------------------|
| Transit embedded success | 48% | >95% |
| e–h file transit success | 0–50% | >90% |
| Short-move (1 rank) success | 0–30% | >90% |
| Descend Z offset | 11–12mm | <6mm |
| Descend inference patch | Required | Can be removed |
| Ascend drift at h-file | 8mm (patched) | <5mm (no patch needed) |
| Full-board 3-stage success | ~40% | >90% |
