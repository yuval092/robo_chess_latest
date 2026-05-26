# Training

## Overview

Each of the three movement specialists (transit, descend, ascend) is trained independently using SAC fine-tuning from the pretrained `FetchPickAndPlace-v4` checkpoint. Training is launched via:

```bash
robo-chess-train train --stage transit
robo-chess-train train --stage descend
robo-chess-train train --stage ascend
```

After training, update `configs/deployed_models.yaml` with the checkpoint path,
then verify stage accuracy with `robo-chess-eval-stage`.

---

## Why SAC?

Soft Actor-Critic (SAC) is an off-policy, maximum-entropy RL algorithm. It is well-suited to this task because:

1. **Continuous action space**: The arm's 4-D action (Δx, Δy, Δz, gripper) is continuous.
2. **Sample efficiency**: Off-policy learning with replay buffer allows reuse of past experience.
3. **Entropy regularisation**: The maximum-entropy objective encourages exploration and prevents premature convergence to a suboptimal deterministic policy.
4. **Works well with dense rewards**: SAC's actor-critic architecture converges reliably with the dense distance-shaping rewards used here.
5. **Compatible with stable-baselines3**: Well-tested implementation with MultiInputPolicy for dict observation spaces.

---

## Training Infrastructure

### `SACTrainer` (`training/trainer.py`)

Central class orchestrating a full training run for one stage.

```python
trainer = SACTrainer(stage="transit", num_envs=4, debug=False)
trainer.train(model_path="models/pretrained/sac-FetchPickAndPlace-v4.zip",
              save_dir="checkpoints/transit_20260524_142519/")
```

**Training loop:**
1. Create N parallel training environments via `SubprocVecEnv` (or `DummyVecEnv` if N=1).
2. Create one evaluation environment via `DummyVecEnv`.
3. Load the base model with overridden hyperparameters (`custom_objects`).
4. Reset entropy coefficient to `initial_ent_coef` (prevents starting with zero entropy).
5. Call `model.learn(total_timesteps, callbacks=...)`.
6. Save `final_{stage}.zip` to `save_dir`.

### Parallelism

With `num_envs=4` (the default), `SubprocVecEnv` spawns 4 worker processes using `fork`. Each worker runs its own independent `ChessTaskEnv` instance with its own random seed. Experience from all workers is merged into a shared replay buffer. The effective timestep rate is approximately 4× single-env throughput.

**Note from training history:** Using 2 environments (as in the v2 ascend model) can produce better-generalising models than 10+ environments, likely because fewer parallel environments means each worker gets more varied experience relative to the model update frequency. The v5 ascend training uses 10 environments for speed but matches v2's 1M total timestep budget.

### Evaluation Environment

A separate `DummyVecEnv` with `force_drift_limit = eval_drift_limit (10 mm)` evaluates the policy every `eval_freq` (50K) steps. The strict drift limit ensures evaluation matches production conditions.

---

## Environment Factory (`training/envs/__init__.py`)

### `make_train_env(stage, drift_curriculum_steps, debug, fixed_drift)`

Returns a factory function (not the env itself) suitable for `SubprocVecEnv`. Each call to the factory:
1. Creates `gym.make("ChessFetchTask-v0", force_scenario=stage, drift_curriculum_steps=..., ...)`
2. Wraps with the stage-specific wrapper (`TransitTrainEnv`, `DescendTrainEnv`, `AscendTrainEnv`)
3. Wraps with `Monitor` for episode stats collection

### `make_eval_env(stage, eval_drift_limit, debug)`

Same structure, but with `force_drift_limit=eval_drift_limit` and no drift curriculum.

---

## Drift Curriculum

The drift curriculum progressively tightens the tube constraint during training:

```
drift_limit(t) = drift_limit_start - (drift_limit_start - drift_limit_end) * min(t / curriculum_steps, 1.0)
```

| Parameter | Value | Meaning |
|---|---|---|
| `drift_limit_start` | 100 mm | Very loose — model only needs to move in the Z direction |
| `drift_limit_end` | 8 mm | Production-tight — must stay within 8 mm of the tube centre |
| `drift_curriculum_steps` | 30,000 per worker | Steps at which the curriculum completes |

With 4 workers and 30K steps/worker, the curriculum completes at 120K total timesteps. The remaining ~480K steps (out of 600K) train at the strict limit, consolidating generalisation.

**Curriculum rationale:** If training starts with the strict 8 mm limit, the policy immediately crashes on every episode (tube breach) and receives only −500 crash penalties. The curriculum allows the model to first learn Z-axis movement (transit horizontally, descend/ascend vertically) before being required to maintain lateral precision.

`fixed_drift=True` skips the curriculum entirely and uses `drift_limit_end` from step 0. Useful for resume training from a checkpoint that already reached the strict limit.

---

## Reward Function (Recap)

```
r = -dist_weight * ||grip_pos - goal_pos||         # primary: reduce distance
  - z_weight    * |grip_z - goal_z|                # secondary: Z precision
  - xy_weight   * ||grip_xy - goal_xy||            # secondary: XY precision
  - brake_weight * speed     (if dist < brake_dist) # velocity near goal
  - jitter_weight * ||action[:3]||^2               # smooth actions
  + floor_penalty             (if near floor)
```

On crash: `r = crash_penalty (-500)`
On success: `r += success_bonus (+500)`

The success bonus of +500 dominates any achievable episode reward, creating a strong incentive to reach the goal stably. The crash penalty discourages any boundary-violating trajectory.

**Braking reward:** When the gripper is within `braking_dist` of the goal, a velocity penalty is applied. This discourages the arm from rushing past the goal at high speed and encourages the policy to decelerate for a stable hold. The braking distance and weight differ per stage (see [01_mujoco_environment.md](01_mujoco_environment.md#reward-function)).

**Jitter penalty:** Penalises high-norm actions, discouraging oscillatory or noisy movement. Weight of 0.003 was retained from the v2 baseline.

---

## Callbacks

### `DetailedLoggingCallback` (`training/callbacks.py`)

Logs rolling statistics every `log_freq` (2000) steps to `logs/training_progress.log`:
- Rolling mean reward over `moving_avg_window` (100) episodes
- Rolling mean episode length
- Rolling success rate

### `SuccessRateEvalCallback`

Extends `EvalCallback` (stable-baselines3). After each evaluation run:
1. Saves `latest_model_{stage}.zip` — always updated
2. If current success rate > historical best: saves `best_model_{stage}.zip`

The best-model checkpoint is what should typically be deployed, though for descend the `final_{stage}.zip` has historically performed better than `best_model` (see training lessons).

---

## Hyperparameter Choices

### Learning Rate: 5e-5

Very conservative compared to typical SAC runs (1e-3). This preserves the pretrained features in the early layers. Aggressive learning rates would quickly overwrite the pretrained weights and require many more steps to re-learn movement skills from scratch.

### Target Entropy: -4.0

For a 4-D action space, `target_entropy = -dim(action) = -4` is the standard SB3 heuristic. With the entropy reset to `initial_ent_coef=0.1`, this means the policy starts with significant randomness and gradually becomes more deterministic as training progresses.

### Buffer Size: 300,000

Holds approximately 300K environment steps of experience. With a 600K total budget, the buffer covers the second half of training with historical data, reducing correlation in mini-batches.

### Batch Size: 512

Larger than default (256). Larger batches reduce gradient variance and are more suitable for the dense, smooth rewards used here.

---

## Saved Artifacts

After training, the following files exist in `checkpoints/{stage}_{timestamp}/`:

| File | Description |
|---|---|
| `best_model_{stage}.zip` | Highest success-rate checkpoint during training |
| `latest_model_{stage}.zip` | Most recent checkpoint |
| `final_{stage}.zip` | Model at the end of full training |

The timestamp format is `YYYYMMDD_HHMMSS`. Logs are written to `logs/{stage}_{timestamp}/tensorboard/`.

---

## Evaluating a Trained Model

After training, run the evaluation suite before deploying:

```bash
# Per-stage accuracy (20 episodes each)
robo-chess-eval-stage --stage all --episodes 20

# Targeted hard-cell coverage (corners, edges, known-bad destinations)
robo-chess-eval-flow --mode complex

# Full pick-and-place sequence check
robo-chess-eval-stage --stage full_move --episodes 20
```

Results are reported as success rate, crash breakdown, and per-stage timing.

**Warning from training experience:** The callback-level evaluation during training is unreliable. Episodes are 2–17 steps long (an artifact of the evaluation environment), producing inflated success rates. Always use `robo-chess-eval-stage` and `robo-chess-eval-flow` with the production checkpoint for ground-truth evaluation.

---

## Training Lessons Learned

The following hard-won lessons were accumulated across multiple training runs:

### 1. Use the Right Checkpoint

- **Transit**: `best_model_transit.zip` outperforms `final_transit.zip`.
- **Descend**: `final_descend.zip` outperforms `best_model_descend.zip`.
- **Ascend**: Check both; performance varies between runs.

Do not assume one rule applies to all stages.

### 2. Callback Evaluation Is Unreliable

The `SuccessRateEvalCallback` reports artificially high success rates during training (episodes end too quickly). Always evaluate with `robo-chess-eval-stage` + `robo-chess-eval-flow --mode complex` using the saved checkpoint and the full 300-step episode budget.

### 3. Random Goal Sampling Over-Estimates Performance

`robo-chess-eval-stage` uses random board positions, which are biased toward the centre of the board. Corner squares (a1, a8, h1, h8) are harder to reach and under-represented. `robo-chess-eval-flow --mode complex` uses a corner-heavy destination set and gives the true production success rate.

### 4. VERTICAL_QUAT Must Be Consistent

Training without `VERTICAL_QUAT` enforcement and then enforcing it at inference creates a distribution shift. However, a model trained without VERTICAL_QUAT enforcement can still achieve 97%+ at inference with VQ (the v2 ascend result). Models trained with VQ enforcement in `step()` have so far performed worse — this remains under investigation.

### 5. Braking Zones for Ascend Hurt, Not Help

The v2 ascend model had `ascend_braking_dist = success_threshold = 10 mm` (effectively no braking zone outside the success zone). Gravity helps the arm decelerate on ascent. v3 (30 mm braking) and v4 (12 mm braking) both performed significantly worse. v5 returns to v2's original configuration.

### 6. More Environments ≠ Better

v2 used 2 environments and achieved 97.2% on key squares. v3 and v4 used 10 environments and performed worse. The cause is unclear — it may be that fewer environments produce higher-quality, more diverse experience relative to update frequency, or that the `fork` start method behaves differently at scale. v5 uses 10 environments for speed (acceptable if quality is maintained).

### 7. Longer Training Matters for Ascend

v2 trained for 1M steps. v3 and v4 trained for 600K. The extra 400K steps in v2 may have been critical for the ascend model to generalise to corner squares. v5 also uses 1M steps.

---

## Resume Training from Checkpoint

```bash
robo-chess-train train \
    --stage transit \
    --model checkpoints/transit_20260524_142519/best_model_transit.zip \
    --timesteps 300000 \
    --save-dir checkpoints/transit_resume_v2/
```

`SACTrainer.train(model_path=...)` loads the checkpoint and resets the entropy coefficient. The learning rate and other hyperparameters are overridden via `custom_objects`. Total timestep counter resets to 0 (new training run).

---

## Updating Deployed Models

1. Train the specialist: `robo-chess-train train --stage ascend`
2. Evaluate the checkpoint: `robo-chess-train eval --stage ascend --model <path>`
3. Run targeted flow coverage: `robo-chess-eval-flow --mode complex --ascend-model <path>`
4. Edit `configs/deployed_models.yaml`:
   ```yaml
   ascend: "checkpoints/ascend_20260525_133712/final_ascend.zip"
   ```
5. Verify physics and stage accuracy: `robo-chess-eval-physics && robo-chess-eval-stage --stage all`
6. Test end-to-end: `robo-chess-play`
