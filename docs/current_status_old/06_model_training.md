# Model Training

## Algorithm: SAC (Soft Actor-Critic)

The project uses SAC from `stable_baselines3`. SAC is an off-policy actor-critic algorithm with maximum entropy regularization. Key properties relevant here:

- **Maximum entropy**: Encourages exploration via entropy bonus, balancing reward maximization with policy randomness.
- **Off-policy**: Uses a replay buffer; efficient sample use.
- **Deterministic evaluation**: `model.predict(obs, deterministic=True)` returns the mean action (no sampling). Used for all evaluation scripts.
- **Action space**: 4D continuous, clipped to [-1, 1].

---

## Pretrained Weights

Starting point: `models/sac-FetchPickAndPlace-v4.zip` — a community-trained SAC model from Hugging Face Hub for the standard `FetchPickAndPlace-v4` environment.

**Why start from pretrained?** The Fetch arm kinematics are complex — learning from scratch would require millions of steps just to learn to move. The pretrained model already knows how to navigate to XYZ targets. We fine-tune on the new board size, 3-scenario setup, and custom reward.

**Key challenge**: The pretrained model's observation space was `obs_dim = 25` (same as our Phase 9 obs). However, the pretrained model expects the standard Fetch observation layout (indices 3-5 = actual cube position relative to grip). Our "Holding Object" trick remaps those indices to fool the model into its "carrying" mode.

---

## Training Infrastructure

### Vectorized Environments

```python
train_env = SubprocVecEnv([make_env(curriculum_steps) for _ in range(num_envs)])
```

8 parallel training environments (SubprocVecEnv = separate processes). Each environment independently samples scenarios randomly, contributing transitions from all three phases to the shared replay buffer.

```python
eval_transit = DummyVecEnv([make_eval_env("transit")])
eval_descend = DummyVecEnv([make_eval_env("descend")])
eval_ascend  = DummyVecEnv([make_eval_env("ascend")])
```

3 separate single-environment evaluation instances, each locked to one scenario with `force_drift_limit=eval_drift_limit=0.005` (5mm tube radius — stricter than curriculum end value of 10mm).

### Training Loop

```python
model.learn(total_timesteps=1_000_000, callback=callbacks, reset_num_timesteps=True)
```

With 8 parallel envs, 1M total steps = 125K steps per env. Each step = 40ms sim time (2ms × 20 substeps). The curriculum completes at 500K total steps (62.5K per env).

---

## Curriculum: Drift Limit Tightening

For `descend` and `ascend` scenarios, the arm must stay within a radial "tube" around `tube_center_xy` (the target XY column). The tube starts wide and tightens:

```python
dp = min(total_env_steps / DRIFT_CURRICULUM_STEPS, 1.0)
current_drift_limit = DRIFT_LIMIT_START - (DRIFT_LIMIT_START - DRIFT_LIMIT_END) * dp
# Start: 0.100m (10cm) → End: 0.010m (1cm)
```

`DRIFT_CURRICULUM_STEPS = target_curriculum_total / num_envs = 500,000 / 8 = 62,500` per worker.

**Why a curriculum?** Early in training, the model doesn't know how to descend vertically. A wide tube allows it to develop basic vertical movement. As the tube tightens, precision increases.

---

## Reward Function

Computed in `compute_reward` and modified in `step`:

```python
# Base reward
dist_to_goal = ||achieved_goal - desired_goal||₂
reward = -DIST_REWARD_WEIGHT × dist_to_goal    # -1.0 × distance

# Z accuracy penalty
z_err = |achieved_goal[2] - desired_goal[2]|
reward -= Z_REWARD_WEIGHT × z_err              # -1.5 × z_error

# XY accuracy penalty  
xy_err = ||achieved_goal[:2] - desired_goal[:2]||₂
reward -= XY_REWARD_WEIGHT × xy_err            # -2.0 × xy_error

# Braking penalty (near goal)
if dist_to_goal < BRAKING_DIST:                # < 10mm from goal
    reward -= BRAKING_REWARD_WEIGHT × speed    # -0.15 × speed

# Jitter penalty (applied in step)
reward -= JITTER_PENALTY_WEIGHT × ||action[:3]||²  # -0.003 × action_norm²

# Floor proximity penalty
if gripper_pos[2] < FLOOR_LIMIT + FLOOR_PROXIMITY_THRESHOLD:  # < 0.425m
    reward += FLOOR_PENALTY                    # -0.5 flat penalty

# Crash
if crashed:
    reward = CRASH_PENALTY                     # -500.0

# Success
if success:
    reward += SUCCESS_BONUS                    # +500.0
```

**Design intent**: The dense reward encourages the arm to move toward the goal while remaining stable:
- The -1.0×distance term provides a continuous signal to approach.
- The -1.5×z_error and -2.0×xy_error terms provide extra pressure for precision (especially important for descend/ascend which require sub-cm accuracy).
- The braking penalty prevents the arm from sliding through the goal at high speed.
- The jitter penalty prevents oscillating high-frequency actions (which would look unnatural and cause instability).
- The ±500 success/crash bonuses dominate the episode total — they are the primary terminal signal.

---

## Callbacks

### `DetailedLoggingCallback`
Logs every 2000 steps:
- Rolling mean reward (100-episode window)
- Rolling success rate (100-episode window)
- Per-scenario success rates (30-episode window, min 10 episodes)

### `SuccessRateEvalCallback` (per scenario)
Evaluates 50 episodes every 10,000 steps on the locked scenario env. Saves:
- `best_model_{scenario}.zip`: Best success rate for that scenario
- `latest_model_{scenario}.zip`: Most recent evaluation

### `CombinedSuccessCallback`
Monitors the average of all three scenario success rates. Saves `best_model_combined.zip` when the average improves. The "combined" model is the primary deliverable for evaluation.

---

## Checkpoints

Saved to `./checkpoints/pure_movement_v6_{timestamp}/`:
- `best_model_transit.zip`: Best transit success
- `best_model_descend.zip`: Best descend success
- `best_model_ascend.zip`: Best ascend success
- `best_model_combined.zip`: Best average across all scenarios
- `latest_model_*.zip`: Most recent evaluations

On training restart, `find_best_checkpoint()` automatically selects the most recent `best_model_transit.zip` as the starting point.

---

## Known Training Limitations

1. **Transit ↔ Descend/Ascend conflict**: The model uses the same weights for all 3 scenarios. High descent/ascend success can come at the cost of transit performance (different motion styles).
2. **"Holding Object" trick fragility**: The observation hack works for pretrained weights with the specific FetchPickAndPlace-v4 training distribution. Different base models might not respond the same way.
3. **Curriculum end value**: The final 10mm tube is fairly strict for an RL model. Evaluation uses 5mm, which is even more demanding.
4. **Grasp is scripted, not RL**: The RL model only learns transit/descend/ascend — the actual grasp is entirely scripted. This is intentional to avoid the extreme difficulty of RL-based grasping.
