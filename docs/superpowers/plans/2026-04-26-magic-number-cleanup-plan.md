# Magic Number Cleanup and Debug Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate hard-coded constants and validate debug mode across all scripts.

**Architecture:** Configuration-driven attribute injection for environment and training classes.

**Tech Stack:** Python 3.12, MuJoCo, Gymnasium Robotics, Stable Baselines3, PyYAML.

---

### Task 1: Expand Configuration Files

**Files:**
- Modify: `configs/env.yaml`
- Modify: `configs/training.yaml`
- Modify: `configs/physics.yaml`

- [ ] **Step 1: Expand env.yaml**
Add reward weights and task thresholds.

```yaml
# configs/env.yaml
# (Keep existing values, add these)

# Task Thresholds
stability_vel_threshold: 0.05  # Max velocity in m/s to consider the arm stable
braking_dist: 0.03             # Distance in meters at which the arm starts braking
floor_proximity_threshold: 0.025 # Safety margin above the floor/table

# Reward Weights & Penalties
z_reward_weight: 0.5           # Multiplier for Z-axis accuracy reward
jitter_penalty_weight: 0.003   # Multiplier for penalizing high-frequency actions
floor_penalty: -0.5            # Constant penalty for being too close to the surface
```

- [ ] **Step 2: Expand training.yaml**
Add evaluation and logging parameters.

```yaml
# configs/training.yaml
# (Keep existing values, add these)

# Evaluation Settings
eval_freq: 10000               # How often to run evaluation (total steps)
n_eval_episodes: 50            # Number of episodes per evaluation scenario

# Logging Settings
log_freq: 2000                 # How often to print progress to console (total steps)
moving_avg_window: 100         # Window size for calculating mean rewards/success
```

- [ ] **Step 3: Expand physics.yaml**
Add simulation setup parameters.

```yaml
# configs/physics.yaml
# (Keep existing values, add these)

# Simulation Setup
env_setup_steps: 10            # Physics steps to run during initial env setup
max_goal_retries: 100          # Max attempts to sample a valid non-overlapping goal
pos_ctrl_scale: 0.015          # Max distance in meters the arm can move per step
```

- [ ] **Step 4: Commit config expansion**

```bash
git add configs/*.yaml
git commit -m "chore: expand configuration schemas with documented constants"
```

### Task 2: Refactor Environment Classes

**Files:**
- Modify: `src/chess_env/simulation.py`
- Modify: `src/chess_env/task.py`

- [ ] **Step 1: Update simulation.py**
Replace magic numbers in `_sample_goal`, `_set_action`, and `_env_setup`.

```python
# In simulation.py __init__:
self.ENV_SETUP_STEPS = self.physics_cfg.get("env_setup_steps", 10)
self.MAX_GOAL_RETRIES = self.physics_cfg.get("max_goal_retries", 100)
self.POS_CTRL_SCALE = self.physics_cfg.get("pos_ctrl_scale", 0.015)

# Update methods to use self.ENV_SETUP_STEPS, self.MAX_GOAL_RETRIES, self.POS_CTRL_SCALE
```

- [ ] **Step 2: Update task.py**
Replace magic numbers in `compute_reward` and `step`.

```python
# In task.py __init__:
self.STABILITY_VEL_THRESHOLD = self.env_cfg["stability_vel_threshold"]
self.BRAKING_DIST = self.env_cfg["braking_dist"]
self.FLOOR_PROXIMITY_THRESHOLD = self.env_cfg["floor_proximity_threshold"]
self.Z_REWARD_WEIGHT = self.env_cfg["z_reward_weight"]
self.JITTER_PENALTY_WEIGHT = self.env_cfg["jitter_penalty_weight"]
self.FLOOR_PENALTY = self.env_cfg["floor_penalty"]

# Update compute_reward and step to use these attributes
```

- [ ] **Step 3: Commit environment refactor**

```bash
git add src/chess_env/
git commit -m "refactor: replace magic numbers with config attributes in environment"
```

### Task 3: Refactor Training and Callbacks

**Files:**
- Modify: `src/training/callbacks.py`
- Modify: `src/training/trainer.py`

- [ ] **Step 1: Update DetailedLoggingCallback**
Load constants from `training.yaml`.

```python
# In callbacks.py DetailedLoggingCallback __init__:
from src.utils.config import load_config
train_cfg = load_config("training")
self.log_freq = train_cfg.get("log_freq", 2000)
self.moving_avg_window = train_cfg.get("moving_avg_window", 100)

# Update _on_step to use self.log_freq and self.moving_avg_window
```

- [ ] **Step 2: Update trainer.py**
Use `eval_freq` and `n_eval_episodes` from config.

```python
# In trainer.py train():
eval_freq = max(self.train_cfg.get("eval_freq", 10000) // self.num_envs, 1)
n_eval_episodes = self.train_cfg.get("n_eval_episodes", 50)
```

- [ ] **Step 3: Commit training refactor**

```bash
git add src/training/
git commit -m "refactor: eliminate magic numbers in training and callbacks"
```

### Task 4: Debug Validation

- [ ] **Step 1: Verify Physics**
Run `python scripts/verify_physics.py --debug` and verify [DEBUG] logs appear for reset and steps.

- [ ] **Step 2: Verify Evaluation**
Run `python scripts/eval.py --n-episodes 2 --debug` and verify per-step debug logs.

- [ ] **Step 3: Verify Visualization**
Run `python scripts/visualize.py --episodes 1 --debug` and verify slow-motion action logs.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "test: validate debug mode across all scripts"
```
