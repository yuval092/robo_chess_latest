# Spec: RoboChess Magic Number Cleanup and Debug Validation

**Date:** 2026-04-26
**Topic:** Code Robustness, Configuration, and Logging

## 1. Objective
Eliminate all "magic numbers" (hard-coded constants) from the RoboChess codebase by moving them to grouped, documented YAML configurations. Ensure the recently implemented debug mode is fully validated across all entry scripts.

## 2. Configuration Schema

### 2.1 `configs/env.yaml`
Groups task-specific limits and reward weights.
*   `success_threshold`: Distance in meters for a successful reach.
*   `stability_vel_threshold`: Max velocity in m/s to consider the arm stable.
*   `braking_dist`: Distance in meters at which the arm starts braking.
*   `floor_proximity_threshold`: Safety margin above the floor/table.
*   `success_bonus`: One-time reward for reaching the goal.
*   `crash_penalty`: Terminal penalty for collision or out-of-bounds.
*   `z_reward_weight`: Multiplier for Z-axis accuracy reward.
*   `jitter_penalty_weight`: Multiplier for penalizing high-frequency actions.
*   `floor_penalty`: Constant penalty for being too close to the surface.

### 2.2 `configs/training.yaml`
Groups training loops and reporting parameters.
*   `eval_freq`: How often to run evaluation (total steps across all envs).
*   `n_eval_episodes`: Number of episodes per evaluation scenario.
*   `log_freq`: How often to print progress to console.
*   `moving_avg_window`: Window size for calculating mean rewards/success.

### 2.3 `configs/physics.yaml`
Groups MuJoCo simulation and sampling parameters.
*   `env_setup_steps`: Physics steps to run during initial env setup.
*   `max_goal_retries`: Max attempts to sample a valid non-overlapping goal.
*   `pos_ctrl_scale`: Max distance in meters the arm can move per step.

## 3. Refactoring Strategy
1.  **Config Loading**: Update `SACTrainer` and `ChessTaskEnv` to read the expanded YAML schemas.
2.  **Attribute Injection**: Classes will assign these config values to uppercase attributes (e.g., `self.SUCCESS_THRESHOLD`) during initialization to maintain readability.
3.  **Callback Update**: `DetailedLoggingCallback` will be updated to use `log_freq` and `moving_avg_window` from the config.

## 4. Verification & Validation
The following commands will be run in sequence to verify the "debug mode" and the refactored logic:
1.  `scripts/verify_physics.py --debug`
2.  `scripts/eval.py --n-episodes 2 --debug`
3.  `scripts/visualize.py --episodes 1 --debug`
