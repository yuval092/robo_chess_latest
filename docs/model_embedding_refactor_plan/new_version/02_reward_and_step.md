# Reward Function & `step()` Safety Checks

## Overview

The current `ChessTaskEnv.step()` is a stub — it applies a zero action and returns `reward=0.0`.
This must be replaced with a **full RL step** for training to work. However, we must be careful:
the **scripted controller never calls `env.step()`** — it calls `env._mujoco_step()` directly
inside `_run_movement_loop`. So the new `step()` only affects training.

> [!IMPORTANT]
> The existing scripted controller pipeline is NOT affected by changes to `step()`.
> `ScriptedController` drives the arm via `_move_mocap_to` → `_mujoco_step` directly.
> `step()` is only called by the SB3 training loop.

---

## 1. Full `step()` Implementation

Replace the stub in `src/chess_env/task.py`:

```python
def step(self, action):
    """
    Full RL training step. Used by SB3 training loop ONLY.
    ScriptedController bypasses this via _mujoco_step() directly.
    
    Flow:
      1. Enforce finger state machine (scenario-dependent)
      2. Apply action (scaled, clipped)
      3. Step physics
      4. Check crash conditions (per scenario)
      5. Compute reward
      6. Check success (proximity + velocity gate)
      7. Return obs, reward, terminated, truncated, info
    """
    self.total_env_steps += 1
    self.episode_steps += 1

    # ── 1. Finger State Machine ───────────────────────────────────────────
    # Descend: arm lowers with fingers OPEN (ready to grasp)
    # Transit/Ascend: arm moves with fingers CLOSED (holding or empty-handed)
    if self.current_scenario == "descend":
        self.finger_target_joint = self.FINGER_OPEN_JOINT
    else:
        self.finger_target_joint = self.FINGER_CLOSED_JOINT

    # ── 2. Apply Action ───────────────────────────────────────────────────
    action_copy = np.clip(action.copy(), self.action_space.low, self.action_space.high)
    action_copy[3] = -1.0  # Suppress gripper control; finger_target_joint handles it
    self._set_action(action_copy)
    self._mujoco_step(action_copy)

    # ── 3. Observe ────────────────────────────────────────────────────────
    obs = self._get_obs()
    # Extract from Phase-9 obs vector
    grip_pos = obs["achieved_goal"]  # indices 0-2 of observation = grip_pos
    grip_vel = obs["observation"][20:23]  # indices 20-22 = grip_velp

    terminated = False
    crash_reason = None

    # ── 4. Finger Fault Check ─────────────────────────────────────────────
    # Detects actuator failure (finger joint diverged from commanded target)
    l_finger = self._utils.get_joint_qpos(
        self.model, self.data, "robot0:l_gripper_finger_joint"
    ).item()
    if abs(l_finger - self.finger_target_joint) > 0.003:
        terminated = True
        crash_reason = f"FINGER_FAULT (actual={l_finger:.4f}, target={self.finger_target_joint:.4f})"

    # ── 5. Scenario-Specific Crash Checks ────────────────────────────────
    if not terminated:
        current_drift_limit = self._get_current_drift_limit()

        if self.current_scenario == "transit":
            if grip_pos[2] < self.FLOOR_LIMIT:
                terminated = True
                crash_reason = f"FLOOR_HIT (z={grip_pos[2]:.4f})"

        elif self.current_scenario in {"descend", "ascend"}:
            if self.tube_center_xy is not None:
                drift = float(np.linalg.norm(grip_pos[:2] - self.tube_center_xy))
                if drift > current_drift_limit:
                    terminated = True
                    crash_reason = f"TUBE_BREACH (drift={drift*1000:.1f}mm > limit={current_drift_limit*1000:.1f}mm)"
            if not terminated and grip_pos[2] < self.TABLE_SURFACE_Z:
                terminated = True
                crash_reason = f"TABLE_HIT (z={grip_pos[2]:.4f})"

    # ── 6. Reward Computation ─────────────────────────────────────────────
    if terminated and crash_reason:
        reward = float(self.env_cfg.get("crash_penalty", -500.0))
        success = 0.0
    else:
        reward = self.compute_reward(grip_pos, self.goal_pos, {})
        
        # Braking reward: penalises high speed when close to goal
        dist = float(np.linalg.norm(grip_pos - self.goal_pos))
        speed = float(np.linalg.norm(grip_vel))
        if dist < self.env_cfg.get("braking_dist", 0.010):
            reward -= self.env_cfg.get("braking_reward_weight", 0.15) * speed
        
        # Jitter penalty: discourages high-frequency oscillating actions
        reward -= self.env_cfg.get("jitter_penalty_weight", 0.003) * float(np.linalg.norm(action_copy[:3]) ** 2)
        # Floor proximity penalty
        floor_prox_thresh = self.env_cfg.get("floor_proximity_threshold", 0.025)
        if grip_pos[2] < self.FLOOR_LIMIT + floor_prox_thresh:
            reward += float(self.env_cfg.get("floor_penalty", -0.5))

        # ── 7. Success Check ──────────────────────────────────────────────
        # Requires BOTH spatial proximity AND low velocity (arm must be stopping, not passing through)
        is_near = bool(self._is_success(grip_pos, self.goal_pos))
        is_stable = float(np.linalg.norm(grip_vel)) < float(self.env_cfg.get("stability_vel_threshold", 0.02))
        success = 1.0 if (is_near and is_stable) else 0.0

        if success:
            terminated = True
            reward += float(self.env_cfg.get("success_bonus", 500.0))

    info = {
        "is_success": success,
        "scenario":   self.current_scenario,
        "crash_reason": crash_reason,
    }

    if self.debug and terminated:
        outcome = f"CRASH ({crash_reason})" if crash_reason else ("SUCCESS" if success else "TIMEOUT")
        self.logger.debug(
            f"[EP {self.episode_number} END] Scenario={self.current_scenario} "
            f"Outcome={outcome} Reward={reward:.2f}"
        )

    return obs, float(reward), terminated, False, info


def _get_current_drift_limit(self) -> float:
    """Computes the current drift limit, applying the curriculum schedule if active."""
    if self.force_drift_limit is not None:
        return float(self.force_drift_limit)
    if self.fixed_drift or self.drift_curriculum_steps is None:
        return float(self.DRIFT_LIMIT_END)
    # Linear curriculum: starts wide, tightens over DRIFT_CURRICULUM_STEPS total env steps
    progress = min(self.total_env_steps / self.DRIFT_CURRICULUM_STEPS, 1.0)
    limit = self.DRIFT_LIMIT_START - (self.DRIFT_LIMIT_START - self.DRIFT_LIMIT_END) * progress
    return float(limit)
```

---

## 2. `compute_reward()` — Multi-Component Dense Reward

Replace the existing stub in `src/chess_env/task.py`:

```python
def compute_reward(self, achieved_goal, desired_goal, info):
    """
    Dense multi-component reward. Called by step() and by HER (if used).
    
    Components:
      1. Distance:  -DIST_WEIGHT * L2(grip, goal)        [primary signal]
      2. Z error:   -Z_WEIGHT * |grip_z - goal_z|        [height precision]
      3. XY error:  -XY_WEIGHT * L2_xy(grip, goal)       [horizontal precision]
      4. Braking:   -BRAKE_WEIGHT * speed (near goal)     [smooth stop]
    
    Jitter and floor penalties are added in step() on top.
    """
    achieved_goal = np.asarray(achieved_goal, dtype=np.float64)
    desired_goal  = np.asarray(desired_goal,  dtype=np.float64)
    scalar = achieved_goal.ndim == 1
    if scalar:
        achieved_goal = achieved_goal[None]
        desired_goal  = desired_goal[None]

    dist = np.linalg.norm(achieved_goal - desired_goal, axis=1)
    reward = -self.env_cfg.get("dist_reward_weight", 1.0) * dist

    z_err  = np.abs(achieved_goal[:, 2] - desired_goal[:, 2])
    reward -= self.env_cfg.get("z_reward_weight", 1.5) * z_err

    xy_err = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
    reward -= self.env_cfg.get("xy_reward_weight", 2.0) * xy_err

    return float(reward[0]) if scalar else reward
```

---

## 3. Reward Config in `configs/env.yaml`

Add these keys (many already exist, verify they are present):

```yaml
# --- RL Training Reward Weights ---
dist_reward_weight: 1.0        # Primary distance-to-goal multiplier
z_reward_weight: 1.5           # Extra penalty for Z inaccuracy
xy_reward_weight: 2.0          # Extra penalty for XY inaccuracy
braking_reward_weight: 0.15    # Speed penalty when near goal (encourages halting)
braking_dist: 0.010            # Distance threshold to activate braking reward
jitter_penalty_weight: 0.003   # Penalise oscillating actions (||action||^2)
floor_penalty: -0.5            # Constant penalty for being near the table
floor_proximity_threshold: 0.025  # Z margin above FLOOR_LIMIT for floor penalty
stability_vel_threshold: 0.02  # Max grip speed (m/s) to count as "stable" at goal
success_bonus: 500.0           # Terminal reward on success
crash_penalty: -500.0          # Terminal reward on crash

# --- Drift Curriculum ---
drift_limit_start: 0.100       # Starting drift limit (wide — lets model learn gross motion)
drift_limit_end: 0.010         # Final drift limit (tight — production accuracy)
drift_curriculum_steps: 62500  # Steps per worker to reach drift_limit_end
                               # (500k total / 8 workers)
```

---

## 4. Rationale for Per-Scenario Specialisation

Each specialist model learns distinct dynamics:

| Model | Key Challenge | Critical Reward Component |
|---|---|---|
| **Transit** | Long horizontal movement, Z discipline | `FLOOR_HIT` crash penalty |
| **Descend** | Precise tube constraint, smooth halt at `HOVER_Z` | Braking reward + `TUBE_BREACH` crash |
| **Ascend** | Tube constraint while rising, momentum control | `TUBE_BREACH` crash + jitter penalty |

Since each model trains exclusively on one scenario (via `force_scenario=...`), it never has to
generalise across conflicting reward landscapes. The `tube_center_xy` and `goal_pos` are always
set correctly by `ChessTaskEnv._reset_sim()` for the locked scenario.

---

## 5. The `_is_success()` Function (unchanged)

The existing implementation is correct and is already used:

```python
def _is_success(self, achieved_goal, desired_goal):
    d_xy = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
    d_z  = abs(achieved_goal[2]  - desired_goal[2])
    return float(d_xy < self.SUCCESS_THRESHOLD and d_z < self.SUCCESS_THRESHOLD)
```

`SUCCESS_THRESHOLD = 0.010` (10mm) from `env.yaml`. Note that `step()` also requires
low velocity (`< STABILITY_VEL_THRESHOLD`) in addition to proximity for success — this is new
and prevents the model from being rewarded for just "passing through" the goal.
