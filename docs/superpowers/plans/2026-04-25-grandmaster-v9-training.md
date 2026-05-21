# Grandmaster v9 Training Architecture

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refine the RL training environment for the Grandmaster movement engine to provide 10mm precision, clean braking, and perfect vertical alignment with a 10-stage waypoint system.

**Architecture:** Remove physical "sag" compensations because they are actually geometric offsets. The agent now targets the true Z-coordinates (0.415 for GRASP, 0.550 for SAFE). The verticality penalty is removed as the orientation is forced by the physics wrapper. Braking is incentivized purely through a localized velocity penalty.

**Tech Stack:** Python, Gymnasium Robotics, MuJoCo

---

### Task 1: Update Physical Constants and Thresholds

**Files:**
- Modify: `chess_env/chess_fetch_dense_env.py`

- [ ] **Step 1: Update GRASP_Z and SUCCESS_THRESHOLD**

In `chess_env/chess_fetch_dense_env.py`, locate the class constants and update `GRASP_Z` and `SUCCESS_THRESHOLD`.

```python
    # --- Action Targets ---
    # GRASP_Z (0.415m): 15mm above the table surface site.
    # Targets the dead center of the 3cm piece.
    GRASP_Z = 0.415

    # SAFE_Z (0.550m): 15cm above the table surface.
    # Provides absolute clearance over any 3cm pieces
    SAFE_Z = 0.550

    # --- Success Metrics ---
    # SUCCESS_THRESHOLD (0.010m): 10mm precision for alignment.
    SUCCESS_THRESHOLD = 0.010
    SUCCESS_BONUS = 500.0
```

- [ ] **Step 2: Commit**

```bash
git add chess_env/chess_fetch_dense_env.py
git commit -m "chore: update environment constants for 10mm precision and exact GRASP_Z"
```

### Task 2: Refine the Reward and Penalty Logic

**Files:**
- Modify: `chess_env/chess_fetch_dense_env.py`

- [ ] **Step 1: Simplify compute_reward**

In `chess_env/chess_fetch_dense_env.py`, replace the entire `compute_reward` method. We are removing the redundant verticality penalty and implementing a clean, localized braking penalty.

```python
    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)
        scalar_input = achieved_goal.ndim == 1
        if scalar_input:
            achieved_goal, desired_goal = achieved_goal[None, :], desired_goal[None, :]

        # 1. DOMINANT DISTANCE REWARD
        dist_to_goal = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        reward = -1.0 * dist_to_goal
        
        # 2. LOCALIZED BRAKING PENALTY
        # Encourage the agent to zero its velocity when approaching the 10mm success threshold.
        gripper_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        g_speed = np.linalg.norm(gripper_vel)
        if np.any(dist_to_goal < 0.03):
             reward -= 1.0 * g_speed

        return float(reward[0]) if scalar_input else reward
```

- [ ] **Step 2: Verify _is_success**

Ensure the `_is_success` method (which was modified earlier) does not contain any "sag compensation" hacks.

```python
    def _is_success(self, achieved_goal, desired_goal):
        achieved_goal = np.asarray(achieved_goal)
        desired_goal  = np.asarray(desired_goal)
        d_xy = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        d_z = abs(achieved_goal[2] - desired_goal[2])

        # SYMMETRIC SUCCESS:
        # We require 10mm XY precision and 10mm Z precision.
        return float(d_xy < self.SUCCESS_THRESHOLD and d_z < self.SUCCESS_THRESHOLD)
```

- [ ] **Step 3: Commit**

```bash
git add chess_env/chess_fetch_dense_env.py
git commit -m "feat: restructure rewards for clean braking and distance magnetism"
```
