# V7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify the environment and training scripts to implement the "Geometric Alignment" strategy, stabilizing the 3D movement model to allow fine-tuning over 5M steps.

**Architecture:** Modify `ChessFetchDenseEnv` observation spaces, reward calculations, and physics limits. Modify `fine_tune_dense.py` to correctly apply hyperparameter configurations, learning rate schedules, and checkpoint callbacks. Include bug fixes highlighted in previous reviews.

**Tech Stack:** Python, Stable Baselines 3 (SAC), Gymnasium Robotics, NumPy.

---

### Task 1: Environment Observation and Constants Overhaul

**Files:**
- Modify: `chess_env/chess_fetch_dense_env.py`

- [ ] **Step 1: Update Physical Constants and Constraints**
Update `GRASP_Z`, `FLOOR_LIMIT`, `SUCCESS_BONUS`, and `CRASH_PENALTY`.
Remove `RELEASE_Z` and redundant `TABLE_Z`. Update `DRIFT_CURRICULUM_STEPS`.

```python
    # ...
    # --- Action Targets ---
    GRASP_Z = 0.430  # 10mm clearance above the 2cm piece (0.420 max height)
    SAFE_Z = 0.550

    # --- Success Metrics ---
    SUCCESS_THRESHOLD = 0.020
    SUCCESS_BONUS = 500.0

    # --- Constraints (The Boundaries) ---
    DRIFT_LIMIT = 0.035
    DRIFT_LIMIT_START = 0.150
    DRIFT_LIMIT_END   = 0.035
    
    # 5M total steps / 14 workers = ~357k per worker. 
    # We want it to tighten over the first 2M total steps.
    # 2,000,000 / 14 = 142,857 steps.
    DRIFT_CURRICULUM_STEPS = 142_857 

    FLOOR_LIMIT = 0.400  # The actual table surface
    CRASH_PENALTY = -500.0
    # ...
```

- [ ] **Step 2: Remove references to `TABLE_Z` and `RELEASE_Z`**
Replace `self.TABLE_Z` with `self.TABLE_SURFACE_Z` (or ensure `TABLE_Z` is aliased in `ChessFetchEnv`). Remove `self.RELEASE_Z`. Ensure the init logic uses `self.GRASP_Z` for DESCEND goals.

- [ ] **Step 3: Update `_build_phase9_observation`**
Implement the "Virtual Object" mapping.

```python
    def _build_phase9_observation(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        scenario_map = {
            "transit": [1.0, 0.0, 0.0],
            "descend": [0.0, 1.0, 0.0],
            "ascend": [0.0, 0.0, 1.0]
        }
        scenario_id_vec = scenario_map.get(self.current_scenario, [0.0, 0.0, 0.0])
        
        if self.goal_pos is not None:
            relative_to_goal = self.goal_pos - grip_pos
            target_pos = self.goal_pos
        else:
            relative_to_goal = np.zeros(3)
            target_pos = np.zeros(3)

        # Indices 0-2: Grip Pos
        # Indices 3-5: Target Pos (Replaces Object Pos)
        # Indices 6-8: Goal-Relative Vector (Replaces Object Rel Pos)
        # Indices 9-10: Gripper State
        # Indices 11-13: Scenario ID (Replaces Object Rot)
        # Indices 14-19: Zero Mask (Replaces Object Vel)
        # Indices 20-24: Robot Vel
        obs = np.concatenate(
            [
                grip_pos,                 # 0, 1, 2
                target_pos,               # 3, 4, 5
                relative_to_goal,         # 6, 7, 8
                gripper_state,            # 9, 10
                scenario_id_vec,          # 11, 12, 13
                np.zeros(3),              # 14, 15, 16
                np.zeros(3),              # 17, 18, 19
                grip_velp,                # 20, 21, 22
                gripper_vel,              # 23, 24
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": grip_pos.copy(),
            "desired_goal": self.goal.copy(),
        }
```

- [ ] **Step 4: Update `_is_success` for Stateless Calculation**
Remove the dependency on `self.current_scenario`.

```python
    def _is_success(self, achieved_goal, desired_goal):
        achieved_goal = np.asarray(achieved_goal)
        desired_goal  = np.asarray(desired_goal)
        d_xy = np.linalg.norm(achieved_goal[:2] - desired_goal[:2])
        d_z  = abs(achieved_goal[2] - desired_goal[2])

        is_transit_goal = abs(desired_goal[2] - self.SAFE_Z) < 0.005
        if is_transit_goal:
            return float(d_xy < self.SUCCESS_THRESHOLD and d_z < 0.030)
        else:
            return float(np.linalg.norm(achieved_goal - desired_goal) < self.SUCCESS_THRESHOLD)
```

- [ ] **Step 5: Add Soft Floor Penalty and Info Scenario in `step`**
Apply the soft floor penalty and include the scenario in the info dictionary. Remove `TABLE_Z` references and replace with `TABLE_SURFACE_Z`.

```python
    def step(self, action):
        # ... (keep action setting and mujoco step) ...
        
        obs = self._get_obs()
        terminated, truncated = False, False
        gripper_pos = obs["observation"][0:3]
        
        reward = self.compute_reward(gripper_pos, self.goal_pos, {})
        reward -= 0.01 * np.linalg.norm(action_copy[0:3]) ** 2 
        
        # Soft Floor Penalty
        if gripper_pos[2] < self.FLOOR_LIMIT + 0.010:
            reward -= 0.5

        success = self._is_success(gripper_pos, self.goal_pos)
        crashed = False

        if self.current_scenario == "transit":
            if gripper_pos[2] < self.FLOOR_LIMIT:
                crashed = True
        elif self.current_scenario in {"descend", "ascend"}:
            drift = np.linalg.norm(gripper_pos[:2] - self.tube_center_xy)
            
            if self.force_drift_limit is not None:
                current_drift_limit = self.force_drift_limit
            else:
                drift_progress = min(self.total_env_steps / self.drift_curriculum_steps, 1.0)
                current_drift_limit = (self.DRIFT_LIMIT_START
                                       - (self.DRIFT_LIMIT_START - self.DRIFT_LIMIT_END) * drift_progress)

            if drift > current_drift_limit or gripper_pos[2] < self.TABLE_SURFACE_Z:
                crashed = True
                self.logger.warning(
                    "TERMINATED (%s): Boundary Crash at step %d (drift=%.3fm, limit=%.3fm)",
                    self.current_scenario.upper(), self.episode_steps, drift, current_drift_limit
                )

        if crashed:
            terminated = True
            reward = self.CRASH_PENALTY 
            success = 0.0

        if success:
            terminated = True
            self.logger.info("SUCCESS (%s): Goal reached in %d steps.",
                             self.current_scenario.upper(), self.episode_steps)

        info = {
            "is_success": float(success),
            "scenario": self.current_scenario,
        }
        return obs, reward, terminated, truncated, info
```

---

### Task 2: Training Script Hyperparameters & Fixes

**Files:**
- Modify: `scripts/fine_tune_dense.py`

- [ ] **Step 1: Update Global Variables**

```python
TOTAL_TIMESTEPS = 5_000_000
NUM_ENVS = 14
BASE_MODEL = "models/sac-FetchPickAndPlace-v4.zip"
DRIFT_CURRICULUM_STEPS = 142_857  # ~2M total steps / 14 workers
```

- [ ] **Step 2: Update `SuccessRateEvalCallback` Bug**
Ensure `evaluations_successes` is populated correctly by using `_is_success_buffer` directly.

```python
    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self._is_success_buffer:
                latest = float(np.mean(self._is_success_buffer))
                self.last_mean_success = latest
                if latest > self._best_success_rate:
                    self._best_success_rate = latest
                    os.makedirs(self.success_save_path, exist_ok=True)
                    save_path = os.path.join(self.success_save_path, f"best_model_{self.name}")
                    self.model.save(save_path)
                    progress_logger.info(
                        "*** NEW BEST EVAL (%s) *** | Success: %.1f%% -> saved to %s.zip",
                        self.name.upper(), latest * 100.0, save_path
                    )
        return result
```

- [ ] **Step 3: Update `fine_tune_dense` function logic**
Implement the slower learning rate, target entropy, and increased learning starts.

```python
def fine_tune_dense(base_model_path=BASE_MODEL, num_envs=NUM_ENVS, fresh_start=False):
    # ... (keep timestamp, run_name, logging setup) ...

    if base_model_path == BASE_MODEL and not fresh_start:
        base_model_path = find_best_checkpoint()

    # ... (keep env creation and callbacks) ...

    # UPDATED HYPERPARAMETERS
    model = SAC.load(
        base_model_path, 
        env=train_env, 
        learning_rate=3e-5, 
        batch_size=512, 
        target_entropy=-6.0,
        verbose=1
    )
    model.tensorboard_log = f"./logs/{run_name}/tensorboard/"
    model.learning_starts = 10_000
    
    # We DO NOT reset the replay buffer if loading a checkpoint, but we DO if fresh start
    if fresh_start:
        model.replay_buffer.reset()

    # Remove the manual ent_coef reset as target_entropy handles it
    
    print(f"Beginning {TOTAL_TIMESTEPS:,} steps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, reset_num_timesteps=True, progress_bar=True)
    # ...
```
