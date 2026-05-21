# V9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the V9 architectural improvements to address gripper kinematics, precision catching (the Funnel), and natural transit (Safety Corridor) without modifying the core project logic at this time (planning stage only).

**Architecture:** 
1. Add a `WRIST_SAFE_Z` constraint to the environment to prevent the bulky wrist from sweeping the board.
2. Introduce a dynamic drift limit (the Funnel) for `descend` and `ascend` scenarios that scales with the Z-axis, forcing perfectly vertical drops.
3. Add a Z-variance penalty during the `transit` scenario to train the agent to perform safe, flat point-to-point sweeps.
4. Prepare for XML geometry modifications to the Fetch robot's gripper to fix the finger/wrist offset.

**Tech Stack:** Python, Stable Baselines 3 (SAC), Gymnasium Robotics, NumPy, MuJoCo.

---

### Task 1: Implement the "Safety Corridor" for Transit

**Files:**
- Modify: `chess_env/chess_fetch_dense_env.py`

- [ ] **Step 1: Add the Z-variance penalty to `step()`**

```python
        # Add safety corridor penalty for transit
        if self.current_scenario == "transit":
            z_error = abs(gripper_pos[2] - self.SAFE_Z)
            reward -= 10.0 * z_error
```

- [ ] **Step 2: Verify penalty logic does not crash training**
Run: `python scripts/per_scenario_eval.py --model checkpoints/... --n 1`
Expected: Script completes successfully.

---

### Task 2: Implement the "Funnel" (Dynamic Drift Limit)

**Files:**
- Modify: `chess_env/chess_fetch_dense_env.py`

- [ ] **Step 1: Update the drift limit calculation in `step()`**

```python
        elif self.current_scenario in {"descend", "ascend"}:
            drift = np.linalg.norm(gripper_pos[:2] - self.tube_center_xy)
            
            if self.force_drift_limit is not None:
                current_drift_limit = self.force_drift_limit
            else:
                drift_progress = min(self.total_env_steps / self.drift_curriculum_steps, 1.0)
                base_drift_limit = (self.DRIFT_LIMIT_START
                                       - (self.DRIFT_LIMIT_START - self.DRIFT_LIMIT_END) * drift_progress)
                
                # The Funnel: Limit shrinks as Z approaches GRASP_Z
                z_progress = max(0.0, min(1.0, (self.SAFE_Z - gripper_pos[2]) / (self.SAFE_Z - self.GRASP_Z)))
                # At SAFE_Z (z_progress=0), drift is base_drift_limit
                # At GRASP_Z (z_progress=1), drift is 0.005m (0.5cm)
                current_drift_limit = base_drift_limit - (base_drift_limit - 0.005) * z_progress

            if drift > current_drift_limit or gripper_pos[2] < self.TABLE_SURFACE_Z:
                crashed = True
                # ... logger warning ...
```

---

### Task 3: Implement the `WRIST_SAFE_Z` Constraint

**Files:**
- Modify: `chess_env/chess_fetch_dense_env.py`

- [ ] **Step 1: Define `WRIST_SAFE_Z` constant**

```python
    # WRIST_SAFE_Z (0.470m): Ensures the 12.3cm wide wrist stays above the tallest piece.
    WRIST_SAFE_Z = 0.470
```

- [ ] **Step 2: Apply the `WRIST_SAFE_Z` penalty in `step()`**

```python
        # Ensure the wrist does not drop into the board clutter unless perfectly centered
        if self.current_scenario in {"transit"}:
            if gripper_pos[2] < self.WRIST_SAFE_Z:
                crashed = True
```

---

### Task 4: XML Geometry Fix (Placeholder)

**Files:**
- Modify: `chess_env/assets/robot.xml`

- [ ] **Step 1: Adjust the Fetch Gripper Geometry**
*(Note: To be executed later when codebase changes are allowed.)*
Rotate the `gripper_link` to orient fingers strictly downwards, or add a Z-offset between the wrist and the fingers to ensure the fingers are the absolute lowest point of the arm. Once this is done, `GRASP_Z` in `chess_fetch_dense_env.py` will be safely lowered to `0.415`.
