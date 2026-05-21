# Grandmaster v10: High-Fidelity Training Architecture

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a high-precision, collision-aware training environment for robotic chess, optimized for a 10-stage waypoint system with rigid transit and safe descent.

**Architecture:**
- **Z-Variance Penalty:** Prevents "U-moves" in transit (scaled safely to not overwhelm distance reward).
- **Safety Grasp:** GRASP_Z = 0.425 to provide a 6.5mm fingertip-board clearance.
- **Physics Hardening:** High friction and 0.5kg weighted pieces (verified to be within arm's lift capacity).
- **Transition Control:** Mandatory physics settle (runs outside episode step count) and a waypoint controller handoff.

**Tech Stack:** Python, Gymnasium Robotics, MuJoCo

---

### Task 1: Physical Environment Hardening (XML)

**Files:**
- Modify: `chess_env/assets/pick_and_place.xml`
- Modify: `chess_env/assets/robot.xml`

- [ ] **Step 1: Weight the chess piece and increase friction**
In `chess_env/assets/pick_and_place.xml`, update the `object0` geom.
```xml
<geom size="0.015 0.015 0.015" type="box" condim="4" name="object0" material="block_mat" mass="0.5" friction="2.0 0.5 0.01"></geom>
```

- [ ] **Step 2: Increase finger friction**
In `chess_env/assets/robot.xml`, update the `l_gripper_finger_link` and `r_gripper_finger_link` geoms.
```xml
<geom ... friction="10.0 0.5 0.01"></geom>
```

- [ ] **Step 3: Commit**
```bash
git add chess_env/assets/pick_and_place.xml chess_env/assets/robot.xml
git commit -m "phys: increase piece mass and finger friction for stable grasping"
```

### Task 2: Advanced Reward & Transition Logic

**Files:**
- Modify: `chess_env/chess_fetch_dense_env.py`

- [ ] **Step 1: Update GRASP_Z and add SETTLE_STEPS**
Update the constants for safety. The 6.5mm gap prevents the 77mm fingers from striking the board.
```python
    GRASP_Z = 0.425  # fingertip safety margin
    SETTLE_STEPS = 25
```

- [ ] **Step 2: Implement Z-Variance and Collision Rewards**
Update `compute_reward` to include transit-rigidity and braking. The `0.5` scaling ensures the distance reward remains dominant.
```python
    def compute_reward(self, achieved_goal, desired_goal, info):
        dist_to_goal = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        reward = -1.0 * dist_to_goal
        
        # Transit Rigidity: Punish vertical deviation in transit softly
        if self.current_scenario == "transit":
            z_err = abs(achieved_goal[..., 2] - self.SAFE_Z)
            reward -= 0.5 * z_err
            
        # Localized Braking
        gripper_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        g_speed = np.linalg.norm(gripper_vel)
        if np.any(dist_to_goal < 0.03):
             reward -= 1.0 * g_speed
             
        return reward
```

- [ ] **Step 3: Update step() with Settle and Velocity Gate**
Ensure episodes start clean. *Note: `SETTLE_STEPS` in `_reset_sim` execute the physics engine but do NOT increment the RL agent's `episode_steps` or `total_env_steps`, preventing timeouts.*
```python
    def _reset_sim(self):
        # ... existing reset logic ...
        # Add Settle steps at the end of reset
        for _ in range(self.SETTLE_STEPS):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        return True
```

- [ ] **Step 4: Commit**
```bash
git add chess_env/chess_fetch_dense_env.py
git commit -m "feat: implement scaled Z-variance penalty and physics settle logic"
```

### Task 3: Waypoint Controller Handoff (Design Document)

**Files:**
- Create: `docs/superpowers/specs/waypoint_controller_handoff.md`

- [ ] **Step 1: Document the Post-RL Settle Phase**
The RL agent terminates its episode when velocity drops below `0.05m/s`. This is a "slow crawl," not a dead stop. If the gripper closes immediately, it may knock the piece. Document the requirement for the production controller to implement a scripted "Dead Stop Settle" after every RL action.

```markdown
# Waypoint Controller: The "Dead Stop" Handoff

When the SAC model returns `done=True` (or `is_success=True`), the arm's velocity is `< 0.05m/s`. 
Before the waypoint controller transitions to the next stage (e.g., from **Descend** to **Grasp**), it MUST execute a scripted settle loop:

```python
# Pseudo-code for production controller
def execute_rl_stage(scenario):
    # Run RL until success
    while not done:
        action = model.predict(obs)
        obs, reward, done, info = env.step(action)
    
    # POST-RL SCRIPTED SETTLE
    # Hold the final action (or a zero-action) until velocity is dead zero
    while np.linalg.norm(env.get_velocity()) > 0.005:
        env.step(np.array([0, 0, 0, current_gripper_state]))
    
    # Now it is safe to change gripper state
    trigger_next_stage()
```
```

- [ ] **Step 2: Commit**
```bash
git add docs/superpowers/specs/waypoint_controller_handoff.md
git commit -m "docs: add waypoint controller post-rl settle specification"
```