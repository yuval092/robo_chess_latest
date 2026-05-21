# Actuator Enforcement & Production Certification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the "Policy Fixpoint Trap" and implement deterministic, scenario-aware gripper control for RoboChess.

**Architecture:** We use a centralized YAML configuration for geometry/rewards, override the MuJoCo finger actuators in the base simulation class to ensure immobility during RL, and implement a scripted "Pre-RL" phase in the task reset loop for finger transitions.

**Tech Stack:** MuJoCo, Gymnasium Robotics, Stable Baselines 3 (SAC).

---

### Task 1: Configuration & Geometry Updates

**Files:**
- Modify: `configs/env.yaml`
- Modify: `configs/training.yaml`

- [ ] **Step 1: Update environment geometry and reward constants in `env.yaml`**

```yaml
# configs/env.yaml updates
finger_open_joint: 0.0181
finger_closed_joint: 0.0000
finger_outer_offset: 0.033
braking_dist: 0.015
braking_reward_weight: 0.15
```

- [ ] **Step 2: Update training scale and base model in `training.yaml`**

```yaml
# configs/training.yaml updates
total_timesteps: 1000000
num_envs: 15
base_model: "models/sac-FetchPickAndPlace-v4.zip"
```

- [ ] **Step 3: Commit configuration changes**

```bash
git add configs/env.yaml configs/training.yaml
git commit -m "config: update geometry, reward weights, and training scale for production run"
```

---

### Task 2: Simulation-Level Actuator Enforcement

**Files:**
- Modify: `src/chess_env/simulation.py`

- [ ] **Step 1: Implement absolute finger position enforcement in `_set_action`**

```python
    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        
        pos_ctrl *= self.POS_CTRL_SCALE
        rot_ctrl = np.zeros(4)

        # Retrieve target joint position (to be set by Task class)
        target_qpos = getattr(self, "finger_target_joint", 0.0)
        
        # Override MuJoCo control with absolute target
        self.data.ctrl[0] = target_qpos
        self.data.ctrl[1] = target_qpos
        
        # Absolute enforcement: Zero velocity and force qpos
        self._utils.set_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint", target_qpos)
        self._utils.set_joint_qpos(self.model, self.data, "robot0:r_gripper_finger_joint", target_qpos)
        
        # Apply mocap position update
        mocap_action = np.concatenate([pos_ctrl, rot_ctrl])
        self._utils.mocap_set_action(self.model, self.data, mocap_action)
```

- [ ] **Step 2: Commit simulation changes**

```bash
git add src/chess_env/simulation.py
git commit -m "feat: implement absolute simulation-level finger enforcement"
```

---

### Task 3: Task-Level State Machine & Scripted Transitions

**Files:**
- Modify: `src/chess_env/task.py`

- [ ] **Step 1: Define new constants and reset state machine in `__init__`**

```python
        # In __init__:
        self.FINGER_OPEN_JOINT = self.env_cfg["finger_open_joint"]
        self.FINGER_CLOSED_JOINT = self.env_cfg["finger_closed_joint"]
        self.FINGER_OUTER_OFFSET = self.env_cfg["finger_outer_offset"]
        self.finger_target_joint = self.FINGER_CLOSED_JOINT # Initial
```

- [ ] **Step 2: Implement scripted transitions in `_reset_sim`**

```python
    def _reset_sim(self):
        # ... (Existing scenario selection logic) ...

        # 1. Settle arm at starting waypoint with fingers CLOSED for stability
        self.finger_target_joint = self.FINGER_CLOSED_JOINT
        self._settle_arm_to_start(arm_start_pos)

        # 2. Execute Scripted Phase
        if self.current_scenario == "descend":
            # Start closed, open scripted
            self.finger_target_joint = self.FINGER_OPEN_JOINT
            for _ in range(20):
                self._mujoco_step(None) # Move fingers while arm is static
        elif self.current_scenario == "ascend":
            # Start open, close scripted
            self._utils.set_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint", self.FINGER_OPEN_JOINT)
            self._utils.set_joint_qpos(self.model, self.data, "robot0:r_gripper_finger_joint", self.FINGER_OPEN_JOINT)
            self.finger_target_joint = self.FINGER_CLOSED_JOINT
            for _ in range(20):
                self._mujoco_step(None)
        
        # 3. Final Validation
        l_pos = self._utils.get_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint").item()
        if abs(l_pos - self.finger_target_joint) > 0.0005:
            self.logger.error(f"Reset Failed: Finger joint at {l_pos}, target {self.finger_target_joint}")
            return False # Triggers environment retry

        return True
```

- [ ] **Step 3: Update `step()` for enforcement and high-fidelity logging**

```python
    def step(self, action):
        # ... (Existing step logic) ...
        
        # Tube Breach with footprint check
        drift_center = np.linalg.norm(gripper_pos[:2] - self.tube_center_xy)
        drift_outer = drift_center + self.FINGER_OUTER_OFFSET
        
        crashed = False
        if self.current_scenario in {"descend", "ascend"}:
             if drift_center > current_drift_limit or drift_outer > current_drift_limit:
                 crashed = True
        
        # Finger Fault check
        l_finger = self._utils.get_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint").item()
        finger_fault = abs(l_finger - self.finger_target_joint) > 0.003

        # Detailed Logging
        if self.debug:
            info_log = { "success": success, "fault": finger_fault, "limit": current_drift_limit }
            print(f"[DEBUG TASK] Step {self.episode_steps} | Grip: {gripper_pos} | OuterDist: {drift_outer:.4f} | Fingers: {l_finger:.4f} | Info: {info_log}")
```

- [ ] **Step 4: Commit task changes**

```bash
git add src/chess_env/task.py
git commit -m "feat: implement scripted gripper transitions and footprint-aware safety checks"
```

---

### Task 4: Curriculum Scaling & Production Training

**Files:**
- Modify: `src/training/trainer.py`

- [ ] **Step 1: Ensure curriculum steps are correctly distributed for 15 workers**

```python
    # In trainer.py train():
    drift_curriculum_steps = self.target_curriculum_total // self.num_envs
    # Verify num_envs is 15 in logs
    progress_logger.info(f"Workers: {self.num_envs} | Curriculum steps per worker: {drift_curriculum_steps}")
```

- [ ] **Step 2: Run verification unit test (100 episodes)**

```bash
PYTHONPATH=. python scripts/eval.py --model models/sac-FetchPickAndPlace-v4.zip --n-episodes 100 --debug
```

- [ ] **Step 3: Launch Production 1M Step Training**

```bash
PYTHONPATH=. python src/training/trainer.py --fresh-start
```
