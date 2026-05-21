# Chess Physics and Verticality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the generic Pick-and-Place engine into a strict, vertical "crane-style" chess piece mover with heavy physics and velocity limits.

**Architecture:** Modifies the XML assets to introduce a heavy, high-friction 3cm cube and an iron-grip actuator. Overrides the action application in the `ChessFetchEnv` base class to strictly enforce a vertical quaternion and a maximum action speed (1.5cm/step).

**Tech Stack:** MuJoCo XML, Python, Gymnasium Robotics.

---

### Task 1: Overhaul Object and Gripper Physics

**Files:**
- Modify: `chess_env/assets/pick_and_place.xml`

- [ ] **Step 1: Modify the Block**
Change `object0` to be a 3x3x3cm cube (size 0.015). Increase mass to 0.5kg (heavy enough to stay put, light enough to lift). Add heavy damping to its free joint.

```xml
		<body name="object0" pos="0.01 0.01 0.01">
			<joint name="object0:joint" type="free" damping="0.1"></joint>
			<geom size="0.015 0.015 0.015" type="box" condim="4" name="object0" material="block_mat" mass="0.5" friction="2.0 0.5 0.01"></geom>
			<site name="object0" pos="0 0 0" size="0.015 0.015 0.015" rgba="1 0 0 1" type="sphere"></site>
		</body>
```

- [ ] **Step 2: Upgrade the Gripper Actuators**
Increase the `kp` (proportional gain / strength) of the gripper actuators to give it a firm grip.

```xml
	<actuator>
		<position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:l_gripper_finger_joint" kp="60000" name="robot0:l_gripper_finger_joint" user="1"></position>
		<position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:r_gripper_finger_joint" kp="60000" name="robot0:r_gripper_finger_joint" user="1"></position>
	</actuator>
```

- [ ] **Step 3: Verify XML Compilation**
Run: `PYTHONPATH=. .venv/bin/python3 scripts/v8_latest/verify_scene.py`
Expected: "PASS: XML Compiled Successfully"

### Task 2: Enforce Verticality and Action Scaling

**Files:**
- Modify: `chess_env/chess_fetch_env.py`

- [ ] **Step 1: Override `_set_action` in `ChessFetchEnv`**
Override `_set_action` to scale the movement and enforce the vertical quaternion `[-0.789, 0.0, -0.614, 0.0]`.

```python
    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()
        
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        
        # STRICT SCALE: Max 1.5cm per step (down from baseline 5cm)
        pos_ctrl *= 0.015  

        # PERFECT VERTICAL QUATERNION
        # Derived via grid search to point X-axis (fingers) straight down [0, 0, -1]
        rot_ctrl = [-0.78914, 0.0, -0.61421, 0.0]
        
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        if self.block_gripper:
            gripper_ctrl = np.zeros_like(gripper_ctrl)
            
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply to simulation via underlying utils
        self._utils.ctrl_set_action(self.model, self.data, action)
        self._utils.mocap_set_action(self.model, self.data, action)
```

- [ ] **Step 2: Override `_env_setup` to start vertical**
Ensure the arm starts in the vertical pose immediately upon reset.

```python
    def _env_setup(self, initial_qpos):
        # Call super first to let it do the default setup
        super()._env_setup(initial_qpos)
        
        # Then forcefully correct the mocap quaternion to be vertical
        vertical_quat = np.array([-0.78914, 0.0, -0.61421, 0.0])
        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", vertical_quat)
        
        # Step a few times to let the IK settle into the vertical pose
        for _ in range(10):
            mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
```

- [ ] **Step 3: Test verticality visually**
Run: `PYTHONPATH=. .venv/bin/python3 scripts/v8_latest/visualize_scenarios.py --scenario transit`
Expected: The arm should be pointing straight down like a crane.

### Task 3: Adjust Dense Env Step Limits and Rewards

Because the agent now moves at 1.5cm/step max (instead of 5cm), it needs more steps to cross the 64cm board.

**Files:**
- Modify: `chess_env/chess_fetch_dense_env.py`

- [ ] **Step 1: Update Reward Scaling**
Because we take more steps, accumulating distance penalties at every step will result in massive negative rewards. Scale down the distance penalty and jitter penalty.

```python
    def compute_reward(self, achieved_goal, desired_goal, info):
        achieved_goal = np.asarray(achieved_goal)
        desired_goal = np.asarray(desired_goal)
        scalar_input = achieved_goal.ndim == 1
        if scalar_input:
            achieved_goal, desired_goal = achieved_goal[None, :], desired_goal[None, :]

        dist_to_goal = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        
        # REDUCED PENALTY: Multiply by 0.3 because we take ~3x more steps
        reward = -0.3 * dist_to_goal
        
        return float(reward[0]) if scalar_input else reward

    def step(self, action):
        # ... inside step() ...
        # (Replace the old jitter line with this)
        reward -= 0.003 * np.linalg.norm(action_copy[0:3]) ** 2 # Jitter penalty reduced
```

### Task 4: Environment Registration Update

**Files:**
- Modify: `chess_env/__init__.py`

- [ ] **Step 1: Increase `max_episode_steps`**
Update the registration to allow the agent more time to cross the board at the slower speed.

```python
register(
    id="ChessFetchDense-v0",
    entry_point="chess_env.chess_fetch_dense_env:ChessFetchDenseEnv",
    max_episode_steps=250, # Increased from 175 due to strict 1.5cm action scaling
)
```

- [ ] **Step 2: Test basic environment creation**
Run: `PYTHONPATH=. .venv/bin/python3 -c "import gymnasium as gym; import chess_env; env=gym.make('ChessFetchDense-v0'); env.reset(); print('OK')"`
Expected: Prints "OK"
