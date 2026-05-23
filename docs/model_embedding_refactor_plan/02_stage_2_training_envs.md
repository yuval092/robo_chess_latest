# Training Environments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Gym environment wrappers (`TransitTrainEnv`, `AscendTrainEnv`, `DescendTrainEnv`) to train the SAC models safely and correctly, rewarding them appropriately while maintaining strict safety constraints.

**Architecture:** Create a `training/envs/` module. The wrappers inherit from `gymnasium.Wrapper`. The wrappers modify `reset()` to start at valid random configurations, and modify `step()` to yield custom rewards and handle episode termination (success, fail on drift/drop).

**Tech Stack:** Python, Gymnasium

---

### Task 1: Setup Training Package and Base Reward Logic

**Files:**
- Create: `training/__init__.py`
- Create: `training/envs/__init__.py`
- Create: `training/envs/base_train_env.py`
- Create: `tests/training/test_base_train_env.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_base_train_env.py
import pytest
import numpy as np
from training.envs.base_train_env import ChessTrainWrapper
import gymnasium as gym

def test_base_wrapper_initialization():
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Dict({"observation": gym.spaces.Box(-1, 1, shape=(10,))})
            self.action_space = gym.spaces.Box(-1, 1, shape=(4,))
            self.unwrapped = self
            self.SAFE_Z = 0.530
            self.HOVER_Z = 0.460
            self.TABLE_SURFACE_Z = 0.400
            
            class DummyUtils:
                def get_site_xpos(self, model, data, site):
                    return np.array([0.0, 0.0, 0.530])
                def get_site_xvelp(self, model, data, site):
                    return np.array([0.0, 0.0, 0.0])
            self._utils = DummyUtils()
            self.model = None
            self.data = None

    env = DummyEnv()
    wrapper = ChessTrainWrapper(env)
    
    assert wrapper.env == env
    assert hasattr(wrapper, "get_grip_state")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_base_train_env.py -v`
Expected: FAIL (File/module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# training/envs/base_train_env.py
import gymnasium as gym
import numpy as np

class ChessTrainWrapper(gym.Wrapper):
    """Base wrapper providing common helpers for training environments."""
    
    def __init__(self, env):
        super().__init__(env)
        
    def get_grip_state(self):
        """Returns grip position and velocity."""
        env = self.unwrapped
        pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
        vel = env._utils.get_site_xvelp(env.model, env.data, "robot0:grip").copy()
        return pos, vel
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/training/test_base_train_env.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
mkdir -p training/envs tests/training
touch training/__init__.py training/envs/__init__.py tests/training/__init__.py
git add training tests/training
git commit -m "feat: setup training package and ChessTrainWrapper"
```

### Task 2: Implement TransitTrainEnv

**Files:**
- Create: `training/envs/transit_env.py`
- Create: `tests/training/test_transit_env.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_transit_env.py
import pytest
import numpy as np
import gymnasium as gym
from training.envs.transit_env import TransitTrainEnv

def test_transit_train_env_reset():
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Dict({"observation": gym.spaces.Box(-1, 1, shape=(10,))})
            self.action_space = gym.spaces.Box(-1, 1, shape=(4,))
            self.unwrapped = self
            self.SAFE_Z = 0.530
            
            class DummyData:
                mocap_pos = np.zeros((1, 3))
            self.data = DummyData()
            self.model = None
            
            class DummyUtils:
                def get_site_xpos(self, model, data, site):
                    return np.array([0.8, 0.2, 0.530])
                def get_site_xvelp(self, model, data, site):
                    return np.array([0.0, 0.0, 0.0])
            self._utils = DummyUtils()
            self.np_random = np.random.default_rng()

        def reset(self, **kwargs):
            return {"observation": np.zeros(10)}, {}

    env = DummyEnv()
    wrapper = TransitTrainEnv(env)
    
    obs, info = wrapper.reset()
    assert "observation" in obs
    assert hasattr(wrapper, "target_xy")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_transit_env.py -v`
Expected: FAIL (File not found)

- [ ] **Step 3: Write minimal implementation**

```python
# training/envs/transit_env.py
import numpy as np
from training.envs.base_train_env import ChessTrainWrapper

class TransitTrainEnv(ChessTrainWrapper):
    """
    Training env for the Transit model.
    Goal: Move from random XY at SAFE_Z to target XY at SAFE_Z.
    """
    def __init__(self, env):
        super().__init__(env)
        self.target_xy = np.zeros(2)
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        unwrapped = self.unwrapped
        
        # Randomize target XY within board
        self.target_xy = unwrapped.np_random.uniform(
            low=[0.600, -0.016], high=[1.160, 0.544]
        )
        
        # Randomize start XY within board
        start_xy = unwrapped.np_random.uniform(
            low=[0.600, -0.016], high=[1.160, 0.544]
        )
        
        # Update environment's goal state
        unwrapped.goal_pos = np.array([self.target_xy[0], self.target_xy[1], unwrapped.SAFE_Z])
        unwrapped.goal = unwrapped.goal_pos.copy()
        
        # Override mocap to start position
        unwrapped.data.mocap_pos[0][:2] = start_xy
        unwrapped.data.mocap_pos[0][2] = unwrapped.SAFE_Z
        
        # Sync physics
        import mujoco
        mujoco.mj_forward(unwrapped.model, unwrapped.data)

```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/training/test_transit_env.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/envs/transit_env.py tests/training/test_transit_env.py
git commit -m "feat: implement TransitTrainEnv reset logic"
```

### Task 3: Implement TransitTrainEnv Step and Reward

**Files:**
- Modify: `training/envs/transit_env.py`
- Modify: `tests/training/test_transit_env.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/training/test_transit_env.py
def test_transit_train_env_step():
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Dict({"observation": gym.spaces.Box(-1, 1, shape=(10,))})
            self.action_space = gym.spaces.Box(-1, 1, shape=(4,))
            self.unwrapped = self
            self.SAFE_Z = 0.530
            self.data = type('DummyData', (), {'mocap_pos': np.zeros((1, 3))})()
            self._utils = type('DummyUtils', (), {
                'get_site_xpos': lambda m, d, s: np.array([0.8, 0.2, 0.530]),
                'get_site_xvelp': lambda m, d, s: np.array([0.0, 0.0, 0.0])
            })()
            self.np_random = np.random.default_rng()

        def reset(self, **kwargs): return {"observation": np.zeros(10)}, {}
        def step(self, action): return {"observation": np.zeros(10)}, 0.0, False, False, {}

    env = DummyEnv()
    wrapper = TransitTrainEnv(env)
    wrapper.reset()
    
    wrapper.unwrapped._utils.get_site_xpos = lambda m, d, s: np.array([wrapper.target_xy[0], wrapper.target_xy[1], 0.530])
    obs, reward, terminated, truncated, info = wrapper.step(np.zeros(4))
    
    assert terminated is True
    assert reward > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_transit_env.py -v`
Expected: FAIL (step logic not implemented, reward is 0)

- [ ] **Step 3: Write minimal implementation**

```python
# Add to training/envs/transit_env.py
    def step(self, action):
        # Force gripper closed
        action = np.copy(action)
        action[3] = -1.0
        
        obs, _, terminated, truncated, info = self.env.step(action)
        
        pos, vel = self.get_grip_state()
        xy_dist = np.linalg.norm(pos[:2] - self.target_xy)
        z_drift = abs(pos[2] - self.unwrapped.SAFE_Z)
        
        # Dense reward
        reward = -xy_dist * 10.0 - 0.1 # Distance penalty + step penalty
        
        # Terminate if successful
        if xy_dist < 0.004:
            terminated = True
            reward += 100.0
            info["is_success"] = True
            
        # Fail on huge Z drift
        elif z_drift > 0.05:
            terminated = True
            reward -= 100.0
            info["is_success"] = False
            
        return obs, float(reward), terminated, truncated, info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/training/test_transit_env.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/envs/transit_env.py tests/training/test_transit_env.py
git commit -m "feat: implement TransitTrainEnv step and reward logic"
```

### Task 4: Implement AscendTrainEnv and DescendTrainEnv

**Files:**
- Create: `training/envs/vertical_envs.py`
- Create: `tests/training/test_vertical_envs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_vertical_envs.py
import pytest
import numpy as np
import gymnasium as gym
from training.envs.vertical_envs import AscendTrainEnv, DescendTrainEnv

def test_vertical_envs_reset_and_step():
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Dict({"observation": gym.spaces.Box(-1, 1, shape=(10,))})
            self.action_space = gym.spaces.Box(-1, 1, shape=(4,))
            self.unwrapped = self
            self.SAFE_Z = 0.530
            self.HOVER_Z = 0.460
            self.data = type('DummyData', (), {'mocap_pos': np.zeros((1, 3))})()
            self._utils = type('DummyUtils', (), {
                'get_site_xpos': lambda m, d, s: np.array([0.8, 0.2, 0.460]),
                'get_site_xvelp': lambda m, d, s: np.array([0.0, 0.0, 0.0])
            })()
            self.np_random = np.random.default_rng()

        def reset(self, **kwargs): return {"observation": np.zeros(10)}, {}
        def step(self, action): return {"observation": np.zeros(10)}, 0.0, False, False, {}

    env = DummyEnv()
    
    # Test Ascend
    ascend_env = AscendTrainEnv(env)
    ascend_env.reset()
    assert ascend_env.unwrapped.data.mocap_pos[0][2] == 0.460
    
    ascend_env.unwrapped._utils.get_site_xpos = lambda m, d, s: np.array([ascend_env.target_xy[0], ascend_env.target_xy[1], 0.530])
    _, rew, term, _, _ = ascend_env.step(np.zeros(4))
    assert term is True
    
    # Test Descend
    descend_env = DescendTrainEnv(env)
    descend_env.reset()
    assert descend_env.unwrapped.data.mocap_pos[0][2] == 0.530
    
    descend_env.unwrapped._utils.get_site_xpos = lambda m, d, s: np.array([descend_env.target_xy[0], descend_env.target_xy[1], 0.460])
    _, rew, term, _, _ = descend_env.step(np.zeros(4))
    assert term is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_vertical_envs.py -v`
Expected: FAIL (File not found)

- [ ] **Step 3: Write minimal implementation**

```python
# training/envs/vertical_envs.py
import numpy as np
from training.envs.base_train_env import ChessTrainWrapper

class VerticalTrainEnv(ChessTrainWrapper):
    """Base class for Ascend and Descend environments."""
    def __init__(self, env, start_z, target_z):
        super().__init__(env)
        self.start_z = start_z
        self.target_z = target_z
        self.target_xy = np.zeros(2)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        unwrapped = self.unwrapped
        
        self.target_xy = unwrapped.np_random.uniform(low=[0.600, -0.016], high=[1.160, 0.544])
        
        # Update environment's goal state
        unwrapped.goal_pos = np.array([self.target_xy[0], self.target_xy[1], self.target_z])
        unwrapped.goal = unwrapped.goal_pos.copy()
        
        unwrapped.data.mocap_pos[0][:2] = self.target_xy
        unwrapped.data.mocap_pos[0][2] = self.start_z
        
        # Sync physics
        import mujoco
        mujoco.mj_forward(unwrapped.model, unwrapped.data)


    def step(self, action):
        action = np.copy(action)
        action[3] = 1.0 # Force open
        
        obs, _, terminated, truncated, info = self.env.step(action)
        
        pos, vel = self.get_grip_state()
        z_dist = abs(pos[2] - self.target_z)
        xy_drift = np.linalg.norm(pos[:2] - self.target_xy)
        
        reward = -z_dist * 10.0 - 0.1
        
        if z_dist < 0.004:
            terminated = True
            reward += 100.0
            info["is_success"] = True
        elif xy_drift > 0.010: # Tube drift limit
            terminated = True
            reward -= 100.0
            info["is_success"] = False
            
        return obs, float(reward), terminated, truncated, info

class AscendTrainEnv(VerticalTrainEnv):
    def __init__(self, env):
        # Ensure we have the base env wrapped
        inner = env.unwrapped if hasattr(env, 'unwrapped') else env
        super().__init__(env, inner.HOVER_Z, inner.SAFE_Z)

class DescendTrainEnv(VerticalTrainEnv):
    def __init__(self, env):
        inner = env.unwrapped if hasattr(env, 'unwrapped') else env
        super().__init__(env, inner.SAFE_Z, inner.HOVER_Z)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/training/test_vertical_envs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/envs/vertical_envs.py tests/training/test_vertical_envs.py
git commit -m "feat: implement AscendTrainEnv and DescendTrainEnv"
```
