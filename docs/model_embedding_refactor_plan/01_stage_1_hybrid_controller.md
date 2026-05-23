# Hybrid Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a `ModelEmbeddedController` that loads three SAC models and uses them for `run_transit`, `run_ascend`, and `run_descend`, acting as a drop-in replacement for `ScriptedController`.

**Architecture:** Create a new `ModelEmbeddedController` class matching the interface of `ScriptedController`. When models are loaded, it queries them for actions and applies the `[dx, dy, dz]` movement deltas while suppressing the gripper action. Unimplemented models fall back to proportional logic.

**Tech Stack:** Python, Gymnasium, Stable Baselines3 (SAC), MuJoCo

---

### Task 1: Create ModelEmbeddedController Base Class

**Files:**
- Create: `src/chess_env/model_controller.py`
- Modify: `tests/chess_env/test_model_controller.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/chess_env/test_model_controller.py
import pytest
import numpy as np
from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_env.controller import StageResult

def test_model_controller_initialization():
    class DummyEnv:
        def __init__(self):
            self.SAFE_Z = 0.530
            self.HOVER_Z = 0.460
            self.env = self
    
    env = DummyEnv()
    controller = ModelEmbeddedController(env)
    assert controller.transit_model is None
    assert controller.ascend_model is None
    assert controller.descend_model is None
    assert hasattr(controller, 'run_transit')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chess_env/test_model_controller.py::test_model_controller_initialization -v`
Expected: FAIL (File/module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/chess_env/model_controller.py
from typing import Optional
import numpy as np
from src.chess_env.controller import ScriptedController, StageResult

class ModelEmbeddedController(ScriptedController):
    """
    Controller that uses loaded RL models for movement stages.
    Falls back to ScriptedController behavior if a model is not loaded.
    """
    def __init__(self, env, drift_limit: float = 0.010, render_fn=None, render_delay: float = 0.0):
        super().__init__(env, drift_limit, render_fn, render_delay)
        self.transit_model = None
        self.ascend_model = None
        self.descend_model = None

    def load_transit_model(self, model_path: str):
        pass # To be implemented
        
    def load_ascend_model(self, model_path: str):
        pass # To be implemented
        
    def load_descend_model(self, model_path: str):
        pass # To be implemented
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chess_env/test_model_controller.py::test_model_controller_initialization -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/chess_env/test_model_controller.py src/chess_env/model_controller.py
git commit -m "feat: add ModelEmbeddedController base class"
```

### Task 2: Implement Model Loading Logic

**Files:**
- Modify: `src/chess_env/model_controller.py:17-25`
- Modify: `tests/chess_env/test_model_controller.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/chess_env/test_model_controller.py
from stable_baselines3 import SAC

def test_model_controller_loads_models(tmp_path):
    # Create dummy env and model
    class DummyEnv:
        def __init__(self):
            self.SAFE_Z = 0.530
            self.env = self
            
    env = DummyEnv()
    controller = ModelEmbeddedController(env)
    
    # We just need to mock SAC.load to return a string or mock object
    import unittest.mock as mock
    with mock.patch('stable_baselines3.SAC.load') as mock_load:
        mock_load.return_value = "dummy_model"
        
        controller.load_transit_model("fake_path.zip")
        assert controller.transit_model == "dummy_model"
        
        controller.load_ascend_model("fake_path.zip")
        assert controller.ascend_model == "dummy_model"
        
        controller.load_descend_model("fake_path.zip")
        assert controller.descend_model == "dummy_model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chess_env/test_model_controller.py::test_model_controller_loads_models -v`
Expected: FAIL (model is None)

- [ ] **Step 3: Write minimal implementation**

```python
# Replace load methods in src/chess_env/model_controller.py
    def load_transit_model(self, model_path: str, norm_path: str = None):
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        self.transit_model = SAC.load(model_path)
        if norm_path:
            # We need a dummy env to wrap VecNormalize around
            venv = DummyVecEnv([lambda: self._env])
            self.transit_norm = VecNormalize.load(norm_path, venv)
            self.transit_norm.training = False
            self.transit_norm.norm_reward = False

    # Repeat for load_ascend_model and load_descend_model ...

    def _run_rl_movement_loop(self, target_pos: np.ndarray, model, normalizer=None, tolerance: float, max_steps: int, abort_fn=None) -> StageResult:
        import time
        env = self._env
        for step in range(max_steps):
            # ... (safety checks same as before)

            obs = env._get_obs()
            if normalizer:
                obs = normalizer.normalize_obs(obs)

            action, _ = model.predict(obs, deterministic=True)
            # ...


            if self._render_fn is not None:
                self._render_fn()
                if self._render_delay > 0:
                    time.sleep(self._render_delay)

        grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
        dist = float(np.linalg.norm(target_pos - grip_pos))
        return StageResult(success=False, steps=max_steps, crash_reason="TIMEOUT", final_pos=grip_pos.copy(), error_mm=dist * 1000)

    def run_transit(self, target_xy: np.ndarray) -> StageResult:
        if self.transit_model is None:
            return super().run_transit(target_xy)
            
        env = self._env
        env._debug_current_phase = "transit"
        target = np.array([target_xy[0], target_xy[1], env.SAFE_Z])

        def transit_abort(grip_pos):
            if grip_pos[2] < self.FLOOR_LIMIT:
                return True, f"FLOOR_HIT (z={grip_pos[2]:.4f})"
            if env.grasp_mode:
                held, reason = env._check_cube_held(grip_pos)
                if not held:
                    return True, reason
            return False, None

        return self._run_rl_movement_loop(
            target_pos=target,
            model=self.transit_model,
            tolerance=self.TRANSIT_TOLERANCE_M,
            max_steps=self.TRANSIT_MAX_STEPS,
            abort_fn=transit_abort
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chess_env/test_model_controller.py::test_model_controller_run_transit_with_model -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/chess_env/model_controller.py tests/chess_env/test_model_controller.py
git commit -m "feat: implement RL inference loop and run_transit override"
```

### Task 4: Implement Ascend and Descend Overrides

**Files:**
- Modify: `src/chess_env/model_controller.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/chess_env/test_model_controller.py
def test_model_controller_ascend_descend_with_model():
    class DummyModel:
        def predict(self, obs, deterministic=True):
            return np.array([0.0, 0.0, 0.01, 1.0]), None
            
    class DummyEnv:
        def __init__(self):
            self.SAFE_Z = 0.530
            self.HOVER_Z = 0.460
            self.env = self
            self.grasp_mode = False
            self.step_count = 0
            self.tube_center_xy = np.array([0.0, 0.0])
            self.TABLE_SURFACE_Z = 0.400
            
            class DummyData:
                mocap_pos = np.zeros((1, 3))
                mocap_quat = np.zeros((1, 4))
            self.data = DummyData()
            self.VERTICAL_QUAT = np.zeros(4)
            
            class DummyUtils:
                def get_site_xpos(self, model, data, site):
                    return np.array([0.0, 0.0, 0.460])
            self._utils = DummyUtils()
            self.model = None

        def _get_obs(self):
            return {"observation": np.zeros(10)}

        def _set_action(self, action):
            self.last_action = action

        def _mujoco_step(self, action):
            self.step_count += 1
            self._utils.get_site_xpos = lambda m, d, s: np.array([0.0, 0.0, 0.530])

        def _check_cube_held(self, pos):
            return True, None

    env = DummyEnv()
    controller = ModelEmbeddedController(env)
    controller.ascend_model = DummyModel()
    controller.descend_model = DummyModel()
    
    # Test Ascend
    res_ascend = controller.run_ascend()
    assert res_ascend.success is True
    
    # Test Descend
    env.step_count = 0
    env._utils.get_site_xpos = lambda m, d, s: np.array([0.0, 0.0, 0.530])
    env._mujoco_step = lambda a: setattr(env._utils, 'get_site_xpos', lambda m,d,s: np.array([0.0, 0.0, 0.460]))
    
    res_descend = controller.run_descend()
    assert res_descend.success is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chess_env/test_model_controller.py::test_model_controller_ascend_descend_with_model -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# Add to src/chess_env/model_controller.py
    def run_descend(self) -> StageResult:
        if self.descend_model is None:
            return super().run_descend()
            
        env = self._env
        env._debug_current_phase = "descend"
        target_xy = env.tube_center_xy
        target = np.array([target_xy[0], target_xy[1], env.HOVER_Z])

        def descend_abort(grip_pos):
            drift = float(np.linalg.norm(grip_pos[:2] - target_xy))
            if drift > self.drift_limit:
                return True, f"TUBE_BREACH (drift={drift*1000:.1f}mm)"
            if grip_pos[2] < env.TABLE_SURFACE_Z:
                return True, f"TABLE_HIT (z={grip_pos[2]:.4f})"
            return False, None

        return self._run_rl_movement_loop(
            target_pos=target,
            model=self.descend_model,
            tolerance=self.VERTICAL_TOLERANCE_M,
            max_steps=self.VERTICAL_MAX_STEPS,
            abort_fn=descend_abort
        )

    def run_ascend(self) -> StageResult:
        if self.ascend_model is None:
            return super().run_ascend()
            
        env = self._env
        env._debug_current_phase = "ascend"
        target_xy = env.tube_center_xy
        target = np.array([target_xy[0], target_xy[1], env.SAFE_Z])

        def ascend_abort(grip_pos):
            drift = float(np.linalg.norm(grip_pos[:2] - target_xy))
            if drift > self.drift_limit:
                return True, f"TUBE_BREACH (drift={drift*1000:.1f}mm)"
            if env.grasp_mode:
                held, reason = env._check_cube_held(grip_pos)
                if not held:
                    return True, reason
            return False, None

        return self._run_rl_movement_loop(
            target_pos=target,
            model=self.ascend_model,
            tolerance=self.VERTICAL_TOLERANCE_M,
            max_steps=self.VERTICAL_MAX_STEPS,
            abort_fn=ascend_abort
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chess_env/test_model_controller.py::test_model_controller_ascend_descend_with_model -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/chess_env/model_controller.py tests/chess_env/test_model_controller.py
git commit -m "feat: implement ascend and descend overrides in ModelEmbeddedController"
```
