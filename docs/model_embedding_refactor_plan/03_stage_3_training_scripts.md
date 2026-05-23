# Training & Evaluation Scripts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide the top-level scripts to train the SAC models using Stable Baselines3, and update existing evaluation tools to benchmark the `ModelEmbeddedController`.

**Architecture:** Create `training/train_models.py` which initializes the respective wrapper environments and runs `SAC.learn()`. Modify `scripts/eval_stages.py` to optionally load and use models instead of the scripted logic.

**Tech Stack:** Python, Stable Baselines3, Gymnasium

---

### Task 1: Create the Training Script

**Files:**
- Create: `training/train_models.py`
- Modify: `tests/training/test_train_models.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/training/test_train_models.py
import pytest
from training.train_models import get_env_for_stage

def test_get_env_for_stage():
    from training.envs.transit_env import TransitTrainEnv
    from training.envs.vertical_envs import AscendTrainEnv, DescendTrainEnv
    
    transit_env = get_env_for_stage("transit")
    assert isinstance(transit_env, TransitTrainEnv)
    
    ascend_env = get_env_for_stage("ascend")
    assert isinstance(ascend_env, AscendTrainEnv)
    
    descend_env = get_env_for_stage("descend")
    assert isinstance(descend_env, DescendTrainEnv)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/training/test_train_models.py -v`
Expected: FAIL (Module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# training/train_models.py
import gymnasium as gym
from stable_baselines3 import SAC
import os
import argparse

from training.envs.transit_env import TransitTrainEnv
from training.envs.vertical_envs import AscendTrainEnv, DescendTrainEnv

def get_env_for_stage(stage: str):
    import src.chess_env.task # Ensure env is registered
    base_env = gym.make("ChessFetchTask-v0", show_chess_pieces=False)
    
    if stage == "transit":
        return TransitTrainEnv(base_env)
    elif stage == "ascend":
        return AscendTrainEnv(base_env)
    elif stage == "descend":
        return DescendTrainEnv(base_env)
    else:
        raise ValueError(f"Unknown stage: {stage}")

def train(stage: str, total_timesteps: int, save_dir: str):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    os.makedirs(save_dir, exist_ok=True)
    
    # Create wrapped env
    def make_env():
        return get_env_for_stage(stage)
    
    venv = DummyVecEnv([make_env])
    # Add normalization for observations and rewards
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    model = SAC("MultiInputPolicy", venv, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    
    # Save both model and normalizer
    model.save(os.path.join(save_dir, f"sac_{stage}"))
    venv.save(os.path.join(save_dir, f"vec_normalize_{stage}.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["transit", "ascend", "descend"], required=True)
    parser.add_argument("--timesteps", type=int, default=1000000) # Increased to 1M
    args = parser.parse_args()
    
    train(args.stage, args.timesteps, "training/models")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/training/test_train_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add training/train_models.py tests/training/test_train_models.py
git commit -m "feat: implement training script for SAC models"
```

### Task 2: Inject ModelEmbeddedController into GameOrchestrator

**Files:**
- Modify: `src/chess_game/game_orchestrator.py:100-110`
- Modify: `src/physical/plan_executor.py`
- Modify: `src/physical/movement_executor.py`

*Note: We need to allow `GameOrchestrator` to optionally use the models.*

- [ ] **Step 1: Write the failing test**

```python
# tests/chess_game/test_game_orchestrator.py (Modify to test model loading)
def test_orchestrator_can_use_model_controller(app_config):
    from src.chess_game.game_orchestrator import GameOrchestrator
    
    app_config["use_rl_models"] = True
    app_config["transit_model_path"] = None
    app_config["ascend_model_path"] = None
    app_config["descend_model_path"] = None
    
    orchestrator = GameOrchestrator(app_config, env_cfg={}, physics_cfg={})
    
    from src.chess_env.model_controller import ModelEmbeddedController
    assert isinstance(orchestrator.physical_executor.movement_exec.controller, ModelEmbeddedController)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chess_game/test_game_orchestrator.py -v -k test_orchestrator_can_use_model_controller`
Expected: FAIL (is ScriptedController)

- [ ] **Step 3: Write minimal implementation**

```python
# Modify src/physical/movement_executor.py
# Change `self.controller = ScriptedController(...)` to conditional:
    def __init__(self, env, use_rl_models=False, transit_model_path=None, ascend_model_path=None, descend_model_path=None):
        self.env = env
        if use_rl_models:
            from src.chess_env.model_controller import ModelEmbeddedController
            self.controller = ModelEmbeddedController(env)
            if transit_model_path:
                self.controller.load_transit_model(transit_model_path)
            if ascend_model_path:
                self.controller.load_ascend_model(ascend_model_path)
            if descend_model_path:
                self.controller.load_descend_model(descend_model_path)
        else:
            from src.chess_env.controller import ScriptedController
            self.controller = ScriptedController(env)
        # ... rest of init

# Modify src/physical/plan_executor.py
    def __init__(self, env, use_rl_models=False, **model_paths):
        self.movement_exec = MovementExecutor(env, use_rl_models=use_rl_models, **model_paths)
        # ... rest of init
        
# Modify src/chess_game/game_orchestrator.py
    def __init__(self, app_config, env_cfg, physics_cfg):
        # Extract model configs safely
        use_rl = app_config.get("use_rl_models", False)
        transit = app_config.get("transit_model_path")
        ascend = app_config.get("ascend_model_path")
        descend = app_config.get("descend_model_path")
        
        self.physical_executor = PhysicalPlanExecutor(
            self.env, 
            use_rl_models=use_rl,
            transit_model_path=transit,
            ascend_model_path=ascend,
            descend_model_path=descend
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chess_game/test_game_orchestrator.py -v -k test_orchestrator_can_use_model_controller`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/chess_game/game_orchestrator.py src/physical/plan_executor.py src/physical/movement_executor.py tests/chess_game/test_game_orchestrator.py
git commit -m "feat: integrate ModelEmbeddedController into execution pipeline"
```

### Task 3: Update `eval_stages.py` for RL benchmarking

**Files:**
- Modify: `scripts/eval_stages.py`

- [ ] **Step 1: Write the failing test**

```bash
# Verify it runs (no failure expected, just setup validation)
python scripts/eval_stages.py --help
```

- [ ] **Step 2: Run test to verify it fails**
N/A (Just checking arguments)

- [ ] **Step 3: Write minimal implementation**

```python
# In scripts/eval_stages.py, add to argparse:
    parser.add_argument("--use-models", action="store_true", help="Use ModelEmbeddedController instead of ScriptedController")
    parser.add_argument("--model-dir", type=str, default="training/models", help="Directory containing SAC models")

# Change controller instantiation:
    if args.use_models:
        from src.chess_env.model_controller import ModelEmbeddedController
        ctrl = ModelEmbeddedController(env, drift_limit=args.drift_limit, render_fn=render_fn, render_delay=args.delay)
        import os
        transit_path = os.path.join(args.model_dir, "sac_transit.zip")
        if os.path.exists(transit_path): ctrl.load_transit_model(transit_path)
        ascend_path = os.path.join(args.model_dir, "sac_ascend.zip")
        if os.path.exists(ascend_path): ctrl.load_ascend_model(ascend_path)
        descend_path = os.path.join(args.model_dir, "sac_descend.zip")
        if os.path.exists(descend_path): ctrl.load_descend_model(descend_path)
    else:
        from src.chess_env.controller import ScriptedController
        ctrl = ScriptedController(env, drift_limit=args.drift_limit, render_fn=render_fn, render_delay=args.delay)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python scripts/eval_stages.py --use-models --stages transit --n-episodes 1`
Expected: Output showing the stage evaluated (even if using scripted fallback because models don't exist yet).

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_stages.py
git commit -m "feat: add --use-models flag to eval_stages for benchmarking"
```
