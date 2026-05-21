# RoboChess Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up the `robo_chess_fine_tuning` project by refactoring environment/training logic into a modular `src/` structure, centralizing constants into YAML configs, and consolidating dozens of loose scripts into four robust entry points.

**Architecture:** A configuration-driven architecture where `configs/*.yaml` dictate simulation, training, and environment parameters. The core logic resides in `src/chess_env/` and `src/training/`, while user-facing execution happens through `scripts/[train|eval|visualize|verify_physics].py`.

**Tech Stack:** Python 3.12, MuJoCo, Gymnasium Robotics, Stable Baselines3 (SAC), PyYAML, NumPy.

---

### Task 1: Setup Configuration System

**Files:**
- Create: `configs/env.yaml`
- Create: `configs/training.yaml`
- Create: `configs/physics.yaml`
- Create: `src/utils/config.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add PyYAML dependency**
Ensure PyYAML is in `requirements.txt` if not already present.

```bash
echo "PyYAML>=6.0" >> requirements.txt
pip install PyYAML
```

- [ ] **Step 2: Create env.yaml**
Extract board geometry, Z-constants, and success limits from `chess_env/chess_fetch_env.py` and `chess_env/chess_fetch_dense_env.py`.

```yaml
# configs/env.yaml
cube_height: 0.030
cube_z: 0.415  # TABLE_Z + cube_height/2
grasp_z: 0.430
safe_z: 0.550
success_threshold: 0.010
success_bonus: 500.0
drift_limit_start: 0.100
drift_limit_end: 0.015
floor_limit: 0.400
crash_penalty: -500.0
hidden_object_pos: [2.0, 2.0, -1.0]
open_finger_pos: 0.05
closed_finger_pos: 0.010
table_surface_z: 0.400
```

- [ ] **Step 3: Create training.yaml**
Extract hyperparameters from `scripts/fine_tune_dense.py`.

```yaml
# configs/training.yaml
total_timesteps: 1000000
num_envs: 12
base_model: "models/sac-FetchPickAndPlace-v4.zip"
target_curriculum_total: 500000
learning_rate: 0.00005
batch_size: 512
target_entropy: -4.0
learning_starts: 10000
initial_ent_coef: 0.1
ent_coef_lr: 0.001
```

- [ ] **Step 4: Create physics.yaml**
Extract mujoco parameters from `chess_env/chess_fetch_env.py` and `chess_env/chess_fetch_dense_env.py`.

```yaml
# configs/physics.yaml
max_settle_steps: 500
settle_steps: 25
settle_tolerance: 0.003
settle_gain: 0.3
vertical_quat: [1.0, 0.0, 1.0, 0.0]
```

- [ ] **Step 5: Create config loader**
Create `src/utils/config.py` to load these YAMLs safely.

```python
# src/utils/config.py
import yaml
import os

def load_config(config_name: str) -> dict:
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', f'{config_name}.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

- [ ] **Step 6: Commit config system**

```bash
git add configs/ src/utils/config.py requirements.txt
git commit -m "feat: add centralized YAML configuration system"
```

### Task 2: Refactor Environment to `src/chess_env/`

**Files:**
- Create: `src/chess_env/__init__.py`
- Create: `src/chess_env/simulation.py`
- Create: `src/chess_env/task.py`

- [ ] **Step 1: Create simulation.py (from chess_fetch_env.py)**
Move `ChessFetchEnv` into `src/chess_env/simulation.py`. Replace hardcoded values like `VERTICAL_QUAT` with config values.

```python
# src/chess_env/simulation.py
import os
import numpy as np
import mujoco
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv
from src.utils.config import load_config

class ChessSimulationEnv(MujocoFetchPickAndPlaceEnv):
    def __init__(self, **kwargs):
        self.env_cfg = load_config('env')
        self.phys_cfg = load_config('physics')
        # Replace hardcoded constants with config values
        self.VERTICAL_QUAT = np.array(self.phys_cfg['vertical_quat'])
        
        # Override XML path
        if 'model_path' not in kwargs:
            kwargs['model_path'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'chess_env', 'assets', 'pick_and_place.xml'))
        super().__init__(**kwargs)
        
    # (Copy the rest of the logic from chess_fetch_env.py, adapting naming as needed)
    # Ensure _set_action, _env_setup, _sample_board_position, generate_mujoco_observations are present.
```
*(Note: Use `read_file` to get the full logic of `chess_env/chess_fetch_env.py` and adapt it during execution.)*

- [ ] **Step 2: Create task.py (from chess_fetch_dense_env.py)**
Move `ChessFetchDenseEnv` into `src/chess_env/task.py`, inheriting from `ChessSimulationEnv`. Replace hardcoded constants with `self.env_cfg` values.

```python
# src/chess_env/task.py
import logging
import os
import mujoco
import numpy as np
from src.chess_env.simulation import ChessSimulationEnv
from src.utils.config import load_config

os.makedirs("logs/env_debug", exist_ok=True)

class ChessTaskEnv(ChessSimulationEnv):
    def __init__(self, force_scenario=None, drift_curriculum_steps=285714, force_drift_limit=None, hide_object=True, **kwargs):
        self.env_cfg = load_config('env')
        self.phys_cfg = load_config('physics')
        
        # Use config values
        self.GRASP_Z = self.env_cfg['grasp_z']
        self.SAFE_Z = self.env_cfg['safe_z']
        # ... map all other config values
        
        # Keep logging setup from previous work
        # Call super().__init__(**kwargs)
        
    # (Copy the rest of the logic from chess_fetch_dense_env.py, replacing constants with config lookups)
```
*(Note: The executing agent must copy the methods from `chess_env/chess_fetch_dense_env.py` and replace constants with dictionary lookups).*

- [ ] **Step 3: Create gym registry in __init__.py**

```python
# src/chess_env/__init__.py
from gymnasium.envs.registration import register

register(
    id="ChessFetchTask-v0",
    entry_point="src.chess_env.task:ChessTaskEnv",
    max_episode_steps=200,
)
```

- [ ] **Step 4: Commit environment refactor**

```bash
git add src/chess_env/
git commit -m "refactor: move environment logic to src/chess_env and integrate configs"
```

### Task 3: Refactor Training and Utilities

**Files:**
- Create: `src/utils/logger.py`
- Create: `src/training/callbacks.py`
- Create: `src/training/trainer.py`

- [ ] **Step 1: Create logger utility**

```python
# src/utils/logger.py
import logging
import os

def setup_logger(name: str, log_file: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
        logger.addHandler(handler)
    return logger
```

- [ ] **Step 2: Create callbacks.py**
Move `DetailedLoggingCallback`, `SuccessRateEvalCallback`, and `CombinedSuccessCallback` from `scripts/fine_tune_dense.py` to `src/training/callbacks.py`.

```python
# src/training/callbacks.py
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from src.utils.logger import setup_logger

progress_logger = setup_logger("training_progress", "logs/training_progress_detailed.log")

# (Copy classes from fine_tune_dense.py, updating logger references)
```

- [ ] **Step 3: Create trainer.py**
Move `find_best_checkpoint`, `make_env`, `make_eval_env`, and the core `fine_tune_dense` logic into a `Trainer` class in `src/training/trainer.py`.

```python
# src/training/trainer.py
import os
import glob
from datetime import datetime
import gymnasium as gym
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import src.chess_env  # registers env
from src.training.callbacks import DetailedLoggingCallback, SuccessRateEvalCallback, CombinedSuccessCallback, CallbackList, progress_logger
from src.utils.config import load_config

class SACTrainer:
    def __init__(self, fresh_start=False, num_envs=None):
        self.train_cfg = load_config('training')
        self.num_envs = num_envs if num_envs is not None else self.train_cfg['num_envs']
        self.fresh_start = fresh_start
        self.base_model_path = self.train_cfg['base_model']
        self.target_curriculum_total = self.train_cfg['target_curriculum_total']
        self.total_timesteps = self.train_cfg['total_timesteps']
        
    # Implement find_best_checkpoint, make_env, make_eval_env as static or instance methods
    # Implement train() method adapting fine_tune_dense logic
```
*(Note: Full logic adaptation to be performed by executing subagent)*

- [ ] **Step 4: Commit training refactor**

```bash
git add src/utils/logger.py src/training/
git commit -m "refactor: extract training logic and callbacks into src/training"
```

### Task 4: Create Entry Scripts

**Files:**
- Create: `scripts/train.py`
- Create: `scripts/eval.py`
- Create: `scripts/verify_physics.py`
- Create: `scripts/visualize.py`

- [ ] **Step 1: Create scripts/train.py**

```python
# scripts/train.py
import argparse
import sys
import os
sys.path.append(os.getcwd())
from src.training.trainer import SACTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=None, help="Number of environments")
    parser.add_argument("--fresh", action="store_true", help="Force cold start from base model")
    args = parser.parse_args()
    
    trainer = SACTrainer(fresh_start=args.fresh, num_envs=args.envs)
    trainer.train()
```

- [ ] **Step 2: Create scripts/eval.py**
Adapt logic from `scripts/v8_latest/evaluate_production.py` and incorporate `argparse`. Include `--visualize` flag which adds `render_mode="human"` to `gym.make`.

```python
# scripts/eval.py
import argparse
import gymnasium as gym
import sys
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
sys.path.append(os.getcwd())
import src.chess_env

# Implement run_eval(model_path, n_episodes, scenarios, visualize, deterministic)
# Print summary table at the end
```

- [ ] **Step 3: Create scripts/verify_physics.py**
Adapt logic from `scripts/v8_latest/verify_scene.py` and `scripts/verify_grasp_z.py`.

```python
# scripts/verify_physics.py
import argparse
import mujoco
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from src.chess_env.simulation import ChessSimulationEnv
from src.utils.config import load_config

def run_checks():
    # 1. XML Compilation Check
    # 2. Static Stability Check
    # 3. Kinematic Reachability Check
    # 4. Teleport Verification Check
    pass

if __name__ == "__main__":
    run_checks()
```

- [ ] **Step 4: Create scripts/visualize.py**
Adapt from `scripts/v8_latest/visual_execution.py`. Simplify to only render the model executing scenarios continuously.

```python
# scripts/visualize.py
import argparse
import gymnasium as gym
import time
import sys
import os
from stable_baselines3 import SAC
sys.path.append(os.getcwd())
import src.chess_env

def run_visual(model_path, scenario, delay, wait):
    # Implement continuous rendering loop with loaded model
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # add args
    # ...
```

- [ ] **Step 5: Make scripts executable and commit**

```bash
chmod +x scripts/*.py
git add scripts/train.py scripts/eval.py scripts/verify_physics.py scripts/visualize.py
git commit -m "feat: add standardized entry scripts for train, eval, verify, visualize"
```

### Task 5: Execute Cleanup and Documentation

**Files:**
- Modify: `README.md`
- Delete: Obsolete scripts and `scripts/v8_latest`

- [ ] **Step 1: Delete obsolete scripts**

```bash
rm -rf scripts/v8_latest/
rm scripts/fine_tune*.py scripts/evaluate_unit*.py scripts/diag_*.py scripts/repro_*.py scripts/test_*.py scripts/check_*.py scripts/visual_*.py scripts/diagnose_*.py scripts/verify_env.py scripts/verify_model.py scripts/verify_grasp_z.py scripts/zero_shot_eval.py scripts/waypoint_controller.py scripts/demo_waypoints_live.py scripts/interactive_move.py scripts/save_scenario_visuals.py scripts/download_model.py scripts/upgrade_model.py scripts/final_eval.py scripts/per_scenario_eval*.py scripts/training_visualizer*.py scripts/visualize_trajectory.py scripts/coordinate_mapping.py scripts/debug_ids.py scripts/controller_scrutiny*.py scripts/evaluate_phase9.py scripts/evaluate_dense_quality.py
rm chess_env/chess_fetch_env.py chess_env/chess_fetch_dense_env.py
```

- [ ] **Step 2: Update Master README.md**

```markdown
# RoboChess Fine-Tuning

This project provides a robust, configuration-driven Reinforcement Learning environment for fine-tuning a Fetch robotic arm to manipulate chess pieces using MuJoCo and Stable Baselines 3.

## Architecture
- `configs/`: Modular YAML configurations (`env.yaml`, `physics.yaml`, `training.yaml`).
- `src/`: Core environment and training logic.
- `scripts/`: Entry points for interacting with the system.

## Usage
- **Training:** `python scripts/train.py --envs 12 --fresh`
- **Evaluation:** `python scripts/eval.py --model path/to/model.zip --scenarios transit descend ascend --visualize`
- **Verification:** `python scripts/verify_physics.py`
- **Visualization:** `python scripts/visualize.py --model path/to/model.zip --scenario descend`
```

- [ ] **Step 3: Test Imports**
Verify that the entry points load without syntax errors.

```bash
python scripts/train.py --help
python scripts/eval.py --help
python scripts/verify_physics.py --help
python scripts/visualize.py --help
```

- [ ] **Step 4: Final Commit**

```bash
git add -A
git commit -m "chore: cleanup obsolete scripts, replace old env files, and update documentation"
```
