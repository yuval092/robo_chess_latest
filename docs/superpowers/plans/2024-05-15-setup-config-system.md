# Setup Configuration System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a centralized YAML-based configuration system for the RoboChess project to replace hardcoded constants.

**Architecture:** Use YAML files for configuration data and a lightweight Python utility to load them into dictionaries.

**Tech Stack:** Python, PyYAML

---

### Task 1: Environment Preparation

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add PyYAML to requirements.txt**

```text
mujoco
gymnasium-robotics
stable-baselines3
huggingface_hub
numpy
PyYAML>=6.0
```

- [ ] **Step 2: Install PyYAML**

Run: `pip install PyYAML>=6.0`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add PyYAML to requirements"
```

### Task 2: Create Configuration Files

**Files:**
- Create: `configs/env.yaml`
- Create: `configs/training.yaml`
- Create: `configs/physics.yaml`

- [ ] **Step 1: Create configs directory**

Run: `mkdir -p configs`

- [ ] **Step 2: Create configs/env.yaml**

```yaml
cube_height: 0.030
cube_z: 0.415
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

- [ ] **Step 3: Create configs/training.yaml**

```yaml
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

- [ ] **Step 4: Create configs/physics.yaml**

```yaml
max_settle_steps: 500
settle_steps: 25
settle_tolerance: 0.003
settle_gain: 0.3
vertical_quat: [1.0, 0.0, 1.0, 0.0]
```

- [ ] **Step 5: Commit**

```bash
git add configs/
git commit -m "feat: add YAML configuration files"
```

### Task 3: Create Configuration Utility

**Files:**
- Create: `src/utils/config.py`

- [ ] **Step 1: Create src/utils directory**

Run: `mkdir -p src/utils`

- [ ] **Step 2: Create src/utils/config.py**

```python
import yaml
import os

def load_config(config_name: str) -> dict:
    # Resolve path relative to this file
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', f'{config_name}.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```

- [ ] **Step 3: Verify configuration loading**

Create a temporary test script `test_config.py`:
```python
from src.utils.config import load_config
env_cfg = load_config('env')
print(f"Cube height: {env_cfg['cube_height']}")
assert env_cfg['cube_height'] == 0.030
print("Config loading verified!")
```
Run: `python test_config.py`
Expected output: `Config loading verified!`

- [ ] **Step 4: Commit and Cleanup**

Run: `rm test_config.py`
```bash
git add src/utils/config.py
git commit -m "feat: add centralized YAML configuration system"
```
