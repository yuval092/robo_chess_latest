# Scripts Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a clean, up-to-date scripts folder (`scripts/v8_latest`) for verification, evaluation, and visualization.

**Architecture:** Independent utility scripts that share a common dependency on `ChessFetchDenseEnv-v0` and the latest trained SAC model.

**Tech Stack:** MuJoCo, Gymnasium, Stable Baselines3, NumPy.

---

### Task 1: Scene Verification Script

**Files:**
- Create: `scripts/v8_latest/verify_scene.py`

- [ ] **Step 1: Implement `verify_scene.py`**

```python
import mujoco
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())
from chess_env.chess_fetch_env import ChessFetchEnv

def verify():
    print("--- XML Scene Verification ---")
    model_path = os.path.join("chess_env", "assets", "pick_and_place.xml")
    if not os.path.exists(model_path):
        print(f"FAILED: Model file not found at {model_path}")
        return
    
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        print("PASS: XML Compiled Successfully")
    except Exception as e:
        print(f"FAILED: XML Compilation Error: {e}")
        return

    # Verify Table Position
    table_id = model.body("table0").id
    table_pos = model.body_pos[table_id]
    expected_table_pos = ChessFetchEnv.TABLE_CENTER_XY
    print(f"Table Pos: {table_pos[:2]} (Expected: {expected_table_pos})")
    if not np.allclose(table_pos[:2], expected_table_pos, atol=0.001):
        print("FAILED: Table position mismatch")
    else:
        print("PASS: Table position correct")

    # Check for initial collisions
    mujoco.mj_step(model, data)
    if data.ncon > 0:
        print(f"WARNING: Found {data.ncon} contacts in initial state. Checking for depth...")
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.dist < -0.001:
                geom1 = model.geom(contact.geom1).name
                geom2 = model.geom(contact.geom2).name
                print(f"  CRITICAL COLLISION: {geom1} and {geom2} (dist: {contact.dist:.4f})")
    else:
        print("PASS: No initial collisions found")

if __name__ == "__main__":
    verify()
```

- [ ] **Step 2: Run verification**

Run: `PYTHONPATH=. .venv/bin/python3 scripts/v8_latest/verify_scene.py`
Expected: "PASS: XML Compiled Successfully", "PASS: Table position correct", "PASS: No initial collisions found".

### Task 2: Scenario Start/Goal Visualizer

**Files:**
- Create: `scripts/v8_latest/visualize_scenarios.py`

- [ ] **Step 1: Implement `visualize_scenarios.py`**

```python
import argparse
import gymnasium as gym
import mujoco
import numpy as np
import os
import sys
import time

sys.path.append(os.getcwd())
import chess_env

def visualize(scenario):
    print(f"--- Visualizing Scenario: {scenario.upper()} ---")
    env = gym.make("ChessFetchDense-v0", force_scenario=scenario, render_mode="human")
    obs, info = env.reset()
    
    # The goal is stored in env.unwrapped.goal
    goal = env.unwrapped.goal_pos
    print(f"Start Position: {obs['achieved_goal']}")
    print(f"Goal Position:  {goal}")
    
    print("Close the window to exit.")
    while True:
        try:
            env.render()
            time.sleep(0.1)
        except:
            break
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["transit", "descend", "ascend"], default="transit")
    args = parser.parse_args()
    visualize(args.scenario)
```

### Task 3: High-Fidelity Evaluation Script

**Files:**
- Create: `scripts/v8_latest/evaluate_production.py`

- [ ] **Step 1: Implement `evaluate_production.py`**

```python
import argparse
import gymnasium as gym
import numpy as np
import os
import sys
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.getcwd())
import chess_env

def run_eval(model_path, n_episodes):
    print(f"--- Production Evaluation: {model_path} ---")
    
    # Load model with custom_objects to handle potential loading issues
    custom_objects = {"ent_coef": 0.001}
    try:
        model = SAC.load(model_path, custom_objects=custom_objects)
    except:
        model = SAC.load(model_path)

    scenarios = ["transit", "descend", "ascend"]
    results = {}

    for sc in scenarios:
        print(f"Evaluating {sc}...")
        env = DummyVecEnv([lambda s=sc: Monitor(gym.make("ChessFetchDense-v0", 
                                                   force_scenario=s, 
                                                   force_drift_limit=0.035))])
        
        successes = 0
        crashes = 0
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_reward += reward[0]
                if done[0]:
                    if info[0].get("is_success"):
                        successes += 1
                    elif ep_reward < -400: # Threshold for CRASH_PENALTY
                        crashes += 1
        
        results[sc] = {"success": successes / n_episodes, "crash": crashes / n_episodes}
        env.close()

    print("\n" + "="*30)
    print("PRODUCTION EVALUATION SUMMARY")
    print("="*30)
    for sc, res in results.items():
        print(f"{sc.upper():8s} | Success: {res['success']:>6.1%} | Crashes: {res['crash']:>6.1%}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chess_fetch_pure_v6_20260424_232625.zip")
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()
    run_eval(args.model, args.n)
```

### Task 4: Visual Execution Script (Slow Motion)

**Files:**
- Create: `scripts/v8_latest/visual_execution.py`

- [ ] **Step 1: Implement `visual_execution.py`**

```python
import argparse
import gymnasium as gym
import numpy as np
import os
import sys
import time
from stable_baselines3 import SAC

sys.path.append(os.getcwd())
import chess_env

def run_visual(model_path, scenario, delay):
    print(f"--- Visual Execution: {scenario.upper()} (Delay: {delay}s) ---")
    env = gym.make("ChessFetchDense-v0", force_scenario=scenario, render_mode="human")
    
    custom_objects = {"ent_coef": 0.001}
    try:
        model = SAC.load(model_path, custom_objects=custom_objects)
    except:
        model = SAC.load(model_path)

    while True:
        obs, info = env.reset()
        done = False
        print("Episode Start")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()
            if delay > 0:
                time.sleep(delay)
        
        print(f"Episode End | Success: {info.get('is_success')}")
        time.sleep(1.0) # Pause between episodes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chess_fetch_pure_v6_20260424_232625.zip")
    parser.add_argument("--scenario", choices=["transit", "descend", "ascend"], default="descend")
    parser.add_argument("--delay", type=float, default=0.05, help="Delay between steps in seconds")
    args = parser.parse_args()
    run_visual(args.model, args.scenario, args.delay)
```

### Task 5: README for the Folder

**Files:**
- Create: `scripts/v8_latest/README.md`

- [ ] **Step 1: Implement `README.md`**

```markdown
# V8 Latest Scripts

This folder contains the authoritative scripts for the current state of the RoboChess Fine-Tuning project.

## Scripts

1.  **`verify_scene.py`**: Validates the XML environment, checks for initial collisions, and confirms coordinate mapping.
2.  **`visualize_scenarios.py`**: Shows the starting position and goal for each scenario without moving.
    *   Usage: `python scripts/v8_latest/visualize_scenarios.py --scenario [transit|descend|ascend]`
3.  **`evaluate_production.py`**: High-fidelity evaluation of the trained model with tightest drift constraints.
    *   Usage: `python scripts/v8_latest/evaluate_production.py --model path_to_model.zip --n 100`
4.  **`visual_execution.py`**: Real-time visualization of the model's movement with adjustable delay.
    *   Usage: `python scripts/v8_latest/visual_execution.py --scenario descend --delay 0.05`

## Latest Model
The default model used by these scripts is `chess_fetch_pure_v6_20260424_232625.zip`.
```
