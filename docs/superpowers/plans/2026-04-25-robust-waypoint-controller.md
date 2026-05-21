# Robust Waypoint Controller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `scripts/v8_latest/test_full_sequence.py` as a production-grade move evaluator with 10 discrete stages, stability handoffs, and error handling.

**Architecture:** A `GrandmasterController` class that orchestrates the state machine. It manages waypoint transitions based on distance/velocity feedback and handles scripted physics-settling phases.

**Tech Stack:** Python, Gymnasium, Stable Baselines3, MuJoCo.

---

### Task 1: The Move Controller Engine

**Files:**
- Create: `scripts/v8_latest/test_full_sequence.py`

- [ ] **Step 1: Implement the `GrandmasterController` structure**
Define the class, its initialization, and the 10-stage state sequence.

```python
import argparse
import gymnasium as gym
import numpy as np
import os
import sys
import time
import logging
from stable_baselines3 import SAC

sys.path.append(os.getcwd())
import chess_env
from scripts.coordinate_mapping import ChessCoordinateMapper

# --- CONSTANTS ---
GRIP_CLOSED = 0.005 # Tight for safety
GRIP_OPEN   = 0.020 # 4cm total span (tight clearance)
STABILITY_THRESH = 0.03 # m/s
PRECISION_THRESH = 0.005 # 5mm
SETTLE_STEPS = 30
GRASP_Z = 0.435
SAFE_Z = 0.550

class GrandmasterController:
    def __init__(self, env, model, mapper, abort_on_fail=True, delay=0.0):
        self.env = env
        self.model = model
        self.mapper = mapper
        self.abort_on_fail = abort_on_fail
        self.delay = delay
        self.logger = logging.getLogger("Controller")
        
    def get_grip_action(self, width):
        # Map 0..0.05 meters to -1..1 action
        return (width / 0.025) - 1.0

    def execute_move(self, start_sq, end_sq):
        p_start = self.mapper.notation_to_world(start_sq)
        p_end = self.mapper.notation_to_world(end_sq)
        
        # Teleport object for testing
        obj_id = self.env.unwrapped.model.joint("object0:joint").id
        qpos_idx = self.env.unwrapped.model.jnt_qposadr[obj_id]
        self.env.unwrapped.data.qpos[qpos_idx : qpos_idx+2] = p_start[:2]
        self.env.unwrapped.data.qpos[qpos_idx + 2] = 0.415
        
        # Define 10-Stage Sequence
        # [Name, Scenario, Target, GripWidth, Type]
        sequence = [
            ("BOOT",     "transit", [0.6, 0.264, SAFE_Z],  GRIP_CLOSED, "settle"),
            ("APPROACH", "transit", [p_start[0], p_start[1], SAFE_Z], GRIP_CLOSED, "rl"),
            ("OPEN",     "transit", [p_start[0], p_start[1], SAFE_Z], GRIP_OPEN,   "settle"),
            ("DESCEND",  "descend", [p_start[0], p_start[1], GRASP_Z], GRIP_OPEN,   "rl"),
            ("GRASP",    "descend", [p_start[0], p_start[1], GRASP_Z], GRIP_CLOSED, "settle"),
            ("LIFT",     "ascend",  [p_start[0], p_start[1], SAFE_Z],  GRIP_CLOSED, "rl"),
            ("CROSS",    "transit", [p_end[0],   p_end[1],   SAFE_Z],  GRIP_CLOSED, "rl"),
            ("DELIVER",  "descend", [p_end[0],   p_end[1],   GRASP_Z], GRIP_CLOSED, "rl"),
            ("RELEASE",  "descend", [p_end[0],   p_end[1],   GRASP_Z], GRIP_OPEN,   "settle"),
            ("RECOVER",  "ascend",  [p_end[0],   p_end[1],   SAFE_Z],  GRIP_OPEN,   "rl"),
            ("CLOSE",    "transit", [p_end[0],   p_end[1],   SAFE_Z],  GRIP_CLOSED, "settle"),
            ("PARK",     "transit", [0.6, 0.264, SAFE_Z],  GRIP_CLOSED, "rl"),
        ]

        print(f"\n>>> EXECUTING MOVE: {start_sq} -> {end_sq}")
        for stage_name, scenario, target, grip, stage_type in sequence:
            success = self._run_stage(stage_name, scenario, target, grip, stage_type)
            if not success and self.abort_on_fail:
                print(f"!!! CRITICAL FAILURE IN {stage_name}. ABORTING.")
                return False
        return True

    def _run_stage(self, name, scenario, target, grip, stage_type):
        print(f"[{name:10s}] Target: {np.round(target,3)} | Grip: {grip:.3f}")
        
        # Update Environment Internals
        self.env.unwrapped.current_scenario = scenario
        self.env.unwrapped.goal = np.array(target)
        self.env.unwrapped.goal_pos = np.array(target)
        if scenario in ["descend", "ascend"]:
            self.env.unwrapped.tube_center_xy = np.array([target[0], target[1]])
        else:
            self.env.unwrapped.tube_center_xy = None
        
        grip_action = self.get_grip_action(grip)
        
        if stage_type == "settle":
            for _ in range(SETTLE_STEPS):
                self.env.step(np.array([0,0,0, grip_action]))
                if self.env.render_mode == "human": self.env.render()
                if self.delay > 0: time.sleep(self.delay)
            return True

        # RL Phase
        steps = 0
        while steps < 150:
            obs = self.env.unwrapped._get_obs()
            action, _ = self.model.predict(obs, deterministic=True)
            action[3] = grip_action # Force gripper state
            
            obs_new, reward, term, trunc, info = self.env.step(action)
            if self.env.render_mode == "human": self.env.render()
            if self.delay > 0: time.sleep(self.delay)
            
            # FEEDBACK
            g_pos = obs_new["observation"][0:3]
            g_vel = np.linalg.norm(obs_new["observation"][20:23])
            dist = np.linalg.norm(g_pos - target)
            
            if dist < PRECISION_THRESH and g_vel < STABILITY_THRESH:
                print(f"  √ STABLE SUCCESS at step {steps} (Dist: {dist:.4f}, Vel: {g_vel:.4f})")
                return True
            
            if term and not info.get("is_success"):
                print(f"  X CRITICAL CRASH: Boundary breached! (Dist: {dist:.4f})")
                return False
                
            steps += 1
        
        print(f"  X TIMEOUT: Failed to stabilize within 150 steps.")
        return False
```

- [ ] **Step 2: Implement Argument Parsing and Main Entry**

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--start", default="e2")
    parser.add_argument("--end", default="e4")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--continue-on-fail", action="store_true")
    args = parser.parse_args()

    # Model Auto-Selection logic
    if not args.model:
        import glob
        runs = sorted(glob.glob("checkpoints/pure_movement_v6_*"))
        if runs:
            args.model = os.path.join(runs[-1], "best_model_combined.zip")
            print(f"Auto-selected model: {args.model}")

    env = gym.make("ChessFetchDense-v0", render_mode=None if args.no_render else "human", hide_object=False)
    model = SAC.load(args.model, env=env)
    mapper = ChessCoordinateMapper()
    
    controller = GrandmasterController(env, model, mapper, 
                                       abort_on_fail=not args.continue_on_fail,
                                       delay=args.delay)
    
    successes = 0
    for i in range(args.trials):
        if controller.execute_move(args.start, args.end):
            successes += 1
    
    print(f"\nFinal Result: {successes}/{args.trials} successful moves.")
    env.close()
```

- [ ] **Step 3: Verification**
Run: `PYTHONPATH=. .venv/bin/python3 scripts/v8_latest/test_full_sequence.py --trials 2 --no-render`
Expected: Full sequence logs for e2->e4, showing "√ STABLE SUCCESS" for all RL stages.

---

### Task 2: README Documentation Update

**Files:**
- Modify: `scripts/v8_latest/README.md`

- [ ] **Step 1: Update README.md**
Add full documentation for `test_full_sequence.py` arguments and stage definitions.
