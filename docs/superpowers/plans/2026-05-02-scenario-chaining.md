# Scenario Chaining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and evaluate the robotic arm + RL model when multiple scenarios are executed back-to-back without a full simulation reset.

**Architecture:** Extend `ChessTaskEnv` with a `soft_reset` mechanism that handles velocity halting, waypoint alignment, and gripper transitions. Use a dedicated `waypoints.py` utility for goal derivation and chain validation.

**Tech Stack:** Python, MuJoCo, Gymnasium, Stable Baselines3 (SAC).

---

## File Structure

- `src/chess_env/waypoints.py`: (New) Centralized logic for valid transitions, scenario exit/entry heights, and nominal goal derivation.
- `src/chess_env/task.py`: (Modify) Add `transition_validate` (diagnostics) and `soft_reset` (state machine orchestration).
- `scripts/eval_sequence.py`: (New) Primary evaluation tool for running and logging multi-scenario chains.

---

### Task 1: Waypoint Utilities

**Files:**
- Create: `src/chess_env/waypoints.py`
- Test: `tests/chess_env/test_waypoints.py`

- [ ] **Step 1: Write tests for waypoint utilities**

```python
import numpy as np
import pytest
from src.chess_env.waypoints import validate_chain, derive_goal_pos, exit_waypoint

def test_validate_chain_valid():
    # Should not raise
    validate_chain(["transit", "descend", "ascend"])

def test_validate_chain_invalid():
    with pytest.raises(ValueError, match="Invalid chain transition"):
        validate_chain(["transit", "ascend"])

def test_derive_goal_pos():
    cell = np.array([0.4, 0.4])
    goal = derive_goal_pos("descend", cell)
    assert np.allclose(goal, [0.4, 0.4, 0.430]) # GRASP_Z

def test_exit_waypoint():
    cell = np.array([0.5, 0.5])
    exit_wp = exit_waypoint("transit", cell)
    assert np.allclose(exit_wp, [0.5, 0.5, 0.550]) # SAFE_Z
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chess_env/test_waypoints.py`
Expected: ModuleNotFoundError or ImportError.

- [ ] **Step 3: Implement waypoint utilities**

```python
import numpy as np
from src.utils.config import load_config

_cfg = load_config("env")
SAFE_Z  = _cfg["safe_z"]    # 0.550m
GRASP_Z = _cfg["grasp_z"]   # 0.430m

SCENARIO_EXIT_Z = {
    "transit": SAFE_Z,
    "descend": GRASP_Z,
    "ascend":  SAFE_Z,
}
SCENARIO_ENTRY_Z = {
    "transit": SAFE_Z,
    "descend": SAFE_Z,
    "ascend":  GRASP_Z,
}

VALID_TRANSITIONS = {
    ("transit", "transit"): True,
    ("transit", "descend"): True,
    ("descend", "ascend"):  True,
    ("ascend",  "transit"): True,
    ("ascend",  "descend"): True,
}

CHAIN_SHORTCUTS = {
    "full_move": ["transit", "descend", "ascend", "transit", "descend", "ascend"],
    "pick":      ["transit", "descend", "ascend"],
    "place":     ["transit", "descend", "ascend"],
    "vertical":  ["descend", "ascend"],
}

def validate_chain(chain: list[str]) -> None:
    for i in range(len(chain) - 1):
        key = (chain[i], chain[i + 1])
        if not VALID_TRANSITIONS.get(key, False):
            raise ValueError(f"Invalid chain transition at pos {i}: {chain[i]} -> {chain[i+1]}")

def derive_goal_pos(scenario: str, cell_xy: np.ndarray) -> np.ndarray:
    z = SCENARIO_ENTRY_Z[scenario] if scenario == "ascend" else SCENARIO_EXIT_Z[scenario]
    # Actually, the goal for descend is GRASP_Z, transit is SAFE_Z, ascend is SAFE_Z
    z_map = {"transit": SAFE_Z, "descend": GRASP_Z, "ascend": SAFE_Z}
    return np.array([cell_xy[0], cell_xy[1], z_map[scenario]])

def exit_waypoint(scenario: str, cell_xy: np.ndarray) -> np.ndarray:
    return np.array([cell_xy[0], cell_xy[1], SCENARIO_EXIT_Z[scenario]])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chess_env/test_waypoints.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/chess_env/waypoints.py tests/chess_env/test_waypoints.py
git commit -m "feat: add waypoint utility and validation"
```

---

### Task 2: Task Environment Additions (Diagnostics)

**Files:**
- Modify: `src/chess_env/task.py`
- Test: `tests/chess_env/test_task_chaining.py`

- [ ] **Step 1: Write test for transition diagnostics**

```python
import gymnasium as gym
import numpy as np

def test_transition_validate():
    env = gym.make("ChessFetchTask-v0", render_mode=None)
    env.reset()
    diag = env.unwrapped.transition_validate()
    assert "grip_speed_mm_s" in diag
    assert "is_velocity_ok" in diag
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chess_env/test_task_chaining.py`
Expected: AttributeError: 'ChessTaskEnv' object has no attribute 'transition_validate'

- [ ] **Step 3: Implement transition_validate in task.py**

```python
    def transition_validate(self, nominal_exit_pos: np.ndarray | None = None) -> dict:
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip").copy()
        speed = float(np.linalg.norm(grip_vel))
        l_finger = self._utils.get_joint_qpos(self.model, self.data, "robot0:l_gripper_finger_joint").item()

        # Threshold from Phase 1 plan
        HALT_VEL_THRESHOLD = 0.0005 

        result = {
            "grip_pos": grip_pos.tolist(),
            "grip_speed_mm_s": speed * 1000,
            "is_velocity_ok": speed < HALT_VEL_THRESHOLD,
            "error_from_nominal_mm": None,
            "finger_state": l_finger,
        }
        if nominal_exit_pos is not None:
            result["error_from_nominal_mm"] = float(np.linalg.norm(grip_pos - nominal_exit_pos) * 1000)
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chess_env/test_task_chaining.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/chess_env/task.py tests/chess_env/test_task_chaining.py
git commit -m "feat: add transition diagnostic helper to task environment"
```

---

### Task 3: Task Environment Additions (Soft Reset)

**Files:**
- Modify: `src/chess_env/task.py`
- Test: `tests/chess_env/test_task_chaining.py`

- [ ] **Step 1: Write test for soft_reset**

```python
import gymnasium as gym
import numpy as np

def test_soft_reset_flow():
    env = gym.make("ChessFetchTask-v0", render_mode=None)
    env.reset()
    
    # Simulate a success state
    uw = env.unwrapped
    uw.current_scenario = "transit"
    nominal_xy = np.array([0.5, 0.5])
    exit_wp = np.array([0.5, 0.5, 0.550])
    next_goal = np.array([0.5, 0.5, 0.430]) # descend
    
    obs = uw.soft_reset(
        new_scenario="descend",
        new_goal_pos=next_goal,
        nominal_exit_pos=exit_wp,
        nominal_xy=nominal_xy
    )
    
    assert uw.current_scenario == "descend"
    assert np.allclose(uw.goal, next_goal)
    assert uw.episode_steps == 0
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chess_env/test_task_chaining.py`
Expected: AttributeError: 'ChessTaskEnv' object has no attribute 'soft_reset'

- [ ] **Step 3: Implement soft_reset in task.py**

```python
    def soft_reset(self, new_scenario: str, new_goal_pos: np.ndarray,
                   nominal_exit_pos: np.ndarray,
                   nominal_xy: np.ndarray | None = None) -> dict:
        import mujoco
        ROBOT_DOF = 15
        HALT_VEL_THRESHOLD = 0.0005
        HALT_HOLD_MAX_STEPS = 100
        ALIGN_TOLERANCE_M = 0.003
        ALIGN_MAX_STEPS = 200
        ALIGN_GAIN = 0.8
        ALIGN_MAX_STEP_M = 0.005

        # Phase 1: Halt
        zero_action = np.zeros(4)
        for _ in range(HALT_HOLD_MAX_STEPS):
            grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
            if np.linalg.norm(grip_vel) < HALT_VEL_THRESHOLD:
                break
            self._set_action(zero_action)
            self._mujoco_step(None)

        self.data.qvel[:ROBOT_DOF] = 0.0
        self.data.qacc[:ROBOT_DOF] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Phase 2: Align
        converged = False
        for _ in range(ALIGN_MAX_STEPS):
            grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
            error = nominal_exit_pos - grip_pos
            dist = np.linalg.norm(error)
            if dist < ALIGN_TOLERANCE_M:
                self.data.qvel[:ROBOT_DOF] = 0.0
                self.data.qacc[:ROBOT_DOF] = 0.0
                mujoco.mj_forward(self.model, self.data)
                converged = True
                break
            step_vec = ALIGN_GAIN * error
            if np.linalg.norm(step_vec) > ALIGN_MAX_STEP_M:
                step_vec = step_vec / np.linalg.norm(step_vec) * ALIGN_MAX_STEP_M
            self.data.mocap_pos[0][:3] += step_vec
            self._mujoco_step(None)

        if not converged:
            raise RuntimeError(f"soft_reset ALIGN_FAILED: did not converge to {nominal_exit_pos}")

        # Phase 3: State Update
        prev_scenario = self.current_scenario
        self.current_scenario = new_scenario
        self.goal_pos = new_goal_pos.copy()
        self.goal = self.goal_pos.copy()
        self.episode_steps = 0
        
        # Reset TimeLimit wrapper
        curr_env = self
        while hasattr(curr_env, "env"):
            if hasattr(curr_env, "_elapsed_steps"):
                curr_env._elapsed_steps = 0
            curr_env = curr_env.env

        if new_scenario in {"descend", "ascend"}:
            self.tube_center_xy = nominal_xy.copy()
        else:
            self.tube_center_xy = None

        self._utils.set_mocap_quat(self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT)

        # Phase 4: Gripper Transition
        needs_open = (new_scenario == "descend" and prev_scenario != "descend")
        needs_close = (new_scenario in {"ascend", "transit"} and prev_scenario == "descend")
        dummy_action = np.zeros(4)

        if needs_open:
            self.finger_target_joint = self.FINGER_OPEN_JOINT
            for _ in range(50):
                self._set_action(dummy_action)
                self._mujoco_step(None)
        elif needs_close:
            self.finger_target_joint = self.FINGER_CLOSED_JOINT
            for _ in range(50):
                self._set_action(dummy_action)
                self._mujoco_step(None)

        return self._get_obs()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chess_env/test_task_chaining.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/chess_env/task.py tests/chess_env/test_task_chaining.py
git commit -m "feat: implement soft_reset for scenario chaining"
```

---

### Task 4: Evaluation Script

**Files:**
- Create: `scripts/eval_sequence.py`

- [ ] **Step 1: Implement scripts/eval_sequence.py**

```python
import argparse
import logging
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import src.chess_env
from src.chess_env.waypoints import validate_chain, derive_goal_pos, exit_waypoint, CHAIN_SHORTCUTS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def run_chain(model_path, chain_str, n_episodes, drift_limit, debug=False):
    env = gym.make("ChessFetchTask-v0")
    model = SAC.load(model_path, env=env)
    
    chain = CHAIN_SHORTCUTS.get(chain_str, chain_str.split(","))
    validate_chain(chain)
    
    success_count = 0
    for ep in range(n_episodes):
        obs, _ = env.reset()
        uw = env.unwrapped
        uw.eval_drift_limit = drift_limit
        
        # Simplified: one dst for whole chain (pick/place logic)
        dst_xy = uw._sample_board_position()[:2]
        
        chain_success = True
        for i, scenario in enumerate(chain):
            goal = derive_goal_pos(scenario, dst_xy)
            if i > 0:
                obs = uw.soft_reset(scenario, goal, exit_waypoint(chain[i-1], dst_xy), dst_xy)
            else:
                uw.current_scenario = scenario
                uw.goal_pos = goal
                uw.goal = goal
                if scenario in {"descend", "ascend"}: uw.tube_center_xy = dst_xy
                obs = uw._get_obs()

            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            if not info.get("is_success"):
                chain_success = False
                break
        
        if chain_success: success_count += 1
        log.info(f"Episode {ep+1}: {'SUCCESS' if chain_success else 'FAILED'}")
    
    log.info(f"Final Success Rate: {success_count/n_episodes*100:.1f}%")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--chain", required=True)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--drift-limit", type=float, default=0.010)
    args = p.parse_args()
    run_chain(args.model, args.chain, args.n_episodes, args.drift_limit)
```

- [ ] **Step 2: Run a smoke test**

Run: `python scripts/eval_sequence.py --model models/latest_model.zip --chain transit,transit --n-episodes 2`
Expected: Logs showing execution and a final success rate.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_sequence.py
git commit -m "feat: add eval_sequence.py for multi-scenario validation"
```
