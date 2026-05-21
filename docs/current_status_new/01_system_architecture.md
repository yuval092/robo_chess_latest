# System Architecture

## Project Overview

RoboChess is a robotic pick-and-place simulation built on the Fetch robot arm in MuJoCo. It is based on the Gymnasium-Robotics `FetchPickAndPlace-v4` environment and extends it with a 70×70cm playing surface, a scripted deterministic controller, and a waypoint-based movement pipeline. The goal is a full chess-playing robot; the current form manipulates a single 30mm cube as a stand-in for chess pieces.

The codebase is in a **scripted-only phase** — the RL training pipeline has been archived and movement is handled entirely by `ScriptedController`. This phase exists to establish reliable, deterministic arm motion before reintroducing RL for high-level chess decision-making.

---

## Directory Structure

```
chess_env/assets/
├── pick_and_place.xml  ← Main MuJoCo scene: floor, table, object, actuators
├── shared.xml          ← Assets, materials, mesh references, equality/contact defaults
└── robot.xml           ← Full Fetch robot body hierarchy, mocap body, arm placement

configs/
├── env.yaml            ← Task geometry, Z-levels, gripper constants, table dimensions
└── physics.yaml        ← Simulation settings, settle tolerance, VERTICAL_QUAT

src/
├── chess_env/
│   ├── __init__.py     ← Registers "ChessFetchTask-v0" with Gymnasium
│   ├── simulation.py   ← ChessSimulationEnv: MuJoCo setup, mocap control, env setup
│   ├── task.py         ← ChessTaskEnv: scenarios, scripted actions, grasp, reset
│   ├── controller.py   ← ScriptedController: high-level movement sequences
│   └── waypoints.py    ← Z-level constants, valid transitions, chain definitions
└── utils/
    ├── args.py         ← Shared argparse: add_common_args(), make_env()
    ├── config.py       ← YAML config loader
    └── logger.py       ← File-based logger factory

scripts/
├── visualize.py        ← Human-render loop using ScriptedController
├── eval_stages.py      ← Per-stage accuracy evaluation (transit / descend / ascend)
├── eval_sequence.py    ← Full chain evaluation (pick, full_move, vertical)
├── eval_stress.py      ← Corner + grid stress test across board positions
├── verify_physics.py   ← Physics sanity checks (stability, reachability, XML integrity)
└── test_grasp_physics.py ← Grasp physics test (static, lift, transit-held)

tests/
├── conftest.py         ← sys.path setup for PYTHONPATH-free pytest runs
└── chess_env/
    ├── test_task_chaining.py  ← pytest: transition_validate, soft_reset flows
    └── test_waypoints.py      ← pytest: chain validation, goal derivation, exit waypoints

archive/rl_system/      ← Archived RL pipeline (SAC, trainer, callbacks, models)
docs/
├── refactor_plan/      ← Stage-by-stage implementation plans
└── current_status_new/ ← This documentation set
```

---

## Class Inheritance

```
gymnasium_robotics.envs.fetch.pick_and_place.MujocoFetchPickAndPlaceEnv
    └── ChessSimulationEnv  (src/chess_env/simulation.py)
            └── ChessTaskEnv  (src/chess_env/task.py)
```

`ChessSimulationEnv` handles low-level setup: XML patching, torso positioning, mocap orientation enforcement, and the `_env_setup` hook.

`ChessTaskEnv` handles task logic: scenario management, `_reset_sim`, scripted actions (`execute_grasp`, `execute_place`, `soft_reset`), step-level validation, observations, and rewards.

`ScriptedController` (`src/chess_env/controller.py`) is an external class that drives the arm through movement stages by calling into the unwrapped env's physics API. It is not a subclass of any env.

---

## Environment Registration

```python
# src/chess_env/__init__.py
gymnasium.register(
    id="ChessFetchTask-v0",
    entry_point="src.chess_env.task:ChessTaskEnv",
    max_episode_steps=500,
)
```

Instantiate with:
```python
import gymnasium as gym
import src.chess_env  # triggers registration

env = gym.make("ChessFetchTask-v0", render_mode=None, force_scenario="transit")
# or:
env = gym.make("ChessFetchTask-v0", render_mode="human", force_scenario=None)
```

The `make_env(args, force_scenario, hide_object)` helper in `src/utils/args.py` wraps this with the standard argparse-driven config.

---

## Key Coordinate System

All positions in world frame (meters):

| Landmark | World XYZ |
|----------|-----------|
| Arm base (`robot0:base_link`) | (0.56, 0.2641, 0.0) |
| Table center | (0.88, 0.2641, 0.0) |
| Table surface top | (0.88, 0.2641, 0.400) |
| Near table edge (usable, with margin) | x = 0.57 |
| Far table edge (usable, with margin) | x = 1.19 |
| Left table edge (usable, with margin) | y = 0.574 |
| Right table edge (usable, with margin) | y = -0.046 |

The arm's base is at x=0.56, placing it 1cm outside the near table edge (x=0.57). The arm reaches forward (+x direction) to cover the board. The torso is raised to 0.3661m (near the upper range of 0.3861m) so the shoulder can reach near-row chess squares (x≈0.61) that would otherwise fall in a kinematic dead zone at lower torso heights.

---

## Key Z-Level Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `TABLE_SURFACE_Z` | 0.400m | Physical tabletop |
| `GRASP_Z` | 0.430m | Descent target; 30mm above table, 18.5mm finger overlap with cube |
| `HOVER_Z` | 0.460m | Pre-grasp stop height; 60mm above cube top |
| `SAFE_Z` | 0.550m | Transit height; 15cm above table surface |

---

## Data Flow: One Full Chess Move

```
1. env.reset() with force_scenario="transit", force_start_pos=HOME_POS
2. ctrl.run_transit(src_xy)        ← move horizontally to src at SAFE_Z
3. ctrl.transition("descend", ...)  ← soft_reset to descend scenario
4. ctrl.run_descend(src_xy)         ← descend from SAFE_Z to HOVER_Z
5. env.execute_grasp()              ← scripted grasp pipeline (6 phases)
6. ctrl.transition("ascend", ...)   ← soft_reset to ascend scenario
7. ctrl.run_ascend(src_xy)          ← ascend from HOVER_Z to SAFE_Z (cube held)
8. ctrl.run_transit(dst_xy)         ← transit to dst at SAFE_Z (cube held)
9. ctrl.transition("descend", ...)  ← soft_reset to descend scenario
10. ctrl.run_descend(dst_xy)        ← descend to HOVER_Z at dst
11. env.execute_place(dst_xy)       ← scripted place pipeline
12. ctrl.transition("ascend", ...)  ← soft_reset to ascend scenario
13. ctrl.run_ascend(dst_xy)         ← ascend back to SAFE_Z (empty gripper)
```

The `ScriptedController.run_full_move(src_xy, dst_xy)` encapsulates steps 2–13 via `run_pick_sequence` + `run_place_sequence`.
