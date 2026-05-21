# System Architecture

## Project Overview

RoboChess is a robotic pick-and-place system built on the Fetch robot arm. It is grounded in the Gymnasium-Robotics `FetchPickAndPlace-v4` environment and extends it with a larger playing surface, a custom reward structure, a multi-stage "waypoint" movement system, and a hybrid RL + scripted action pipeline. The eventual goal is a full chess-playing robot; in the current form a single cube is manipulated as a stand-in for chess pieces.

---

## High-Level Architecture

```
configs/
├── env.yaml          ← Task geometry, Z-levels, reward weights, gripper constants
├── physics.yaml      ← Simulation settings, settle parameters, quaternion
└── training.yaml     ← SAC hyperparameters, eval frequency, logging

chess_env/assets/
├── pick_and_place.xml  ← Main MuJoCo scene: floor, table, object, actuators
├── shared.xml          ← Assets, materials, mesh references, equality/contact defaults
└── robot.xml           ← Full Fetch robot body hierarchy, mocap body

src/
├── chess_env/
│   ├── __init__.py        ← Registers "ChessFetchTask-v0" with Gymnasium
│   ├── simulation.py      ← ChessSimulationEnv: MuJoCo setup, mocap control, finger enforcement
│   ├── task.py            ← ChessTaskEnv: RL scenarios, rewards, scripted actions, transitions
│   └── waypoints.py       ← Z-level constants, transition rules, chain definitions
├── training/
│   ├── trainer.py         ← SACTrainer: training loop, environment vectorization
│   └── callbacks.py       ← Logging, evaluation, combined-success saving callbacks
└── utils/
    ├── config.py           ← YAML config loader
    └── logger.py           ← File-based logger factory

scripts/
├── train.py            ← Training entry point
├── eval.py             ← RL model evaluation by scenario
├── eval_grasp.py       ← Full pick sequence evaluation (RL + scripted grasp)
├── eval_sequence.py    ← Multi-scenario chain evaluation with ChainEpisodeRunner
├── verify_physics.py   ← Physics sanity checks (stability, reachability, XML integrity)
├── test_corners.py     ← Corner/edge stress test with RL
├── test_grasp_physics.py ← Grasp physics tests (static, lift, transit-held)
├── visualize.py        ← Human-render loop
└── record_move.py      ← Offscreen frame capture

tests/chess_env/
├── test_task_chaining.py  ← pytest: transition_validate, soft_reset flows
└── test_waypoints.py      ← pytest: chain validation, goal derivation, exit waypoints
```

---

## Class Inheritance

```
gymnasium_robotics.envs.fetch.pick_and_place.MujocoFetchPickAndPlaceEnv
    └── ChessSimulationEnv  (src/chess_env/simulation.py)
            └── ChessTaskEnv  (src/chess_env/task.py)
```

`MujocoFetchPickAndPlaceEnv` provides:
- MuJoCo model loading (`mujoco.MjModel`, `mujoco.MjData`)
- Standard Gymnasium Env interface (`reset`, `step`, `render`)
- `_utils` helper object (get/set joint qpos/qvel, mocap state, site positions)
- `_mujoco_step`, `generate_mujoco_observations`
- `_reset_sim`, `_env_setup`, `_set_action` (all overridden below)

`ChessSimulationEnv` overrides:
- `__init__`: Model XML monkey-patching, config loading, table geometry constants
- `_sample_board_position`: Uniform sampling within board XY bounds
- `_sample_goal`: Distance-gated goal sampling
- `_reset_sim`: Places cube at random board position at correct Z
- `_set_action`: Converts 4D RL action → mocap delta + finger enforcement
- `_env_setup`: Forces torso to max height, sets vertical quat, runs settle steps

`ChessTaskEnv` overrides:
- `__init__`: All RL task constants (rewards, thresholds, grasp parameters)
- `_reset_sim`: Full reset sequence with scenario selection, arm settling, scripted gripper transitions
- `_get_obs` / `_build_phase9_observation`: 25D observation with "Holding Object" trick
- `_sample_goal`: Returns stored `goal_pos`
- `_is_success`: Distance check against 10mm XY/Z threshold
- `step`: State-machine finger enforcement, drift/floor checks, reward computation
- `compute_reward`: Dense reward function

Additional methods on `ChessTaskEnv`:
- `_move_mocap_to`: Core movement primitive (proportional controller to target position)
- `_settle_arm_to_start`: Moves arm to start, zeros velocities
- `execute_grasp`: 6-phase scripted grasp pipeline
- `execute_place`: 6-phase scripted place pipeline
- `soft_reset`: Seamless scenario transition without arm teleportation
- `transition_validate`: Diagnostics at scenario boundary

---

## Data Flow

1. **Initialization**: `load_config("env")` + `load_config("physics")` populate all constants in both classes. `_fpp_module.MODEL_XML_PATH` is monkey-patched before calling `super().__init__()` to redirect MuJoCo to the custom XML.

2. **Reset**: `env.reset()` calls `_reset_sim()` → scenario selection → object placement → arm settle → scripted gripper transition → `_get_obs()` returns initial observation.

3. **RL Step**: `env.step(action)` → `_set_action(action)` (applies to mocap) → `_mujoco_step()` → observation → reward → termination checks.

4. **Scripted Step**: `_move_mocap_to(target, quat)` loop, bypasses RL entirely.

5. **Training**: `SACTrainer` creates 8 `SubprocVecEnv` training envs + 3 `DummyVecEnv` eval envs (one per scenario). `SAC.learn()` drives the loop; callbacks save best models per scenario and for combined performance.

---

## Key Dependencies

| Library | Version / Role |
|---------|---------------|
| `mujoco` | Physics simulation (direct API) |
| `gymnasium-robotics` | Base Fetch environment |
| `stable-baselines3` | SAC RL algorithm |
| `numpy` | Array math throughout |
| `scipy` | Rotation transforms in grasp evaluation |
| `torch` | SAC neural network backend |
| `pyyaml` | Config loading |
| `PIL/Pillow` | Frame capture in record_move.py |

---

## Configuration Loading

`src/utils/config.py::load_config(name)` reads `configs/{name}.yaml` relative to the project root, resolved from the module's `__file__` path. All config is loaded at `__init__` time — no lazy loading.

Constants derived from config are stored as instance attributes (e.g., `self.SAFE_Z`, `self.FINGER_OPEN_JOINT`). This means they are fixed per-environment-instance and not hot-reloadable.
