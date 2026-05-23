# RL Model Training Add-On: Overview

## Goal

Train **three specialist SAC models** — `transit`, `descend`, and `ascend` — that can be embedded
into the existing RoboChess robotic arm pipeline. Once trained, the `ModelEmbeddedController`
wraps the existing scripted controller and transparently replaces the three movement stages with
RL inference, while keeping the scripted grasp/place/transition logic untouched.

## Design Philosophy

**Minimal invasiveness.** The production code (`ChessTaskEnv`, `ScriptedController`, `GameOrchestrator`)
should be changed as little as possible. All training machinery lives in a new top-level `training/`
package. Only the minimal observation / reward hooks need to be added to `ChessTaskEnv`.

**Specialist models, not a generalist.** Each model is trained exclusively on its own movement
type and learns it deeply. This makes the models easier to debug, replace, and evaluate
independently. The reference project showed that a single generalist works — but three specialists
give us finer-grained control and clearer failure modes.

**Use the "Holding Object" trick.** The reference project proved that by constructing a 25D
observation that maps `object_pos → grip_pos`, we can activate the stable transport dynamics
from the pretrained `FetchPickAndPlace-v4` weights. We use this same trick for all three models.

**No VecNormalize.** The 25D Phase-9 observation uses raw MuJoCo coordinates. The model
uses `MultiInputPolicy` with the `observation/achieved_goal/desired_goal` dict structure
that matches the pretrained weights exactly.

---

## Document Map

| File | Contents |
|---|---|
| `00_overview.md` | This file |
| `01_observation_and_env.md` | Phase-9 observation, `ChessTaskEnv` changes, training env wrappers |
| `02_reward_and_step.md` | Reward function, `step()` safety checks, crash/success conditions |
| `03_training_infrastructure.md` | `SACTrainer`, callbacks, `SubprocVecEnv`, script CLI |
| `04_model_controller.md` | `ModelEmbeddedController` inference loop, loading, fallback |
| `05_integration_and_eval.md` | Wiring into `GameOrchestrator`, `eval_stages.py`, `eval_sequence.py` |

---

## Key Constants (from `configs/env.yaml` + `configs/physics.yaml`)

| Constant | Value | Where Used |
|---|---|---|
| `SAFE_Z` | 0.530 | Transit entry/exit Z |
| `HOVER_Z` | 0.460 | Descend target Z / Ascend entry Z |
| `GRASP_Z` | 0.430 | Grasp plunge target Z |
| `POS_CTRL_SCALE` | 0.015 | Action scaling (max 15mm/step) |
| `DRIFT_LIMIT_END` | 0.010 | Final tube radius for eval |
| `DRIFT_LIMIT_START` | 0.100 | Initial tube radius (curriculum start) |
| `SUCCESS_THRESHOLD` | 0.010 | Distance threshold for success |
| `STABILITY_VEL_THRESHOLD` | 0.02 | Max speed to call arm "stable" at goal |
| `CRASH_PENALTY` | -500.0 | Terminal reward on crash |
| `SUCCESS_BONUS` | 500.0 | Terminal reward on success |
| Board X range | 0.53–1.23 | Sampling bounds (table_center ± half - margin) |
| Board Y range | -0.085–0.613 | Sampling bounds |

---

## File Layout After Implementation

```
robo_chess_latest/
├── training/
│   ├── __init__.py
│   ├── trainer.py          # SACTrainer class
│   ├── callbacks.py        # SB3 callbacks
│   └── envs/
│       ├── __init__.py
│       ├── transit_env.py  # TransitTrainEnv wrapper
│       ├── ascend_env.py   # AscendTrainEnv wrapper
│       └── descend_env.py  # DescendTrainEnv wrapper
├── scripts/
│   └── train_rl.py         # CLI entry point
├── src/
│   └── chess_env/
│       ├── task.py         # MODIFIED: Phase-9 obs, reward, step()
│       └── model_controller.py   # NEW: ModelEmbeddedController
└── configs/
    └── training.yaml       # NEW: training hyperparameters
```
