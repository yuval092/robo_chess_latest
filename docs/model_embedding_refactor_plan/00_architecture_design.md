# Model-Embedding Refactor: Architectural Design

> **⚠ SUPERSEDED** — This document (and `01_stage_1_hybrid_controller.md`,
> `02_stage_2_training_envs.md`, `03_stage_3_training_scripts.md`) is the **original draft**.
> The authoritative, corrected plan lives in `new_version/`. Do not implement from these files.
> Key problems with this draft: (a) training envs override `_reset_sim` incorrectly, causing
> double-reset; (b) no Phase-9 / Holding Object observation trick — model will see wrong obs;
> (c) uses VecNormalize which breaks pretrained weight matching; (d) missing `run_full_move`
> in the controller.

## Overview
This document outlines the architectural design for embedding three specialized Reinforcement Learning (RL) models (Transit, Ascend, and Descend) into the RoboChess project. The models will replace the corresponding scripted movement stages of the robotic arm, while retaining the robust, scripted procedures for grasping, releasing, and settling.

## 1. Architecture & Integration (Hybrid Controller Wrapper)

### ModelEmbeddedController
We will introduce a `ModelEmbeddedController` that serves as a drop-in replacement for the existing `ScriptedController`.
- **Interface:** It will implement the same public methods (`run_transit`, `run_ascend`, `run_descend`, `run_full_move`, etc.) so that the `MovementExecutor` and `GameOrchestrator` remain completely unaware of the RL models.
- **Model Inference:** During `run_transit`, `run_ascend`, and `run_descend`, the controller will:
  1. Retrieve the current environment observation (`env._get_obs()`).
  2. Query the appropriate loaded SAC model for an action: `action, _ = model.predict(obs, deterministic=True)`.
  3. Extract the `[dx, dy, dz]` movement deltas from the model's output.
  4. Force the `gripper` control part of the action to be `0` (or the appropriate neutral value) to ensure the models do not accidentally open/close the fingers.
  5. Apply the action via `env._set_action(action)` and step the physics via `env._mujoco_step(None)`.
- **Scripted Phases:** The grasp pipeline (`execute_grasp`) and release pipeline (`execute_place`), along with halt and alignment sequences, will remain strictly scripted within `ChessTaskEnv`.

## 2. Training Environments & Observation Space

### Isolation
To ensure the training process does not harm or alter the production environment, all training code will reside in a new `training/` directory at the project root.

### Wrapper Environments
We will create three wrapper environments (`TransitTrainEnv`, `AscendTrainEnv`, `DescendTrainEnv`) that inherit from or wrap `ChessTaskEnv`.
- **Initialization & Reset:** Each wrapper's `reset()` method will randomly spawn the arm in valid starting configurations to ensure robust learning.
  - *Transit:* Random starting XY on the board at `SAFE_Z`. Target is a random destination XY at `SAFE_Z`.
  - *Ascend:* Random XY above the board at `HOVER_Z`. Target is the same XY at `SAFE_Z`.
  - *Descend:* Random XY above the board at `SAFE_Z`. Target is the same XY at `HOVER_Z`.
- **Gripper Simplification:** As requested, the models will be trained without holding an actual piece. To match the expected production state:
  - *Transit:* Fingers will always be initialized and kept closed.
  - *Ascend/Descend:* Fingers will always be initialized and kept open.

### Observation Space
The models require sufficient context to halt smoothly and maintain constraints. The observation space will be a `Dict` space (as currently defined in `ChessTaskEnv`) but we will ensure the models utilize the following key components:
- `grip_pos` (3D current position)
- `grip_vel` (3D current velocity)
- `goal_pos` (3D target position - CRITICAL for goal-directed movement)
- `l_finger` (1D finger state)

**Note on Goal Context:** The current `_get_obs` implementation in `ChessTaskEnv` provides these as separate keys but omits `goal_pos` from the concatenated `observation` vector. The training wrappers will either use the full `Dict` space with `MultiInputPolicy` or we will update `_get_obs` to include `goal_pos` in the flat vector.

**Normalization:** We will utilize SB3's `VecNormalize` wrapper during training to stabilize learning, as raw MuJoCo coordinates and velocities can vary significantly in magnitude.

### Action Space
- **Specialized Action:** The models will output a 3D or 4D action. Although the environment expects 4D `[dx, dy, dz, gripper]`, the `gripper` value is ignored by `_set_action` during movement phases (controlled instead by `finger_target_joint`). We will force this to a neutral `0` or keep it at the model's output, knowing it is ignored.
- **Scaling:** Actions will be scaled by `POS_CTRL_SCALE` (0.015) by the environment.

## 3. Reward & Penalty Design

The SAC models will be trained using dense rewards mixed with severe penalties for violating safety constraints.

### Transit Model
- **Reward:** Dense L2 distance to the target XY coordinate. High terminal reward for reaching the target within `TRANSIT_TOLERANCE_M`.
- **Penalty:** Severe penalty for deviating from `SAFE_Z` (Z-drift penalty).
- **Termination:** Episode succeeds upon reaching target XY. Fails if Z drops below a critical threshold.

### Ascend Model
- **Reward:** Dense Z-distance to `SAFE_Z`.
- **Penalty:** Severe penalty for XY-drift from the initial starting XY (the "tube constraint" to avoid colliding with nearby tall pieces).
- **Termination:** Episode succeeds upon reaching `SAFE_Z`. Fails if XY drift exceeds `drift_limit`.

### Descend Model
- **Reward:** Dense Z-distance to `HOVER_Z`.
- **Penalty:** Severe penalty for XY-drift from the tube center. Additional velocity penalties as the arm approaches `HOVER_Z` to encourage a smooth, controlled halt.
- **Termination:** Episode succeeds upon reaching `HOVER_Z` with low velocity. Fails if XY drift exceeds `drift_limit` or if it crashes below `HOVER_Z`.

### Shared
- **Efficiency:** A small negative reward per step to encourage the arm to move quickly rather than loitering.

## 4. Evaluation Strategy

### Training Evaluation
- Standalone evaluation scripts will be placed in `training/eval/` to test each model's success rate, average steps, and constraint violation rate during training.

### Integration Evaluation
- Existing test scripts (`scripts/eval_stages.py`, `scripts/eval_sequence.py`, `scripts/eval_chess_game_flow.py`) will be updated or parameterized to use the `ModelEmbeddedController`.
- This ensures the RL models are evaluated within the exact same testing harness currently used for the scripted controller.
- Metrics such as XY error, Z error, step counts, and crash reasons will continue to be recorded.
