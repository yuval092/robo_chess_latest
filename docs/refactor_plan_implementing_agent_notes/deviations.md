# Implementation Notes & Plan Deviations

This document tracks deviations from the original `docs/refactor_plan/` made during the implementation process to ensure physical stability, code correctness, and compliance with library interfaces.

---

## Stage 1: Environment Geometry

### 1. Collision Exclusion for Arm Base
- **Deviation**: Added a contact exclusion between `robot0:base_link` and `table0` in `chess_env/assets/shared.xml`.
- **Reason**: Simulation revealed that the robot's base mesh top is at $Z \approx 0.478\text{m}$, which penetrates the table surface slab ($Z = 0.375\text{m} - 0.400\text{m}$). Without this exclusion, the simulation would start in a collision state.

---

## Stage 2: Environment Interface Cleanup

### 2. Observation Space & Key Alignment
- **Deviation**: Included mandatory Gymnasium-Robotics keys (`observation`, `achieved_goal`, `desired_goal`) in the observation dictionary and updated `self.observation_space`.
- **Reason**: Necessary for compatibility with the parent `MujocoFetchEnv` and `RobotEnv` classes.

### 3. Inclusion of `scenario_id` in Observation Dict
- **Deviation**: Added `scenario_id` (Discrete space) to the observation dictionary instead of a string `"scenario"`.
- **Reason**: Gymnasium's standard spaces (like `Box` or `Dict` containing `Box`) do not support string types. Mapping scenario names to integers (1: transit, 2: descend, 3: ascend) provides this information in a valid format.

### 4. Explicit Type Casting to float32
- **Deviation**: Added explicit `.astype(np.float32)` and `np.float32()` casting in `_get_obs()`.
- **Reason**: Ensures exact matches with the `observation_space` definition and prevents Gymnasium warnings.

### 5. `compute_reward` Stub Implementation
- **Deviation**: Added a minimal `compute_reward` method returning `0.0`.
- **Reason**: Required by the `GoalEnv` interface.

### 6. Reduction of Observation Vector (25D → 7D)
- **Deviation**: Reduced the `observation` key vector size from 25 dimensions to 7 (pos, vel, finger).
- **Reason**: RL-specific features were removed. Note that this breaks old evaluation scripts (like `scripts/eval_sequence.py`) until they are rewritten in Stage 4.

---

## Stage 4: Evaluation Scripts

### 13. Removal of Legacy RL Scripts
- **Action**: Deleted `scripts/train.py`, `record_move.py`, `eval.py`, `eval_grasp.py`, and `test_corners.py`.
- **Reason**: These scripts were either hardcoded for RL models or used the old observation vector format (25D), making them incompatible with the new deterministic system.

### 14. Implementation of New Evaluation Suite
- **Action**: Created `eval_stages.py`, `eval_sequence.py`, `eval_stress.py`, and a rewritten `visualize.py`.
- **Note**: These scripts utilize the `ScriptedController` and the shared `args.py` module to provide robust, model-independent evaluation of the physical environment and movement logic.

### 15. ~~Final Arm Base Repositioning (X = 0.35)~~ [REVERTED]
- **~~Deviation~~**: ~~Moved the arm base to $X = 0.35\text{m}$ instead of the plan's proposed $X = 0.60\text{m}$.~~
- **~~Reason~~**: 
    - ~~At $X = 0.60\text{m}$, the far edge of the board ($X \approx 1.18\text{m}$) was at the extreme limit of the Fetch arm's reach ($\approx 0.60\text{m}$ extension). This caused kinematic errors of $\approx 11\text{mm}$ even without collisions.~~
    - ~~At $X = 0.45\text{m}$, reachability improved but still showed $\approx 15\text{mm}$ errors at the lateral edges of the board.~~
    - ~~Moving to $X = 0.35\text{m}$ (only $6\text{cm}$ closer than the original position) ensures that the entire $60\text{cm} \times 60\text{cm}$ board is well within the healthy workspace envelope of the arm, providing high precision and avoiding singularities.~~
- **Correction**: A subsequent review proved that the $X = 0.60\text{m}$ position is correct and achieves $<1\text{mm}$ error across the entire board. Previous reachability failures were artifacts of testing without calling `env.reset()`, which meant the torso was not raised. The base has been reverted to $X = 0.60\text{m}$.

### 16. Mandatory `env.reset()` for Reachability Tests
- **Finding**: Discovered that the Fetch robot's torso starts at a default height in MuJoCo. Calling `env.reset()` is **mandatory** to trigger `_env_setup`, which forces the torso to its maximum height ($0.4\text{m}$). Without this, the arm cannot reach the table surface without colliding with its own shoulder link.
- **Fix**: Updated `scripts/verify_physics.py` to ensure `reset()` is called before reachability checks.

### 17. Torso Lift Range Correction
- **Deviation**: In `_env_setup`, used the actual joint maximum (`self.model.jnt_range[torso_id][1]`) instead of the hardcoded `0.4`.
- **Reason**: The torso lift joint range is $[0.0386, 0.3861]$. Setting it to $0.4$ caused MuJoCo's limit constraint to apply a small corrective impulse each step. Using the exact maximum ensures a stable, within-limit starting state.

---

## Stage 5: Cleanup and Bug Fixes

### 18. Evaluation Script `hide_object` Fix
- **Deviation**: Updated `src/utils/args.py` and evaluation scripts (`eval_sequence.py`, `eval_stress.py`) to explicitly pass `hide_object=False` when performing manipulation tasks.
- **Reason**: The `ChessTaskEnv` defaults to `hide_object=True` (for pure movement). Evaluation scripts that attempt to pick or move the cube would fail if the cube remained at its hidden off-board location.
