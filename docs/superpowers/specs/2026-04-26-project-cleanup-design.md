# Spec: RoboChess Project Cleanup and Reorganization

**Date:** 2026-04-26
**Topic:** Project Structure, Stability, and Configuration

## 1. Objective
Transform the `robo_chess_fine_tuning` project into a robust, organized, and config-driven codebase. This involves removing obsolete diagnostic scripts, refactoring environment logic into a modular source tree, and centralizing all constants into YAML files.

## 2. Architecture

### 2.1 Configuration System
All hard-coded constants in Python will be moved to `configs/*.yaml`.
*   `configs/env.yaml`: Board dimensions, site names, success thresholds, and scenario definitions.
*   `configs/training.yaml`: SAC hyperparameters (LR, batch size, target entropy), total timesteps, and curriculum targets.
*   `configs/physics.yaml`: MuJoCo-specific parameters (substeps, mocap orientation, floor limits).

### 2.2 Source Structure (`src/`)
Core logic moves from the root/scripts into a structured `src/` directory to improve maintainability and testability.
*   `src/chess_env/simulation.py`: The low-level MuJoCo interface (base physics, observation generation).
*   `src/chess_env/task.py`: The high-level RL task (reward functions, termination conditions, curriculum logic).
*   `src/training/`: Encapsulates the SAC training loop and custom callbacks.
*   `src/utils/`: Shared utilities for YAML loading, standardized logging, and coordinate transformations.

## 3. Core Scripts (`scripts/`)
Dozens of loose scripts will be consolidated into four robust, `argparse`-heavy tools:

### 3.1 `train.py`
The primary interface for model training.
*   **Features:** Automated checkpoint resumption, dynamic curriculum calculation, and TensorBoard integration.
*   **Key Args:** `--envs`, `--fresh`, `--config`, `--timesteps`.

### 3.2 `eval.py`
Quantitative performance measurement.
*   **Features:** Per-scenario success/crash rate reporting.
*   **Rendering:** Optional `--visualize` flag to enable human rendering during evaluation.
*   **Key Args:** `--model`, `--scenarios`, `--n-episodes`, `--deterministic`.

### 3.3 `verify_physics.py`
Comprehensive system health check.
*   **Tests:**
    1.  **XML Integrity:** Ensures assets load and geoms are correctly named.
    2.  **Static Stability:** Verifies that a piece placed on the table does not drift/vibrate over time (no-action simulation).
    3.  **Kinematic Reachability:** Validates that the arm can reach picking and transit altitudes without collision.
    4.  **Teleport Verification:** Confirms the object-hiding logic works correctly.

### 3.4 `visualize.py`
Qualitative observation and debugging tool.
*   **Features:** Continuous human rendering of the agent performing the task, with options to add a delay between steps for slow-motion inspection. Does not calculate aggregate statistics.
*   **Key Args:** `--model`, `--scenario`, `--delay`, `--wait`.

## 4. Implementation Strategy

### 4.1 Cleanup
*   Remove all `evaluate_unit*.py`, `test_*`, `diag_*`, `repro_*`, and `check_*` scripts.
*   Absorb `v8_latest/` logic into `src/` and delete the folder.
*   Delete obsolete log files and intermediate checkpoints to free up workspace.

### 4.2 Robustness
*   Add Python type hinting (`typing`) to all core classes.
*   Implement explicit error handling for MuJoCo loading and model I/O.
*   Standardize logging output across all scripts.

## 5. Documentation
*   A master `README.md` will be created in the root directory.
*   A secondary `README.md` inside `scripts/` will document the exact flags and usage for each consolidated tool.
