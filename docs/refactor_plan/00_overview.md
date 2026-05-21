# Refactor Plan Overview: Scripted-Only RoboChess

## Purpose

This document describes the master plan for converting the RoboChess project from a hybrid RL+scripted system to a fully scripted deterministic system. The refactor is a deliberate, temporary step: the goal is to perfect the physical environment and movement system before re-introducing RL for a full chess game.

---

## Why This Refactor

The current system uses a Soft Actor-Critic (SAC) model trained on simulated scenarios to move the arm. While the RL model works reasonably well, it introduces:

1. **Non-determinism**: Stochastic policy sampling means the same state produces different trajectories.
2. **Brittleness at distribution boundaries**: The model struggles near table corners and at unusual starting positions.
3. **Opaque failures**: When the RL model fails, the reason is unclear — it could be a policy deficiency, a physics miscalibration, or a reward misspecification.
4. **Training overhead**: Any change to the environment (new table, repositioned arm) requires retraining from scratch.

Replacing RL with deterministic scripted movement eliminates all of these issues. Once the scripted system works reliably for any board position, the environment and physics will be in a known-good state, making future RL re-introduction straightforward and the comparison clean.

---

## What Is Being Removed

| Component | File(s) | Reason for Removal |
|-----------|---------|-------------------|
| SAC model loading | `scripts/eval*.py`, `src/training/` | No longer needed |
| Observation builder | `task.py::_build_phase9_observation` | RL-specific |
| Observation space (25D Dict) | `simulation.py` | Replace with minimal state |
| Training callbacks and curriculum | `src/training/` | No RL training |
| Drift curriculum (tightening radius) | `task.py` | Constant radius from day 1 |
| `scripts/train.py` | `scripts/train.py` | No training |
| `scripts/record_move.py` | `scripts/record_move.py` | RL-dependent, low priority |
| RL model argument (`--model`) | All eval scripts | No model needed |

---

## What Is Being Kept

| Component | File(s) | Notes |
|-----------|---------|-------|
| MuJoCo physics simulation | `assets/*.xml` | Core infrastructure |
| Gymnasium-Robotics base class | `simulation.py` | Keep inheritance, simplify interface |
| `_move_mocap_to` primitive | `task.py` | Central to all movement |
| `execute_grasp` / `execute_place` | `task.py` | Fully scripted already |
| `soft_reset` | `task.py` | Scenario transitions |
| Waypoint concept | `waypoints.py` | Keep 3-scenario structure |
| Crane mode (VERTICAL_QUAT) | `simulation.py` | Required for chess |
| Physics constants and config | `configs/` | Keep YAML config system |
| Gymnasium env registration | `src/chess_env/__init__.py` | Keep for compatibility |

---

## What Is Being Added / Changed

| Component | File(s) | Notes |
|-----------|---------|-------|
| 4-legged table (60×60cm) | `assets/pick_and_place.xml` | Replace solid-box table |
| Repositioned arm base | `assets/robot.xml` | Move partly under table |
| `ScriptedController` class | `src/chess_env/controller.py` | New: wraps all movement |
| Shared argparse module | `src/utils/args.py` | New: common CLI flags |
| Rewritten eval scripts | `scripts/eval_physics.py`, etc. | No RL model required |
| Stress testing support | `scripts/eval_stress.py` | New: corner/grid coverage |
| Fix test assertions | `tests/chess_env/test_waypoints.py` | z=0.430 → z=HOVER_Z |
| Fix Phase 6 scoping bug | `task.py` | `release_target` scope issue |

---

## Stage Overview

| Stage | Title | Key Deliverable | Validation |
|-------|-------|----------------|------------|
| **Stage 1** | Environment Geometry | New 4-legged table XML, repositioned arm | Physics tests: reachability, stability |
| **Stage 2** | Env Interface Cleanup | Remove RL interface, simplify env | Unit tests pass, env instantiates cleanly |
| **Stage 3** | ScriptedController | Full scripted movement for all waypoints | Per-stage accuracy tests |
| **Stage 4** | Evaluation Scripts | All scripts rewritten, shared argparse | Full pipeline runs without model |
| **Stage 5** | Cleanup & Bug Fixes | Remove training module, fix known bugs | Entire test suite passes |

Each stage is self-contained: it starts from a working state, makes a bounded set of changes, and ends in a verified working state before the next stage begins.

---

## Key Design Principles

### Determinism First
Every movement in the scripted system must be fully reproducible given the same initial physics state. No random sampling, no probability distributions, no stochastic elements.

### Preserve Existing Primitives
The `_move_mocap_to` proportional controller and the 4-step pattern (`_set_action → mocap_pos → mocap_quat → _mujoco_step`) are proven correct. Do not invent new movement mechanisms — build on these.

### Config-Driven Constants
All physical constants (Z-levels, tolerances, step counts) remain in `configs/env.yaml` and `configs/physics.yaml`. No magic numbers in code.

### Stage Isolation
Each stage modifies only the files it needs to. If a later stage reveals a problem in an earlier stage, fix the earlier stage and re-validate before continuing.

### Test Before Proceed
Every stage ends with a concrete set of test commands that must produce specific expected outputs. Do not advance to the next stage until all validation steps pass.

---

## Table Design Summary

The new table replaces the current solid-box table with a 4-legged open table:

| Property | Current | New |
|----------|---------|-----|
| Surface size | 56×56cm (code) | 60×60cm |
| Table style | Solid box body | Surface slab + 4 separate legs |
| Arm base X | 0.2869m | ~0.58m (near table edge) |
| Surface Z | 0.400m | 0.400m (unchanged) |
| Leg visibility | N/A | 4 legs at corners, height ~35cm |

Keeping `TABLE_SURFACE_Z = 0.400` unchanged means all Z-level constants (`GRASP_Z`, `HOVER_Z`, `SAFE_Z`) remain valid and do not require reconfiguration.

---

## Arm Repositioning Summary

The arm is currently positioned far from the table (base at x=0.2869, near table edge at x=0.60). This creates a long effective lever arm and puts the near side of the board near the robot's singular configuration.

New position: base at approximately x=0.58 (at or slightly under the near table edge).

Benefits:
- Far board edge (~1.18m) is now ~0.60m from arm base: well within Fetch reach (~1.0m)
- Near board edge (~0.58m) remains reachable with good arm extension
- All corners reachable in crane mode with adequate force/torque margins

---

## Known Bugs Fixed in This Refactor

1. **`test_waypoints.py` z=0.430 assertions** (Stage 5): Tests written when `hover_z=0.430`; current value is `0.460`. Fix: update assertions to use `HOVER_Z` constant.

2. **`execute_grasp` Phase 6 `release_target` scoping** (Stage 5): Variable `release_target` is set inside Phase 4's loop and used in Phase 6. Rename to `grasp_pos` and assign at Phase 3 end (where the exact GRASP_Z position is first established).

---

## File Structure After Refactor

```
src/
  chess_env/
    __init__.py         (registration unchanged)
    simulation.py       (simplified: remove obs builder, keep mocap)
    task.py             (simplified: remove RL step logic, keep scripted)
    controller.py       (NEW: ScriptedController)
    waypoints.py        (unchanged)
    config_loader.py    (unchanged)
  utils/
    args.py             (NEW: shared argparse)
    
scripts/
  eval_physics.py       (NEW/REWRITE: physics sanity checks, no model)
  eval_stages.py        (NEW/REWRITE: per-scenario accuracy, no model)
  eval_sequence.py      (REWRITE: full chain eval, no model)
  eval_stress.py        (NEW: corner/grid stress test, no model)
  visualize.py          (KEEP: already partially model-independent)

configs/
  env.yaml              (update table_half_x/y: 0.30)
  physics.yaml          (unchanged)
  training.yaml         (DELETE or keep as archive)
  
assets/
  pick_and_place.xml    (CHANGE: new 4-legged table + arm position)
  robot.xml             (CHANGE: arm base position)
  shared.xml            (unchanged)
  
tests/
  chess_env/
    test_waypoints.py   (FIX: z=0.430 → HOVER_Z)
    test_task_chaining.py (review for RL dependencies)
    
docs/
  current_status/       (complete, 11 files)
  refactor_plan/        (this folder, 6 files)
```
