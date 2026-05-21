# Implementation Notes & Plan Deviations

This document tracks deviations from the original `docs/refactor_plan/` made during the implementation process to ensure physical stability, code correctness, and compliance with library interfaces.

---

## Stage 1: Environment Geometry

### 1. Collision Exclusion for Arm Base
- **Deviation**: Added a contact exclusion between `robot0:base_link` and `table0` in `chess_env/assets/shared.xml`.
- **Reason**: The plan (Section 1.1) stated that at X=0.60, there would be "no physical overlap" and "no collision." However, simulation revealed that the top of the robot's base link ($Z \approx 0.478\text{m}$) is higher than the bottom of the table surface slab ($Z = 0.375\text{m}$). Without this exclusion, the environment would initialize in a collision state, causing physics instability or preventing load.

---

## Stage 3: ScriptedController

### 6. Finger Transition Bugs in `task.py`
- **Deviation**: Fixed bugs in `_reset_sim` and `soft_reset` where fingers would not open/close if the arm was already at the target position.
- **Fix**: Added explicit `self._set_gripper_state()` calls before calling `_move_mocap_to` for finger transitions. This ensures the teleport-mode fingers reach their target joint position even if the movement loop exits early.

### 7. Adjustment of `grasp_z`
- **Deviation**: Increased `grasp_z` from `0.425` to `0.430` in `configs/env.yaml`.
- **Reason**: At `0.425`, the finger tips ($Z \approx 0.4065\text{m}$) were too close to the table surface ($Z = 0.400\text{m}$), causing intermittent collisions. Increasing to `0.430` provides $11.5\text{mm}$ of clearance while maintaining a healthy $18.5\text{mm}$ (62%) overlap with the $30\text{mm}$ cube.

### 8. Tuning of Actuator Kp and Finger Stiffness
- **Deviation**: 
    - Reduced Actuator `Kp` from `150000` to `20000` in `chess_env/assets/pick_and_place.xml`.
    - Increased finger stiffness by adding `solref="0.002 1" solimp="0.99 0.999 0.001"` to the `robot0:fetchGripper` class in `chess_env/assets/shared.xml`.
- **Reason**: The original combination of extremely high `Kp` and default (soft) finger constraints caused "numerical tunneling," where fingers would pass through the cube rather than stalling against it. Tuning these values produced a stable grasp that reliably stalls at $j \approx 0.007$ without physical artifacts.

### 9. Fix of `NameError` in `task.py`
- **Deviation**: Fixed a scoping bug in `execute_grasp` and `execute_place` where Phase 6 referenced an undefined `release_target` variable.
- **Fix**: Defined `grasp_pos` (or `place_pos`) explicitly after the plunge phase to provide a stable XY reference for the retract phase. (Note: This was originally slated for Stage 5 but was moved up to Stage 3 to unblock validation tests).

