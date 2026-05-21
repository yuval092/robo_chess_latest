# Design Spec: Actuator Enforcement & Production Certification

This document specifies the structural fixes for the "Policy Fixpoint Trap" and the implementation of deterministic, scenario-aware gripper control for RoboChess.

## 1. Physical Geometry & Footprint
The gripper footprint is calculated to ensure safety within a 7.0cm chess cell (3.5cm radius) while accommodating a 3.0cm piece.

*   **Finger Target (Open):** Joint position `0.0181`.
    *   Internal gap: `38.0mm` (4.0mm margin on each side of a 30mm piece).
    *   External footprint: `66.0mm` width (33.0mm radius).
*   **Finger Target (Closed):** Joint position `0.0000`.
    *   Internal gap: `1.8mm` (fingers effectively touching).
*   **Safety Thresholds:**
    *   **Reset Phase:** fingers must reach within `0.5mm` of target.
    *   **RL Runtime Phase:** fingers must stay within `3.0mm` of target (logged as `finger_fault`).
    *   **Tube Breach:** Terminate episode if `center_dist > limit` OR `(center_dist + 33mm) > limit`.

## 2. Scenario-Aware Control (State Machine)
Gripper state is managed via scripted phases in `_reset_sim` and absolute enforcement in `step`. The RL model (SAC) provides movement vectors but has no control over fingers.

| Scenario | Reset State | Scripted Phase (Pre-RL) | RL Runtime State |
| :--- | :--- | :--- | :--- |
| **Transit** | `Closed` | None | `Closed` |
| **Descend** | `Closed` | `Open` (20 steps) | `Open` |
| **Ascend** | `Open` | `Close` (20 steps) | `Closed` |

## 3. Policy Fixpoint Trap Mitigation
To eliminate the stalling behavior at the 10mm success threshold, the reward surface is re-balanced to favor completion over braking.

*   **Braking Zone (`braking_dist`):** `0.015m` (reduced from 0.030m).
*   **Braking Weight (`braking_reward_weight`):** `0.15` (reduced from 0.40).
*   **Base Model:** Training starts from standard `sac-FetchPickAndPlace-v4.zip` weights.

## 4. Instrumentation & Quality Control
### 4.1 Enhanced Logging
Every `step()` call under `--debug` will log:
*   Robot joint positions (7-DOF).
*   Actual finger joint positions vs. targets.
*   Calculated outer-edge footprint distance.
*   The content of the `info` dictionary (success, scenario, faults).
*   Model's raw output vs. the forced (scripted) action.

### 4.2 Production Training Scale
*   **Workers:** 15 parallel environments.
*   **Total Steps:** 1,000,000.
*   **Curriculum:** 500,000 steps total ($33,333$ steps per worker).

## 5. Implementation Strategy
1.  **Config Update:** Centralize new geometry and reward constants in `env.yaml`.
2.  **Simulator Enforcement:** Modify `ChessSimulationEnv._set_action` to override finger actuators with absolute targets and zero velocity.
3.  **Task Logic:** Refactor `ChessTaskEnv._reset_sim` to implement the 20-step scripted transitions.
4.  **Verification:** Run a 100-episode unit test before committing to the 1M step training run.
