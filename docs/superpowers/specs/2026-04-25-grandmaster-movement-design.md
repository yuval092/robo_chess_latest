# Grandmaster Movement Engine: 56cm Board Design

**Status:** Approved by User
**Date:** 2026-04-25

## 1. Goal
Refine the robotic movement engine to operate on a 56cm x 56cm chessboard (7cm cells) with perfect verticality, high precision (1.5cm threshold), and collision-free pathing.

## 2. Observation Space (25-Dimensional)
Fully compatible with baseline SAC models but tailored for "Pure Movement":

| Indices | Description | Implementation |
| :--- | :--- | :--- |
| 0 - 2 | Gripper Position | Site `robot0:grip` XYZ coordinates. |
| 3 - 5 | Object Position | Set to Gripper Position (Holding Object Trick). |
| 6 - 8 | Object Rel Pos | Set to `[0, 0, 0]`. |
| 9 - 10 | Gripper State | **Masked to `[0, 0]`**. |
| 11 - 13 | Scenario ID | One-hot: [1,0,0] Transit, [0,1,0] Descend, [0,0,1] Ascend. |
| 14 - 19 | Zero Mask | Masked to `[0, 0, 0, 0, 0, 0]`. |
| 20 - 22 | Gripper Velocity | Real-time linear velocity. |
| 23 - 24 | Robot Velocity | **Masked to `[0, 0]`**. |

## 3. Physical Constants & Geometry
*   **Board:** 56cm x 56cm (`TABLE_HALF = 0.28m`).
*   **Cell Size:** 7.0cm.
*   **Chess Piece:** 3x3x3cm cube (0.5kg).
*   **Grasp Height ($GRASP\_Z$):** **0.425m** (2.5mm clearance over board, centered on 3cm cube).
*   **Safety Height ($SAFE\_Z$):** **0.550m** (12cm clearance over board).
*   **Robot Base:** `Slide X = 0.00`. (2.6cm safety gap between base and table).

## 4. Control Logic
*   **Verticality:** Absolute vertical orientation set during reset; **Zero Delta** passed during steps to lock pose.
*   **Speed Limit:** Position delta capped at **1.5cm per step**.
*   **Gripper:** Forced **CLOSED** during training to minimize collision footprint.

## 5. Reward System
*   **Distance Reward:** `-0.3 * ||achieved - desired||`.
*   **Verticality Penalty:** `-0.1 * ||site_x_axis - [0, 0, -1]||` (Soft crane incentive).
*   **Jitter Penalty:** `-0.003 * ||action[:3]||^2`.
*   **Soft Floor Penalty:** `-0.5` if $Z < 0.410$.
*   **Success Bonus:** `+500` (Condition: Dist < 1.5cm AND Vel < 0.05).
*   **Crash Penalty:** `-500` (Floor or Virtual Tube breach).

## 6. Training Configuration
*   **Episode Steps:** 250.
*   **Virtual Tube:** Tightens to **3.5cm** (matches square boundaries).
