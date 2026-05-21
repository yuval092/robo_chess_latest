# V7 Geometric Alignment Design

## 1. Problem Statement
The V6 training failed due to two critical issues:
1.  **Geometric Confusion:** The mapping of the Scenario One-Hot ID into the original `object_pos` observation indices destroyed the pretrained model's "Phase 1: Reaching" logic.
2.  **The "Suicide Trap":** The agent discovered that crashing immediately (Penalty: -100) was mathematically preferable to exploring and timing out (Penalty: ~-175 in accumulated distance rewards). This caused an entropy explosion (`ent_coef` > 2.0).

## 2. Solution: The "Virtual Object" Observation
We will repurpose the 25-dimensional observation space of the `sac-FetchPickAndPlace-v4` model to treat the **Target Waypoint** as the physical object to be reached.

### Observation Mapping (25 Dimensions)
| Index | Original Meaning | V7 Meaning | Reasoning |
|---|---|---|---|
| 0-2 | Gripper Pos | Gripper Pos | Unchanged. |
| 3-5 | Object Pos | **Target Waypoint** | Pretrained weights know how to move to this coordinate. |
| 6-8 | Object Rel Pos | **Goal - Grip Vector** | Provides a direct distance gradient. |
| 9-10 | Gripper State | Gripper State | Unchanged. |
| 11-13| Object Rot | **Scenario One-Hot** | Hijacks an unused slot for context. |
| 14-19| Object Vel | **Zero Mask** | Keeps the model focused on static waypoints. |
| 20-24| Robot Vel | Robot Vel | Unchanged. |

## 3. Reward Surface Stabilization
To prevent the "Suicide Trap," we must ensure that **crashing is always the worst possible outcome**.

*   **Success Bonus:** +500 (Provides a strong pull toward the goal).
*   **Crash Penalty:** -500 (Overrides any distance-based incentive to suicide).
*   **Distance Reward:** -1.0 * Euclidean Distance.
*   **Soft Floor Penalty:** -0.5 if Z < FLOOR_LIMIT + 1cm (Nudges the arm away from danger).

## 4. Environment Constants Updates
*   **GRASP_Z:** Raised to **0.430m** to provide 10mm safety margin above the 2cm piece.
*   **FLOOR_LIMIT:** Lowered to **0.400m** (actual table surface).
*   **Success Criteria:**
    *   `DESCEND`: Strict 2.0cm 3D Euclidean.
    *   `TRANSIT/ASCEND`: 2.0cm XY tolerance, 3.0cm Z tolerance.

## 5. Training Hyperparameters Updates
*   **TOTAL_TIMESTEPS:** 5,000,000.
*   **LEARNING_RATE:** 3e-5 (Slow refinement).
*   **BATCH_SIZE:** 512.
*   **TARGET_ENTROPY:** -6.0 (Forces the model toward determinism as it succeeds).
*   **DRIFT_CURRICULUM_STEPS:** 2,000,000 (per worker average).
