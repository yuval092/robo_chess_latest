# Design Spec: Curriculum & Learnability Optimization

This document specifies the fixes for the "2mm Trap" and the synchronization of the physical footprint check with the learning curriculum.

## 1. Physical Constraints & Curriculum Alignment
The previous implementation enforced a hard 35mm physical boundary for the finger footprint from Step 0. This conflicted with the 100mm starting limit of the curriculum, causing immediate crashes.

*   **Elastic Boundary**: The allowed footprint radius will now be dynamic: `current_drift_limit + FINGER_OUTER_OFFSET`.
*   **Center Precision Target**: The final target for the arm center drift (`drift_limit_end`) is updated to **5.0mm** (0.005m).
*   **Final Physical Radius**: The robot is certified successful if its total footprint stays within **38.0mm** (33mm fingers + 5mm center) at the end of training.

## 2. Robustness Improvements
*   **Transition Duration**: Scripted finger transitions (Open/Close) during the reset phase will be increased from 20 steps to **50 steps** to ensure joint convergence within 0.5mm tolerance.
*   **Evaluation Alignment**: The `eval_drift_limit` is synchronized to the 5mm production target.

## 3. Training Scale
*   **Workers**: 15 parallel environments.
*   **Curriculum Timing**: 500,000 steps total (global).
*   **Total Duration**: 1,000,000 steps.

## 4. Implementation Strategy
1.  **Config**: Update `env.yaml` with the 5mm targets.
2.  **Task Logic**: Refactor `step()` to use a relative `TUBE_BREACH` check.
3.  **Reset Logic**: Increase settle steps in `_reset_sim`.
4.  **Verification**: Restart training and perform log-based forensic analysis.
