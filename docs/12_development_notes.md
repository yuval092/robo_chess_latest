# Development Notes

## Important Invariants

- The project is simulation-only; do not document or add real hardware behavior
  unless a hardware layer is actually implemented.
- The chess board commits after physical success, not before.
- MuJoCo state should be touched from the main simulation thread.
- Static XML/STL assets are source of truth.
- `object0` remains for Fetch compatibility but chess play uses selected chess
  piece freejoints.
- Trained model checkpoints expect the transfer observation shape and the current
  Z/grasp geometry.
- Promotion uses reserve physical pieces; the promoted pawn is moved off-board to
  a promotion reserve slot.

## Common Extension Points

| Task | Files to inspect first |
|---|---|
| Add a UI control | `src/ui/templates/index.html`, `src/ui/static/app.js`, `src/ui/app.py` |
| Change chess move behavior | `src/chess_game/move_planner.py`, `src/chess_game/game_orchestrator.py` |
| Change board geometry | `configs/chess.yaml`, `src/chess_game/board_mapper.py`, asset tests |
| Change grasp/place | `src/chess_env/task_execution.py`, `configs/env.yaml` |
| Change learned stage behavior | `src/chess_env/model_controller.py`, `src/chess_env/task_runtime.py` |
| Add a model checkpoint | `configs/deployed_models.yaml` or runtime model override flags |
| Change scene geometry | `chess_env/assets/*.xml`, `chess_env/stls/`, static/physics tests |

## Config and Code Coupling

Some config keys are read during object construction, while others are loaded at
module import time. For example, waypoint constants in `src/chess_env/waypoints.py`
are initialized from `env.yaml` when the module is imported. Restart Python
processes after changing config.

`BoardMapper` validates geometry on construction. If config geometry is invalid,
startup should fail early.

## Model Compatibility Risks

Changing any of the following may require retraining or at least full physical
validation:

- `safe_z`, `hover_z`, `grasp_z`
- `pos_ctrl_scale`
- Gripper joint targets and grasp ramp values
- Piece cube dimensions
- Board/table placement
- Transfer observation layout
- Wrist orientation enforcement policy

The current controller clamps `VERTICAL_QUAT` during transit inference only.
Changing that should be treated as a model compatibility change.

## Debugging Physical Failures

Useful failure strings:

| Error | Likely source |
|---|---|
| `PRECONDITION_FINGER` | Finger state does not match stage expectation |
| `FLOOR_HIT` / `TABLE_HIT` | Gripper dropped below safe surface |
| `TUBE_BREACH` | Descend/ascend drifted outside `eval_drift_limit` |
| `PIECE_DROPPED_XY` / `PIECE_DROPPED_Z` | Held piece no longer tracks gripper |
| `FINGER_CLOSED_EMPTY` | Grasp closed without contacting a piece |
| `VERIFY_XY_FAILED` / `VERIFY_Z_FAILED` | Grasp verification failed |
| `PLACE_XY_FAILED` / `PLACE_Z_FAILED` | Placement verification failed |
| `XY_RECONCILE_FAILED` / `Z_RECONCILE_FAILED` | Post-move piece pose outside tolerance |
| `HOME_POSTURE_RESET_FAILED` | Return-home posture normalization failed |

Start with the `StageResult` sequence in the physical move result, then inspect
the active piece pose and occupancy maps.

## Documentation Maintenance

When behavior changes, update the matching topic document in `docs/` and leave
`docs/old/` as historical reference. The current docs should describe code as it
exists, not planned behavior.

