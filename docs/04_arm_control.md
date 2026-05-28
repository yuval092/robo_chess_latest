# Arm Control

## Controller Overview

`ModelEmbeddedController` is a hybrid controller:

- Learned SAC stages: `transit`, `descend`, `ascend`.
- Scripted physical stages: `grasp`, `place`.
- Scripted transitions between stages via `env.soft_reset()`.

The full board-to-board move is:

```text
pick sequence:
  transit(src_xy) -> descend(src_xy) -> grasp() -> ascend(src_xy)

place sequence:
  transit(dst_xy) -> descend(dst_xy) -> place(dst_xy) -> ascend(dst_xy)
```

`run_full_move(src_xy, dst_xy)` runs both sequences and returns a
`SequenceResult`.

## Result Types

```python
@dataclass
class StageResult:
    success: bool
    steps: int
    crash_reason: str | None
    final_pos: np.ndarray
    error_mm: float
```

```python
@dataclass
class SequenceResult:
    success: bool
    stage_results: list
    failed_at: str | None
    grasp_quality: dict | None
```

`stage_results` contains `(stage_name, StageResult)` pairs such as
`("transit", result)` and `("grasp", result)`.

## Model Loading

Known model stages:

```python
{"transit", "descend", "ascend"}
```

`load_model(stage, path)` temporarily enables the transfer observation space and
loads a Stable-Baselines3 `SAC` checkpoint:

```python
with transfer_obs_enabled(env):
    SAC.load(path, env=env)
```

This matters because SB3 validates the checkpoint against the environment's
observation space at load time.

## Learned Stage Execution

`_run_stage(stage, target_pos)`:

1. Sets `env.goal_pos`, `env.goal`, and `env.current_scenario`.
2. Sets `tube_center_xy` for `descend` and `ascend`.
3. Sets the expected finger target when not holding a piece.
4. Checks finger preconditions when not in `grasp_mode`.
5. Enables transfer observations.
6. Repeatedly predicts deterministic SAC actions.
7. Overrides the gripper action dimension.
8. Steps MuJoCo directly.
9. Checks production crash conditions.
10. Restores the previous observation mode.

The gripper action override is deterministic:

| Stage | Action dimension 3 |
|---|---:|
| `transit` | `-1.0` |
| `descend` | `1.0` |
| `ascend` | `-1.0` |

Only transit clamps `env.data.mocap_quat[0]` to `VERTICAL_QUAT` during inference,
matching that model's training conditions.

## Inference Crash Checks

| Stage | Checks |
|---|---|
| `transit` | floor hit; held-piece drop if in `grasp_mode` |
| `descend` | tube breach using `eval_drift_limit`; table hit |
| `ascend` | tube breach; table hit; held-piece drop if in `grasp_mode` |

Timeout after `env.rl_max_steps_per_stage` becomes `TIMEOUT`.

## Scripted Grasp

`execute_grasp()` runs after descend reaches `HOVER_Z` over the source piece.

Main phases:

1. Halt and settle the full simulation state.
2. Verify speed, Z position, and open fingers.
3. Read active piece pose and reject dangerously rotated cubes.
4. Align above the piece and enforce vertical wrist.
5. Plunge to `GRASP_Z`.
6. Enable `grasp_mode` and ramp fingers closed.
7. Detect empty grasp if fingers close too far.
8. Hold and verify XY/Z/finger thresholds.
9. Retract to `HOVER_Z` while checking the piece remains held.

The active piece is chosen by `env.set_active_piece(piece_id)` before the move.
`get_cube_position()` and `get_cube_quat()` route to that selected piece.

## Scripted Place

`execute_place(dst_xy)` runs after descend reaches `HOVER_Z` over the
destination square.

Main phases:

1. Halt and settle.
2. Verify speed and Z position.
3. Align above destination and enforce vertical wrist.
4. Plunge to `GRASP_Z`.
5. Ramp fingers open.
6. Disable `grasp_mode`.
7. Verify final piece XY and Z placement.
8. Retract to `HOVER_Z`.

After a successful full move, `MovementExecutor` performs final reconciliation
and snaps the freejoint pose exactly to the destination square.

## Return Home

`PhysicalPlanExecutor.return_to_home()` runs after each committed physical plan:

1. `controller.run_transit(home_xy)`.
2. `env.reset_arm_to_home_posture()` if available.

The home-posture reset restores the exact reset-time joint posture and mocap pose.
This avoids accumulating different redundant wrist/roll joint configurations
after repeated end-effector-only moves.

