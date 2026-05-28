# Configuration Reference

All YAML config files live under `configs/` and are loaded with:

```python
src.utils.io.load_config(name)
```

`name` maps to `configs/{name}.yaml`.

## Files

| File | Purpose |
|---|---|
| `env.yaml` | Task constants, Z levels, rewards, grasp thresholds, controller tolerances |
| `chess.yaml` | Board geometry, piece/reserve layouts, game and engine config |
| `physics.yaml` | MuJoCo setup constants and action scaling |
| `training.yaml` | SAC training schedule and hyperparameters |
| `deployed_models.yaml` | Production model checkpoint paths |

## `env.yaml`

Geometry and Z levels:

| Key | Value | Meaning |
|---|---:|---|
| `cube_height` | `0.030` | Collision cube height |
| `cube_z` | `0.415` | Resting cube center Z |
| `table_surface_z` | `0.400` | Table top |
| `table_center_xy` | `[0.88, 0.2641]` | Table/board center |
| `table_half_x` | `0.35` | Table half width X |
| `table_half_y` | `0.35` | Table half width Y |
| `grasp_z` | `0.430` | Grasp/place plunge target |
| `hover_z` | `0.460` | Stop height above piece |
| `safe_z` | `0.530` | Transit height |
| `home_position_xy` | `[0.88, 0.2641]` | Return-home XY |

Training thresholds and rewards:

| Key | Value |
|---|---:|
| `success_threshold` | `0.010` |
| `descend_success_threshold` | `0.008` |
| `success_bonus` | `500.0` |
| `crash_penalty` | `-500.0` |
| `dist_reward_weight` | `1.0` |
| `z_reward_weight` | `1.5` |
| `xy_reward_weight` | `2.0` |
| `jitter_penalty_weight` | `0.003` |
| `floor_penalty` | `-0.5` |
| `stability_vel_threshold` | `0.02` |

Drift and braking:

| Key | Value |
|---|---:|
| `drift_limit_start` | `0.100` |
| `drift_limit_end` | `0.008` |
| `drift_curriculum_steps` | `30000` |
| `eval_drift_limit` | `0.010` |
| `transit_braking_dist` | `0.030` |
| `descend_braking_dist` | `0.025` |
| `ascend_braking_dist` | `0.010` |
| `transit_braking_reward_weight` | `0.50` |
| `descend_braking_reward_weight` | `0.30` |
| `ascend_braking_reward_weight` | `0.15` |

Grasp/place:

| Key | Value |
|---|---:|
| `grasp_align_tolerance` | `0.001` |
| `grasp_close_steps` | `24` |
| `grasp_ramp_end` | `0.010` |
| `empty_grasp_threshold` | `0.011` |
| `grasp_hold_steps` | `2` |
| `grasp_plunge_step_m` | `0.006` |
| `grasp_retract_step_m` | `0.006` |
| `release_ramp_steps` | `4` |
| `release_settle_steps` | `2` |
| `grasp_verify_xy_threshold` | `0.015` |
| `grasp_verify_z_threshold` | `0.020` |
| `cube_held_xy_limit` | `0.030` |
| `cube_held_z_limit` | `0.020` |

Execution tolerances:

| Key | Value |
|---|---:|
| `rl_max_steps_per_stage` | `300` |
| `reconcile_xy_tolerance_m` | `0.020` |
| `reconcile_z_tolerance_m` | `0.010` |
| `halt_vel_threshold` | `0.0005` |
| `finger_open_joint` | `0.0181` |
| `finger_closed_joint` | `0.0000` |

## `chess.yaml`

Board:

| Key | Value |
|---|---:|
| `board.cell_size_m` | `0.08` |
| `board.board_size` | `8` |
| `board.center_xy` | `[0.88, 0.2641]` |
| `board.width_m` | `0.64` |
| `board.margin_on_table_m` | `0.03` |
| `board.orientation.white_side` | `low_y` |
| `board.orientation.file_axis` | `x` |
| `board.orientation.rank_axis` | `y` |

Pieces:

| Key | Value |
|---|---:|
| `pieces.cube_half_extent_m` | `0.015` |
| `pieces.cube_height_m` | `0.030` |
| `pieces.freejoint_damping` | `8.0` |

Game:

| Key | Default |
|---|---|
| `game.human_color` | `white` |
| `game.auto_computer_reply` | `true` |

Engine:

| Key | Default |
|---|---|
| `engine.stockfish_path` | `stockfish` |
| `engine.skill_level` | `5` |
| `engine.think_time_s` | `0.5` |
| `engine.fallback_depth` | present but unused |

## `physics.yaml`

| Key | Value | Meaning |
|---|---:|---|
| `vertical_quat` | `[0.7071068, 0, 0.7071068, 0]` | Downward gripper orientation |
| `initial_qpos` | `[-0.05, 0.00]` | Initial Fetch base slides override |
| `env_setup_steps` | `10` | Startup settle steps |
| `max_goal_retries` | `100` | Random goal sampling attempts |
| `pos_ctrl_scale` | `0.015` | Max relative action displacement per step |
| `max_settle_steps` | `500` | Reset settle limit |
| `settle_tolerance` | `0.003` | Reset settle tolerance |
| `settle_gain` | `0.3` | Legacy settle gain config |
| `settle_steps_final` | `25` | Final settle config |

## `deployed_models.yaml`

```yaml
transit: "models/transit.zip"
descend: "models/descend.zip"
ascend: "models/ascend.zip"
```

`resolve_model_paths()` requires all three stages to resolve to a string path.

## Validation

`src/utils/config_validation.py` checks:

- Required `env.yaml` keys.
- Board cell size and board width consistency.
- Presence of `reachability_expected`.
- Deployed model stage keys and string values.
- Existence of the base training model.

Run:

```bash
pytest tests/test_config_schema.py
```

## Utility Modules

`src/utils/io.py` provides:

| Function | Purpose |
|---|---|
| `load_config(name)` | Load `configs/{name}.yaml` |
| `resolve_model_paths(overrides)` | Merge CLI model overrides with deployed model config |
| `setup_logger(name, log_file)` | Create an idempotent file-backed logger |

`src/utils/logger.py` is a backward-compatible shim that re-exports
`setup_logger` from `src.utils.io`.
