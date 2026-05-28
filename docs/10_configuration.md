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
| `cube_z` | `0.415` | Resting cube center Z (`table_surface_z + cube_height/2`) |
| `table_surface_z` | `0.400` | Table top; equals `floor_limit` |
| `floor_limit` | `0.400` | Minimum gripper Z for transit; below this triggers `FLOOR_HIT` |
| `table_center_xy` | `[0.88, 0.2641]` | Table/board center |
| `table_half_x` | `0.35` | Table half width X |
| `table_half_y` | `0.35` | Table half width Y |
| `edge_margin` | `0.04` | Inset margin from table edges when sampling random training positions |
| `min_goal_dist` | `0.10` | Minimum XY distance required between object and sampled training goal |
| `torso_height` | `0.3700` | Fetch torso lift height, tuned for full-board arm reachability |
| `grasp_z` | `0.430` | Gripper Z target for grasp and place plunge phases |
| `hover_z` | `0.460` | Gripper Z where descend/ascend stages stop; arm idles here before grasp/place |
| `safe_z` | `0.530` | Transit height; arm travels between squares at this Z |
| `home_position_xy` | `[0.88, 0.2641]` | Return-home XY after each move |

Training thresholds and rewards:

| Key | Value | Meaning |
|---|---:|---|
| `success_threshold` | `0.010` | Distance from goal (m) for RL stage success |
| `descend_success_threshold` | `0.008` | Tighter Z threshold for descend success |
| `success_bonus` | `500.0` | Sparse reward on episode success |
| `crash_penalty` | `-500.0` | Sparse penalty on crash termination |
| `dist_reward_weight` | `1.0` | Dense reward weight for total distance to goal |
| `z_reward_weight` | `1.5` | Dense reward weight for Z-axis proximity |
| `xy_reward_weight` | `2.0` | Dense reward weight for XY proximity |
| `jitter_penalty_weight` | `0.003` | Penalty coefficient for squared XYZ action magnitude |
| `floor_penalty` | `-0.5` | Added when gripper is within `floor_proximity_threshold` of `floor_limit` (value is negative) |
| `floor_proximity_threshold` | `0.025` | Safety margin above `floor_limit` that triggers the floor penalty |
| `stability_vel_threshold` | `0.02` | Grip speed (m/s) below which the arm is stable for success |

Drift and braking:

| Key | Value | Meaning |
|---|---:|---|
| `drift_limit_start` | `0.100` | Initial XY tube radius for descend/ascend curriculum |
| `drift_limit_end` | `0.008` | Final XY tube radius after curriculum |
| `drift_curriculum_steps` | `30000` | Per-worker steps to tighten the tube to `drift_limit_end` |
| `eval_drift_limit` | `0.010` | Fixed tube radius used during evaluation |
| `transit_braking_dist` | `0.030` | Transit distance from goal at which velocity penalty begins |
| `descend_braking_dist` | `0.025` | Descend distance from goal at which velocity penalty begins |
| `ascend_braking_dist` | `0.010` | Ascend distance from goal at which velocity penalty begins |
| `braking_dist` | `0.010` | Fallback braking distance when no stage-specific key is present |
| `transit_braking_reward_weight` | `0.50` | Transit velocity penalty weight |
| `descend_braking_reward_weight` | `0.30` | Descend velocity penalty weight |
| `ascend_braking_reward_weight` | `0.15` | Ascend velocity penalty weight |
| `braking_reward_weight` | `0.15` | Fallback braking weight when no stage-specific key is present |

Grasp/place:

| Key | Value | Meaning |
|---|---:|---|
| `grasp_align_tolerance` | `0.001` | Maximum XY error (m) allowed before the plunge begins |
| `grasp_close_steps` | `24` | Steps to ramp fingers from open to grip target |
| `grasp_ramp_end` | `0.010` | Finger joint target at end of close ramp |
| `empty_grasp_threshold` | `0.011` | Finger joint below this during close means cube is absent |
| `grasp_hold_steps` | `2` | Steps to hold grip after close so contact impulses settle |
| `grasp_plunge_step_m` | `0.006` | Vertical step size (m) per physics step during plunge |
| `grasp_retract_step_m` | `0.006` | Vertical step size (m) per physics step during retract |
| `release_ramp_steps` | `4` | Steps to ramp fingers open during place |
| `release_settle_steps` | `2` | Steps to hold fingers open before placement is verified |
| `grasp_verify_xy_threshold` | `0.015` | Maximum cube-to-grip XY distance (m) for a successful grasp |
| `grasp_verify_z_threshold` | `0.020` | Maximum cube-to-grip Z distance (m) for a successful grasp |
| `cube_held_xy_limit` | `0.030` | XY deviation (m) that triggers `CUBE_DROPPED_XY` |
| `cube_held_z_limit` | `0.020` | Z deviation (m) that triggers `CUBE_DROPPED_Z` |

Execution tolerances:

| Key | Value | Meaning |
|---|---:|---|
| `rl_max_steps_per_stage` | `300` | Maximum SAC inference steps per stage before `TIMEOUT` |
| `reconcile_xy_tolerance_m` | `0.020` | Maximum XY error between final piece position and destination |
| `reconcile_z_tolerance_m` | `0.010` | Maximum Z error between final piece position and table surface |
| `halt_vel_threshold` | `0.0005` | Maximum gripper speed (m/s) to consider the arm stopped |
| `finger_open_joint` | `0.0181` | Finger joint position for fully open fingers |
| `finger_closed_joint` | `0.0000` | Finger joint position for fully closed fingers |

## `chess.yaml`

Board:

| Key | Value | Meaning |
|---|---:|---|
| `board.cell_size_m` | `0.08` | Physical width of one chess square (m) |
| `board.board_size` | `8` | Squares along one edge |
| `board.center_xy` | `[0.88, 0.2641]` | World XY of the board centre |

Pieces:

| Key | Value | Meaning |
|---|---:|---|
| `pieces.cube_height_m` | `0.030` | Full collision cube height; piece centre at `table_z + 0.015` |
| `pieces.freejoint_damping` | `8.0` | MuJoCo freejoint damping for every chess piece body |

Game:

| Key | Default |
|---|---|
| `game.human_color` | `white` |
| `game.auto_computer_reply` | `true` |

Engine:

| Key | Default | Meaning |
|---|---|---|
| `engine.stockfish_path` | `stockfish` | Executable name on PATH or full path; `null` disables engine |
| `engine.skill_level` | `5` | UCI Skill Level (0–20) |
| `engine.think_time_s` | `0.5` | Seconds Stockfish may search per move |

## `physics.yaml`

| Key | Value | Meaning |
|---|---:|---|
| `vertical_quat` | `[0.7071068, 0, 0.7071068, 0]` | Unit quaternion for downward gripper orientation |
| `initial_qpos` | `[-0.05, 0.00]` | Initial Fetch base slide positions |
| `env_setup_steps` | `10` | Physics settle steps during initial env setup |
| `max_goal_retries` | `100` | Random goal sampling attempts |
| `pos_ctrl_scale` | `0.015` | Max arm displacement (m) per action step |
| `settle_tolerance` | `0.003` | Gripper position tolerance (m) for `_move_mocap_to` |

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
