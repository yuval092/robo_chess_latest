# Configuration

All configuration lives in `configs/`. Files are loaded with `src/utils/io.load_config(name)` which reads `configs/{name}.yaml`.

---

## `configs/env.yaml`

Runtime environment parameters. Grouped by concern.

### Geometry & Heights

| Key | Value | Description |
|---|---|---|
| `table_surface_z` | 0.400 | Z of table surface; equals `floor_limit` |
| `table_center_xy` | [0.88, 0.2641] | World XY of table centre |
| `table_half_x/y` | 0.35 | Half-extents of table (70 cm total) |
| `piece_height` | 0.030 | Piece collision box height (m) |
| `grasp_z` | 0.430 | Gripper Z for grasp/place plunge phases |
| `hover_z` | 0.460 | Z where descend/ascend stop; arm idles here |
| `safe_z` | 0.530 | Transit height |
| `floor_limit` | 0.400 | Minimum Z for transit; below triggers `FLOOR_HIT` |
| `torso_height` | 0.370 | Fetch torso joint position |
| `home_position_xy` | [0.88, 0.2641] | XY arm returns to after each move |

### Grasp Pipeline

| Key | Description |
|---|---|
| `grasp_align_tolerance` | Max XY error (m) before plunge begins |
| `hover_speed_threshold` | Max grip speed (m/s) to be "settled" |
| `hover_z_tolerance` | Max Z deviation from `hover_z` to be "at hover height" |
| `grasp_close_steps` | Actuator steps for finger close ramp |
| `grasp_ramp_end` | Finger joint target at end of close ramp (0.010) |
| `empty_grasp_threshold` | Finger joint below this = no piece (0.011) |
| `grasp_hold_steps` | Sim steps to hold grip after close |
| `grasp_plunge_step_m` | Vertical step size (m) per sim step during plunge |
| `grasp_retract_step_m` | Vertical step size (m) per sim step during retract |
| `grasp_verify_xy_threshold` | Max piece–grip XY error for successful grasp |
| `grasp_verify_z_threshold` | Max piece–grip Z error for successful grasp |

### Place Pipeline

| Key | Description |
|---|---|
| `release_ramp_steps` | Steps over which fingers open |
| `release_settle_steps` | Steps to hold fingers open before verifying |
| `place_verify_xy_threshold` | Max piece–target XY drift for successful place |
| `place_verify_z_threshold` | Max piece Z deviation from table for successful place |

### Piece Hold Monitoring (transit/ascend)

| Key | Description |
|---|---|
| `piece_held_xy_limit` | XY distance (m) that triggers `PIECE_DROPPED_XY` |
| `piece_held_z_limit` | Z distance (m) that triggers `PIECE_DROPPED_Z` |

### Post-move Reconciliation

| Key | Description |
|---|---|
| `reconcile_xy_tolerance_m` | Max XY error after place before `XY_LANDING_FAILED` |
| `reconcile_z_tolerance_m` | Max Z error after place before `Z_LANDING_FAILED` |

### RL / Training

| Key | Description |
|---|---|
| `rl_max_steps_per_stage` | Max SAC steps before `TIMEOUT` |
| `success_threshold` | Distance from goal for RL success (m) |
| `eval_drift_limit` | Fixed tube radius during evaluation |
| `stability_vel_threshold` | Speed below which arm is "stable" |
| `drift_limit_start/end` | Curriculum tube radius range |
| `drift_curriculum_steps` | Steps to tighten from start to end |

---

## `configs/chess.yaml`

Board and piece geometry, reserve layouts, and game settings.

| Section | Key | Description |
|---|---|---|
| `board` | `cell_size_m`, `board_size`, `center_xy` | 8×8 board geometry |
| `pieces` | `piece_height_m` | Piece height (matches `env.yaml:piece_height`) |
| `reserves` | `graveyard_slot_spacing_m`, `promotion_slot_spacing_m` | Slot spacing |
| `graveyards` | `white/black.origin_xyz`, `rows`, `cols` | Graveyard grid layout |
| `promotion_reserve` | `white/black.origin_xyz`, `rows`, `cols` | Promotion reserve layout |
| `game` | `human_color`, `auto_computer_reply` | Game behaviour |
| `engine` | `stockfish_path`, `skill_level`, `think_time_s` | Stockfish config |

---

## `configs/physics.yaml`

| Key | Description |
|---|---|
| `vertical_quat` | Gripper downward orientation quaternion |
| `initial_qpos` | Fetch base slide initial positions |
| `env_setup_steps` | Physics settle steps on first env setup |
| `pos_ctrl_scale` | Max arm displacement per action step (0.015 m) |
| `settle_tolerance` | Grip tolerance for `_move_grip_to` waypoints |

---

## `configs/training.yaml`

SAC hyperparameters and training loop settings. See [09_training.md](09_training.md) for full reference.

---

## `configs/deployed_models.yaml`

Paths to the three production SAC model ZIPs. Validated at startup by `src/utils/config_validation.py`.

```yaml
transit: models/transit/model.zip
descend: models/descend/model.zip
ascend:  models/ascend/model.zip
```
