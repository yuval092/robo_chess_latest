# Configuration Reference

## Overview

All configuration lives in YAML files under `configs/`. They are loaded at runtime via `src/utils/io.load_config(name)`, which resolves `configs/{name}.yaml` relative to the project root. No config is baked into module-level constants — all parameters are read from these files at startup.

| File | Purpose |
|---|---|
| `configs/env.yaml` | Physics constants, Z-levels, grasp thresholds, reward weights, controller tuning |
| `configs/chess.yaml` | Board geometry, piece config, graveyard/reserve layout, game settings, Stockfish |
| `configs/physics.yaml` | MuJoCo solver settings, gripper quaternion, initial arm posture |
| `configs/training.yaml` | SAC hyperparameters, training schedule, logging |
| `configs/deployed_models.yaml` | Paths to current production RL model checkpoints |

---

## `configs/env.yaml` — Environment Constants

### Physical Dimensions

| Key | Value | Description |
|---|---|---|
| `cube_height` | `0.030` | Cube geom full height (m) |
| `cube_z` | `0.415` | World Z of cube centre when resting on table (table_z + cube_height/2) |
| `table_surface_z` | `0.400` | Table top surface Z (m) |
| `table_center_xy` | `[0.88, 0.2641]` | World XY of table centre; also the home position |
| `table_half_x` | `0.35` | Table half-extent in X (full width = 0.70 m) |
| `table_half_y` | `0.35` | Table half-extent in Y (full width = 0.70 m) |
| `edge_margin` | `0.04` | Extra margin for goal sampling and evaluation (not chess geometry) |
| `torso_height` | `0.3700` | Fetch torso height; tuned to ensure near-row reachability |

### Z-Level Constants

These form the key heights used throughout arm control:

| Key | Value | Meaning |
|---|---|---|
| `grasp_z` | `0.430` | Target Z for finger contact; 18.5mm finger overlap into cube, 11.5mm table clearance |
| `hover_z` | `0.460` | Safe RL stop height after descent; allows RL variance + finger clearance |
| `safe_z` | `0.530` | Transit altitude; piece centre clears all 32 pieces even at full height |

### RL Task Thresholds

| Key | Value | Description |
|---|---|---|
| `success_threshold` | `0.010` | Radial (3D) success distance in metres (10 mm) |
| `drift_limit_start` | `0.100` | Initial tube radius at start of drift curriculum (100 mm) |
| `drift_limit_end` | `0.008` | Final tube radius after curriculum (8 mm) |
| `drift_curriculum_steps` | `30000` | Per-worker steps to reach `drift_limit_end` |
| `floor_limit` | `0.400` | Minimum allowed gripper Z (table surface); breaching triggers floor penalty |
| `hidden_object_pos` | `[2.0, 2.0, 0.015]` | Park position for the RL training object when in chess mode |
| `eval_drift_limit` | `0.010` | Strict drift limit used during evaluation (10 mm) |
| `min_goal_dist` | `0.10` | Minimum XY distance between goal and gripper start for RL training |
| `descend_success_threshold` | `0.008` | Tighter Z threshold for descend success (8 mm) during training |

### Reward Weights and Penalties

| Key | Value | Description |
|---|---|---|
| `success_bonus` | `500.0` | Terminal reward for reaching goal |
| `crash_penalty` | `-500.0` | Terminal penalty for tube breach (drift) or floor violation |
| `dist_reward_weight` | `1.0` | Multiplier for 3D distance-to-goal reward |
| `z_reward_weight` | `1.5` | Extra weight on Z-axis error (encourages vertical precision) |
| `xy_reward_weight` | `2.0` | Penalty weight on XY drift from tube centre |
| `jitter_penalty_weight` | `0.003` | Penalises large actions; matches v2 original training |
| `floor_penalty` | `-0.5` | Per-step penalty when gripper is near floor |
| `braking_reward_weight` | `0.15` | Fallback velocity penalty weight |
| `transit_braking_reward_weight` | `0.50` | Transit: strong velocity penalty near goal |
| `descend_braking_reward_weight` | `0.30` | Descend: moderate velocity penalty |
| `ascend_braking_reward_weight` | `0.15` | Ascend: matches v2 original (no meaningful braking outside success zone) |
| `floor_proximity_threshold` | `0.025` | Safety margin above floor that triggers floor penalty |

### Stage-Specific Braking Distances

| Key | Value | Description |
|---|---|---|
| `braking_dist` | `0.010` | Fallback braking distance (used for ascend) |
| `transit_braking_dist` | `0.030` | Transit: arm decelerates within 30 mm of goal |
| `descend_braking_dist` | `0.025` | Descend: arm decelerates within 25 mm of HOVER_Z |
| `ascend_braking_dist` | `0.010` | Ascend: equals success threshold — effectively no braking zone |

The ascend braking distance intentionally equals the success threshold because gravity assists deceleration during ascent. Adding a meaningful braking zone (tried in v3/v4 with 30mm/12mm) hurt performance.

### Controller Movement Parameters

| Key | Value | Description |
|---|---|---|
| `transit_tolerance_m` | `0.004` | XY stopping tolerance for scripted transit (4 mm) |
| `vertical_tolerance_m` | `0.004` | Z stopping tolerance for scripted descend/ascend (4 mm) |
| `step_gain` | `1.0` | Proportional gain multiplier for scripted controller |
| `min_step_size_m` | `0.002` | Minimum step size per scripted controller iteration |
| `max_step_size_m` | `0.024` | Maximum step size per scripted controller iteration |
| `transit_max_steps` | `300` | Max steps before scripted transit times out |
| `vertical_max_steps` | `200` | Max steps before scripted descent/ascent times out |
| `rl_max_steps_per_stage` | `300` | Max steps per RL inference stage |
| `stability_vel_threshold` | `0.02` | Max velocity (m/s) for arm to be considered stable/stopped |
| `halt_vel_threshold` | `0.0005` | Max velocity to call arm "stopped" during transition phases |

### Grasp Phase Thresholds

| Key | Value | Description |
|---|---|---|
| `grasp_contact_approach_tolerance` | `0.001` | 1mm tolerance for fine descent step during grasp |
| `grasp_align_tolerance` | `0.001` | 1mm XY alignment tolerance before grasp/place |
| `grasp_close_steps` | `24` | Steps for actuator-driven finger close ramp |
| `grasp_ramp_end` | `0.010` | Secure grip joint target; avoids excessive squeeze |
| `empty_grasp_threshold` | `0.011` | Finger joint value below which grip is considered empty (cube stalls near 0.013) |
| `grasp_hold_steps` | `2` | Hold steps after close to let physics settle |
| `grasp_plunge_step_m` | `0.006` | Vertical increment per step during grasp approach |
| `grasp_retract_step_m` | `0.006` | Vertical increment per step during post-grasp retract |
| `release_ramp_steps` | `4` | Steps to ramp fingers open during place |
| `release_settle_steps` | `2` | Steps to hold fully open before release verification |
| `grasp_verify_xy_threshold` | `0.015` | Max XY error between cube and gripper to confirm hold (15 mm) |
| `grasp_verify_z_threshold` | `0.020` | Max Z error between cube and gripper to confirm hold (20 mm) |
| `grasp_verify_finger_threshold` | `0.016` | Finger joint threshold to confirm hold (above stall at ~0.014) |
| `grasp_verify_drift_mm` | `30.0` | Max XY drift tolerance while verifying grasp |

### Cube Hold Monitoring

| Key | Value | Description |
|---|---|---|
| `cube_held_xy_limit` | `0.030` | XY distance limit cube→gripper to detect drop (30 mm) |
| `cube_held_z_limit` | `0.020` | Z distance limit cube→gripper to detect drop (20 mm) |

### Post-Place Reconciliation

| Key | Value | Description |
|---|---|---|
| `reconcile_xy_tolerance_m` | `0.020` | Max XY error after arm move before reconcile failure (20 mm) |
| `reconcile_z_tolerance_m` | `0.010` | Max Z error after arm move before reconcile failure (10 mm) |

### Gripper Constants

| Key | Value | Description |
|---|---|---|
| `max_gripper_width` | `0.05` | Maximum gripper opening (metres) |
| `finger_open_joint` | `0.0181` | Joint value for fully open fingers |
| `finger_closed_joint` | `0.0000` | Joint value for fully closed fingers |
| `finger_outer_offset` | `0.033` | Spatial offset from gripper centre to finger tips |

### Home Position

| Key | Value | Description |
|---|---|---|
| `home_position_xy` | `[0.88, 0.2641]` | XY for return-to-home after each move (board centre) |

### UI Busy Polling (Documentation)

| Key | Value | Description |
|---|---|---|
| `busy_poll_initial_ms` | `200` | Initial polling interval when arm is busy |
| `busy_poll_max_ms` | `1500` | Maximum polling interval (exponential backoff cap) |

### Logging

| Key | Value | Description |
|---|---|---|
| `logging.level` | `"INFO"` | Default log level |
| `logging.log_dir` | `"logs/env_debug"` | Directory for environment debug logs |
| `sample_debug_freq` | `50` | Enable high-fidelity debug logging every N episodes |

---

## `configs/chess.yaml` — Chess Board and Game Settings

### Board Geometry

| Key | Value | Description |
|---|---|---|
| `board.cell_size_m` | `0.08` | Square side length (80 mm) |
| `board.board_size` | `8` | Board dimension (8×8) |
| `board.center_xy` | `[0.88, 0.2641]` | World XY of board centre |
| `board.width_m` | `0.64` | Total board width (8 × 80 mm) |
| `board.margin_on_table_m` | `0.03` | Minimum margin from board edge to table edge (30 mm) |
| `board.orientation.white_side` | `low_y` | White pieces start at low Y (rank 1–2) |
| `board.orientation.file_axis` | `x` | Files (a–h) map to world X axis |
| `board.orientation.rank_axis` | `y` | Ranks (1–8) map to world Y axis |

Derived board extents:
- File a: X = 0.600 m, File h: X = 1.160 m
- Rank 1: Y ≈ −0.016 m, Rank 8: Y ≈ 0.544 m

The board fits within the 70cm×70cm table with 3cm margins on all sides.

### Geometry Validation

```yaml
board.validation:
  required_cell_size_m: 0.08
  required_board_width_m: 0.64
  required_table_margin_m: 0.03
  geometry_tolerance_m: 1.0e-9
```

`BoardMapper._validate_geometry()` asserts these values at startup. The tolerance is set to floating-point precision — any mismatch is a configuration error.

### Reachability Expectations

```yaml
reachability_expected:
  rank1_y_m: -0.0159
  rank8_y_m: 0.5441
  file_a_x_m: 0.600
  file_h_x_m: 1.160
  geometry_tolerance_m: 1.0e-9
```

Used by `eval_chess_reachability.py` to verify that all 64 squares map to the expected world coordinates.

### Piece Geometry

| Key | Value | Description |
|---|---|---|
| `pieces.cube_half_extent_m` | `0.015` | Physics cube half-size (30 mm cube) |
| `pieces.cube_height_m` | `0.030` | Full cube height |
| `pieces.freejoint_damping` | `8.0` | MuJoCo free joint damping to prevent tumbling |

Visual mesh geoms use `pos="0 0 0.017"` to sit on top of the cube. STL files are pre-scaled in metres.

### Graveyard Layout

Two 4×4 grids, one per colour, holding captured pieces:

```yaml
graveyards:
  white:
    origin_xyz: [0.640, -0.300, 0.015]   # South of board
    rows: 4
    cols: 4
  black:
    origin_xyz: [0.640, 0.680, 0.015]    # North of board
    rows: 4
    cols: 4
```

Slot spacing: 45mm (`reserves.graveyard_slot_spacing_m`). 16 slots per colour supports full captures (max 15 non-king pieces + 1 buffer).

### Promotion Reserve Layout

Two 8×4 grids, one per colour, holding reserve pieces for pawn promotion:

```yaml
promotion_reserve:
  white:
    origin_xyz: [0.820, -0.300, 0.015]   # South of board, east of graveyard
    rows: 8
    cols: 4
  black:
    origin_xyz: [0.820, 0.680, 0.015]    # North of board, east of graveyard
    rows: 8
    cols: 4
```

32 slots per colour: 8 queens + 8 rooks + 8 bishops + 8 knights. Spacing: 45mm.

```yaml
reserves:
  promotion_pieces_per_type: 8
  per_color:
    queen: 8
    rook: 8
    bishop: 8
    knight: 8
```

### Game Settings

| Key | Value | Description |
|---|---|---|
| `game.human_color` | `white` | Default human player colour |
| `game.auto_computer_reply` | `true` | Computer plays automatically after human move |

### Stockfish Engine

| Key | Default | Description |
|---|---|---|
| `engine.stockfish_path` | `stockfish` | Executable name (on PATH) or full path; `null` to disable |
| `engine.skill_level` | `5` | Skill level 0–20 (0 ≈ 800 ELO, 20 ≈ full Stockfish strength) |
| `engine.think_time_s` | `0.5` | Seconds Stockfish may think per move |
| `engine.fallback_depth` | `3` | Minimax depth when Stockfish is unavailable |

---

## `configs/physics.yaml` — MuJoCo Physics Settings

### Simulation Reset

| Key | Value | Description |
|---|---|---|
| `max_settle_steps` | `500` | Maximum physics steps during reset settling |
| `settle_tolerance` | `0.003` | Velocity tolerance to consider simulation settled |
| `settle_gain` | `0.3` | Damping gain applied during settling |
| `env_setup_steps` | `10` | Steps to run on first env setup |
| `max_goal_retries` | `100` | Max attempts to sample a non-overlapping RL goal |
| `settle_steps_final` | `25` | Final settle steps after placing pieces in reset |

### Arm Configuration

| Key | Value | Description |
|---|---|---|
| `vertical_quat` | `[0.7071068, 0.0, 0.7071068, 0.0]` | Quaternion for gripper pointing straight down (90° about Y) |
| `initial_qpos` | `[-0.05, 0.00]` | Initial torso/slider joint positions |
| `pos_ctrl_scale` | `0.015` | Max arm displacement per MuJoCo step (metres) |

`vertical_quat` is applied every transit step in `ModelEmbeddedController` to prevent wrist rotation from drifting during the long horizontal sweep. Without it, pretrained FetchPickAndPlace weights sometimes tilt the wrist, causing grasp failures.

---

## `configs/training.yaml` — SAC Training Hyperparameters

### Model and Parallelism

| Key | Value | Description |
|---|---|---|
| `base_model` | `"models/pretrained/sac-FetchPickAndPlace-v4.zip"` | Starting checkpoint for fine-tuning |
| `num_envs` | `4` | Parallel training environments (SubprocVecEnv) |
| `total_timesteps` | `600000` | Total training steps (recommend 1M for ascend) |

### SAC Hyperparameters

| Key | Value | Rationale |
|---|---|---|
| `learning_rate` | `0.00005` | Very conservative; preserves pretrained features |
| `batch_size` | `512` | Large batch; stable gradients for dense rewards |
| `target_entropy` | `-4.0` | Standard heuristic: `-dim(action_space) = -4` |
| `learning_starts` | `10000` | Steps of random exploration before gradient updates |
| `buffer_size` | `300000` | Holds ~half of training data for off-policy reuse |
| `initial_ent_coef` | `0.1` | Entropy coefficient reset on load; starts exploration high |
| `ent_coef_lr` | `0.001` | Learning rate for automatic entropy coefficient tuning |

### Evaluation Schedule

| Key | Value | Description |
|---|---|---|
| `eval_freq` | `50000` | Evaluate every 50K steps |
| `n_eval_episodes` | `20` | Episodes per evaluation run (5 is too noisy) |

### Logging

| Key | Value | Description |
|---|---|---|
| `log_freq` | `2000` | Log rolling stats every 2000 steps |
| `moving_avg_window` | `100` | Episode window for rolling mean reward/length/success |

---

## `configs/deployed_models.yaml` — Production Model Paths

```yaml
transit: "models/transit.zip"
descend: "models/descend.zip"
ascend:  "models/ascend.zip"
```

These paths are relative to the project root. They are read by `build_controller()` in `run_chess_ui.py` and by evaluation scripts.

**To deploy a new model:**
1. Train: `python scripts/train_rl.py --stage ascend`
2. Evaluate: `python scripts/eval_all_cells_rl.py --mode key`
3. Update this file with the new checkpoint path
4. Validate: `python scripts/validate_deployed_models.py`
5. Test end-to-end: `robo-chess-ui --visualize`

---

## Configuration Loading (`src/utils/io.py`)

```python
def load_config(name: str) -> dict:
    path = project_root / "configs" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
```

Config files are loaded fresh on each call — there is no global cache. This means configuration changes (e.g., during development) take effect on the next call without restart, but also means repeated calls in hot paths incur file I/O. In practice all loading happens at startup.

---

## Configuration Validation (`scripts/validate_config.py`)

Checks that all required keys exist in all config files and verifies internal consistency (e.g., board geometry constraints, model file paths). Run before training or evaluation:

```bash
python scripts/validate_config.py
```

The pytest test suite also runs this via `tests/test_config_schema.py::test_config_schema_is_valid`.
