# Configuration Reference

All configuration is loaded via `src/utils/config.py:load_config(name)`, which reads `configs/{name}.yaml` and returns a plain Python dict. There are three configuration files:

- `configs/env.yaml` — MuJoCo environment, arm physics, grasp parameters
- `configs/chess.yaml` — chess board geometry, piece dimensions, graveyard/reserve zones
- `configs/physics.yaml` — MuJoCo controller settings, quaternion, settle parameters

---

## 1. `configs/env.yaml`

This file controls the simulation environment geometry, grasp parameters, and reward/penalty weights.

### Geometry

| Key | Value | Description |
|---|---|---|
| `cube_height` | `0.030` | Full height of chess piece collision box in metres (30mm) |
| `cube_z` | `0.415` | Z position of piece **centre** when resting on table (`table_surface_z + cube_height/2`) |
| `grasp_z` | `0.430` | Z target for gripper during grasp plunge; exactly at cube top (table_z + cube_height) |
| `hover_z` | `0.460` | Safe hover height above pieces; 30mm above grasp_z |
| `safe_z` | `0.510` | Transit altitude; lower cruise height for edge reachability while clearing pieces and board |
| `table_surface_z` | `0.400` | Table surface Z in world coordinates |
| `table_center_xy` | `[0.88, 0.2641]` | Table physics centre (world XY); also used as arm home position |
| `table_half_x` | `0.35` | Half-extent of table in world X (table is 70cm wide) |
| `table_half_y` | `0.35` | Half-extent of table in world Y (table is 70cm deep) |
| `edge_margin` | `0.04` | Margin from table edge used only for movement/eval sampling, not chess geometry |
| `torso_height` | `0.3700` | Fetch robot torso height; tuned to allow near-row (rank 1/2) reachability |

### Home Position

| Key | Value | Description |
|---|---|---|
| `home_position_xy` | `[0.88, 0.2641]` | XY arm rests at between moves; same as board/table centre |

### Grasp Phase Thresholds

| Key | Value | Description |
|---|---|---|
| `grasp_contact_approach_tolerance` | `0.001` | 1mm fine-descent tolerance during plunge |
| `grasp_align_tolerance` | `0.001` | 1mm tolerance for pre-grasp/place vertical alignment; return-home posture reset keeps starts consistent |
| `grasp_close_steps` | `24` | Number of simulation steps for actuator-driven finger close |
| `grasp_ramp_end` | `0.010` | Target joint value for secure grip (10mm) |
| `empty_grasp_threshold` | `0.011` | Joint value above which gripper is considered empty (no cube contact) |
| `grasp_hold_steps` | `2` | Steps to hold grip after closing, letting physics settle |
| `grasp_plunge_step_m` | `0.006` | Vertical increment per step during plunge (6mm) |
| `grasp_retract_step_m` | `0.006` | Vertical increment per step during retract (6mm) |
| `release_ramp_steps` | `4` | Steps to ramp fingers open during place |
| `release_settle_steps` | `2` | Steps to hold open before verification |
| `grasp_verify_xy_threshold` | `0.015` | 15mm: max XY error between gripper and cube to confirm hold |
| `grasp_verify_z_threshold` | `0.020` | 20mm: max Z error between gripper and cube to confirm hold |
| `grasp_verify_finger_threshold` | `0.016` | Finger joint value above stall (≈0.0141) that proves cube is held |

### Cube Hold Monitoring

| Key | Value | Description |
|---|---|---|
| `cube_held_xy_limit` | `0.030` | 30mm: if cube drifts this far from gripper in XY, it's considered dropped |
| `cube_held_z_limit` | `0.020` | 20mm: tighter Z limit to catch drop before cube falls off arm |

### Safety

| Key | Value | Description |
|---|---|---|
| `success_threshold` | `0.010` | 10mm: RL goal success radius |
| `drift_limit_start` | `0.100` | Max drift at start of RL episode |
| `drift_limit_end` | `0.010` | Max drift at end (tight constraint) |
| `floor_limit` | `0.400` | Crash if arm goes below this Z (table surface) |
| `hidden_object_pos` | `[2.0, 2.0, 0.015]` | Where RL goal object is hidden when not in use |

### Reward Weights (RL; not used in scripted mode)

| Key | Value | Description |
|---|---|---|
| `success_bonus` | `500.0` | Sparse reward for reaching goal |
| `crash_penalty` | `-500.0` | Penalty for hitting the floor |
| `z_reward_weight` | `1.5` | Multiplier for Z accuracy |
| `xy_reward_weight` | `2.0` | Multiplier for XY drift penalty |
| `jitter_penalty_weight` | `0.003` | Penalty for high-frequency actions |
| `floor_penalty` | `-0.5` | Constant penalty near table surface |
| `braking_reward_weight` | `0.15` | Reward for decelerating near goal |
| `dist_reward_weight` | `1.0` | Distance-to-goal reward multiplier |

### Gripper Constants

| Key | Value | Description |
|---|---|---|
| `max_gripper_width` | `0.05` | Maximum gripper opening (50mm) |
| `finger_open_joint` | `0.0181` | Joint value for fully open fingers |
| `finger_closed_joint` | `0.0000` | Joint value for fully closed fingers |
| `finger_outer_offset` | `0.033` | Physical finger outer edge offset |

### Transition Thresholds

| Key | Value | Description |
|---|---|---|
| `halt_vel_threshold` | `0.0005` | Arm velocity (m/s) below which it is considered "stopped" |
| `stability_vel_threshold` | `0.05` | Max velocity to consider arm stable |
| `braking_dist` | `0.010` | Distance at which arm starts decelerating |
| `floor_proximity_threshold` | `0.025` | Safety margin above table surface |

### Eval / Debug

| Key | Value | Description |
|---|---|---|
| `min_goal_dist` | `0.10` | Minimum RL goal distance for sampling |
| `eval_drift_limit` | `0.010` | Radial drift limit during eval runs |
| `sample_debug_freq` | `50` | High-fidelity debug log frequency (episodes) |

### Logging

```yaml
logging:
  level: "INFO"
  log_dir: "logs/env_debug"
```

---

## 2. `configs/chess.yaml`

This file controls chess board geometry and piece/zone layout.

### Board

| Key | Value | Description |
|---|---|---|
| `board.cell_size_m` | `0.08` | 80mm per chess square |
| `board.board_size` | `8` | 8×8 board |
| `board.center_xy` | `[0.88, 0.2641]` | Board centre in world XY (matches table centre) |
| `board.width_m` | `0.64` | Total board width (8 × 80mm) |
| `board.margin_on_table_m` | `0.03` | 30mm margin around board on table (not used for piece placement) |
| `board.orientation.white_side` | `low_y` | White pieces start at low Y (rank 1 = low Y) |
| `board.orientation.file_axis` | `x` | Files (a–h) run along the X axis |
| `board.orientation.rank_axis` | `y` | Ranks (1–8) run along the Y axis |
| `board.use_env_edge_margin_for_board` | `false` | Whether env edge margin applies to board bounds |

### Pieces

| Key | Value | Description |
|---|---|---|
| `pieces.cube_half_extent_m` | `0.015` | Physics collision box half-size (30mm cube → 15mm half) |
| `pieces.cube_height_m` | `0.030` | Full cube height; piece centre is at `table_z + 0.015` |
| `pieces.freejoint_damping` | `8.0` | Damping on piece freejoints to prevent sliding |

**Piece visual heights** (from comments in chess.yaml): pawn 18mm, rook 20mm, knight 21mm, bishop 24mm, queen 23mm, king 26mm. STL meshes are already in metres.

### Reserves

| Key | Value | Description |
|---|---|---|
| `reserves.graveyard_slot_spacing_m` | `0.045` | 45mm between graveyard slots |
| `reserves.promotion_slot_spacing_m` | `0.045` | 45mm between promotion reserve slots |
| `reserves.promotion_pieces_per_type` | `8` | Reserve copies per promotion type per colour |
| `reserves.per_color.queen` | `8` | Queen reserve count per colour |
| `reserves.per_color.rook` | `8` | Rook reserve count per colour |
| `reserves.per_color.bishop` | `8` | Bishop reserve count per colour |
| `reserves.per_color.knight` | `8` | Knight reserve count per colour |

### Graveyards

Two graveyard zones (one per colour) for captured pieces. Each is a grid:

```yaml
graveyards:
  white:
    origin_xyz: [0.640, -0.300, 0.015]   # World position of slot_00
    rows: 4
    cols: 4                               # 16 slots total (max 15 captures + buffer)
  black:
    origin_xyz: [0.640, 0.680, 0.015]
    rows: 4
    cols: 4
```

Slot layout: `slot = row * cols + col`. Position: `origin + [row*spacing, col*spacing, 0]`.

White's graveyard is at negative Y (behind white's side). Black's is at high Y (behind black's side).

### Promotion Reserve

Two promotion reserve zones for reserve pieces awaiting promotion:

```yaml
promotion_reserve:
  white:
    origin_xyz: [0.820, -0.300, 0.015]
    rows: 8
    cols: 4   # 32 slots: 4 types × 8 copies
  black:
    origin_xyz: [0.820, 0.680, 0.015]
    rows: 8
    cols: 4
```

### Game Settings

| Key | Value | Description |
|---|---|---|
| `game.human_color` | `"white"` | Which colour the human plays (`"white"` or `"black"`) |
| `game.auto_computer_reply` | `true` | After every human move, automatically play the computer's response |

---

## 3. `configs/physics.yaml`

This file controls MuJoCo physics integration and the scripted controller.

| Key | Value | Description |
|---|---|---|
| `max_settle_steps` | `500` | Max physics steps during arm settle |
| `settle_tolerance` | `0.003` | 3mm: arm is "settled" when velocity is below this |
| `settle_gain` | `0.3` | Proportional gain for settle nudges |
| `vertical_quat` | `[0.7071068, 0.0, 0.7071068, 0.0]` | Unit quaternion for gripper-down orientation (90° rotation about Y axis) |
| `initial_qpos` | `[-0.05, 0.00]` | Initial finger joint positions |
| `env_setup_steps` | `10` | Physics steps run during initial environment setup |
| `max_goal_retries` | `100` | Max attempts to sample a non-overlapping RL goal |
| `pos_ctrl_scale` | `0.015` | Max arm movement per step in metres (15mm — used as MAX_STEP) |
| `settle_steps_final` | `25` | Physics steps for final settle during environment reset |

### vertical_quat explanation

The Fetch robot's gripper faces forward by default. The quaternion `[0.7071068, 0, 0.7071068, 0]` represents a 90° rotation about the Y axis, rotating the gripper from forward-facing to downward-facing. This is enforced at every arm step (see `01_arm_control.md`).

Numerically: `cos(45°) = sin(45°) ≈ 0.7071068`. The quaternion `[w, x, y, z] = [0.7071068, 0, 0.7071068, 0]` is a unit quaternion: `w² + y² = 0.5 + 0.5 = 1.0`. ✓

---

## 4. Config Loading

`load_config(name: str) → dict`:

```python
def load_config(name: str) -> dict:
    path = Path(__file__).parents[2] / "configs" / f"{name}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)
```

- Config is **not cached** — loaded fresh on each call. This is fine because configs are small and loaded infrequently.
- All configs are loaded at class construction time and stored as instance attributes.
- Do not mutate the returned dict — it is not a copy.

---

## 5. Configuration Relationships

Several values appear in multiple configs and must remain consistent:

| Value | Primary | Secondary |
|---|---|---|
| Table/board centre | `env.yaml: table_center_xy` | `chess.yaml: board.center_xy`, `env.yaml: home_position_xy` |
| Cube height | `env.yaml: cube_height` | `chess.yaml: pieces.cube_height_m` |
| Table Z | `env.yaml: table_surface_z` | (derived: `cube_z = table_surface_z + cube_height/2`) |
| MAX_STEP | `physics.yaml: pos_ctrl_scale` | Used as `ScriptedController.MAX_STEP` |

When changing board geometry, update `chess.yaml: board.center_xy` and `env.yaml: table_center_xy` together.
