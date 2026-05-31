# MuJoCo Environment

## Class Hierarchy

### `ChessSimulationEnv` (`simulation.py`)

Lowest-level chess wrapper. Handles:

- **XML hot-swap** — replaces `gymnasium-robotics`' default XML path with the chess board XML under a threading lock, then restores it after `super().__init__()`.
- **Action scaling** — scales `[dx, dy, dz]` by `pos_ctrl_scale` (0.015 m/step), sets mocap delta.
- **Finger enforcement** — `_apply_finger_target()` sets actuator ctrl targets every step. When `grasp_mode=False`, it also directly teleports the joint positions so fingers never drift during movement.
- **Board sampling** — `_random_board_position()` samples a uniform XY within `BOARD_MIN_XY`/`BOARD_MAX_XY` margins.
- **`IDENTITY_QUAT`** — shared constant `[1, 0, 0, 0]` used for identity piece orientation.

### `ChessBaseEnv` (`base_env.py`)

Mid-level base used by both production and training. Handles:

- **Config loading** — reads `env.yaml` and `chess.yaml` at init.
- **Observation space** — native 7-D dict obs (`grip_pos`, `grip_vel`, `finger_angle`, `goal`). Switches to 25-D pretrained format when `_use_pretrained_obs_format=True`.
- **Task constants** — `GRASP_Z`, `HOVER_Z`, `SAFE_Z`, `FINGER_OPEN_JOINT`, `FINGER_CLOSED_JOINT`, `HOME_POS`, etc.
- **Episode reset** (`_reset_sim`) — picks scenario, places objects, settles arm, validates finger state, calls `_on_reset_settled`.
- **Soft reset** (`soft_reset`) — switches scenario mid-episode without teleporting the arm. Used by `ModelEmbeddedController._prepare_stage` between SAC stages.
- **Low-level helpers** — `_move_grip_to`, `_step_grip_toward`, `_commit_finger_target`, `_settle_arm_to_start`.
- **Chess piece placement** — places all 32 active pieces and reserve pieces via `_reset_chess_piece_bodies`.

### `ChessProductionEnv` (`production_env.py`)

Adds piece tracking and scripted pipelines on top of `ChessBaseEnv`. Handles:

- **Active piece state** — `set_active_piece(piece_id)`, `clear_active_piece()`, `get_active_piece_position()`, `get_active_piece_quat()`.
- **Grasp pipeline** — `execute_grasp()` runs the full scripted GRASP sequence.
- **Place pipeline** — `execute_place(dst_xy)` runs the full scripted PLACE sequence.
- **Home posture** — captures and restores the exact arm configuration after each move via `reset_arm_to_home_posture()`.
- **Piece drop monitoring** — `_check_piece_held()` verifies the piece stays in the gripper during transit.

### `ChessTrainingEnv` (`training_env.py`)

Adds RL training on top of `ChessBaseEnv`. Handles:

- **`step()`** — applies action, computes reward, checks crash, returns gymnasium tuple.
- **Reward shaping** — dense distance reward + braking penalty + jitter penalty + floor penalty + sparse success bonus.
- **Crash detection** — `_check_crash()` returns a reason string for `FINGER_FAULT`, `FLOOR_HIT`, `TUBE_BREACH`, `TABLE_HIT`.
- **Drift curriculum** — tube radius tightens from `drift_limit_start` to `drift_limit_end` over `drift_curriculum_steps`.

## Episode Reset Sequence

1. `_setup_episode()` — picks scenario (transit / descend / ascend), samples `start_xy` and `goal_pos`, sets `tube_center_xy`.
2. `super()._reset_sim()` — calls `_sample_goal()`.
3. `_place_objects()` — places dummy cube and all chess pieces, forward-propagates.
4. `_prepare_arm_for_episode()` — moves arm to `arm_start_pos`, opens/closes fingers per scenario.
5. `_validate_finger_state()` — asserts finger is at target (returns `False` to trigger retry if not).
6. `_on_reset_settled()` — hook for subclasses. `ChessProductionEnv` captures home posture here.

## Soft Reset

Used between SAC stages to switch scenario context without teleporting the arm:

1. `_halt_arm()` — zeros robot DOFs, waits up to 30 steps for velocity to drop below `halt_vel_threshold`.
2. `_align_to_exit_pos()` — nudges arm to the nominal exit position of the previous stage.
3. `_update_scenario_state()` — switches `current_scenario`, `goal_pos`, `tube_center_xy`, `episode_steps`.
4. `_transition_fingers_for_scenario()` — opens fingers for descend, closes for ascend/transit.

## Grasp Mode

`self.grasp_mode` controls finger physics:

| `grasp_mode` | Finger behaviour |
|---|---|
| `False` | Teleport mode — joint positions forced directly, no physics drift |
| `True` | Actuator mode — contact forces active, piece can push back against fingers |

Set to `True` just before `_close_fingers()` in `execute_grasp`, set back to `False` after `_open_fingers()` in `execute_place`.

## Height Levels

| Constant | Typical value | Role |
|---|---|---|
| `TABLE_SURFACE_Z` | 0.400 m | Table surface / floor limit for transit |
| `GRASP_Z` | 0.430 m | Gripper Z for both grasp and place plunge |
| `HOVER_Z` | 0.460 m | Where descend/ascend stages stop; arm idles here |
| `SAFE_Z` | 0.530 m | Transit height |
