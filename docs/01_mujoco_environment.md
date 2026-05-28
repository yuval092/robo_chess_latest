# MuJoCo Environment

## Overview

The simulation is built on top of `MujocoFetchPickAndPlaceEnv` from `gymnasium-robotics`. The Fetch robot arm — a 7-DOF manipulator with a parallel-jaw gripper — operates on a 70cm × 70cm table containing an 8×8 chess board. The environment is modified in several important ways from the stock `FetchPickAndPlace-v4`:

- The table is larger (64cm chess board with 3cm margins vs. the stock 25cm workspace)
- The robot's base position and torso height are re-tuned for board reachability
- The physics step and action scaling are adjusted for smooth, slow movement
- The scene XML is checked-in and injected at startup
- A "tube constraint" limits lateral drift during vertical stages
- Gripper state is explicitly controlled (teleport mode vs. actuator-driven mode)

---

## Class Hierarchy

```
MujocoFetchPickAndPlaceEnv   (gymnasium-robotics)
  └── ChessSimulationEnv     (src/chess_env/simulation.py)
        └── ChessTaskEnv     (src/chess_env/task.py)
              assembled from:
                GraspPlaceMixin    (src/chess_env/task_execution.py)
                TaskStateMixin     (src/chess_env/task_state.py)
                TaskRuntimeMixin   (src/chess_env/task_runtime.py)
                ChessSimulationEnv
```

Python's MRO ensures that mixin methods are called in the order listed in the class declaration.

---

## `ChessSimulationEnv` (`src/chess_env/simulation.py`)

### Responsibility
Low-level MuJoCo environment setup: model loading, board sampling, action scaling, and gripper enforcement.

### Gymnasium Registration
`ChessFetchTask-v0` is registered in `src/chess_env/__init__.py`. The environment is created with:
```python
gym.make("ChessFetchTask-v0", force_scenario="transit", ...)
```

### XML Injection Trick
MuJoCo requires the XML path to be set before the parent class is initialised. `ChessSimulationEnv.__init__` temporarily overrides the module-level `MODEL_XML_PATH` variable in `gymnasium_robotics.envs.fetch.pick_and_place` using a threading lock, calls `super().__init__()`, then restores the original path. This ensures our custom assets are loaded without monkey-patching the installed package permanently.

```python
with _xml_path_lock:
    _original_path = _fpp_module.MODEL_XML_PATH
    _fpp_module.MODEL_XML_PATH = asset_path   # chess_env/assets/pick_and_place.xml
    try:
        super().__init__(**kwargs)
    finally:
        _fpp_module.MODEL_XML_PATH = _original_path
```

### Key Fields and Constants

| Field | Source | Description |
|---|---|---|
| `TABLE_CENTER_XY` | `env.yaml:table_center_xy` | World XY of board/table center: `[0.88, 0.2641]` |
| `TABLE_HALF_X` | `env.yaml:table_half_x` | Table half-extent in X: 0.35 m |
| `TABLE_HALF_Y` | `env.yaml:table_half_y` | Table half-extent in Y: 0.35 m |
| `TABLE_SURFACE_Z` | `env.yaml:table_surface_z` | Table top altitude: 0.400 m |
| `CUBE_HEIGHT` | `env.yaml:cube_height` | Chess piece collision body height: 0.030 m |
| `EDGE_MARGIN` | `env.yaml:edge_margin` | Board sampling/eval margin: 0.04 m |
| `MIN_GOAL_DIST` | `env.yaml:min_goal_dist` | Minimum goal–object separation for sampling: 0.10 m |
| `POS_CTRL_SCALE` | `physics.yaml:pos_ctrl_scale` | Max displacement per sim step: 0.015 m |
| `VERTICAL_QUAT` | `physics.yaml:vertical_quat` | `[0.7071, 0, 0.7071, 0]` — gripper points straight down |
| `ENV_SETUP_STEPS` | `physics.yaml:env_setup_steps` | Physics settle steps at startup: 10 |
| `torso_height` | `env.yaml:torso_height` | Robot torso lift joint: 0.3700 m |

### `_set_action(action)` — Action Interface

The RL action space is a 4-vector `[dx, dy, dz, gripper]`. `_set_action` translates this into MuJoCo mocap control:

1. **Position control**: `pos_ctrl = action[:3] * POS_CTRL_SCALE` — scales the displacement down to at most 15mm per step.
2. **Rotation**: Always zero-delta. `_set_action()` does not apply rotational action; setup and selected controllers enforce `VERTICAL_QUAT` where required.
3. **Gripper enforcement**:
   - When `grasp_mode=False` (transit/descend/ascend without a piece): finger joints are **teleported** to `finger_target_joint`. Both `ctrl` and `qpos`/`qvel` are set directly. This bypasses contact physics.
   - When `grasp_mode=True` (carrying a piece): only `ctrl[0]` and `ctrl[1]` are set. The MuJoCo position actuator drives the fingers, allowing contact forces with the held piece.

### `_reset_sim()` — Board Object Placement

Samples a random board position for the MuJoCo `object0` (used during RL training). In chess mode, `object0` is hidden (placed far off-screen) because chess pieces are tracked separately.

### `_env_setup(initial_qpos)` — First Setup

Called once at environment construction. Forces the torso to the configured height, enforces vertical quaternion on the mocap body, and runs `ENV_SETUP_STEPS` physics steps to let the arm settle into a stable resting configuration.

### `_render_callback()`

Overrides the parent's goal-marker rendering to suppress the floating target sphere that FetchPickAndPlace normally shows.

---

## `ChessTaskEnv` (`src/chess_env/task.py`)

### Responsibility
High-level RL task logic: observation construction, reward computation, episode reset, terminal conditions, scenario management, and drift curriculum.

### Constructor Parameters

| Parameter | Default | Description |
|---|---|---|
| `force_scenario` | `None` | Lock env to `'transit'`, `'descend'`, or `'ascend'` |
| `hide_object` | `True` | Teleport `object0` off-screen (pure arm movement, no cube) |
| `show_chess_pieces` | `False` | Place chess pieces at starting positions (chess game mode) |
| `debug` | `False` | Enable verbose per-step file logging |
| `drift_curriculum_steps` | from `env.yaml` | Total per-worker steps over which drift limit tightens |
| `force_drift_limit` | `None` | Override curriculum; lock drift limit to this value |
| `fixed_drift` | `False` | Skip curriculum; always use `drift_limit_end` |

### Observation Space

During **RL training and inference**, the environment switches to the 25-D transfer observation space defined in `src/chess_env/transfer_obs.py`:

```python
TRANSFER_OBS_SPACE = spaces.Dict({
    "observation":    Box(-inf, inf, shape=(25,), dtype=float64),
    "achieved_goal":  Box(-inf, inf, shape=(3,),  dtype=float64),
    "desired_goal":   Box(-inf, inf, shape=(3,),  dtype=float64),
})
```

During **scripted control and status queries**, a 7-D native observation is used:

```python
{
    "observation":    Box(shape=(7,))   # [grip_pos(3), grip_vel(3), l_finger(1)]
    "achieved_goal":  Box(shape=(3,))   # grip_pos
    "desired_goal":   Box(shape=(3,))   # goal_pos
    "grip_pos":       Box(shape=(3,))
    "grip_vel":       Box(shape=(3,))
    "l_finger":       Box(shape=())
    "goal_pos":       Box(shape=(3,))
    "scenario_id":    Discrete(4)       # 0=None, 1=transit, 2=descend, 3=ascend
}
```

The flag `_use_transfer_obs` switches between the two. The training wrappers set it to `True` permanently; `ModelEmbeddedController` temporarily sets it to `True` during model inference.

### `_build_transfer_observation()` — The "Holding Object" Trick

```python
fake_object_pos = grip_pos.copy()   # Pretend object is at gripper
goal = self.goal
rel_to_goal = goal - grip_pos

obs_vec = np.concatenate([
    grip_pos,            # [0:3]   gripper position
    fake_object_pos,     # [3:6]   fake object = gripper (the trick)
    rel_to_goal,         # [6:9]   vector to goal
    np.zeros(2),         # [9:11]  gripper state (zeroed)
    np.zeros(3),         # [11:14] object rot (zeroed)
    np.zeros(3),         # [14:17] object velp (zeroed)
    np.zeros(3),         # [17:20] object velr (zeroed)
    grip_velp,           # [20:23] gripper velocity
    np.zeros(2),         # [23:25] gripper vel (zeroed)
])
```

This 25-D vector matches the exact layout expected by the pretrained `FetchPickAndPlace-v4` `MultiInputPolicy`. By setting `object_pos = grip_pos`, the policy perceives "the object is at my gripper" and applies its pick-up transport knowledge to pure gripper motion.

### Scenarios

Three movement scenarios are defined, corresponding to distinct arm motion phases:

| Scenario | Entry Z | Exit Z | Constraint | Finger State |
|---|---|---|---|---|
| `transit` | SAFE_Z (0.530 m) | SAFE_Z | Floor (grip Z > 0.400 m) | Closed (0.000) |
| `descend` | SAFE_Z | HOVER_Z (0.460 m) | Tube XY + table surface | Open (0.0181) |
| `ascend` | HOVER_Z | SAFE_Z | Tube XY | Closed or held |

When `force_scenario` is set, episodes always use the same scenario. Otherwise, the scenario is sampled uniformly from the three options.

### Drift Curriculum

The "tube constraint" for descend and ascend starts wide (`drift_limit_start = 0.100 m`) and tightens linearly to `drift_limit_end = 0.008 m` over `drift_curriculum_steps` per-worker steps. This curriculum lets the model first learn the Z-axis movement before being penalised for lateral drift.

```python
progress = min(total_env_steps / DRIFT_CURRICULUM_STEPS, 1.0)
limit = DRIFT_LIMIT_START - (DRIFT_LIMIT_START - DRIFT_LIMIT_END) * progress
```

At inference time, the controller uses `eval_drift_limit = 0.010 m` (10 mm), which is the strict production constraint.

### `step(action)` — RL Training Step

1. Increment step counters.
2. Set `finger_target_joint` based on scenario (open for descend, closed otherwise).
3. Clip action to action space bounds; force gripper dimension to `-1.0`.
4. Call `_set_action(action)` and `_mujoco_step(action)`.
5. Check terminal conditions:
   - `FINGER_FAULT`: finger joint deviates from target by >3 mm
   - `FLOOR_HIT` (transit): grip Z < 0.400 m
   - `TUBE_BREACH` (descend/ascend): XY drift from tube center > current_drift_limit
   - `TABLE_HIT` (descend/ascend): grip Z < TABLE_SURFACE_Z
6. Compute reward (see Reward section below).
7. Detect success: near goal AND stable (velocity < `stability_vel_threshold`).
8. Return `(obs, reward, terminated, False, info)`.

**Note:** The scripted controller and `ModelEmbeddedController` bypass `step()` entirely, calling `_set_action()` and `_mujoco_step()` directly. The reward function is only used during RL training.

### Reward Function

```
reward = -dist_reward_weight * ||grip_pos - goal_pos||     # distance to goal

       - z_reward_weight * |grip_z - goal_z|               # Z axis precision

       - xy_reward_weight * ||grip_xy - goal_xy||          # XY drift penalty

       - braking_weight * speed   (if dist < braking_dist) # velocity near goal

       - jitter_penalty * ||action[:3]||^2                 # action smoothness

       + floor_penalty   (if grip_z < floor_limit + floor_proximity_threshold)

crash:  reward = crash_penalty  (-500)
success: reward += success_bonus  (+500)
```

Per-scenario braking parameters:

| Scenario | `braking_dist` | `braking_weight` |
|---|---|---|
| transit | 0.030 m | 0.50 |
| descend | 0.025 m | 0.30 |
| ascend  | 0.010 m | 0.15 |

The success bonus of +500 is large enough to dominate episode reward when the model reaches the goal stably. The crash penalty of −500 strongly discourages boundary violations.

### `_is_success(achieved_goal, desired_goal)` — Success Criterion

For transit and ascend:
```
d_xy < SUCCESS_THRESHOLD (10 mm)  AND  d_z < SUCCESS_THRESHOLD (10 mm)
```

For descend, a tighter Z criterion is used:
```
d_xy < SUCCESS_THRESHOLD (10 mm)  AND  d_z < descend_success_threshold (8 mm)
```

Success also requires the arm velocity to be below `stability_vel_threshold = 0.02 m/s`.

---

## `TaskRuntimeMixin` (`src/chess_env/task_runtime.py`)

### `_reset_sim()` — Full Episode Reset

The full reset sequence has three explicit phases:

**Phase 1 — Settle arm CLOSED for stability:**
```
force finger_target_joint = FINGER_CLOSED_JOINT
_set_gripper_state()          ← teleport fingers closed
_settle_arm_to_start(arm_start_pos)  ← proportional control to start position
```

**Phase 2 — Scripted gripper transitions:**
- `descend`: fingers closed → open (finger_target_joint = FINGER_OPEN_JOINT)
- `ascend`: fingers closed → open → closed (full open/close cycle)
- `transit`: stay closed

**Phase 3 — Finger validation:**
Final finger joint position is checked: `|actual - target| < 0.5 mm`. If failed, reset returns False.

Also captures home posture when starting from the home position (used for `reset_arm_to_home_posture` after each move).

### `soft_reset()` — Scenario Transition Without Arm Teleport

Used between consecutive scenarios (e.g., transit→descend) during a full pick-and-place sequence. The arm is already near the correct position and must not be teleported.

**Four phases:**
1. **Halt**: Zero robot DOF velocities and accelerations; wait for `HALT_VEL_THRESHOLD` convergence.
2. **Waypoint alignment**: Proportional control to `nominal_exit_pos` (at most 80 steps, 4 mm tolerance).
3. **State update**: Update `current_scenario`, `goal_pos`, `tube_center_xy`.
4. **Gripper transition**: Open fingers before descend; close after descend→ascend/transit.

### `compute_reward()` — Vectorised Reward

Accepts batched arrays for HER (Hindsight Experience Replay) compatibility, even though RoboChess does not use HER. Returns a scalar for single inputs.

### `transition_validate()` — Transition Diagnostics

Returns a dict with grip position, speed, is_velocity_ok, and error from the nominal exit position. Used by diagnostic tooling.

---

## `TaskStateMixin` (`src/chess_env/task_state.py`)

### Active Piece Tracking

For chess game execution, one chess piece is "active" at a time (the piece being moved). The mixin stores:
- `active_piece_id`: e.g., `"white_pawn_e"`
- `active_piece_body_name`: e.g., `"piece_white_pawn_e"`
- `active_piece_joint_name`: e.g., `"piece_white_pawn_e:joint"`

These are set by `set_active_piece(piece_id)` before an arm move and cleared by `clear_active_piece()` after.

### Home Posture Capture and Restore

After the first full episode reset, the exact arm joint positions are captured in `_home_posture_qpos`. After each move's return-to-home transit, `reset_arm_to_home_posture()` interpolates all joints back to this posture over 10 steps, then snaps them exactly. This prevents the arm from gradually drifting into a "twisted" configuration across many moves.

### `_check_cube_held(grip_pos)` — In-Flight Piece Monitoring

During `grasp_mode=True` stages (ascend, transit with piece), the piece position is checked every step:
```
xy_error = ||piece_xy - grip_xy|| > CUBE_HELD_XY_LIMIT (30 mm) → CUBE_DROPPED_XY
z_error  = |piece_z - (grip_z - 0.015)| > CUBE_HELD_Z_LIMIT (20 mm) → CUBE_DROPPED_Z
```

The 15mm Z offset accounts for the piece's centre of mass hanging below the grip site.

### `_reset_chess_piece_bodies()` — Scene Reset

Repositions all 32 active pieces and 64 reserve pieces for a new game. In chess game mode (`show_chess_pieces=True`), pieces are placed at starting squares. In RL training mode, all pieces are moved to a hidden grid far off-screen to avoid interfering with arm movement.

---

## `GraspPlaceMixin` (`src/chess_env/task_execution.py`)

### Low-Level Movement Primitives

**`_move_mocap_to(target_pos, target_quat, max_steps, tolerance)`**
Proportional control loop driving the gripper site to an exact position. Used for:
- Pre-grasp alignment over the cube
- Pre-place alignment over the destination
- Soft-reset waypoint alignment

**`_plunge_to_z(xy, target_z, step_m, should_render)`**
Incremental Z descent in fixed steps. Moves the gripper down from current Z to `target_z` in `step_m`-sized steps, holding XY fixed.

**`_retract_to_hover(xy_provider, start_z, step_m, should_render, verify_held)`**
Incremental Z ascent back to HOVER_Z. The `xy_provider` callable returns the current XY to track (used to follow the piece's live position during lift).

### `execute_grasp()` — 6-Phase Grasp Pipeline

Full documentation in [02_arm_control.md](02_arm_control.md#stage-4-grasp-pipeline).

### `execute_place(dst_xy)` — 6-Phase Place Pipeline

Full documentation in [02_arm_control.md](02_arm_control.md#stage-5-place-pipeline).

---

## Z-Level Constants

| Name | Value | Description |
|---|---|---|
| `TABLE_SURFACE_Z` | 0.400 m | Table top (collision surface) |
| `CUBE_Z` | 0.415 m | Piece centre-of-mass height on table |
| `GRASP_Z` | 0.430 m | Finger plunge depth for grasp/place |
| `HOVER_Z` | 0.460 m | RL stopping altitude above piece |
| `SAFE_Z` | 0.530 m | Transit altitude; arm moves horizontally here |
| `FLOOR_LIMIT` | 0.400 m | Emergency abort: below table surface |

The gap between `HOVER_Z` (460mm) and `GRASP_Z` (430mm) = 30mm. The scripted grasp pipeline descends this 30mm in 6mm steps, giving 5 steps of fine contact.

---

## Physics Configuration (`configs/physics.yaml`)

| Key | Value | Description |
|---|---|---|
| `vertical_quat` | `[0.7071, 0, 0.7071, 0]` | 90° Y-axis rotation → gripper faces down |
| `initial_qpos` | `[-0.05, 0.00]` | Robot base x/y offset |
| `max_settle_steps` | 500 | Max steps for arm settle during reset |
| `settle_tolerance` | 0.003 m | Convergence tolerance during settle |
| `pos_ctrl_scale` | 0.015 | Action scale: max 15 mm displacement per step |
| `env_setup_steps` | 10 | Steps at construction to let arm settle |
| `max_goal_retries` | 100 | Max sampling attempts for non-overlapping goal |
| `settle_gain` | 0.3 | Proportional gain for settle loop |
| `settle_steps_final` | 25 | Final settle physics steps after reset |

---

## MuJoCo XML Assets

### `chess_env/assets/pick_and_place.xml`

The main scene XML is based on the FetchPickAndPlace XML and contains three
static chess sections:

| Section | Content |
|---|---|
| Board squares | 64 coloured visual-only box geoms |
| Chess pieces | 32+64 freejoint body+geom pairs |
| Zone markers | 4 graveyard/reserve visual-only zone geoms |

These sections are edited as normal source and are not regenerated at startup.

### `chess_env/assets/robot.xml`

Defines the Fetch robot URDF: torso, arm joints, gripper finger joints, mocap body, and actuators. The mocap body `robot0:mocap` is the control target for position control.

### STL Meshes

Six checked-in meshes live in `chess_env/stls/chess/`: pawn, rook, knight,
bishop, queen, king. See [07_scene_generation.md](07_scene_generation.md) for
asset ownership details.
