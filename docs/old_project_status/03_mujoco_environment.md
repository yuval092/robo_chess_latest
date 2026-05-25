# MuJoCo, Gymnasium-Robotics, and the Fetch Environment

## Background: MuJoCo

[MuJoCo](https://mujoco.org/) (Multi-Joint dynamics with Contact) is a physics simulator specialised for robotics. It models rigid bodies, joints, contact forces, and actuators using a reduced-coordinate forward dynamics formulation. RoboChess loads MuJoCo via the Python bindings (`import mujoco`) and drives it through the `gymnasium-robotics` framework.

---

## Gymnasium-Robotics and the Fetch Robot

[gymnasium-robotics](https://robotics.farama.org/) provides pre-built RL environments for the OpenAI Fetch robot — a 7-DOF torso-mounted manipulator with a parallel-jaw gripper. The specific base class used is `MujocoFetchPickAndPlaceEnv` from `gymnasium_robotics.envs.fetch.pick_and_place`.

The Fetch arm has:
- 3 linear slides (x, y, z torso) + 1 torso lift
- 7 revolute arm joints (shoulder pan/lift, elbow flex, wrist flex/roll × 2, forearm roll)
- 2 gripper finger joints (l/r)
- A **mocap body** (`robot0:mocap`) — a kinematic target that the arm tracks via impedance control. By setting `mocap_pos` and `mocap_quat`, we command the arm's desired end-effector pose without computing inverse kinematics explicitly.

---

## ChessSimulationEnv (`src/chess_env/simulation.py`)

`ChessSimulationEnv` extends `MujocoFetchPickAndPlaceEnv` with chess-specific geometry and physics constants. Key responsibilities:

### XML Injection
MuJoCo requires a static XML file at construction time. The base class hardcodes `MODEL_XML_PATH`. `ChessSimulationEnv.__init__` temporarily overwrites this module-level variable to point to the chess scene, protected by a `threading.Lock`:

```python
# simulation.py:61
with _xml_path_lock:
    _original_path = _fpp_module.MODEL_XML_PATH
    _fpp_module.MODEL_XML_PATH = asset_path   # chess_env/assets/pick_and_place.xml
    try:
        super().__init__(**kwargs)
    finally:
        _fpp_module.MODEL_XML_PATH = _original_path
```

### Vertical Quaternion
`VERTICAL_QUAT` is loaded from `configs/physics.yaml:vertical_quat` and normalised:
```python
raw_quat = np.array(self.physics_cfg["vertical_quat"])  # [0.7071068, 0, 0.7071068, 0]
self.VERTICAL_QUAT = raw_quat / np.linalg.norm(raw_quat)
```
This enforces the gripper pointing straight down throughout all moves.

### `_set_action(action)`
Converts a 4-element action `[dx, dy, dz, gripper]` into MuJoCo mocap control:
- `pos_ctrl *= POS_CTRL_SCALE` (0.015 m/step) — scales the delta for smooth motion.
- In **teleport mode** (`grasp_mode=False`): finger joints are directly set to `finger_target_joint` and their velocities zeroed. No contact forces on fingers.
- In **grasp mode** (`grasp_mode=True`): only `ctrl[0/1]` is set; MuJoCo's Kp controller drives the fingers with contact physics, allowing the fingers to stall against the piece.

### Board Geometry
The chess board is 64 cm × 64 cm, larger than the standard 50 cm Fetch table. `TABLE_HALF_X = TABLE_HALF_Y = 0.35 m`. The arm's torso height is tuned to `0.3661 m` (from `configs/env.yaml:torso_height`) to ensure reach across all 64 squares.

### `_env_setup`
Called once at first reset. Forces the mocap quaternion to vertical and lets `ENV_SETUP_STEPS` (10) physics steps settle the arm before the first episode.

---

## ChessTaskEnv (`src/chess_env/task.py`)

`ChessTaskEnv` extends `ChessSimulationEnv` with all chess-specific task logic.

### Chess Piece Bodies
The MuJoCo scene contains 96 physics bodies: 32 active pieces (starting positions) + 64 reserve pieces (for promotions). Each piece body has:
- A `free joint` (`type="free"`, `damping=8.0`) — the piece can fall, slide, or be teleported.
- A **cube geom** (30 mm × 30 mm × 30 mm box) — the physics collision body that the fingers interact with.
- A **mesh geom** (STL visual) — the rendered chess piece shape.
- A `site` — used for sensor/debug purposes.

### Active Piece
Before a move, `env.set_active_piece(piece_id)` stores references to the active piece's body, joint, and joint address. All `get_cube_position()`, `get_cube_quat()`, `get_active_piece_position()` calls use these references.

### Finger Control
Two modes:
- **Teleport mode** (`grasp_mode=False`): `_set_action` directly writes `finger_target_joint` to both finger joints' qpos and zeroes qvel. The fingers are glued to their target, unaffected by contact. Used during transit/descend/ascend to prevent the open gripper from accidentally disturbing pieces.
- **Grasp mode** (`grasp_mode=True`): only `ctrl` is set; the finger actuators drive the joints with physical contact forces. Used during the grasp close/hold phases.

### `soft_reset(new_scenario, new_goal_pos, nominal_exit_pos)`
Transitions between scenarios mid-episode:
1. Halts the arm (halt loop up to 30 steps until velocity < `HALT_VEL_THRESHOLD`).
2. Drives the arm to `nominal_exit_pos` with a proportional controller (up to 80 alignment steps).
3. Updates scenario, goal, tube centre, gripper state.

### Debug Callback
`_debug_step_callback` — if set to a callable `f(phase: str)`, it is called from `_mujoco_step` every physics step. Used by `debug_one_move.py` to capture per-step telemetry to a JSONL file.

### Logging
Logger `chess_task_{pid}` uses a `FileHandler` only when `debug=True`; otherwise a `NullHandler` to prevent empty log file accumulation.

---

## Gymnasium Registration

`src/chess_env/__init__.py` registers the environment:
```python
register(
    id='ChessFetchTask-v0',
    entry_point='src.chess_env.task:ChessTaskEnv',
    max_episode_steps=500,
)
```

All scripts and tests use `gymnasium.make("ChessFetchTask-v0", ...)` to construct the environment.

---

## Environment Construction Parameters

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `force_scenario` | str | `None` | Lock env to a specific scenario (`"transit"`, `"descend"`, `"ascend"`) |
| `hide_object` | bool | `True` | Teleport the legacy `object0` cube to `[2, 2, 0.015]` (hidden) |
| `show_chess_pieces` | bool | `False` | Place all 32 pieces at starting positions on reset |
| `debug` | bool | `False` | Enable per-step verbose logging |
| `render_mode` | str | `None` | `"human"` for live visualisation window |

---

## Physics Solver Settings

From `configs/physics.yaml`:
- `max_settle_steps = 500`: Maximum steps for the arm settle loop after `_env_setup`.
- `settle_tolerance = 0.003 m`: Arm is settled when all joint velocities < this.
- `settle_gain = 0.3`: Damping gain used during settling.
- `env_setup_steps = 10`: Steps run in `_env_setup` before first episode.
- `settle_steps_final = 25`: Additional settle steps at the end of reset.
- `pos_ctrl_scale = 0.015`: Metres moved per action unit — limits per-step displacement.

---

## Coordinate Frames

All positions are in the **world frame** of the MuJoCo model. Key reference points:
- Table surface: Z = 0.400 m
- Piece cube centre (resting): Z = 0.400 + 0.015 = 0.415 m
- HOVER_Z: 0.460 m (45 mm above table, 45 mm above piece centre)
- SAFE_Z: 0.530 m (110 mm above table)
- Board centre: X = 0.880 m, Y = 0.2641 m
- Board extent: X ∈ [0.560, 1.200] m, Y ∈ [-0.076, 0.604] m (table edges)
- Board cells: X ∈ [0.600, 1.160] m, Y ∈ [-0.016, 0.544] m (first/last square centres)

## Environment Utilities

`unwrap_env(env)` and `reset_elapsed_steps(env)` live in `src/chess_env/simulation.py`. They centralize wrapper traversal for controllers and physical helpers.

`transfer_obs.py` owns the 25-D transfer-observation space and context manager used when loading SAC specialist models. `ModelEmbeddedController` no longer imports `training/`.

`ModelRegistry` (`src/chess_env/model_registry.py`) owns specialist SAC checkpoint loading and storage by stage.
