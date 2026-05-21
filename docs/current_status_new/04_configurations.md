# Configurations

All configuration files are in `configs/`. They are loaded at runtime via `src/utils/config.py`'s `load_config("env")` and `load_config("physics")` helpers, which return Python dicts.

---

## configs/env.yaml

### Geometry Constants

```yaml
table_surface_z: 0.400    # Top of table surface (meters)
table_center_xy: [0.88, 0.2641]   # World XY of table center
table_half_x: 0.35        # Half-extent in X (70cm table)
table_half_y: 0.35        # Half-extent in Y (70cm table)
edge_margin: 0.04         # 4cm safety margin from edges for sampling

grasp_z: 0.430            # Target Z for grasp (cube top; 30mm above table)
hover_z: 0.460            # Pre-grasp stop height (30mm above cube top)
safe_z: 0.550             # Transit height (15cm above table)
cube_height: 0.030        # Full cube height (meters)
cube_z: 0.415             # Cube center Z when resting on table

torso_height: 0.3661      # Torso lift joint target height (meters)
                           # Joint range: [0.0386, 0.3861]
                           # 0.3661m (94.8% of max) required to clear near-row dead zone;
                           # combined with arm_x=0.56, gives max 2.9mm error across 64 squares
```

### Arm Home Position

```yaml
home_position_xy: [0.680, 0.2641]
```

The arm's starting XY for each episode when `force_start_pos = HOME_POS` is set. The home position is at x=0.680, 11cm inside the near edge of the table (x=0.57), centered on the table's Y axis.

**Why not x=0.56 (arm base)?** x=0.56 is outside the usable board region (near edge = 0.57). HOME_POS is at x=0.680, which is reachable and well within the transit training range [0.640, 1.120].

### Simulation Constants

```yaml
hidden_object_pos: [2.0, 2.0, 0.015]   # Cube position when hidden (off-board)
floor_limit: 0.400                       # Minimum Z during transit (= TABLE_SURFACE_Z)
min_goal_dist: 0.10                      # Minimum start-to-goal distance (10cm)
```

### Grasp Phase Thresholds

```yaml
grasp_contact_approach_tolerance: 0.001  # 1mm final descent tolerance in execute_grasp
grasp_close_steps: 150                   # Physics steps for actuator-driven close
grasp_hold_steps: 50                     # Hold steps after close to let physics settle
grasp_verify_xy_threshold: 0.015         # Max cube-to-grip XY error to confirm hold (15mm)
grasp_verify_z_threshold: 0.020          # Max cube-to-grip Z error to confirm hold (20mm)
grasp_verify_finger_threshold: 0.016     # If both fingers stall at ≥16mm, grasp failed
                                          # Normal stall for 30mm cube: ~14mm
```

### Cube Hold Monitoring

Active when `grasp_mode = True` (cube is being carried):

```yaml
cube_held_xy_limit: 0.030   # 30mm: if cube drifts >30mm from grip XY, it's dropped
cube_held_z_limit: 0.020    # 20mm: if cube Z is not ~15mm below grip, it's dropped
```

The `_check_cube_held` function computes:
- `xy_error = ||cube_xy - grip_xy||`
- `z_error = |cube_z - (grip_z - 0.015)|`  — cube CoM is ~15mm below grip when held

### Drift Limits

```yaml
drift_limit_start: 0.100    # Starting drift limit (10cm) during RL training ramp
drift_limit_end: 0.010      # Final drift limit (10mm) — enforced in scripted controller
eval_drift_limit: 0.010     # Drift limit used during evaluation scripts
```

`ScriptedController` always uses `drift_limit=0.010` (10mm). The ramp variables are RL training artifacts retained for compatibility.

### Reward Weights (Legacy)

These fields are retained for compatibility but are not used in the scripted-only pipeline:
```yaml
success_threshold: 0.010
success_bonus: 500.0
crash_penalty: -500.0
z_reward_weight: 1.5
xy_reward_weight: 2.0
dist_reward_weight: 1.0
```

### Gripper Constants

```yaml
finger_open_joint: 0.0181    # 18.1mm — open position for descent/approach
finger_closed_joint: 0.0000  # 0mm — fully closed (transit/ascend non-grasp)
finger_outer_offset: 0.033   # Finger geometry offset (used for collision checking)
```

### Transition Thresholds (used by soft_reset)

```yaml
halt_vel_threshold: 0.0005   # 0.5mm/s — arm considered stopped for transition
```

---

## configs/physics.yaml

```yaml
settle_tolerance: 0.003    # 3mm — arm settlement threshold during _settle_arm_to_start
```

The `VERTICAL_QUAT` is defined here (as a list loaded into `ChessSimulationEnv`):
```yaml
vertical_quat: [0.0, 1.0, 0.0, 0.0]   # 180° rotation around X → gripper points down
```

---

## How Configs Are Loaded

```python
# src/utils/config.py
def load_config(name: str) -> dict:
    path = Path(__file__).parents[2] / "configs" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
```

In `ChessTaskEnv.__init__`:
```python
self.env_cfg = load_config("env")
self.physics_cfg = load_config("physics")
```

Constants used frequently are cached on the instance at init time:
```python
self.SAFE_Z = self.env_cfg["safe_z"]          # 0.550
self.HOVER_Z = self.env_cfg["hover_z"]        # 0.460
self.GRASP_Z = self.env_cfg["grasp_z"]        # 0.430
self.TABLE_SURFACE_Z = self.env_cfg["table_surface_z"]  # 0.400
self.TABLE_Z = self.env_cfg["table_surface_z"]
self.CUBE_HEIGHT = self.env_cfg["cube_height"]
self.HOME_POS = np.array([*self.env_cfg["home_position_xy"], self.SAFE_Z])
self.VERTICAL_QUAT = np.array(self.physics_cfg["vertical_quat"])
self.FINGER_OPEN_JOINT = self.env_cfg["finger_open_joint"]      # 0.0181
self.FINGER_CLOSED_JOINT = self.env_cfg["finger_closed_joint"]  # 0.0
self.SETTLE_TOLERANCE = self.physics_cfg["settle_tolerance"]    # 0.003
self.GRASP_VERIFY_FINGER_THRESHOLD = self.env_cfg["grasp_verify_finger_threshold"]  # 0.016
```

Constants also exported from `src/chess_env/waypoints.py` for external use:
```python
from src.chess_env.waypoints import SAFE_Z, HOVER_Z, GRASP_Z
```

---

## Changing Table Size

To expand or contract the table:

1. `chess_env/assets/pick_and_place.xml`: Change `table0_surface` geom `size` and all four leg `pos` attributes
2. `configs/env.yaml`: Change `table_half_x` and `table_half_y`
3. `scripts/verify_physics.py`: Update the `test_table_geometry` assertion to match new values

The leg positions should be `(half_extent - 0.04)` from center in each axis, keeping legs 4cm inside the surface edges.

## Changing Torso Height

Only `configs/env.yaml`:
```yaml
torso_height: 0.25   # Change this value
```

Both `simulation.py._env_setup` and `task.py._reset_sim` read `env_cfg.get("torso_height", 0.25)`. The value must be within `[0.0386, 0.3861]` (joint range from robot.xml). Setting it outside this range causes MuJoCo's joint limit constraint to push the torso back, creating an invalid initial state.
