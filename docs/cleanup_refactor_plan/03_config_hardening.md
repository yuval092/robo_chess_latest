# Stage 3 — Configuration Hardening: Eliminate Magic Numbers and Fallback Values

**Objective**: Every numeric constant used in logic must either come from a YAML config file or be a named class-level constant loaded from config. `cfg.get("key", fallback)` calls are banned — replace with `cfg["key"]` so missing keys raise `KeyError` immediately. No silent fallbacks.

---

## 3.1 Rule Definitions

**Magic number**: A numeric literal embedded directly in a code expression. Examples:
- `if z > 0.550:` — magic number; should be `self.SAFE_Z`
- `busyPollDelay = 200` — this is JavaScript in the UI; acceptable there.
- `result["close_steps_used"] * 1000.0` — the 1000.0 multiplier is a unit conversion constant; name it.

**Fallback value**: A `dict.get(key, value)` call where `value` is not `None` and not an empty collection. Examples that must be fixed:
- `self.env_cfg.get("cube_height", 0.030)` → `self.env_cfg["cube_height"]`
- `self.env_cfg.get("hover_z", 0.460)` → `self.env_cfg["hover_z"]`

**Exception**: `dict.get(key)` with no fallback (returns `None` if missing) is acceptable where `None` is a meaningful absence. So are `dict.get(key, {})` and `dict.get(key, [])` where an empty container is a valid default.

---

## 3.2 New Config Keys to Add

The following constants are currently magic numbers in Python code. They must be added to the appropriate YAML file.

### Add to `configs/env.yaml`

```yaml
# Controller movement parameters (used by ScriptedController)
transit_tolerance_m: 0.004       # 4mm XY success threshold for transit
vertical_tolerance_m: 0.004      # 4mm Z success threshold for descend/ascend
step_gain: 1.0                   # Proportional gain: full error applied per step
min_step_size_m: 0.002           # 2mm floor to prevent slow final-approach creep
max_step_size_m: 0.024           # 24mm ceiling per step; ensures edge-square reachability
transit_max_steps: 300           # Max steps for transit (longest diagonal + margin)
vertical_max_steps: 200          # Max steps for descend/ascend (90mm / 6mm per step + margin)
grasp_verify_drift_mm: 30.0      # Max XY drift (mm) for post-grasp cube-held check

# Post-place reconciliation (used by MovementExecutor)
reconcile_xy_tolerance_m: 0.020  # 20mm: max XY error before rejecting a place result
reconcile_z_tolerance_m: 0.010   # 10mm: max Z error before rejecting a place result

# Busy poll timing (used by QueuedUIBackend / JS; document here for reference)
# Note: JS values are in app.js — document only; not loaded from YAML
busy_poll_initial_ms: 200        # Initial polling interval when arm is busy
busy_poll_max_ms: 1500           # Maximum polling interval (exponential backoff cap)
```

### Add to `configs/chess.yaml`

```yaml
# Board geometry validation constants (used by BoardMapper._validate_geometry)
required_cell_size_m: 0.08       # Must equal cell_size_m; validated at construction
required_board_width_m: 0.64     # Must equal board_size * cell_size_m
required_table_margin_m: 0.03    # Minimum margin around board on table
geometry_tolerance_m: 1.0e-9     # Floating-point tolerance for geometry checks

# Reachability eval expected coordinates (used by eval_chess_reachability.py)
reachability_expected:
  rank1_y_m: -0.0159
  rank8_y_m: 0.5441
  file_a_x_m: 0.600
  file_h_x_m: 1.160
  geometry_tolerance_m: 1.0e-9
```

### Add to `configs/physics.yaml`

No new keys needed — all physics constants are already present.

---

## 3.3 Fix `configs/env.yaml` — Ensure All Keys Exist

Verify these keys exist (they do based on current file; listing for completeness):
- `cube_height`, `cube_z`, `grasp_z`, `hover_z`, `safe_z`
- `success_threshold`, `drift_limit_end`, `floor_limit`
- `hidden_object_pos`, `table_surface_z`, `table_center_xy`
- `table_half_x`, `table_half_y`, `edge_margin`, `torso_height`
- `min_goal_dist`, `home_position_xy`
- `grasp_contact_approach_tolerance`, `grasp_close_steps`, `grasp_ramp_end`
- `empty_grasp_threshold`, `grasp_hold_steps`, `grasp_plunge_step_m`
- `grasp_retract_step_m`, `release_ramp_steps`, `release_settle_steps`
- `grasp_verify_xy_threshold`, `grasp_verify_z_threshold`, `grasp_verify_finger_threshold`
- `cube_held_xy_limit`, `cube_held_z_limit`
- `halt_vel_threshold`, `stability_vel_threshold`, `braking_dist`
- `floor_proximity_threshold`, `max_gripper_width`
- `finger_open_joint`, `finger_closed_joint`, `finger_outer_offset`
- `transit_tolerance_m`, `vertical_tolerance_m`, `step_gain`
- `min_step_size_m`, `max_step_size_m`, `transit_max_steps`, `vertical_max_steps`
- `grasp_verify_drift_mm`, `reconcile_xy_tolerance_m`, `reconcile_z_tolerance_m`

---

## 3.4 Fix `src/chess_env/simulation.py` — Remove All `.get()` Fallbacks

**Current code (lines 43–69)**: 7 instances of `cfg.get("key", default)`.

Replace every instance:

| Before | After |
|---|---|
| `self.env_cfg.get("cube_height", 0.030)` | `self.env_cfg["cube_height"]` |
| `self.env_cfg.get("max_gripper_width", 0.05)` | `self.env_cfg["max_gripper_width"]` |
| `self.physics_cfg.get("env_setup_steps", 10)` | `self.physics_cfg["env_setup_steps"]` |
| `self.physics_cfg.get("max_goal_retries", 100)` | `self.physics_cfg["max_goal_retries"]` |
| `self.physics_cfg.get("pos_ctrl_scale", 0.015)` | `self.physics_cfg["pos_ctrl_scale"]` |
| `self.physics_cfg.get("initial_qpos", [-0.05, 0.00])` | `self.physics_cfg["initial_qpos"]` |
| `self.env_cfg.get("torso_height", 0.25)` (line ~183) | `self.env_cfg["torso_height"]` |

---

## 3.5 Fix `src/chess_env/task.py` — Remove All `.get()` Fallbacks

**Current code (lines 83–142)**: 21 instances of `cfg.get("key", default)`.

Replace every instance:

| Before | After |
|---|---|
| `self.env_cfg.get("hover_z", 0.460)` | `self.env_cfg["hover_z"]` |
| `self.env_cfg.get("halt_vel_threshold", 0.0005)` | `self.env_cfg["halt_vel_threshold"]` |
| `self.env_cfg.get("home_position_xy", [...])` | `self.env_cfg["home_position_xy"]` |
| `self.env_cfg.get("grasp_contact_approach_tolerance", 0.001)` | `self.env_cfg["grasp_contact_approach_tolerance"]` |
| `self.env_cfg.get("grasp_close_steps", 150)` | `self.env_cfg["grasp_close_steps"]` |
| `self.env_cfg.get("grasp_ramp_end", 0.010)` | `self.env_cfg["grasp_ramp_end"]` |
| `self.env_cfg.get("empty_grasp_threshold", ...)` | `self.env_cfg["empty_grasp_threshold"]` |
| `self.env_cfg.get("grasp_hold_steps", 50)` | `self.env_cfg["grasp_hold_steps"]` |
| `self.env_cfg.get("grasp_plunge_step_m", 0.002)` | `self.env_cfg["grasp_plunge_step_m"]` |
| `self.env_cfg.get("grasp_retract_step_m", 0.002)` | `self.env_cfg["grasp_retract_step_m"]` |
| `self.env_cfg.get("release_ramp_steps", 12)` | `self.env_cfg["release_ramp_steps"]` |
| `self.env_cfg.get("release_settle_steps", 8)` | `self.env_cfg["release_settle_steps"]` |
| `self.env_cfg.get("grasp_verify_xy_threshold", 0.015)` | `self.env_cfg["grasp_verify_xy_threshold"]` |
| `self.env_cfg.get("grasp_verify_z_threshold", 0.020)` | `self.env_cfg["grasp_verify_z_threshold"]` |
| `self.env_cfg.get("cube_held_xy_limit", 0.030)` | `self.env_cfg["cube_held_xy_limit"]` |
| `self.env_cfg.get("cube_held_z_limit", 0.020)` | `self.env_cfg["cube_held_z_limit"]` |

For the `logging` sub-dict (lines 140–142):
```python
# Before
log_cfg = self.env_cfg.get("logging", {})
log_level = logging.DEBUG if debug else getattr(logging, log_cfg.get("level", "INFO"))
log_dir = log_cfg.get("log_dir", "logs/env_debug")

# After
log_cfg = self.env_cfg["logging"]
log_level = logging.DEBUG if debug else getattr(logging, log_cfg["level"])
log_dir = log_cfg["log_dir"]
```

---

## 3.6 Fix `src/chess_env/controller.py` — Replace Magic Numbers with Config-Loaded Constants

**Current class-level constants** (lines 37–44):
```python
TRANSIT_TOLERANCE_M   = 0.004
VERTICAL_TOLERANCE_M  = 0.004
STEP_GAIN             = 1.0
MIN_STEP_SIZE_M       = 0.002
MAX_STEP_SIZE_M       = 0.024
TRANSIT_MAX_STEPS     = 300
VERTICAL_MAX_STEPS    = 200
FLOOR_LIMIT           = 0.400
GRASP_VERIFY_DRIFT_MM = 30.0
```

These are already named constants — that is correct. However, they are hardcoded instead of loaded from config. Replace with config loading in `__init__`:

```python
def __init__(self, env, drift_limit: float = 0.010, render_fn=None, render_delay: float = 0.0):
    """..."""
    # ... (env unwrapping) ...
    cfg = load_config("env")
    self.TRANSIT_TOLERANCE_M   = cfg["transit_tolerance_m"]
    self.VERTICAL_TOLERANCE_M  = cfg["vertical_tolerance_m"]
    self.STEP_GAIN             = cfg["step_gain"]
    self.MIN_STEP_SIZE_M       = cfg["min_step_size_m"]
    self.MAX_STEP_SIZE_M       = cfg["max_step_size_m"]
    self.TRANSIT_MAX_STEPS     = cfg["transit_max_steps"]
    self.VERTICAL_MAX_STEPS    = cfg["vertical_max_steps"]
    self.FLOOR_LIMIT           = cfg["floor_limit"]
    self.GRASP_VERIFY_DRIFT_MM = cfg["grasp_verify_drift_mm"]
    self._drift_limit          = drift_limit
    self._render_fn            = render_fn
    self._render_delay         = render_delay
```

Remove the class-level attribute declarations entirely.

**Remaining magic numbers in controller.py** (format strings using `1000.0` for mm conversion):
- Lines 172, 197: `f"... {error:.1f}mm"` where error is already in mm from `* 1000.0`

These are format strings, not logic constants. The `1000.0` multiplier for m→mm conversion should be named:
```python
M_TO_MM = 1000.0  # Unit conversion constant; not a tunable parameter
```

Define this as a module-level constant in `controller.py` (not in config, since it's a physical unit, not a tunable value).

---

## 3.7 Fix `src/chess_env/environment_generation.py` — Name the STL Geometry Constants

The piece geometry functions (`pawn_triangles`, `rook_triangles`, etc.) contain many floating-point literals like `0.0115`, `0.0085`, `0.007`, etc. These are piece visual dimensions in metres.

These are **not** physics parameters — they are fixed visual design values derived from the STL spec. They should be named module-level constants grouped by piece type:

```python
# --- Pawn geometry (all dimensions in metres) ---
_PAWN_BASE_R       = 0.0115   # bottom frustum base radius
_PAWN_WAIST_R      = 0.0085   # frustum top / waist radius
_PAWN_NECK_R       = 0.0070   # neck radius below head
_PAWN_HEAD_R       = 0.0100   # sphere head radius
_PAWN_BASE_H       = 0.0080   # frustum base height
_PAWN_WAIST_H      = 0.0060   # waist box height
_PAWN_NECK_H       = 0.0030   # neck height
_PAWN_HEAD_CENTRE  = 0.0155   # Z centre of head sphere

# --- Rook geometry ---
_ROOK_BASE_R       = 0.0120
# ... (continue for all pieces)
```

The functions themselves then reference the named constants, not literals.

**Note**: The geometry constants in `environment_generation.py` at module level (lines 25–28) are already named (`BASE_R`, `WAIST_R`, `BASE_TOP`, `SEGS`) but use abbreviated names. Rename to be self-documenting:

```python
# Before
BASE_R   = 0.022
WAIST_R  = 0.010
BASE_TOP = 0.032
SEGS     = 32

# After (module-level STL geometry constants)
_STL_DEFAULT_SEGS     = 32    # Triangulation segments for circular cross-sections
_STL_BASE_RADIUS      = 0.022  # Default piece base radius in metres
_STL_WAIST_RADIUS     = 0.010  # Default piece waist radius in metres
_STL_BASE_TOP_HEIGHT  = 0.032  # Default height of base frustum top in metres
```

---

## 3.8 Fix `scripts/eval_chess_reachability.py` — Load Expected Coordinates from Config

**Current code** (lines 14–16):
```python
REQUIRED_RANK1_Y = -0.0159
REQUIRED_RANK8_Y = 0.5441
REQUIRED_FILE_A_X = 0.600
REQUIRED_FILE_H_X = 1.160
GEOMETRY_TOLERANCE_M = 1e-9
```

**After** — load from `chess.yaml`:
```python
def _load_expected_geometry() -> dict:
    """Load expected board geometry from config for reachability validation."""
    return load_config("chess")["reachability_expected"]
```

And in `validate_board_geometry()`:
```python
def validate_board_geometry(mapper: BoardMapper) -> None:
    """Assert that computed board geometry matches expected config values."""
    expected = _load_expected_geometry()
    tol = expected["geometry_tolerance_m"]
    checks = [
        ("rank1_y", mapper.square_name_to_xy("a1")[1], expected["rank1_y_m"]),
        ("rank8_y", mapper.square_name_to_xy("a8")[1], expected["rank8_y_m"]),
        ("file_a_x", mapper.square_name_to_xy("a1")[0], expected["file_a_x_m"]),
        ("file_h_x", mapper.square_name_to_xy("h1")[0], expected["file_h_x_m"]),
    ]
    for name, actual, required in checks:
        if not math.isclose(actual, required, abs_tol=tol):
            raise SystemExit(f"{name}: expected {required:.4f}m, got {actual:.6f}m")
```

---

## 3.9 Fix `src/chess_game/board_mapper.py` — Load Validation Constants from Config

**Current code**: `REQUIRED_CELL_SIZE_M = 0.08`, `REQUIRED_BOARD_WIDTH_M = 0.64`, etc. are class-level constants.

**After** — load from `chess.yaml` in `from_configs()`:
```python
@classmethod
def from_configs(cls) -> "BoardMapper":
    """Construct BoardMapper from the chess.yaml and env.yaml config files."""
    chess_cfg = load_config("chess")
    env_cfg = load_config("env")
    board_cfg = chess_cfg["board"]
    pieces_cfg = chess_cfg["pieces"]
    geometry = BoardGeometry(
        center_xy=tuple(board_cfg["center_xy"]),
        cell_size_m=float(board_cfg["cell_size_m"]),
        board_size=int(board_cfg["board_size"]),
        table_surface_z=float(env_cfg["table_surface_z"]),
        cube_height_m=float(pieces_cfg["cube_height_m"]),
        table_half_x=float(env_cfg["table_half_x"]),
        table_half_y=float(env_cfg["table_half_y"]),
    )
    # Load validation thresholds from config
    required_cell = float(board_cfg["required_cell_size_m"])
    required_width = float(board_cfg["required_board_width_m"])
    required_margin = float(board_cfg["required_table_margin_m"])
    tol = float(board_cfg["geometry_tolerance_m"])
    return cls(geometry, required_cell, required_width, required_margin, tol)
```

Update `BoardMapper.__init__` to accept these validation parameters rather than having them as class-level constants.

---

## 3.10 Fix `src/physical/movement_executor.py` — Remove Inline Tolerances

**Current code** (lines 68–71):
```python
if xy_error > 0.020:
    return PhysicalMoveResult(False, ..., f"XY_RECONCILE_FAILED {xy_error*1000:.1f}mm")
if z_error > 0.010:
    return PhysicalMoveResult(False, ..., f"Z_RECONCILE_FAILED {z_error*1000:.1f}mm")
```

**After** — load from config in `__init__`:
```python
cfg = load_config("env")
self._reconcile_xy_tol = cfg["reconcile_xy_tolerance_m"]
self._reconcile_z_tol = cfg["reconcile_z_tolerance_m"]
```

And in the check:
```python
M_TO_MM = 1000.0
if xy_error > self._reconcile_xy_tol:
    return PhysicalMoveResult(False, ..., f"XY_RECONCILE_FAILED {xy_error * M_TO_MM:.1f}mm")
if z_error > self._reconcile_z_tol:
    return PhysicalMoveResult(False, ..., f"Z_RECONCILE_FAILED {z_error * M_TO_MM:.1f}mm")
```

---

## 3.11 Fix `scripts/run_chess_ui.py` — Remove Hardcoded drift_limit

**Current code** (line ~97):
```python
controller = ScriptedController(env, drift_limit=0.010, ...)
```

The `0.010` is `drift_limit_end` from `env.yaml`. Replace:
```python
cfg = load_config("env")
controller = ScriptedController(env, drift_limit=cfg["drift_limit_end"], ...)
```

---

## Stage 3 — Full Validation Checklist

```bash
# 1. No naked numeric literals in key logic files
grep -n "= 0\.[0-9]\|> 0\.[0-9]\|< 0\.[0-9]" src/chess_env/controller.py src/chess_env/task.py src/physical/movement_executor.py

# 2. No .get() with non-None fallback values
grep -n "\.get(\"" src/chess_env/simulation.py src/chess_env/task.py

# 3. All new config keys are present
python -c "
from src.utils.io import load_config
cfg = load_config('env')
required = ['transit_tolerance_m', 'vertical_tolerance_m', 'step_gain',
            'min_step_size_m', 'max_step_size_m', 'transit_max_steps',
            'vertical_max_steps', 'grasp_verify_drift_mm',
            'reconcile_xy_tolerance_m', 'reconcile_z_tolerance_m']
for k in required:
    assert k in cfg, f'Missing key: {k}'
print('All env.yaml keys present')
"

python -c "
from src.utils.io import load_config
cfg = load_config('chess')
board = cfg['board']
required = ['required_cell_size_m', 'required_board_width_m',
            'required_table_margin_m', 'geometry_tolerance_m']
for k in required:
    assert k in board, f'Missing board key: {k}'
assert 'reachability_expected' in cfg
print('All chess.yaml keys present')
"

# 4. Full test suite
python -m pytest tests/ -v

# 5. Smoke test: controller loads config correctly
python -c "
import gymnasium as gym
import src.chess_env
env = gym.make('ChessFetchTask-v0', render_mode=None, force_scenario='transit')
env.reset()
from src.chess_env.controller import ScriptedController
ctrl = ScriptedController(env)
print('TRANSIT_MAX_STEPS:', ctrl.TRANSIT_MAX_STEPS)
print('MIN_STEP_SIZE_M:', ctrl.MIN_STEP_SIZE_M)
env.close()
print('OK')
"
```
