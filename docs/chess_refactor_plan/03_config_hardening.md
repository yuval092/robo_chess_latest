# Stage 3 — Configuration Hardening: Eliminate Magic Numbers and Fallback Values

**Objective**: Every numeric constant used in logic must either come from a YAML config file or be a named class-level constant loaded from config. `cfg.get("key", fallback)` calls are banned — replace with `cfg["key"]` so missing keys raise `KeyError` immediately. No silent fallbacks.

---

## 3.1 Rule Definitions

**Tunable runtime parameter → YAML config**: Values that a developer legitimately adjusts to tune behavior.
- `self.env_cfg.get("cube_height", 0.030)` — `0.030` is tunable, must be in YAML
- `self.env_cfg.get("hover_z", 0.460)` — `0.460` is tunable, must be in YAML

**Domain invariant → named module constant**: Fixed facts about the chess domain or physical schema that cannot meaningfully vary.
- Chess board size `8` — a chess rule, not a tunable; keep as `BOARD_SIZE = 8` near where it's used
- Observation vector dimension `25` — a model architecture fact; named constant, not YAML
- Action shape `4`, gripper DOF `2` — hardware/model schema; named constants
- Unit conversion `1000.0` (metres to millimetres) — physical constant; `M_TO_MM = 1000.0` in module

**Fallback value (banned in config access)**: A `dict.get(key, value)` call where `value` is not `None` and not an empty collection. Must be replaced with `cfg["key"]` so misconfigured runs fail loudly.

**Exception**: `dict.get(key)` with no fallback (returns `None`) is acceptable where `None` is a meaningful absence. So are `dict.get(key, {})` and `dict.get(key, [])` where an empty container is a valid default. Scenario-specific multi-level lookups (`cfg.get("transit_braking_dist", cfg.get("braking_dist"))`) are permitted when documented.

---

## 3.2 Config Key Status: What Already Exists vs What Needs Adding

**Important**: Many keys that earlier plan iterations thought were missing are ALREADY present in `configs/env.yaml`. The work for those keys is simpler: just replace `.get("key", fallback)` with `cfg["key"]` (the fallback is dead code).

### Keys already in `configs/env.yaml` — just replace `.get()` with `[]`

The following keys exist and their corresponding `.get()` fallbacks must be dropped:
- `hover_z`, `halt_vel_threshold`, `home_position_xy`
- All grasp phase thresholds: `grasp_contact_approach_tolerance`, `grasp_close_steps`, `grasp_ramp_end`, `empty_grasp_threshold`, `grasp_hold_steps`, `grasp_plunge_step_m`, `grasp_retract_step_m`, `release_ramp_steps`, `release_settle_steps`, `grasp_verify_xy_threshold`, `grasp_verify_z_threshold`, `grasp_verify_finger_threshold`, `cube_held_xy_limit`, `cube_held_z_limit`
- `stability_vel_threshold`, `eval_drift_limit`, `drift_limit_start`, `drift_limit_end`, `drift_curriculum_steps`
- `grasp_align_tolerance`
- `logging.level`, `logging.log_dir` (nested under `logging:` key)

For `model_controller.py` specifically:
- `.get("stability_vel_threshold", 0.02)` → `cfg["stability_vel_threshold"]`
- `.get("eval_drift_limit", 0.010)` → `cfg["eval_drift_limit"]`

### Keys that MUST BE ADDED to `configs/env.yaml` (truly new)

These are currently hardcoded magic numbers in Python source files. Add them:

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

# RL inference step limit
rl_max_steps_per_stage: 300      # Maximum inference steps for one RL stage (transit/descend/ascend)
```

---

## 3.3 Fix `configs/env.yaml` — Complete Key List

After adding the keys above, verify these all exist:
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
- `halt_vel_threshold`, `stability_vel_threshold`, `eval_drift_limit`, `braking_dist`
- `drift_limit_start`, `drift_limit_end`, `drift_curriculum_steps`, `grasp_align_tolerance`
- `floor_proximity_threshold`, `max_gripper_width`
- `finger_open_joint`, `finger_closed_joint`, `finger_outer_offset`
- `transit_tolerance_m`, `vertical_tolerance_m`, `step_gain`
- `min_step_size_m`, `max_step_size_m`, `transit_max_steps`, `vertical_max_steps`
- `grasp_verify_drift_mm`, `reconcile_xy_tolerance_m`, `reconcile_z_tolerance_m`
- `rl_max_steps_per_stage`

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

**Current code**: Many instances of `cfg.get("key", default)`. For keys that already exist in `env.yaml`, the fallback is dead code — replace with direct key access.

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

For `drift_limit_start` and `drift_limit_end` specifically (already in `env.yaml`):
```python
# Before (dead fallback — key already exists)
DRIFT_LIMIT_START = self.env_cfg.get("drift_limit_start", 0.060)
DRIFT_LIMIT_END   = self.env_cfg.get("drift_limit_end",   0.010)

# After
DRIFT_LIMIT_START = self.env_cfg["drift_limit_start"]
DRIFT_LIMIT_END   = self.env_cfg["drift_limit_end"]
```

For the `logging` sub-dict (nested key access):
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

## 3.5a Fix `src/chess_env/task.py` — Legacy Constructor Param Cleanup

Lines 28–36 of `task.py` consume legacy kwargs to suppress gymnasium warnings:
```python
kwargs.pop('drift_curriculum_steps', None)
kwargs.pop('force_drift_limit', None)
kwargs.pop('fixed_drift', None)
```

**Action (in Stage 3, before Stage 6 removes them entirely)**:

Step 1 — Replace the `drift_curriculum_steps` kwarg path with config loading (the key is already in `env.yaml`):
```python
# Before
self.drift_curriculum_steps = drift_curriculum_steps or self.env_cfg.get("drift_curriculum_steps", 200000)

# After
self.drift_curriculum_steps = self.env_cfg["drift_curriculum_steps"]
```

Step 2 — Verify that `force_drift_limit` and `fixed_drift` are not passed by any caller:
```bash
grep -rn "force_drift_limit\|fixed_drift" src/ scripts/ tests/ training/
# If nothing found: these are truly dead params — remove in Stage 6.3.1
# If callers found: fix those callers first
```

Step 3 — If no callers use them, remove the `kwargs.pop` lines. Otherwise, leave them for Stage 6.3.1 to handle after confirming all callers are updated.

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
The `1000.0` multiplier for m→mm conversion should be named:
```python
M_TO_MM = 1000.0  # Unit conversion constant; not a tunable parameter
```

Define this as a module-level constant in `controller.py` (not in config, since it's a physical unit, not a tunable value).

---

## 3.7 Fix `src/chess_env/model_controller.py` — Remove Unnecessary Fallbacks

`model_controller.py` uses `.get()` with fallbacks for keys that already exist in `env.yaml`:

```python
# Before (key exists in env.yaml — fallback is dead code)
self._stability_vel_threshold = self._env.env_cfg.get("stability_vel_threshold", 0.02)
self._eval_drift_limit        = self._env.env_cfg.get("eval_drift_limit", 0.010)

# After
self._stability_vel_threshold = self._env.env_cfg["stability_vel_threshold"]
self._eval_drift_limit        = self._env.env_cfg["eval_drift_limit"]
```

Also remove `MAX_STEPS = 300` (hardcoded constant) and replace with:
```python
self._max_steps = self._env.env_cfg["rl_max_steps_per_stage"]
```

---

## 3.8 Fix `src/chess_env/environment_generation.py` — Name the STL Geometry Constants

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

## 3.9 Fix `scripts/eval_chess_reachability.py` — Load Expected Coordinates from Config

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

Add to `configs/chess.yaml`:
```yaml
reachability_expected:
  rank1_y_m: -0.0159
  rank8_y_m: 0.5441
  file_a_x_m: 0.600
  file_h_x_m: 1.160
  geometry_tolerance_m: 1.0e-9
```

---

## 3.10 Fix `src/chess_game/board_mapper.py` — Load Validation Constants from Config

**Current code**: `REQUIRED_CELL_SIZE_M = 0.08`, `REQUIRED_BOARD_WIDTH_M = 0.64`, etc. are class-level constants.

**Step 1**: Define a `BoardValidation` dataclass to carry validation parameters cleanly:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class BoardValidation:
    required_cell_size_m: float
    required_board_width_m: float
    required_table_margin_m: float
    geometry_tolerance_m: float
```

**Step 2**: Update `BoardMapper.__init__` to accept `BoardValidation` as an optional argument (with a class-level default for backward compatibility). The default keeps existing callers working while `from_configs()` passes the config-loaded values:

```python
_DEFAULT_VALIDATION = BoardValidation(
    required_cell_size_m=0.08,
    required_board_width_m=0.64,
    required_table_margin_m=0.03,
    geometry_tolerance_m=1.0e-9,
)

class BoardMapper:
    def __init__(self, geometry: BoardGeometry, validation: BoardValidation = _DEFAULT_VALIDATION):
        ...
```

**Step 3**: Load from `chess.yaml` — keys are under `board.validation` (see section 3.9):

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
    val_cfg = board_cfg["validation"]
    validation = BoardValidation(
        required_cell_size_m=float(val_cfg["required_cell_size_m"]),
        required_board_width_m=float(val_cfg["required_board_width_m"]),
        required_table_margin_m=float(val_cfg["required_table_margin_m"]),
        geometry_tolerance_m=float(val_cfg["geometry_tolerance_m"]),
    )
    return cls(geometry, validation)
```

Add to `configs/chess.yaml` under the `board:` section:
```yaml
board:
  # ... existing keys ...
  validation:
    required_cell_size_m: 0.08
    required_board_width_m: 0.64
    required_table_margin_m: 0.03
    geometry_tolerance_m: 1.0e-9
```

---

## 3.11 Fix `src/physical/movement_executor.py` — Remove Inline Tolerances

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

## 3.12 Fix `scripts/run_chess_ui.py` — Remove Hardcoded drift_limit

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

## 3.13 Split `configs/training.yaml` — Separate Hyperparameters from Deployed Model Paths

**Problem**: `configs/training.yaml` currently mixes two completely different concerns:
1. **Training hyperparameters** (`total_timesteps`, `n_envs`, `learning_rate`, etc.) — change rarely, only during training experiments
2. **Deployed model paths** (`deployed_models.transit`, `deployed_models.descend`, etc.) — change every time a new model is deployed to production

Production code (`model_controller.py`) reads deployed model paths to know which checkpoints to load. It has no business knowing about training hyperparameters.

**Action**: Split into two files:

`configs/training.yaml` — training hyperparameters only:
```yaml
base_model: "models/pretrained/sac-FetchPickAndPlace-v4.zip"
num_envs: 4
total_timesteps: 600000
learning_rate: 0.0003
batch_size: 256
target_entropy: "auto"
buffer_size: 1000000
learning_starts: 1000
initial_ent_coef: 0.1
ent_coef_lr: 0.0001
eval_freq: 10000
n_eval_episodes: 30
log_freq: 1000
moving_avg_window: 100
drift_curriculum_steps: 200000
```

**⚠ Key name**: Use `num_envs` (not `n_envs`). `training/trainer.py` reads `cfg.get("num_envs", 8)`. Renaming requires updating `trainer.py` in the same commit.

`configs/deployed_models.yaml` — production inference paths only:
```yaml
transit: "checkpoints/transit_20260523_164127/final_transit.zip"
descend: "checkpoints/descend_20260523_222907/final_descend.zip"
ascend:  "checkpoints/ascend_20260523_222849/best_model_ascend.zip"
```

**Update callers**:
- `src/chess_env/model_controller.py` → `load_config("deployed_models")` (not `training`)
- `scripts/train_rl.py` → `load_config("training")` (hyperparameters only)
- `scripts/eval_targeted.py` → `load_config("deployed_models")` after this split (see Stage 7.14)

**Why**: After Stage 10 removes the `training/` import from `model_controller.py`, this stage removes the final logical dependency — the production code no longer reads the training config at all.

---

## Stage 3 — Full Validation Checklist

```bash
# 1. No naked numeric literals in key logic files
grep -n "= 0\.[0-9]\|> 0\.[0-9]\|< 0\.[0-9]" src/chess_env/controller.py src/chess_env/task.py src/physical/movement_executor.py

# 2. No .get() with non-None fallback values in ALL of src/
# The no-silent-fallbacks rule applies to all production src/ code.
# training/ and scripts/ are exempt (they may have legitimate fallbacks).
grep -rn '\.get("[^"]*", [^N{[]' src/
# Expected: zero results from src/ (None, {}, and [] defaults are acceptable; numeric/string defaults are not)

# 3. All new config keys are present
python -c "
from src.utils.io import load_config
cfg = load_config('env')
required = ['transit_tolerance_m', 'vertical_tolerance_m', 'step_gain',
            'min_step_size_m', 'max_step_size_m', 'transit_max_steps',
            'vertical_max_steps', 'grasp_verify_drift_mm',
            'reconcile_xy_tolerance_m', 'reconcile_z_tolerance_m',
            'rl_max_steps_per_stage']
for k in required:
    assert k in cfg, f'Missing key: {k}'
print('All new env.yaml keys present')
"

# 4. Keys that already existed are still there
python -c "
from src.utils.io import load_config
cfg = load_config('env')
existing = ['hover_z', 'halt_vel_threshold', 'home_position_xy',
            'stability_vel_threshold', 'eval_drift_limit',
            'drift_limit_start', 'drift_limit_end', 'drift_curriculum_steps']
for k in existing:
    assert k in cfg, f'Pre-existing key gone: {k}'
print('All pre-existing env.yaml keys still present')
"

python -c "
from src.utils.io import load_config
cfg = load_config('chess')
board_validation = cfg['board']['validation']
required = ['required_cell_size_m', 'required_board_width_m',
            'required_table_margin_m', 'geometry_tolerance_m']
for k in required:
    assert k in board_validation, f'Missing board.validation key: {k}'
assert 'reachability_expected' in cfg
print('All chess.yaml keys present')
"

# 5. deployed_models.yaml exists and is loadable
python -c "
from src.utils.io import load_config
cfg = load_config('deployed_models')
assert 'transit' in cfg and 'descend' in cfg and 'ascend' in cfg
print('deployed_models.yaml OK')
"

# 6. Full test suite
python -m pytest tests/ -v

# 7. Smoke test: controller loads config correctly
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
