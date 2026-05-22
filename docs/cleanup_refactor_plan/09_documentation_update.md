# Stage 9 — Documentation Update

**Objective**: After all refactor stages are complete, update `docs/current_status/` to accurately reflect the new project structure, file layout, and design decisions introduced by the cleaning refactor.

---

## 9.1 What Changed During the Refactor

The following structural changes must be documented:

| Change | Introduced in | Affects documents |
|---|---|---|
| `archive/` deleted | Stage 1 | `00_overview.md` |
| 5 unused XML assets deleted | Stage 1 | `04_scene_generation.md` |
| `visualize_chess_setup.py` merged into `visualize.py` | Stage 1 | `10_evaluation.md` |
| 4 generate_*.py merged into `generate_scene.py` | Stage 1 | `10_evaluation.md` |
| `models.py` merged into `chess_service.py` | Stage 1 | `05_chess_logic.md` |
| `config.py` + `logger.py` merged into `io.py` | Stage 1 | `00_overview.md`, `08_configuration.md` |
| `pyproject.toml` added | Stage 2 | `00_overview.md` |
| All `sys.path` hacks removed | Stage 2 | `10_evaluation.md` |
| `scripts/__init__.py` added | Stage 2 | `10_evaluation.md` |
| All `.get()` fallbacks removed | Stage 3 | `08_configuration.md` |
| New config keys added to `env.yaml` and `chess.yaml` | Stage 3 | `08_configuration.md` |
| Docstrings added everywhere | Stage 4 | All files (no doc change needed) |
| PEP8 pass complete | Stage 5 | No structural change |
| `task_execution.py` extracted from `task.py` | Stage 6 | `03_mujoco_environment.md`, `01_arm_control.md` |
| `NoOpPhysicalExecutor` added to `plan_executor.py` | Stage 6 | `06_physical_layer.md` |
| `GameOrchestrator.create_headless()` added | Stage 6 | `05_chess_logic.md` |
| `QueuedUIBackend` moved to `src/ui/queued_backend.py` | Stage 6 | `07_ui.md` |
| `unwrap_env()` utility added | Stage 6 | `03_mujoco_environment.md` |
| `eval_draw_conditions.py` refactored with argparse | Stage 7 | `10_evaluation.md` |
| `eval_special_moves.py` refactored with argparse | Stage 7 | `10_evaluation.md` |
| `eval_game_logic.py` added (new script) | Stage 7 | `10_evaluation.md` |
| `test_grasp_physics.py` renamed to `eval_grasp_physics.py` | Stage 7 | `10_evaluation.md` |
| `regenerate_environment()` called on `import src.chess_env` | Stage 8 | `04_scene_generation.md` |
| All STL geometry constants named | Stage 8 | `04_scene_generation.md` |

---

## 9.2 Document-by-Document Update Instructions

### `docs/current_status/00_overview.md`

**Update the file map section**:
- Remove `archive/` from the directory tree.
- Add `pyproject.toml` to the project root listing.
- Update `src/utils/` section: replace `config.py` + `logger.py` with `io.py`.
- Update `scripts/` section to reflect merged and renamed scripts.

**Update the "Architecture Diagram"**:
- No structural change to the architecture itself.

**Add a new section: "How to Run"**:
```markdown
## Setup

Install the project as an editable package (required once):
```bash
pip install -e .
```

This allows all `src.*` imports to work from any directory without path hacks.

## Regenerating Scene Assets

Assets are auto-generated on every `import src.chess_env`. To regenerate manually:
```bash
python scripts/generate_scene.py all
```
```

---

### `docs/current_status/04_scene_generation.md`

**Update Section 1 (Overview)**:
- Note that `regenerate_environment()` is called automatically on `import src.chess_env` (via `__init__.py`).
- Remove the manual run instructions as the primary path; keep as "to regenerate manually".

**Update Section 2 (STL Generation)**:
- List the named geometry constants by piece type (referencing the `_PAWN_*`, `_ROOK_*` etc. constants from Stage 8).
- Remove any mention of raw float literals.

**Update the XML asset list**:
- Remove `push.xml`, `reach.xml`, `slide.xml`, `robot.xml`, `shared.xml` from the file listing. They no longer exist.

---

### `docs/current_status/05_chess_logic.md`

**Update Section 2 (ChessService)**:
- Remove references to `src/chess_game/models.py` as a separate file.
- Note that `GameStatus` is defined in `chess_service.py`.

**Add a new subsection to Section 7 (GameOrchestrator)**:
```markdown
### create_headless() Factory

For testing and logic-only evaluation (no MuJoCo required):

```python
orchestrator = GameOrchestrator.create_headless(
    human_color="white",
    auto_computer_reply=False,
)
```

This creates an orchestrator backed by `NoOpPhysicalExecutor`, which accepts all
plans without running the simulation. Used by eval scripts and tests.
```

---

### `docs/current_status/06_physical_layer.md`

**Add a new section: NoOpPhysicalExecutor**:
```markdown
## 8. NoOpPhysicalExecutor

`NoOpPhysicalExecutor` (in `src/physical/plan_executor.py`) is a testing stub that
accepts all plans and reports success without calling any simulation code.

```python
from src.physical.plan_executor import NoOpPhysicalExecutor
executor = NoOpPhysicalExecutor()
result = executor.execute(plan)   # always succeeds
```

Use it anywhere you need to test chess logic without MuJoCo.
```

**Update Section 2 (PhysicalPlanExecutor)**:
- Document the handler registry pattern (from Stage 6 SOLID fix).
- Note that new command types can be added by registering a handler in `_handlers`.

---

### `docs/current_status/07_ui.md`

**Update Section 6 (Server Entry Point)**:
- Note that `QueuedUIBackend` is now in `src/ui/queued_backend.py`, not defined inline in `run_chess_ui.py`.
- Update the import path.

---

### `docs/current_status/08_configuration.md`

**Update Section 1 (env.yaml)**:
Add the new keys introduced in Stage 3:

```markdown
### Controller Movement Parameters (new in cleaning refactor)

| Key | Value | Description |
|---|---|---|
| `transit_tolerance_m` | `0.004` | 4mm XY success threshold for transit stage |
| `vertical_tolerance_m` | `0.004` | 4mm Z success threshold for descend/ascend |
| `step_gain` | `1.0` | Proportional gain: full error applied per step |
| `min_step_size_m` | `0.002` | 2mm floor to prevent slow approach creep |
| `max_step_size_m` | `0.024` | 24mm ceiling per step |
| `transit_max_steps` | `300` | Max steps for transit |
| `vertical_max_steps` | `200` | Max steps for descend/ascend |
| `grasp_verify_drift_mm` | `30.0` | Max XY drift for post-grasp cube-held check |
| `reconcile_xy_tolerance_m` | `0.020` | Max XY error before rejecting a place result |
| `reconcile_z_tolerance_m` | `0.010` | Max Z error before rejecting a place result |
```

**Update Section 2 (chess.yaml)**:
Add the new keys:
```markdown
| `board.required_cell_size_m` | `0.08` | Validation: cell size must equal this |
| `board.required_board_width_m` | `0.64` | Validation: board width must equal this |
| `board.required_table_margin_m` | `0.03` | Validation: minimum board-to-edge margin |
| `board.geometry_tolerance_m` | `1e-9` | Floating-point tolerance for geometry checks |
| `reachability_expected.*` | see below | Expected board geometry for reachability eval |
```

**Add a section: "Config Access Policy"**:
```markdown
## 5. Config Access Policy

All config values are accessed via `cfg["key"]` (not `cfg.get("key", default)`).
If a required key is missing from the YAML file, the code raises `KeyError` immediately.
There are no silent fallbacks.

This means:
- Adding a new required config key requires updating the YAML file in the same commit.
- Deprecated keys should be removed from both the YAML file and the code together.
- The config files are the single source of truth for all numeric parameters.
```

---

### `docs/current_status/09_testing.md`

**Update Section 1 (Test Structure)**:
- Remove `tests/conftest.py` path hack note — it's now empty (or a pure docstring).

**Update Section 6 (Test Gaps)**:
- Add: `NoOpPhysicalExecutor` is now the recommended mock — remove any mention of per-script mock classes.
- Add: `eval_game_logic.py` covers the GameOrchestrator logical flow without MuJoCo.

---

### `docs/current_status/10_evaluation.md`

**Update Section 1 (Overview table)**:
- Add `eval_game_logic.py` row.
- Add `eval_grasp_physics.py` row (renamed from `test_grasp_physics.py`).
- Remove the four separate `generate_*.py` script rows; replace with `generate_scene.py`.
- Update `visualize.py` to note the `--setup` flag.

**Update Section 3 (eval_stress.py), Section 7 (eval_special_moves.py), and Section 8 (eval_draw_conditions.py)**:
- Note that these scripts now have argparse and proper `main()` guards.
- Update usage examples to reflect the new `--verbose` flag.

**Add Section 11.5: `eval_game_logic.py`**:
```markdown
### `eval_game_logic.py` (new)

**Purpose**: Logical-only evaluation of `GameOrchestrator` through complete game sequences. No MuJoCo required.

**Usage**:
```bash
python scripts/eval_game_logic.py --verbose
```

**Sequences tested**:
- Scholar's Mate (7 moves, checkmate)
- Castling sequence (kingside castle for white)
- En passant sequence
- Pawn promotion sequence (promotion to queen)

All moves go through the full `submit_human_move` → `MovePlanner` → `NoOpPhysicalExecutor` pipeline.
```

---

### `docs/current_status/03_mujoco_environment.md`

**Add a section: Environment Utilities**:
```markdown
## 6. Environment Utilities

### unwrap_env(env)

Located in `src/utils/env_utils.py` (or `src/chess_env/simulation.py`).
Unwraps nested gymnasium wrappers to reach the innermost `ChessTaskEnv`.
Used by `PieceTeleporter`, `MovementExecutor`, `ScriptedController`, and `PhysicalPlanExecutor`.

```python
from src.utils.env_utils import unwrap_env
inner = unwrap_env(wrapped_env)
```
```

### `docs/current_status/01_arm_control.md`

**Add a note on `GraspPlaceMixin`**:
```markdown
## Note on Code Organisation

The grasp and place execution code was extracted from `ChessTaskEnv` into a separate
`GraspPlaceMixin` class in `src/chess_env/task_execution.py` during the cleaning refactor.
`ChessTaskEnv` now inherits from both `ChessSimulationEnv` and `GraspPlaceMixin`.
The algorithm is unchanged; only the file location differs.
```

---

## 9.3 Validation

```bash
# 1. All docs render without broken links or references to deleted files
grep -r "archive/\|generate_board_xml\|generate_pieces_xml\|generate_zones_xml\|generate_chess_stls\|visualize_chess_setup\|models\.py\|config\.py\|logger\.py" docs/current_status/

# 2. All new entities are documented
grep -l "NoOpPhysicalExecutor" docs/current_status/
grep -l "create_headless" docs/current_status/
grep -l "QueuedUIBackend" docs/current_status/
grep -l "eval_game_logic" docs/current_status/

# 3. Configuration doc lists all new keys
grep "transit_tolerance_m\|reconcile_xy_tolerance_m\|required_cell_size_m" docs/current_status/08_configuration.md
```

---

## Final Refactor Completion Checklist

After all 9 stages are complete:

```bash
# Full test suite — must show 0 failures
python -m pytest tests/ -v

# Ruff — must show 0 errors
ruff check src/ scripts/ tests/

# No sys.path hacks
grep -rn "sys.path" src/ scripts/ tests/

# No fallback .get() calls
grep -rn '\.get("' src/chess_env/simulation.py src/chess_env/task.py

# No magic numbers in logic files
grep -n "= 0\.[0-9][0-9][0-9]\b\|> 0\.[0-9][0-9][0-9]\b" src/chess_env/controller.py src/physical/movement_executor.py

# All scripts have main guards
for f in scripts/eval_*.py scripts/run_chess_ui.py scripts/visualize.py; do
    python -c "import ast; t=ast.parse(open('$f').read()); print('$f:', 'OK' if any(isinstance(n,ast.If) and getattr(getattr(n.test,'left',None),'id','')=='__name__' for n in ast.walk(t)) else 'MISSING')"
done

# Full smoke test
python -c "
import src.chess_env
from src.chess_game.game_orchestrator import GameOrchestrator
o = GameOrchestrator.create_headless()
r = o.submit_human_move('e2', 'e4')
assert r.accepted, r.error
print('Smoke test: OK')
"
```
