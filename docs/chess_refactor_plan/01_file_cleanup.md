# Stage 1 — Dead-File Removal and Small-File Merges

**Objective**: Delete every file that is not referenced by any active code path, and merge files that are too small to stand alone without blurring responsibility. Assumes Stage 0 (archive cleanup) is already complete.

---

## 1.1 Merge the Four `generate_*.py` Scripts into `scripts/generate_scene.py`

**What they are**: Four nearly identical 24–32-line CLI wrappers, each calling one function from `environment_generation.py`:

- `generate_board_xml.py` → `build_board_fragment` / `update_board_scene`
- `generate_pieces_xml.py` → `build_pieces_fragment` / `update_pieces_scene`
- `generate_zones_xml.py` → `build_zones_fragment` / `update_zones_scene`
- `generate_chess_stls.py` → `PIECE_BUILDERS` / `regenerate_stls`

**Action**: Create `scripts/generate_scene.py` with sub-commands, then delete the four originals.

```python
"""CLI tool for regenerating chess scene assets (XML fragments and STL meshes)."""
from __future__ import annotations
import argparse
from pathlib import Path

SCENE_PATH = Path("chess_env/assets/pick_and_place.xml")


def cmd_board(args):
    from src.chess_env.environment_generation import build_board_fragment, update_board_scene
    if args.write:
        update_board_scene(SCENE_PATH)
        print(f"Updated board fragment in {SCENE_PATH}")
    else:
        print(build_board_fragment())


def cmd_pieces(args):
    from src.chess_env.environment_generation import build_pieces_fragment, update_pieces_scene
    if args.write:
        update_pieces_scene(SCENE_PATH)
    else:
        print(build_pieces_fragment())


def cmd_zones(args):
    from src.chess_env.environment_generation import build_zones_fragment, update_zones_scene
    if args.write:
        update_zones_scene(SCENE_PATH)
    else:
        print(build_zones_fragment())


def cmd_stls(args):
    from src.chess_env.environment_generation import PIECE_BUILDERS, regenerate_stls
    regenerate_stls(Path(args.out_dir))
    print(f"Generated {len(PIECE_BUILDERS)} STL meshes in {args.out_dir}")


def cmd_all(args):
    from src.chess_env.environment_generation import regenerate_environment
    regenerate_environment()
    print("Full environment regeneration complete.")


def main():
    parser = argparse.ArgumentParser(description="Regenerate RoboChess scene assets.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_board = sub.add_parser("board"); p_board.add_argument("--write", action="store_true"); p_board.set_defaults(func=cmd_board)
    p_pieces = sub.add_parser("pieces"); p_pieces.add_argument("--write", action="store_true"); p_pieces.set_defaults(func=cmd_pieces)
    p_zones = sub.add_parser("zones"); p_zones.add_argument("--write", action="store_true"); p_zones.set_defaults(func=cmd_zones)
    p_stls = sub.add_parser("stls"); p_stls.add_argument("--out-dir", default="chess_env/stls/chess"); p_stls.set_defaults(func=cmd_stls)
    p_all = sub.add_parser("all"); p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

```bash
rm scripts/generate_board_xml.py
rm scripts/generate_pieces_xml.py
rm scripts/generate_zones_xml.py
rm scripts/generate_chess_stls.py
```

---

## 1.2 Merge `visualize_chess_setup.py` into `visualize.py`

`visualize.py` opens the MuJoCo viewer. `visualize_chess_setup.py` (24 lines, no argparse) also opens it but additionally teleports pieces to start. Add a `--setup` flag to `visualize.py` and delete the second file.

```bash
rm scripts/visualize_chess_setup.py
```

In `scripts/visualize.py`, add:
```python
parser.add_argument("--setup", action="store_true",
                    help="Teleport all pieces to starting squares before rendering.")
```
Then conditionally call the piece teleporter if `args.setup`.

---

## 1.3 Move `GameStatus` from `models.py` into `chess_service.py`; Keep Shim

`models.py` is a ~20-line file containing only the `GameStatus` dataclass. Move it to the top of `chess_service.py`, then replace `models.py` with a backward-compat re-export shim (same pattern as section 1.4 for `config.py`/`logger.py`):

```bash
grep -rn "from src.chess_game.models import\|from .models import" src/ scripts/ tests/
# New code must import from chess_service; shim keeps old imports working:
# from src.chess_game.models import GameStatus  ← still works via shim
# from src.chess_game.chess_service import GameStatus  ← preferred going forward
```

**Replace `src/chess_game/models.py`** with:
```python
"""Backward-compatible shim — import GameStatus from src.chess_game.chess_service instead."""
from src.chess_game.chess_service import GameStatus  # noqa: F401
```

Do **not** delete `models.py` in this stage — the shim keeps `game_orchestrator.py` and any test imports working without a simultaneous broad update.

---

## 1.4 Create `src/utils/io.py` from `config.py` and `logger.py`; Keep Shims

Both are tiny utilities (10 and 12 lines). The implementation moves into `src/utils/io.py`, but the old module paths are preserved as one-line re-export shims so existing code and dev snippets keep working without a simultaneous broad import rewrite. The shims are removed in a later cleanup pass after all callers migrate.

**Create `src/utils/io.py`**:
```python
"""Shared I/O utilities: YAML config loading and logging setup."""
from __future__ import annotations
import logging
import os
from pathlib import Path
import yaml

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"  # absolute: .../robo_chess_latest/configs


def load_config(config_name: str) -> dict:
    """Load and return a YAML config by name from configs/."""
    path = _CONFIG_ROOT / f"{config_name}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Create a file-backed logger at INFO level, idempotent on repeated calls."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
        )
        logger.addHandler(handler)
    return logger
```

**Replace `src/utils/config.py` with a re-export shim**:
```python
"""Backward-compatible shim — import load_config from src.utils.io instead."""
from src.utils.io import load_config  # noqa: F401
```

**Replace `src/utils/logger.py` with a re-export shim**:
```python
"""Backward-compatible shim — import setup_logger from src.utils.io instead."""
from src.utils.io import setup_logger  # noqa: F401
```

Migrate callers gradually (new code must use `src.utils.io`):
```bash
grep -rn "from src.utils.config\|from src.utils.logger\|from .config\|from .logger" src/ scripts/ tests/ training/
# Update each occurrence to import from src.utils.io
# The shims keep everything working during migration
```

Do **not** delete `config.py` or `logger.py` in this stage — the shims stay until Stage 9 confirms all callers are migrated.

---

## 1.5 Move `scripts/eval_rl_stages.py` to `scripts/diagnostics/`

**Why it cannot be simply deleted**: `eval_rl_stages.py` bypasses `ModelEmbeddedController` and tests a checkpoint directly under the same training wrapper used during training. This is a distinct and legitimate use case: it isolates whether a checkpoint itself is bad versus whether `ModelEmbeddedController` integration is bad. Deleting it would remove that diagnostic capability.

**Why it must not stay in `scripts/`**: It imports from `training/`, which violates the training/production boundary (Guiding Principle #6). It must not be treated as a production evaluation script.

**Action**: Move to `scripts/diagnostics/` alongside the other diagnostic tools:
```bash
git mv scripts/eval_rl_stages.py scripts/diagnostics/eval_rl_stages_direct.py
```

Add a header comment to the moved file:
```python
"""
Direct training-wrapper diagnostic for RL model checkpoints.

Tests a model checkpoint directly under its training wrapper (bypassing
ModelEmbeddedController). Use this to distinguish:
  - checkpoint-level failures (model policy is wrong)
  - integration failures (ModelEmbeddedController wiring is wrong)

For production integration testing, use:
  PYTHONPATH=. python scripts/eval_stages.py --stages transit --n-episodes 30
"""
```

Update `scripts/diagnostics/README.md` to document this tool's diagnostic purpose.

**Production path**: For production integration testing, `eval_stages.py` (RL default after Stage 7) is the correct tool. The direct diagnostic is only for training-debug isolation.

---

## 1.6 Organise Diagnostic Scripts

The following scripts were created during RL model debugging sessions:
- `scripts/diagnose_transit_full.py`
- `scripts/diagnose_descend_afile.py`
- `scripts/diagnose_transit_c2.py`

These are not evaluation scripts (they don't report pass/fail metrics) but are useful for debugging specific failure modes.

**Action**: Move them to `scripts/diagnostics/`:
```bash
mkdir -p scripts/diagnostics
mv scripts/diagnose_transit_full.py scripts/diagnostics/
mv scripts/diagnose_descend_afile.py scripts/diagnostics/
mv scripts/diagnose_transit_c2.py scripts/diagnostics/
touch scripts/diagnostics/__init__.py
```

Add a `scripts/diagnostics/README.md` explaining that these are debug tools, not production evaluation scripts.

---

## 1.7 Rename `scripts/test_grasp_physics.py` → `scripts/eval_grasp_physics.py`

Scripts prefixed `test_` are collected by pytest. Rename to align with the `eval_*` convention.

```bash
git mv scripts/test_grasp_physics.py scripts/eval_grasp_physics.py
```

---

## 1.8 Delete Truly Unused MuJoCo XML Assets

**⚠ Critical**: `robot.xml` and `shared.xml` are NOT unused. `pick_and_place.xml` includes both:
- Line 8: `<include file="shared.xml">`
- Line 12: `<include file="robot.xml">`

Deleting them would break `gym.make("ChessFetchTask-v0", ...)`, every eval script, and all tests.

Only three files are standalone demo scenes from upstream gymnasium-robotics with no references in this project:

```bash
# Verify the three demo scenes are truly unreferenced before deleting:
grep -rn "push\.xml\|reach\.xml\|slide\.xml" src/ scripts/ tests/ chess_env/
# Must return nothing, then:
rm chess_env/assets/push.xml
rm chess_env/assets/reach.xml
rm chess_env/assets/slide.xml
```

`robot.xml` and `shared.xml` must be kept. The grep check in the original plan included them by mistake.

---

## 1.9 Add Module Docstrings to Empty `__init__.py` Files

```python
# src/chess_game/__init__.py
"""Chess game logic: service, orchestrator, move planner, and board mapper."""

# src/physical/__init__.py
"""Physical execution layer: arm movement, piece teleporter, and occupancy tracking."""

# src/ui/__init__.py
"""Web UI: Flask application and JSON serialisation."""

# src/utils/__init__.py
"""Shared utilities: config loading and logging."""

# training/__init__.py  (already has content — just add docstring at top)
"""SAC specialist model training infrastructure. Not imported by production src/ code."""

# scripts/diagnostics/__init__.py
"""Step-by-step diagnostic scripts for RL model debugging. Not for production evaluation."""
```

`src/chess_env/__init__.py` gains a docstring but its `register()` call stays unchanged.

---

## 1.10 Handle `eval_targeted.py`

`scripts/eval_targeted.py` is a targeted square evaluation script that uses RL models to evaluate specific problem squares (those identified as failing in earlier diagnostics). It has a `main()` function, argparse, and imports from the RL pipeline.

**Assessment**:
- Has `main()` and argparse: qualifies as a proper `eval_*` script.
- Uses RL models for evaluation: belongs at `scripts/` root alongside `eval_all_cells_rl.py`.
- Has a `sys.path` hack: fixed in Stage 2.
- Loads deployed model paths from `training.yaml["deployed_models"]` directly: fixed in Stage 7.14 after Stage 3.12 creates `deployed_models.yaml`.

**Conclusion**: Keep `scripts/eval_targeted.py` in `scripts/` root. Do not move it to diagnostics. Fix its issues in the later stages as noted.

```bash
# Verify it is a proper eval script (not a diagnostic tool):
python scripts/eval_targeted.py --help
# Must show argparse help without error
```

---

## Stage 1 — Full Validation Checklist

```bash
# 1. No references to deleted/moved scripts (old names gone from src/ and tests/)
grep -r "generate_board_xml\|generate_pieces_xml\|generate_zones_xml\|generate_chess_stls" src/ scripts/ tests/
grep -r "visualize_chess_setup\|test_grasp_physics" src/ scripts/ tests/

# 2. eval_rl_stages moved (not deleted) — in diagnostics, not in scripts/ root
ls scripts/diagnostics/eval_rl_stages_direct.py
test -f scripts/eval_rl_stages.py && echo "FAIL: still in scripts/" || echo "OK: moved"

# 3. No references to models.py
grep -r "from src.chess_game.models\|from .models" src/ scripts/ tests/

# 4. src/utils/io.py exists and is the canonical import
python -c "from src.utils.io import load_config, setup_logger; print('OK')"
# Shims still work (must not break existing code)
python -c "from src.utils.config import load_config; print('shim OK')"
python -c "from src.utils.logger import setup_logger; print('shim OK')"

# 5. Only the three standalone demo XMLs removed; robot.xml and shared.xml present
test -f chess_env/assets/robot.xml && echo "OK" || echo "FAIL: robot.xml missing"
test -f chess_env/assets/shared.xml && echo "OK" || echo "FAIL: shared.xml missing"
test ! -f chess_env/assets/push.xml && echo "OK" || echo "push.xml still present"
test ! -f chess_env/assets/reach.xml && echo "OK" || echo "reach.xml still present"
test ! -f chess_env/assets/slide.xml && echo "OK" || echo "slide.xml still present"

# 6. MuJoCo env still loads (proves robot.xml/shared.xml intact)
PYTHONPATH=. python -c "import src.chess_env, gymnasium as gym; env = gym.make('ChessFetchTask-v0'); env.close(); print('env load OK')"

# 7. Diagnostic scripts in new location
ls scripts/diagnostics/

# 8. generate_scene.py works
python scripts/generate_scene.py --help
python scripts/generate_scene.py board

# 9. Full test suite
python -m pytest tests/ -v

# 10. Import smoke tests
python -c "from src.chess_game.chess_service import ChessService, GameStatus; print('OK')"

# 11. eval_targeted.py is a proper eval script at scripts/ root
python scripts/eval_targeted.py --help
```
