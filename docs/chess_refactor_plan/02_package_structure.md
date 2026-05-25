# Stage 2 — Installable Package and Import Cleanup

**Objective**: Make the project installable as a Python package via `pip install -e .` so that all `src/` modules are importable without `sys.path` hacks. Remove all `sys.path.append` / `sys.path.insert` calls from scripts and tests.

---

## 2.1 Background: Why Path Hacks Are a Problem

Every script currently does one of:
```python
sys.path.append(os.getcwd())         # requires user to run from project root
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

These hacks:
- Fail silently if the working directory is wrong.
- Make scripts non-runnable via `python -m`.
- Pollute `sys.path` with the full project root, causing hard-to-debug import shadowing.

The fix is to install the project as an editable package once, after which all `from src.X import Y` imports work from anywhere.

---

## 2.2 Production vs Training Package Boundary

**Critical rule**: `training/` is not a production package. It contains training infrastructure (SB3 wrappers, callbacks, trainer). It must not be imported by any `src/` module. The violation that exists today — `src/chess_env/model_controller.py` importing `from training.envs import WRAPPER_MAP` — is fixed in Stage 10, not here.

For the `pyproject.toml`, `training/` is intentionally excluded from the installable package. After `pip install -e .`, `from training.envs import ...` still works (because the repo root is on `sys.path` in editable mode), but the intent is clear: `training/` is a development tool, not a production module.

---

## 2.3 Create `pyproject.toml`

Create a `pyproject.toml` at the project root. This replaces the need for a `setup.py`.

**File: `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "robo-chess"
version = "0.1.0"
description = "RoboChess: MuJoCo-based robot arm chess controller"
requires-python = ">=3.10"
dependencies = [
    "gymnasium",
    "gymnasium-robotics",
    "mujoco",
    "numpy",
    "python-chess",
    "flask",
    "pyyaml",
    "stable-baselines3",
    "huggingface_hub",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*", "chess_env*"]
# scripts/ and training/ excluded: scripts/ mixes production and training-dependent tools;
# using src/cli/ for installable entry points avoids packaging training-importing scripts.

[tool.setuptools.package-data]
"chess_env" = ["assets/*.xml", "stls/**/*.stl"]

[project.scripts]
robo-chess-ui = "src.cli.run_chess_ui:main"
robo-chess-generate = "src.cli.generate_scene:main"
robo-chess-visualize = "src.cli.visualize:main"
```

**Why `scripts*` is NOT included**: `scripts/` contains both production tools (UI, generate) and training-dependent tools (`train_rl.py`, `diagnostics/eval_rl_stages_direct.py`). Packaging the whole directory would include modules that import `training/`, which is intentionally excluded from the distribution — creating broken entry points in a wheel install.

**Why `stable-baselines3` and `huggingface_hub` are in main `dependencies`**: RL inference is the default production path (Guiding Principle #7). If a future scripted-only install profile is needed, move those libs to a separate optional extra and guard RL imports explicitly — but pick one strategy and do not duplicate them in both `dependencies` and `optional-dependencies`.

**Solution — `src/cli/`**: Move the production CLI implementations into packaged modules under `src/cli/`. The key architectural rule is:

> `src/cli/` modules must contain the **real implementation logic**. `scripts/` wrappers are thin one-liners that call `src.cli.X.main()`.

This is necessary because `scripts/` is excluded from the installed distribution. If `src/cli/` merely forwarded to `scripts/`, installed entry points (`robo-chess-ui`, etc.) would fail with `ModuleNotFoundError` outside the repo.

**File ownership after this stage:**

```text
src/cli/run_chess_ui.py      # real UI entry point implementation (moved from scripts/)
src/cli/generate_scene.py    # real asset-generation CLI implementation (moved from scripts/)
src/cli/visualize.py         # real visualization CLI implementation (moved from scripts/)
scripts/run_chess_ui.py      # thin dev wrapper: from src.cli.run_chess_ui import main; main()
scripts/generate_scene.py    # thin dev wrapper: from src.cli.generate_scene import main; main()
scripts/visualize.py         # thin dev wrapper: from src.cli.visualize import main; main()
```

The dev wrappers in `scripts/` should contain only:
```python
"""Dev-only wrapper — calls src.cli.<module>.main()."""
from src.cli.<module> import main
if __name__ == "__main__":
    main()
```

Direct execution from the repo root still works (`python scripts/run_chess_ui.py`) while installed entry points depend only on packaged `src.*` modules. All other `eval_*` scripts, `train_rl.py`, and diagnostic tools remain in `scripts/` only (they are not production entry points).

**After creating this file, install the package once:**
```bash
pip install -e .
```

This makes `from src.chess_game.X import Y` work in any terminal, regardless of working directory.

---

## 2.4 Remove All `sys.path` Hacks from Scripts

**Generate the complete list of affected files first** — do not rely on a static list:

```bash
# Find every script (recursive, including diagnostics/) that has a sys.path hack
grep -rln "sys\.path" scripts/
```

Run this before starting work to capture the full current inventory. At the time of writing the known affected files include:

- `scripts/analyze_debug_log.py`
- `scripts/debug_one_move.py`
- `scripts/eval_all_cells_rl.py`
- `scripts/eval_all_square_moves.py`
- `scripts/eval_chess_game_flow.py`
- `scripts/eval_chess_piece_move.py`
- `scripts/eval_chess_reachability.py`
- `scripts/eval_draw_conditions.py`
- `scripts/eval_rl_stages.py`
- `scripts/eval_sequence.py`
- `scripts/eval_special_moves.py`
- `scripts/eval_stages.py`
- `scripts/eval_stress.py`
- `scripts/eval_targeted.py`
- `scripts/train_rl.py`
- `scripts/run_chess_ui.py`
- `scripts/verify_physics.py`
- `scripts/visualize.py`
- `scripts/generate_scene.py` (new file from Stage 1 — must not have the hack)
- `scripts/diagnostics/*.py` (all diagnostic scripts)

**The grep command is the authoritative source.** If the grep output differs from the list above, fix every file that grep finds.

**Action**: In each file, remove the following lines entirely:

```python
import sys, os                          # remove os/sys if only used for path hack
sys.path.append(os.getcwd())            # DELETE
sys.path.insert(0, os.getcwd())         # DELETE
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # DELETE
import os                               # keep only if used elsewhere
import sys                              # keep only if used elsewhere
```

**Example: `scripts/eval_stages.py` before:**
```python
import sys, os, argparse, time
import numpy as np
sys.path.append(os.getcwd())
import src.chess_env
```

**After:**
```python
"""Per-stage arm accuracy evaluation using ScriptedController."""
from __future__ import annotations

import argparse
import time

import numpy as np

import src.chess_env
from src.chess_env.controller import ScriptedController
from src.utils.args import add_common_args, make_env
```

Apply the same pattern to every script: `from __future__ import annotations` first, then stdlib, then third-party, then local `src.*` imports — no path manipulation.

---

## 2.5 Remove `sys.path` Hack from `tests/conftest.py`

**Current `tests/conftest.py`** (2 lines):
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```

**After**: Replace with an empty file (or a single module docstring), since `pip install -e .` handles imports:
```python
"""Pytest configuration and shared fixtures for the RoboChess test suite."""
```

If any test fixtures are needed (none currently exist), they go here.

---

## 2.6 Create a `scripts/__init__.py`

For `python -m scripts.X` invocation to work, `scripts/` must be a package:

```bash
touch scripts/__init__.py
```

**Content of `scripts/__init__.py`**:
```python
"""RoboChess evaluation, generation, and UI launch scripts."""
```

---

## 2.7 Verify Script Invocation Patterns

**`scripts/` is not installed** — `python -m scripts.X` requires the repo root to be on `sys.path`. It works when run from the repo root but not from an arbitrary directory after a wheel install.

After this stage there are two valid ways to run scripts, depending on context:

```bash
# From the repo root (dev context only — scripts/ is NOT packaged)
python scripts/eval_stages.py --n-episodes 5
python -m scripts.eval_stages --n-episodes 5   # requires running from repo root

# From anywhere, using installed entry points (packaged src/cli/ only)
robo-chess-ui --help
robo-chess-generate --help
robo-chess-visualize --help
```

The `python -m scripts.X` form is a convenience for developers running from the repo root; it is not a guarantee of portability. The only invocation patterns that work from arbitrary directories are the `robo-chess-*` entry points backed by `src/cli/`.

---

## 2.8 Update `src/utils/args.py` (Already Exists)

`src/utils/args.py` already contains `add_common_args()` and `make_env()`. It must also import cleanly without path hacks. Verify it has no path manipulation:

```bash
grep "sys.path" src/utils/args.py
```

Must return nothing. If found, remove.

Ensure it has a module docstring and the `from __future__ import annotations` guard.

---

## Stage 2 — Full Validation Checklist

```bash
# 1. Install the package
pip install -e .

# 2. No sys.path hacks remaining in ANY script (recursive scan)
grep -rln "sys\.path" scripts/ tests/ src/
# Must return nothing. This is the authoritative check — every file in scripts/**/*.py
# must have had its hack removed.

# 3. All imports work from a clean Python session
python -c "from src.chess_game.chess_service import ChessService; print('OK')"
python -c "from src.chess_env.controller import ScriptedController; print('OK')"
python -c "from src.utils.io import load_config; print('OK')"
python -c "from src.chess_env.model_controller import ModelEmbeddedController; print('OK')"

# 4. src/cli/ modules contain real implementations (not just imports from scripts/)
python -c "from src.cli.run_chess_ui import main; print('OK')"
python -c "from src.cli.generate_scene import main; print('OK')"
python -c "from src.cli.visualize import main; print('OK')"

# 5. training/ is NOT in the installed package distribution (dev-only)
python -c "import pkg_resources; d = pkg_resources.get_distribution('robo-chess'); print([p for p in d.files if 'training' in str(p)])"
# Must return [] (no training/ files in the installed distribution)

# 6. Scripts runnable as modules FROM THE REPO ROOT (not from arbitrary dirs)
python -m scripts.eval_stages --help      # run from repo root only
# Installed entry points work from anywhere after pip install -e .:
robo-chess-generate --help
robo-chess-ui --help
robo-chess-visualize --help

# 7. Full test suite
python -m pytest tests/ -v

# 8. Scripts still run directly
python scripts/eval_stages.py --help

# 9. stable-baselines3 not duplicated in both deps and optional-dependencies
grep -c "stable-baselines3" pyproject.toml
# Must return 1 (appears exactly once)
```
