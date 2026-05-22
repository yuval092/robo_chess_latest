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

## 2.2 Create `pyproject.toml`

Create a `pyproject.toml` at the project root. This replaces the need for a `setup.py`.

**File: `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

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
]

[tool.setuptools.packages.find]
where = ["."]
include = ["src*", "chess_env*"]

[tool.setuptools.package-data]
"chess_env" = ["assets/*.xml", "stls/**/*.stl"]

[project.scripts]
robo-chess-ui = "scripts.run_chess_ui:main"
robo-chess-generate = "scripts.generate_scene:main"
robo-chess-visualize = "scripts.visualize:main"
```

**After creating this file, install the package once:**
```bash
pip install -e .
```

This makes `from src.chess_game.X import Y` work in any terminal, regardless of working directory.

---

## 2.3 Remove All `sys.path` Hacks from Scripts

**Affected files** (15 scripts):
- `scripts/analyze_debug_log.py`
- `scripts/debug_one_move.py`
- `scripts/eval_chess_game_flow.py`
- `scripts/eval_chess_piece_move.py`
- `scripts/eval_chess_reachability.py`
- `scripts/eval_draw_conditions.py`
- `scripts/eval_sequence.py`
- `scripts/eval_special_moves.py`
- `scripts/eval_stages.py`
- `scripts/eval_stress.py`
- `scripts/run_chess_ui.py`
- `scripts/verify_physics.py`
- `scripts/visualize.py`
- `scripts/generate_scene.py` (new file — must not have the hack)
- `scripts/debug_one_move.py`

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

## 2.4 Remove `sys.path` Hack from `tests/conftest.py`

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

## 2.5 Create a `scripts/__init__.py`

For `python -m scripts.X` invocation to work, `scripts/` must be a package:

```bash
touch scripts/__init__.py
```

**Content of `scripts/__init__.py`**:
```python
"""RoboChess evaluation, generation, and UI launch scripts."""
```

---

## 2.6 Verify Script Invocation Patterns

After this stage, scripts must be runnable in two equivalent ways:

```bash
# Method 1: direct file execution (from project root)
python scripts/eval_stages.py --n-episodes 5

# Method 2: module invocation (from anywhere, after pip install -e .)
python -m scripts.eval_stages --n-episodes 5
```

Both must work without any `sys.path` manipulation.

---

## 2.7 Update `src/utils/args.py` (Already Exists)

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

# 2. No sys.path hacks remaining
grep -rn "sys.path" scripts/ tests/ src/

# 3. All imports work from a clean Python session
python -c "from src.chess_game.chess_service import ChessService; print('OK')"
python -c "from src.chess_env.controller import ScriptedController; print('OK')"
python -c "from src.utils.io import load_config; print('OK')"

# 4. Scripts runnable as modules
python -m scripts.eval_stages --help
python -m scripts.generate_scene --help

# 5. Full test suite
python -m pytest tests/ -v

# 6. Scripts still run directly
python scripts/eval_stages.py --help
```
