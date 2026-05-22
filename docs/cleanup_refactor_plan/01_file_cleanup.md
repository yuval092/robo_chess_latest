# Stage 1 — Dead-File Removal and Small-File Merges

**Objective**: Delete every file that is not referenced by any active code path, and merge files that are too small to stand alone without blurring responsibility.

---

## 1.1 Delete `archive/`

**What it is**: The `archive/rl_system/` folder contains two Python files (`training/callbacks.py`, `training/trainer.py`) and a `training.yaml` left over from a previous RL training iteration. Nothing in `src/`, `scripts/`, or `tests/` imports or references these files.

**Action**: Delete the entire `archive/` directory tree.

```bash
rm -rf archive/
```

**Validation**: `grep -r "archive" src/ scripts/ tests/` returns nothing.

---

## 1.2 Delete Unused MuJoCo XML Assets

**What they are**: The `chess_env/assets/` folder contains six XML files. Only `pick_and_place.xml` is loaded at runtime (via `environment_generation.py:DEFAULT_SCENE_PATH`). The other five are Fetch robot demo scenes from the upstream gymnasium-robotics package that were copied in but never adapted for chess:

- `push.xml`
- `reach.xml`
- `slide.xml`
- `robot.xml`
- `shared.xml`

**Action**: Delete the five unused XML files.

```bash
rm chess_env/assets/push.xml
rm chess_env/assets/reach.xml
rm chess_env/assets/slide.xml
rm chess_env/assets/robot.xml
rm chess_env/assets/shared.xml
```

**Validation**:
```bash
grep -r "push.xml\|reach.xml\|slide.xml\|robot.xml\|shared.xml" src/ scripts/ tests/ chess_env/
```
Returns nothing.

---

## 1.3 Merge `visualize_chess_setup.py` into `visualize.py`

**What they are**: Two scripts that both open the MuJoCo viewer. `visualize.py` opens the environment without pieces. `visualize_chess_setup.py` (24 lines, no argparse) additionally teleports all pieces to their starting squares. They share identical boilerplate.

**Action**: Extend `scripts/visualize.py` with a `--setup` flag that triggers the piece reset, then delete `scripts/visualize_chess_setup.py`.

**New `scripts/visualize.py` structure**:
```python
"""Open a MuJoCo viewer for the chess environment, optionally with pieces reset to start."""
from __future__ import annotations

import argparse
import gymnasium as gym

import src.chess_env
from src.chess_env.environment_generation import regenerate_environment
from src.physical.piece_registry import PieceRegistry
from src.physical.piece_teleport import PieceTeleporter
from src.chess_game.board_mapper import BoardMapper
from src.utils.args import add_common_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise the RoboChess MuJoCo scene.")
    add_common_args(parser)
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Teleport all pieces to starting squares before rendering.",
    )
    parser.add_argument(
        "--scenario",
        default="transit",
        choices=["transit", "descend", "ascend"],
        help="Force-scenario for the environment (default: transit).",
    )
    args = parser.parse_args()

    regenerate_environment()
    env = gym.make(
        "ChessFetchTask-v0",
        render_mode="human",
        show_chess_pieces=True,
        hide_object=True,
        force_scenario=args.scenario,
        debug=args.debug,
    )
    env.reset()

    if args.setup:
        board_mapper = BoardMapper.from_configs()
        teleporter = PieceTeleporter(env, board_mapper)
        for piece_id, square in PieceRegistry().starting_square_map().items():
            teleporter.teleport_piece_to_square(piece_id, square)

    try:
        while True:
            env.render()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

```bash
rm scripts/visualize_chess_setup.py
```

**Validation**: `python scripts/visualize.py --help` shows `--setup` flag. `python scripts/visualize.py --setup` opens viewer with pieces on start squares (manual check).

---

## 1.4 Merge the Four `generate_*.py` Scripts into `scripts/generate_scene.py`

**What they are**: Four nearly identical 24–32-line CLI wrappers, each calling one function from `environment_generation.py`:

- `generate_board_xml.py` → `build_board_fragment` / `update_board_scene`
- `generate_pieces_xml.py` → `build_pieces_fragment` / `update_pieces_scene`
- `generate_zones_xml.py` → `build_zones_fragment` / `update_zones_scene`
- `generate_chess_stls.py` → `PIECE_BUILDERS` / `regenerate_stls`

They share identical boilerplate and are all run together whenever regeneration is needed. Merging them into one script with sub-commands reduces the scripts/ surface area.

**Action**: Create `scripts/generate_scene.py` with sub-commands, then delete the four originals.

**New `scripts/generate_scene.py` structure**:
```python
"""CLI tool for regenerating chess scene assets (XML fragments and STL meshes)."""
from __future__ import annotations

import argparse
from pathlib import Path


SCENE_PATH = Path("chess_env/assets/pick_and_place.xml")


def cmd_board(args: argparse.Namespace) -> None:
    """Regenerate or print the chess board XML fragment."""
    from src.chess_env.environment_generation import build_board_fragment, update_board_scene
    if args.write:
        update_board_scene(SCENE_PATH)
        print(f"Updated board fragment in {SCENE_PATH}")
    else:
        print(build_board_fragment())


def cmd_pieces(args: argparse.Namespace) -> None:
    """Regenerate or print the chess pieces XML fragment."""
    from src.chess_env.environment_generation import build_pieces_fragment, update_pieces_scene
    if args.write:
        update_pieces_scene(SCENE_PATH)
        print(f"Updated pieces fragment in {SCENE_PATH}")
    else:
        print(build_pieces_fragment())


def cmd_zones(args: argparse.Namespace) -> None:
    """Regenerate or print the chess zone XML fragment."""
    from src.chess_env.environment_generation import build_zones_fragment, update_zones_scene
    if args.write:
        update_zones_scene(SCENE_PATH)
        print(f"Updated zones fragment in {SCENE_PATH}")
    else:
        print(build_zones_fragment())


def cmd_stls(args: argparse.Namespace) -> None:
    """Regenerate STL mesh files for all chess piece types."""
    from src.chess_env.environment_generation import PIECE_BUILDERS, regenerate_stls
    out_dir = regenerate_stls(Path(args.out_dir))
    for name in PIECE_BUILDERS:
        print(f"  {name}: generated")
    print(f"Generated {len(PIECE_BUILDERS)} STL meshes in {out_dir}")


def cmd_all(args: argparse.Namespace) -> None:
    """Regenerate all scene assets: STLs, board XML, pieces XML, zones XML."""
    from src.chess_env.environment_generation import regenerate_environment
    regenerate_environment()
    print("Full environment regeneration complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate RoboChess scene assets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_board = sub.add_parser("board", help="Regenerate board XML.")
    p_board.add_argument("--write", action="store_true")
    p_board.set_defaults(func=cmd_board)

    p_pieces = sub.add_parser("pieces", help="Regenerate pieces XML.")
    p_pieces.add_argument("--write", action="store_true")
    p_pieces.set_defaults(func=cmd_pieces)

    p_zones = sub.add_parser("zones", help="Regenerate zones XML.")
    p_zones.add_argument("--write", action="store_true")
    p_zones.set_defaults(func=cmd_zones)

    p_stls = sub.add_parser("stls", help="Regenerate STL meshes.")
    p_stls.add_argument("--out-dir", default="chess_env/stls/chess")
    p_stls.set_defaults(func=cmd_stls)

    p_all = sub.add_parser("all", help="Regenerate everything.")
    p_all.set_defaults(func=cmd_all)

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

**Validation**:
```bash
python scripts/generate_scene.py --help
python scripts/generate_scene.py board
python scripts/generate_scene.py stls
python scripts/generate_scene.py all
```

---

## 1.5 Merge `src/chess_game/models.py` into `src/chess_game/chess_service.py`

**What it is**: `models.py` is a 20-line file containing only the `GameStatus` dataclass. This dataclass is used exclusively by `ChessService.status()` and returned throughout the chess layer. It is not large enough to warrant its own file.

**Action**: Move the `GameStatus` dataclass to the top of `chess_service.py` (before the `ChessService` class), then delete `models.py`. Update all imports across the project.

**Files that import from `models.py`** (find with grep):
```bash
grep -rn "from src.chess_game.models import\|from .models import" src/ scripts/ tests/
```

Expected results:
- `src/chess_game/game_orchestrator.py`
- `src/ui/app.py` (indirectly via `GameSnapshot`)
- Test files

**Import change** (all affected files):
```python
# Before
from src.chess_game.models import GameStatus

# After
from src.chess_game.chess_service import GameStatus
```

```bash
rm src/chess_game/models.py
```

**Validation**: `python -m pytest tests/ -v` — all pass.

---

## 1.6 Merge `src/utils/config.py` and `src/utils/logger.py` into `src/utils/io.py`

**What they are**: Two tiny utility files (10 and 12 lines respectively). Both are infrastructure utilities with no interdependencies. Merging them into a single `src/utils/io.py` reduces the package surface without blurring responsibility.

**Action**: Create `src/utils/io.py` with the combined content. Update all imports. Delete the originals.

**New `src/utils/io.py`**:
```python
"""Shared I/O utilities: YAML config loading and logging setup."""
from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml


_CONFIG_ROOT = Path(__file__).parents[2] / "configs"


def load_config(config_name: str) -> dict:
    """Load and return a YAML config file by name from the configs/ directory."""
    path = _CONFIG_ROOT / f"{config_name}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Create a file-backed logger at INFO level, idempotent across repeated calls."""
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

**Import change** (all files importing from the old modules):
```python
# Before
from src.utils.config import load_config
from src.utils.logger import setup_logger

# After
from src.utils.io import load_config, setup_logger
```

Find all affected files:
```bash
grep -rn "from src.utils.config\|from src.utils.logger\|from .config\|from .logger" src/ scripts/ tests/
```

```bash
rm src/utils/config.py
rm src/utils/logger.py
```

**Validation**: `python -m pytest tests/ -v` — all pass.

---

## 1.7 Clean Up `__init__.py` Files

The following `__init__.py` files are completely empty or contain only a single comment:

- `src/chess_game/__init__.py` (1 line, empty)
- `src/physical/__init__.py` (1 line, empty)
- `src/ui/__init__.py` (1 line, empty)
- `src/utils/__init__.py` (if it exists, likely empty)

**Action**: These files must remain (they mark directories as Python packages) but should each receive a one-line module docstring:

```python
"""Chess game logic: service, orchestrator, move planner, and board mapper."""
```
```python
"""Physical execution layer: arm movement, piece teleporter, and occupancy tracking."""
```
```python
"""Web UI: Flask application and JSON serialisation."""
```
```python
"""Shared utilities: config loading and logging."""
```

`src/chess_env/__init__.py` currently contains the Gymnasium `register()` call — this is correct and stays as-is, but gains a module docstring.

---

## Stage 1 — Full Validation Checklist

```bash
# 1. No references to deleted files
grep -r "archive\|push\.xml\|reach\.xml\|slide\.xml\|robot\.xml\|shared\.xml" src/ scripts/ tests/

# 2. No references to deleted scripts
grep -r "generate_board_xml\|generate_pieces_xml\|generate_zones_xml\|generate_chess_stls\|visualize_chess_setup" src/ scripts/ tests/ docs/

# 3. No references to models.py
grep -r "from src.chess_game.models\|from .models" src/ scripts/ tests/

# 4. No references to old utils
grep -r "from src.utils.config\|from src.utils.logger\|from .config\|from .logger" src/ scripts/ tests/

# 5. Full test suite
python -m pytest tests/ -v

# 6. Import smoke test
python -c "from src.chess_game.chess_service import ChessService, GameStatus; print('OK')"
python -c "from src.utils.io import load_config, setup_logger; print('OK')"
python -c "python scripts/generate_scene.py --help"
```
