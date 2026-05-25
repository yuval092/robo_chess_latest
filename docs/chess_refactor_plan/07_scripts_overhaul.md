# Stage 7 — Scripts Overhaul

**Objective**: Every script in `scripts/` must: have a module docstring, have a `main()` function guarded by `if __name__ == "__main__"`, use argparse where applicable, work correctly end-to-end, and cover every logical part of the project.

---

## 7.1 Scripts Inventory and Status

After Stage 1 (file cleanup), the scripts folder contains:

| Script | Status | Issues |
|---|---|---|
| `analyze_debug_log.py` | ✓ Has argparse + main | Long lines; needs docstring |
| `debug_one_move.py` | ✓ Has argparse + main | Path hack (fixed in Stage 2) |
| `eval_all_cells_rl.py` | ✓ Has argparse + main | RL-default; no issues |
| `eval_all_square_moves.py` | ✓ Has argparse + main | Path hack; needs docstring |
| `eval_chess_game_flow.py` | ✓ Has argparse + main | Path hack; needs docstring |
| `eval_chess_piece_move.py` | ✓ Has argparse + main | Path hack; needs docstring |
| `eval_chess_reachability.py` | ✓ Has argparse + main | Path hack; magic numbers (fixed in Stage 3) |
| `eval_draw_conditions.py` | ✗ No argparse | No main guard; path hack |
| `eval_sequence.py` | ✓ Has argparse + main | Path hack |
| `eval_special_moves.py` | ✗ No argparse | No main guard; path hack |
| `eval_stages.py` | ✓ Has argparse + main | Path hack; currently scripted-only default — fix in 7.10 |
| `eval_stress.py` | ✓ Has argparse + main | Path hack |
| `eval_targeted.py` | ✓ Has argparse + main | sys.path hack; loads from `training.yaml["deployed_models"]` — fix to use `deployed_models.yaml` after Stage 3.13 |
| `generate_scene.py` | NEW — created in Stage 1 | Clean |
| `run_chess_ui.py` | ✓ Has argparse + main | Path hack; QueuedUIBackend moved in Stage 6 |
| `train_rl.py` | ✓ Has argparse + main | Training-only; no issues |
| `verify_physics.py` | ✓ Has argparse + main | Path hack |
| `visualize.py` | ✓ Has argparse + main (updated Stage 1) | Path hack (fixed Stage 2) |
| `diagnostics/diagnose_transit_full.py` | ✓ Debug tool | Moved to diagnostics/ in Stage 1 |
| `diagnostics/diagnose_descend_afile.py` | ✓ Debug tool | Moved to diagnostics/ in Stage 1 |
| `diagnostics/diagnose_transit_c2.py` | ✓ Debug tool | Moved to diagnostics/ in Stage 1 |

---

## 7.2 Rename `test_grasp_physics.py` → `eval_grasp_physics.py`

This rename was already specified in Stage 1.7 — it is listed here for completeness. The rename is done via `git mv` in Stage 1; no additional action needed here.

```bash
grep -rn "test_grasp_physics" docs/ scripts/ tests/
# must return nothing after Stage 1
```

---

## 7.3 Fix `eval_draw_conditions.py` — Add argparse and main()

**Current**: A flat script (~40 lines) that runs draw-condition checks without argument parsing or a main guard.

**After**:
```python
"""Verify that all chess draw conditions are correctly detected by ChessService."""
from __future__ import annotations

import argparse

from src.chess_game.chess_service import ChessService
from src.chess_game.game_orchestrator import GameOrchestrator


def run_checks(verbose: bool = False) -> int:
    """Run all draw-condition checks; return the number of failures."""
    failures = 0
    checks = [
        ("stalemate",               _check_stalemate),
        ("insufficient_material",   _check_insufficient_material),
        ("fifty_move_claim",        _check_fifty_move),
        ("seventyfive_move_auto",   _check_seventyfive_move),
        ("threefold_claim",         _check_threefold),
        ("fivefold_auto",           _check_fivefold),
    ]
    for name, fn in checks:
        try:
            fn()
            if verbose:
                print(f"  {name}: OK")
        except AssertionError as exc:
            print(f"  FAIL {name}: {exc}")
            failures += 1
    return failures


def _check_stalemate() -> None:
    """Assert stalemate is detected in a known stalemate position."""
    service = ChessService("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
    assert service.status().is_stalemate, "Stalemate not detected"


# ... (one function per draw condition, matching current assertions) ...


def main() -> None:
    """Entry point: parse args and run draw-condition checks."""
    parser = argparse.ArgumentParser(description="Verify chess draw condition detection.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each check result.")
    args = parser.parse_args()

    failures = run_checks(verbose=args.verbose)
    if failures:
        print(f"\n{failures} check(s) FAILED.")
        raise SystemExit(1)
    print("All draw conditions correctly detected.")


if __name__ == "__main__":
    main()
```

---

## 7.4 Fix `eval_special_moves.py` — Add argparse and main()

**Current**: A flat script (~120 lines) that runs special-move checks without argument parsing.

**After**:
```python
"""Verify that all special chess moves generate the correct physical command sequences."""
from __future__ import annotations

import argparse

from src.chess_game.game_orchestrator import GameOrchestrator
from src.chess_game.move_planner import (
    ArmMoveCommand,
    LogicalPieceTracker,
    MovePlanner,
    RemoveFromBoardCommand,
    TeleportCommand,
)
from src.physical.noop_executor import NoOpPhysicalExecutor


def run_checks(checks: list[str], verbose: bool = False) -> int:
    """Run the requested special-move checks; return the number of failures."""
    all_checks = {
        "castling":    _check_castling,
        "en-passant":  _check_en_passant,
        "promotion":   _check_promotion,
    }
    selected = {k: v for k, v in all_checks.items() if k in checks}
    failures = 0
    for name, fn in selected.items():
        try:
            fn()
            if verbose:
                print(f"  {name}: OK")
        except (AssertionError, ValueError) as exc:
            print(f"  FAIL {name}: {exc}")
            failures += 1
    return failures


# ... (_check_castling, _check_en_passant, _check_promotion implementations) ...


def main() -> None:
    """Entry point: parse args and run selected special-move checks."""
    parser = argparse.ArgumentParser(
        description="Verify special chess move command generation.",
    )
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=["castling", "en-passant", "promotion"],
        default=["castling", "en-passant", "promotion"],
        help="Which special moves to test (default: all).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    failures = run_checks(args.checks, verbose=args.verbose)
    if failures:
        print(f"\n{failures} check(s) FAILED.")
        raise SystemExit(1)
    print("All special move checks passed.")


if __name__ == "__main__":
    main()
```

---

## 7.5 Add `eval_game_logic.py` — New Script for Complete Coverage

No existing script exercises the `GameOrchestrator` through a complete game with all move types in logical sequence (without needing MuJoCo). Create this script:

```python
"""
Logical-only evaluation of GameOrchestrator through a complete game sequence.

Tests all move types (normal, capture, castling, en passant, promotion) using
NoOpPhysicalExecutor — no MuJoCo required.
"""
from __future__ import annotations

import argparse

from src.chess_game.game_orchestrator import GameOrchestrator


SCHOLAR_MATE_MOVES = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "a7a6", "h5f7"]
CASTLING_SEQUENCE  = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"]
EN_PASSANT_SEQ     = ["e2e4", "d7d5", "e4e5", "f7f5", "e5f6"]
PROMOTION_SEQ      = [
    "a2a4", "b7b5", "a4b5", "a7a6", "b5a6",
    "b8c6", "a6a7", "c6d4", "a7a8q",
]


def check_sequence(name: str, moves: list[str], verbose: bool) -> bool:
    """Play a move sequence and verify no errors occur; return True on success."""
    orchestrator = GameOrchestrator.create_headless(auto_computer_reply=False)
    for uci in moves:
        src, dst = uci[:2], uci[2:4]
        promotion = uci[4] if len(uci) == 5 else None
        result = orchestrator.submit_human_move(src, dst, promotion)
        if not result.accepted:
            if verbose:
                print(f"  FAIL {name} at {uci}: {result.error}")
            return False
    if verbose:
        print(f"  {name}: OK ({len(moves)} moves)")
    return True


def main() -> None:
    """Entry point: run all logical game sequences and report results."""
    parser = argparse.ArgumentParser(
        description="Logical game-flow evaluation (no MuJoCo required).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    sequences = [
        ("scholar_mate",  SCHOLAR_MATE_MOVES),
        ("castling",      CASTLING_SEQUENCE),
        ("en_passant",    EN_PASSANT_SEQ),
        ("promotion",     PROMOTION_SEQ),
    ]
    failures = sum(
        0 if check_sequence(name, moves, args.verbose) else 1
        for name, moves in sequences
    )
    if failures:
        print(f"\n{failures} sequence(s) FAILED.")
        raise SystemExit(1)
    print("All game logic sequences passed.")


if __name__ == "__main__":
    main()
```

---

## 7.6 Ensure All `eval_*` Scripts Exit with Code 1 on Failure

All evaluation scripts must call `raise SystemExit(1)` (or `sys.exit(1)`) when any check fails. This makes them usable in CI pipelines.

Audit all `eval_*` scripts:
- If any script currently `print`s a failure and continues without exiting with code 1, add `raise SystemExit(1)` at the end of `main()` when `failures > 0`.

---

## 7.7 Standardise Verbose Output Format

All eval scripts must use the same output style:

```
[PASS] description
[FAIL] description: reason
```

Or the simpler two-column format (name: OK / FAIL reason). Pick one and apply it consistently. The simpler format is preferred:

```
  castling kingside: OK
  castling queenside: OK
  en passant: OK
  promotion queen: OK
  ---
  4/4 checks passed.
```

---

## 7.8 Ensure `run_chess_ui.py` Imports from Correct Locations

After Stage 6 moves `QueuedUIBackend` to `src/ui/queued_backend.py`, update `run_chess_ui.py`:

```python
# Before
class QueuedUIBackend:
    ...  # defined inline

# After
from src.ui.queued_backend import QueuedUIBackend
```

Also use the `build_orchestrator` factory consistently.

---

## 7.9 Validate Deployed Model Paths at Startup

When RL is the default, scripts fail immediately with a confusing `FileNotFoundError` if deployed model files are missing (e.g. fresh clone, after cleanup, or after Stage 0 before training completes). Add an explicit startup check with a clear remediation message.

Create `scripts/validate_deployed_models.py`:

```python
"""Validate that all deployed model paths in configs/deployed_models.yaml exist."""
from __future__ import annotations
import sys
from pathlib import Path
from src.utils.io import load_config


def main() -> None:
    cfg = load_config("deployed_models")
    missing = []
    for stage, path in cfg.items():
        if not Path(path).exists():
            missing.append((stage, path))
    if missing:
        print("ERROR: Missing deployed model files:")
        for stage, path in missing:
            print(f"  {stage}: {path}")
        print()
        print("To run with scripted-only mode instead:")
        print("  python scripts/eval_stages.py --use-scripted-only ...")
        sys.exit(1)
    print(f"OK: all {len(cfg)} deployed model files present")


if __name__ == "__main__":
    main()
```

Update `eval_stages.py`, `eval_all_cells_rl.py`, and `eval_chess_piece_move.py` to call this validation at startup before building the controller (when RL mode is active). The check must print the actionable error and exit with code 1 — never silently fall back to scripted mode.

---

## 7.10 Make RL Models the Default in `eval_stages.py` and `eval_chess_piece_move.py`

**Current behaviour**: `eval_stages.py` defaults to scripted-only mode. Passing `--use-rl-models` switches to `ModelEmbeddedController`. This is the wrong default: the integrated RL pipeline is the production path.

**Target behaviour**: RL models are the default. A `--use-scripted-only` flag is the explicit opt-in for the scripted-only path (useful for regression testing if a model is unavailable).

**Changes to `eval_stages.py`**:

```python
# Before
parser.add_argument("--use-rl-models", action="store_true",
                    help="Use ModelEmbeddedController instead of ScriptedController.")

# After
parser.add_argument("--use-scripted-only", action="store_true",
                    help="Use ScriptedController instead of integrated RL models.")
```

In the controller-setup block:
```python
# Before
if args.use_rl_models:
    controller = _load_model_controller(env, cfg)
else:
    controller = ScriptedController(env, ...)

# After
if args.use_scripted_only:
    controller = ScriptedController(env, ...)
else:
    controller = _load_model_controller(env, cfg)
```

Apply the same change to `eval_chess_piece_move.py` and `eval_all_square_moves.py` if they have a similar flag.

**Why**: Guiding Principle #7 — RL is the default. All evaluation and production scripts must default to the integrated RL pipeline. Scripted-only mode is an explicit opt-in, not a default.

---

## 7.11 `eval_all_cells_rl.py` — Confirm RL-Default and Document Modes

`eval_all_cells_rl.py` (created 2026-05-24) already defaults to RL models. Verify it has a module docstring and documents its three modes:

- `pawn` — 48 pawn-move source/destination pairs (48 × 3 reps = 144 episodes)
- `key` — corners, edges, center squares (216 pairs × 3 reps = 648 episodes)
- `full` — every possible source/destination pair on the board (5944 episodes)

Add `--help` output verification to the checklist.

---

## 7.12 `train_rl.py` — Belongs in `scripts/`, Not Production

`scripts/train_rl.py` launches SB3 training. It imports from `training/` — this is correct and intentional (it's a training script, not production code). It must never be imported by `src/`. Verify:

```bash
grep -rn "train_rl\|from scripts.train_rl\|import train_rl" src/ tests/
# Must return nothing
```

---

## 7.13 Script Category Standards

Different script categories have different quality requirements:

| Category | Examples | argparse | main() guard | Exit code 1 | CI-blocking |
|---|---|---|---|---|---|
| `eval_*` | `eval_stages.py`, `eval_all_cells_rl.py`, `eval_targeted.py` | Required | Required | Required | Yes |
| `diagnostics/*` | `diagnose_transit_full.py`, `eval_rl_stages_direct.py` | Preferred | Required | Preferred | No |
| `generate_*` | `generate_scene.py` | Required | Required | Required | Yes |
| `run_*` | `run_chess_ui.py` | Required | Required | n/a | No |
| `train_*.py` | `train_rl.py` | Required | Required | n/a | No |

The Stage 7 validation checklist applies `--help` checks only to `eval_*` scripts; diagnostic scripts under `scripts/diagnostics/` are excluded from CI-blocking checks.

---

## 7.14 Update `eval_targeted.py`

`eval_targeted.py` is a targeted square evaluation script that runs RL models against specific problem squares. After Stage 2 removes the sys.path hack, and after Stage 3.13 creates `deployed_models.yaml`, this script needs one more update.

**Action**:

1. **sys.path hack**: Already handled in Stage 2. Verify it is gone:
   ```bash
   grep "sys.path" scripts/eval_targeted.py
   # Must return nothing
   ```

2. **Config loading update**: After Stage 3.13 splits `training.yaml` into `training.yaml` + `deployed_models.yaml`, update `eval_targeted.py` to load deployed models from the correct config:
   ```python
   # Before (loads from training.yaml which no longer has deployed_models section)
   cfg = load_config("training")
   models = cfg["deployed_models"]

   # After
   models = load_config("deployed_models")
   ```

3. **Module docstring**: If missing, add:
   ```python
   """Targeted RL evaluation for specific problem squares identified in earlier diagnostics."""
   ```

4. **Exit code**: Verify `main()` calls `raise SystemExit(1)` when any evaluation fails. If not, add it.

5. **Smoke test**:
   ```bash
   python scripts/eval_targeted.py --help
   # Must show argparse help without error
   ```

---

## Stage 7 — Full Validation Checklist

```bash
# 1. All eval scripts have main() guards
for f in scripts/eval_*.py; do
    python -c "
import ast, sys
tree = ast.parse(open('$f').read())
has_guard = any(
    isinstance(n, ast.If)
    and isinstance(n.test, ast.Compare)
    and getattr(n.test.left, 'id', '') == '__name__'
    for n in ast.walk(tree)
)
print('$f:', 'OK' if has_guard else 'MISSING main guard')
"
done

# 2. RL is the default — scripted-only is an explicit flag
python scripts/eval_stages.py --help | grep -E "use-scripted-only|use-rl-models"
# Expected: --use-scripted-only present, --use-rl-models absent

# 3. eval_stages.py runs with RL models by default (short sanity check)
PYTHONPATH=. python scripts/eval_stages.py --stages transit --n-episodes 2

# 4. eval_all_cells_rl.py is documented and works
python scripts/eval_all_cells_rl.py --help

# 5. All eval scripts exit with code 1 on failure (manual check: run each and inject failure)
python scripts/eval_draw_conditions.py --verbose
python scripts/eval_special_moves.py --verbose

# 6. New game logic script works
python scripts/eval_game_logic.py --verbose

# 7. No test_*.py in scripts/ (outside diagnostics)
ls scripts/test_*.py 2>/dev/null && echo "FOUND — should be renamed" || echo "None found (OK)"

# 8. Renamed grasp script works
python scripts/eval_grasp_physics.py --help

# 9. validate_deployed_models.py works (rl tier only — requires checkpoint files)
# Run only in rl CI tier, not in fast or sim tiers:
python scripts/validate_deployed_models.py

# 10. train_rl.py not imported by src/
grep -rn "train_rl\|from scripts.train_rl" src/ tests/
# Must return nothing

# 11. Full test suite
python -m pytest tests/ -v

# 12. All eval_* scripts (not diagnostics) show --help without error
for f in scripts/eval_*.py; do python "$f" --help > /dev/null && echo "$f: OK"; done
# Diagnostics are excluded from this CI-blocking check

# 13. eval_targeted.py loads from deployed_models.yaml (not training.yaml)
grep "deployed_models\|training" scripts/eval_targeted.py | head -10
# Should reference load_config("deployed_models"), not load_config("training")

# 14. eval_targeted.py --help works
python scripts/eval_targeted.py --help
```
