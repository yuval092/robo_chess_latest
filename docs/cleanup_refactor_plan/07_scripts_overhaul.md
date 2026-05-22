# Stage 7 — Scripts Overhaul

**Objective**: Every script in `scripts/` must: have a module docstring, have a `main()` function guarded by `if __name__ == "__main__"`, use argparse where applicable, work correctly end-to-end, and cover every logical part of the project.

---

## 7.1 Scripts Inventory and Status

After Stage 1 (file cleanup), the scripts folder contains:

| Script | Status | Issues |
|---|---|---|
| `analyze_debug_log.py` | ✓ Has argparse + main | Long lines; needs docstring |
| `debug_one_move.py` | ✓ Has argparse + main | Path hack (fixed in Stage 2) |
| `eval_chess_game_flow.py` | ✓ Has argparse + main | Path hack; needs docstring |
| `eval_chess_piece_move.py` | ✓ Has argparse + main | Path hack; needs docstring |
| `eval_chess_reachability.py` | ✓ Has argparse + main | Path hack; magic numbers (fixed in Stage 3) |
| `eval_draw_conditions.py` | ✗ No argparse | No main guard; path hack |
| `eval_sequence.py` | ✓ Has argparse + main | Path hack |
| `eval_special_moves.py` | ✗ No argparse | No main guard; path hack |
| `eval_stages.py` | ✓ Has argparse + main | Path hack |
| `eval_stress.py` | ✓ Has argparse + main | Path hack |
| `generate_scene.py` | NEW — created in Stage 1 | Clean |
| `run_chess_ui.py` | ✓ Has argparse + main | Path hack; QueuedUIBackend moved in Stage 6 |
| `test_grasp_physics.py` | ✓ Has argparse + main | Path hack; naming (should be eval_*) |
| `verify_physics.py` | ✓ Has argparse + main | Path hack |
| `visualize.py` | ✓ Has argparse + main (updated Stage 1) | Path hack (fixed Stage 2) |

---

## 7.2 Rename `test_grasp_physics.py` → `eval_grasp_physics.py`

Scripts prefixed with `test_` are confusing — pytest will attempt to collect them as tests. Rename to `eval_grasp_physics.py` to align with the `eval_*` naming convention.

```bash
git mv scripts/test_grasp_physics.py scripts/eval_grasp_physics.py
```

Update any references:
```bash
grep -rn "test_grasp_physics" docs/ scripts/ tests/
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
from src.physical.plan_executor import NoOpPhysicalExecutor


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
from src.utils.args import add_common_args


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

# 2. All eval scripts exit with code 1 on failure (manual check: run each and inject failure)
python scripts/eval_draw_conditions.py --verbose
python scripts/eval_special_moves.py --verbose

# 3. New script works
python scripts/eval_game_logic.py --verbose

# 4. No test_*.py in scripts/
ls scripts/test_*.py 2>/dev/null && echo "FOUND — should be renamed" || echo "None found (OK)"

# 5. Renamed script works
python scripts/eval_grasp_physics.py --help

# 6. Full test suite
python -m pytest tests/ -v

# 7. All eval scripts show --help
for f in scripts/eval_*.py; do python "$f" --help > /dev/null && echo "$f: OK"; done
```
