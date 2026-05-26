# Stage 4 — `robo-chess-eval-flow`

## Objective

Create `src/cli/eval_flow.py` — an installable CLI entry point that evaluates a
full physical arm flow: moving a chess piece from cell A to cell B. It uses the
**exact same production code paths** as the running game (no patching, no mocks).
Three modes cover the full quality spectrum from a quick sanity check to an
exhaustive board sweep.

---

## Source Analysis

### `scripts/eval_chess_piece_move.py` → Mode: `simple`

- Takes a `piece`, `src`, `dst` square.
- Uses `movement_executor` / `plan_executor` directly (production path).
- Checks: source position, final position, displacement of non-moving pieces.
- Returns per-stage results and error messages.
- ~138 lines; single-move only.

### `scripts/eval_targeted.py` → Mode: `complex`

- Defines a fixed set of "problematic" destinations: a5, c1.
- Adds structural coverage: 4 corners, near-corners, edge midpoints, center (d4/e5).
- 10 source squares spread across the board.
- Configurable repetitions per (src, dst) pair.
- Prints destination success %, source→destination breakdown for failures.
- ~206 lines.

### `scripts/eval_stress.py` → Mode: `complex` (merged)

- Specifically tests 4 corners + center, or an NxN grid of positions.
- Overlaps heavily with targeted; merged into `complex` mode.

### `scripts/eval_all_square_moves.py` → Mode: `full`

- Iterates **all** (src, dst) chess cell pairs (64×63 = 4032 moves).
- Tracks passed/failed per move, per stage.
- Drift limits, piece selection, home-check options.
- Returns `MoveCheckSummary` with failure details.
- ~267 lines.

---

## New File: `src/cli/eval_flow.py`

### argparse specification

```
robo-chess-eval-flow [options]

Mode  (required)
  --mode  {simple,complex,full}   REQUIRED.
    simple    One move: optional --src, --dst (random if not given).
    complex   Targeted set: corners, edges, near-corners, center, known-hard cells.
    full      Brute-force: every (src, dst) combination on the board.

Move selection (simple mode only)
  --src   CELL   Source cell in algebraic notation (e.g. e2).
                 Default: random valid chess cell.
  --dst   CELL   Destination cell in algebraic notation (e.g. e4).
                 Default: random valid chess cell (≠ src).
  --piece PIECE  Piece type to move (pawn, rook, etc.).
                 Default: pawn.

Repetition
  --episodes  INT   Repetitions per (src, dst) pair.
                    Default: 1 for simple/full, 3 for complex.

Controller
  --controller  {scripted,model}   Default: model.
  --transit-model  PATH
  --descend-model  PATH
  --ascend-model   PATH

Tolerances
  --drift-limit  FLOAT   Override drift tolerance (metres).

Output
  --debug        Verbose per-step logs.
  --progress     Show progress bar for long runs (auto-enabled for full mode).
```

### Board cell definitions (used internally)

```python
ALL_CELLS = [f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"]   # 64 cells

CORNER_CELLS = ["a1", "h1", "a8", "h8"]
EDGE_MIDPOINTS = ["a4", "h4", "d1", "d8", "e1", "e8"]
NEAR_CORNERS = ["b1", "b2", "a2", "g1", "g2", "h2",
                "b7", "b8", "a7", "g7", "g8", "h7"]
HARD_DESTINATIONS = ["a5", "c1"]          # historically problematic
CENTER_CELLS = ["d4", "d5", "e4", "e5"]

COMPLEX_DESTINATIONS = list(dict.fromkeys(
    CORNER_CELLS + NEAR_CORNERS + EDGE_MIDPOINTS + HARD_DESTINATIONS + CENTER_CELLS
))
COMPLEX_SOURCES = ["b2", "b7", "g2", "g7", "d3", "e3", "d6", "e6", "c4", "f5"]
```

### Per-move result structure

```python
@dataclass
class MoveResult:
    src: str
    dst: str
    success: bool
    stage_results: dict[str, bool]     # {transit: True, descend: False, ...}
    failure_reason: str
    final_error_m: float
    steps: int
```

### Implementation sketch

```python
"""robo-chess-eval-flow — full arm movement evaluation."""

import argparse
import random
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from src.physical.movement_executor import MovementExecutor
from src.chess_game.board_mapper import BoardMapper
from src.chess_env.controller import ScriptedController
from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_env.model_registry import load_model
from src.utils.logger import get_logger
from src.utils.config import load_config


# ── Data ───────────────────────────────────────────────────────────────────

ALL_CELLS = [f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"]

CORNER_CELLS      = ["a1", "h1", "a8", "h8"]
EDGE_MIDPOINTS    = ["a4", "h4", "d1", "d8", "e1", "e8"]
NEAR_CORNERS      = ["b1", "b2", "a2", "g1", "g2", "h2",
                     "b7", "b8", "a7", "g7", "g8", "h7"]
HARD_DESTINATIONS = ["a5", "c1"]
CENTER_CELLS      = ["d4", "d5", "e4", "e5"]

COMPLEX_DESTINATIONS = list(dict.fromkeys(
    CORNER_CELLS + NEAR_CORNERS + EDGE_MIDPOINTS + HARD_DESTINATIONS + CENTER_CELLS
))
COMPLEX_SOURCES = ["b2", "b7", "g2", "g7", "d3", "e3", "d6", "e6", "c4", "f5"]


@dataclass
class MoveResult:
    src: str
    dst: str
    success: bool
    stage_results: dict[str, bool] = field(default_factory=dict)
    failure_reason: str = ""
    final_error_m: float = 0.0
    steps: int = 0


# ── Core move executor ─────────────────────────────────────────────────────

def _execute_one_move(
    src: str,
    dst: str,
    controller,
    executor: MovementExecutor,
    debug: bool,
    drift_limit: float | None,
) -> MoveResult:
    """Run one full piece move through production code. Returns MoveResult."""
    try:
        result_info = executor.execute_move(
            src_cell=src,
            dst_cell=dst,
            controller=controller,
            drift_limit=drift_limit,
            debug=debug,
        )
        return MoveResult(
            src=src,
            dst=dst,
            success=result_info["success"],
            stage_results=result_info.get("stage_results", {}),
            failure_reason=result_info.get("failure_reason", ""),
            final_error_m=result_info.get("final_error_m", 0.0),
            steps=result_info.get("steps", 0),
        )
    except Exception as exc:
        return MoveResult(
            src=src, dst=dst, success=False,
            failure_reason=f"exception: {exc}",
        )


# ── Modes ─────────────────────────────────────────────────────────────────

def _mode_simple(args, controller, executor, log) -> list[MoveResult]:
    src = args.src or random.choice(ALL_CELLS)
    dst = args.dst
    if not dst:
        candidates = [c for c in ALL_CELLS if c != src]
        dst = random.choice(candidates)

    results = []
    log.info(f"[eval-flow] simple  src={src}  dst={dst}  episodes={args.episodes}")
    for _ in range(args.episodes):
        r = _execute_one_move(src, dst, controller, executor,
                               args.debug, args.drift_limit)
        results.append(r)
    return results


def _mode_complex(args, controller, executor, log) -> list[MoveResult]:
    pairs = [
        (src, dst)
        for dst in COMPLEX_DESTINATIONS
        for src in COMPLEX_SOURCES
        if src != dst
    ]
    episodes = args.episodes if args.episodes != 1 else 3
    total = len(pairs) * episodes

    log.info(
        f"[eval-flow] complex  pairs={len(pairs)}  reps={episodes}  total={total}"
    )
    results = []
    with tqdm(total=total, desc="complex", unit="move") as pbar:
        for src, dst in pairs:
            for _ in range(episodes):
                r = _execute_one_move(src, dst, controller, executor,
                                       args.debug, args.drift_limit)
                results.append(r)
                pbar.update(1)
                ok = sum(x.success for x in results)
                pbar.set_postfix({"ok%": f"{100*ok/len(results):.0f}"})
    return results


def _mode_full(args, controller, executor, log) -> list[MoveResult]:
    pairs = [(s, d) for s in ALL_CELLS for d in ALL_CELLS if s != d]   # 4032
    episodes = args.episodes
    total = len(pairs) * episodes

    log.info(
        f"[eval-flow] full  pairs={len(pairs)}  reps={episodes}  total={total}"
    )
    results = []
    with tqdm(total=total, desc="full sweep", unit="move") as pbar:
        for src, dst in pairs:
            for _ in range(episodes):
                r = _execute_one_move(src, dst, controller, executor,
                                       args.debug, args.drift_limit)
                results.append(r)
                pbar.update(1)
                ok = sum(x.success for x in results)
                pbar.set_postfix({"ok%": f"{100*ok/len(results):.0f}"})
    return results


# ── Summary ────────────────────────────────────────────────────────────────

def _print_summary(results: list[MoveResult], mode: str) -> None:
    total = len(results)
    ok = sum(r.success for r in results)
    print()
    print(f"=== robo-chess-eval-flow  mode={mode}  moves={total} ===")
    print(f"  Success : {ok:5d} / {total}  ({100*ok/max(total,1):.1f}%)")
    print(f"  Failures: {total-ok:5d} / {total}  ({100*(total-ok)/max(total,1):.1f}%)")

    # Per-stage breakdown
    stage_ok: dict[str, int] = {}
    stage_total: dict[str, int] = {}
    for r in results:
        for stage, passed in r.stage_results.items():
            stage_total[stage] = stage_total.get(stage, 0) + 1
            if passed:
                stage_ok[stage] = stage_ok.get(stage, 0) + 1
    if stage_total:
        print()
        print("  Per-stage:")
        for stage, n in stage_total.items():
            ok_s = stage_ok.get(stage, 0)
            print(f"    {stage:10s}  {ok_s:5d}/{n}  ({100*ok_s/n:.1f}%)")

    # Failure reasons
    reasons: dict[str, int] = {}
    for r in results:
        if not r.success and r.failure_reason:
            reasons[r.failure_reason] = reasons.get(r.failure_reason, 0) + 1
    if reasons:
        print()
        print("  Failure reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:<50s}  {count:4d}  ({100*count/max(total,1):.1f}%)")

    # Worst destinations (if enough pairs)
    if len({r.dst for r in results}) > 1:
        dst_ok: dict[str, list[bool]] = {}
        for r in results:
            dst_ok.setdefault(r.dst, []).append(r.success)
        dst_rates = {d: sum(v)/len(v) for d, v in dst_ok.items()}
        worst = sorted(dst_rates.items(), key=lambda x: x[1])[:5]
        print()
        print("  Worst destination cells:")
        for cell, rate in worst:
            print(f"    {cell}  {100*rate:.0f}% ok")

    print()


# ── Entry point ────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robo-chess-eval-flow",
        description="Evaluate full arm piece-move flow using production code.",
    )
    p.add_argument("--mode", required=True, choices=["simple", "complex", "full"],
                   help="Evaluation mode.")
    p.add_argument("--src", metavar="CELL",
                   help="Source cell (simple mode). Default: random.")
    p.add_argument("--dst", metavar="CELL",
                   help="Destination cell (simple mode). Default: random.")
    p.add_argument("--piece", default="pawn",
                   help="Piece type to move (default: pawn).")
    p.add_argument("--episodes", type=int, default=1,
                   help="Repetitions per (src,dst) pair (default: 1; complex default: 3).")
    p.add_argument("--controller", choices=["scripted", "model"], default="model")
    p.add_argument("--transit-model", metavar="PATH")
    p.add_argument("--descend-model", metavar="PATH")
    p.add_argument("--ascend-model", metavar="PATH")
    p.add_argument("--drift-limit", type=float, default=None)
    p.add_argument("--debug", action="store_true",
                   help="Verbose per-step logs.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    log = get_logger("eval-flow", debug=args.debug)

    # Build controller
    if args.controller == "scripted":
        controller = ScriptedController(debug=args.debug)
    else:
        overrides = {}
        if args.transit_model:
            overrides["transit"] = args.transit_model
        if args.descend_model:
            overrides["descend"] = args.descend_model
        if args.ascend_model:
            overrides["ascend"] = args.ascend_model
        models = {
            s: load_model(stage=s, override_path=overrides.get(s))
            for s in ["transit", "descend", "ascend"]
        }
        controller = ModelEmbeddedController(models=models, debug=args.debug)

    cfg = load_config()
    executor = MovementExecutor(cfg=cfg)   # production executor — no mock

    if args.mode == "simple":
        results = _mode_simple(args, controller, executor, log)
    elif args.mode == "complex":
        results = _mode_complex(args, controller, executor, log)
    else:
        results = _mode_full(args, controller, executor, log)

    _print_summary(results, args.mode)
    failed = [r for r in results if not r.success]
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
```

---

## pyproject.toml addition

```toml
robo-chess-eval-flow = "src.cli.eval_flow:main"
```

---

## Scripts superseded

| Script | How covered |
|--------|-------------|
| `scripts/eval_chess_piece_move.py` | `--mode simple` |
| `scripts/eval_targeted.py` | `--mode complex` (destinations + sources list) |
| `scripts/eval_stress.py` | `--mode complex` (corners + center ⊂ COMPLEX_DESTINATIONS) |
| `scripts/eval_all_square_moves.py` | `--mode full` (all 64×63 pairs) |

---

## Validation Checklist

- [ ] `robo-chess-eval-flow --help` shows all options, no import errors.
- [ ] `robo-chess-eval-flow --mode simple --episodes 1` runs one random move end-to-end and prints a summary.
- [ ] `robo-chess-eval-flow --mode simple --src e2 --dst e4 --episodes 1` moves from e2 to e4.
- [ ] `robo-chess-eval-flow --mode simple --src e2` (no --dst) selects a random destination.
- [ ] `robo-chess-eval-flow --mode complex --episodes 1` runs all (COMPLEX_SOURCES × COMPLEX_DESTINATIONS) pairs.
- [ ] `robo-chess-eval-flow --mode full --episodes 1` runs all 4032 pairs with a progress bar.
- [ ] Progress bar advances correctly and shows live ok% for both complex and full modes.
- [ ] Failure reasons are printed in the summary when any move fails.
- [ ] Worst destination cells are listed when multiple destinations are tested.
- [ ] `--controller scripted` runs without model loading.
- [ ] `--debug` produces per-step output.
- [ ] `MovementExecutor` is the production class — no mocking, no patching.
- [ ] Exit code is 1 when any move fails; 0 when all succeed.
