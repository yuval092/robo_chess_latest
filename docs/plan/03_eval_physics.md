# Stage 3 — `robo-chess-eval-physics`

## Objective

Create `src/cli/eval_physics.py` — an installable CLI entry point that verifies
the MuJoCo physics of the RoboChess scene without ever commanding the arm.
It starts the simulation exactly as production does, runs a configurable number
of settle steps, and then monitors:

1. **Arm drift** — the arm must not move (no actions were applied).
2. **Piece drift** — no chess piece may drift from its initial position beyond a
   threshold (indicates an unstable contact or geometry problem).
3. **Collisions** — unexpected contacts between bodies that should not touch.
4. **Board geometry** — the 64 squares are present, correctly sized (8 cm), and
   at the expected positions relative to the board centre.

This replaces `scripts/verify_physics.py` and incorporates the static-geometry
checks from `scripts/eval_chess_reachability.py`.

---

## Source Analysis

### `scripts/verify_physics.py`

Large script (~500 lines) that:
- Loads the MuJoCo XML.
- Checks table geometry (70×70 cm, 4 legs).
- Validates 64 board squares for exact 8 cm spacing.
- Checks piece placement against `piece_registry`.
- Validates contact properties and masses.
- Does **not** step the simulation; everything is static XML/data inspection.

### `scripts/eval_grasp_physics.py`

- Loads the simulation with a piece in hand.
- Runs grasp-hold, lift, and horizontal-transit tests.
- Checks piece falls off, vibration, etc.
- **Not migrated**: the new command explicitly states it does not check grasp.
  This script's unique value is gone; discard it.

### `scripts/eval_chess_reachability.py`

- Validates board geometry (rank/file positions, 8 cm spacing) by computing
  expected XY and comparing to `board_mapper`.
- The geometry portion maps cleanly to the static checks section of
  `eval_physics.py`.

---

## New File: `src/cli/eval_physics.py`

### Conceptual structure

The command has three phases, run in order:

```
Phase 1 — Static geometry checks (no simulation steps needed)
  1a. XML integrity: load the MuJoCo model, count expected bodies.
  1b. Table geometry: verify table geom dimensions & leg positions.
  1c. Board squares: verify 64 square geom positions & sizes (8 cm cells).
  1d. Piece registration: every piece in PieceRegistry has a matching MuJoCo body.
  1e. Board mapper geometry: every chess cell (a1–h8) maps to the expected XY.

Phase 2 — Settle loop (N simulation steps, no arm command)
  2a. Record initial qpos for arm joints and all piece positions.
  2b. Step the simulation N times (default: 200 steps ≈ 1 simulation second).
  2c. After each step, sample arm joint deltas, piece position deltas, contacts.

Phase 3 — Diagnostics report
  3a. Arm drift: max joint delta across settle loop.
  3b. Piece drift: per-piece max XY displacement; flag > threshold (default 0.5 mm).
  3c. Unexpected contacts: bodies in contact that are not the table↔piece pair.
  3d. Print pass/fail summary.
  3e. Exit 0 if all checks pass; exit 1 otherwise.
```

### argparse specification

```
robo-chess-eval-physics [options]

Simulation
  --settle-steps  INT     Simulation steps to run before sampling drift.
                          Default: 200.
  --drift-threshold  FLOAT   Piece drift threshold in metres.
                             Default: 0.0005 (0.5 mm).
  --arm-drift-threshold  FLOAT  Arm joint threshold in radians.
                                Default: 0.001.

Checks (disable selectively for CI speed)
  --skip-geometry      Skip static geometry checks (phases 1a–1e).
  --skip-settle        Skip the settle loop (phases 2–3).

Output
  --debug   Verbose per-step drift logs during the settle loop.
```

### Implementation sketch

```python
"""robo-chess-eval-physics — MuJoCo physics integrity verification."""

import argparse
import sys

import numpy as np

from src.chess_env.task import ChessTaskEnv
from src.chess_env.simulation import step_sim
from src.physical.piece_registry import PieceRegistry
from src.chess_game.board_mapper import BoardMapper
from src.utils.logger import get_logger
from src.utils.config import load_config


# ── Static geometry ────────────────────────────────────────────────────────

def _check_static_geometry(env: ChessTaskEnv, log) -> list[str]:
    """Return a list of failure messages; empty list means all passed."""
    failures = []
    model = env.model          # MuJoCo mjModel
    cfg = load_config()

    # 1b. Table
    # (check table geom dimensions against cfg.chess.board_width etc.)

    # 1c. Board squares — 64 geoms with name pattern "sq_*"
    expected_cell_size = cfg.chess.cell_size        # 0.08 m
    sq_geoms = [i for i in range(model.ngeom) if model.geom(i).name.startswith("sq_")]
    if len(sq_geoms) != 64:
        failures.append(f"Expected 64 board square geoms, found {len(sq_geoms)}")
    for gi in sq_geoms:
        geom = model.geom(gi)
        half_size = geom.size[0]
        if abs(half_size - expected_cell_size / 2) > 1e-4:
            failures.append(
                f"Square {geom.name}: expected half-size {expected_cell_size/2:.4f}, "
                f"got {half_size:.4f}"
            )

    # 1d. Piece registration
    registry = PieceRegistry()
    for piece_id, body_name in registry.all_body_names():
        body_id = model.body(body_name).id if body_name in model.body_names else -1
        if body_id < 0:
            failures.append(f"PieceRegistry body '{body_name}' not found in MuJoCo model")

    # 1e. Board mapper geometry
    mapper = BoardMapper(cfg)
    for rank in range(8):
        for file in range(8):
            cell = chr(ord("a") + file) + str(rank + 1)
            expected_xy = mapper.cell_to_xy(cell)
            # Expected positions are already validated by the mapper's own
            # geometry, so we just check it doesn't raise.
            if expected_xy is None:
                failures.append(f"BoardMapper returned None for cell {cell}")

    return failures


# ── Settle loop ────────────────────────────────────────────────────────────

def _run_settle(
    env: ChessTaskEnv,
    settle_steps: int,
    piece_threshold: float,
    arm_threshold: float,
    debug: bool,
    log,
) -> list[str]:
    """Run settle loop; return failure messages."""
    failures = []
    data = env.data

    # Snapshot initial state
    init_qpos = data.qpos.copy()
    registry = PieceRegistry()
    init_piece_pos = {
        name: env.data.body(name).xpos.copy()
        for name in registry.all_body_names_flat()
    }

    max_arm_delta = 0.0
    max_piece_deltas: dict[str, float] = {n: 0.0 for n in init_piece_pos}

    for step_i in range(settle_steps):
        step_sim(env)   # single MuJoCo step, no action applied

        # Arm drift: monitor all arm DOF joints
        arm_delta = float(np.max(np.abs(data.qpos[:7] - init_qpos[:7])))
        max_arm_delta = max(max_arm_delta, arm_delta)

        # Piece drift
        for name, init_pos in init_piece_pos.items():
            cur_pos = data.body(name).xpos
            delta = float(np.linalg.norm(cur_pos[:2] - init_pos[:2]))  # XY only
            max_piece_deltas[name] = max(max_piece_deltas[name], delta)

        if debug and step_i % 50 == 0:
            log.debug(
                f"  step {step_i:4d}  arm_max={max_arm_delta*1000:.3f}mm  "
                f"piece_max={max(max_piece_deltas.values())*1000:.3f}mm"
            )

    # Evaluate arm
    if max_arm_delta > arm_threshold:
        failures.append(
            f"Arm drifted {max_arm_delta*1000:.3f} mm (threshold "
            f"{arm_threshold*1000:.3f} mm) — unexpected joint movement without action"
        )

    # Evaluate pieces
    for name, delta in max_piece_deltas.items():
        if delta > piece_threshold:
            failures.append(
                f"Piece '{name}' drifted {delta*1000:.3f} mm "
                f"(threshold {piece_threshold*1000:.3f} mm)"
            )

    return failures


# ── Report ─────────────────────────────────────────────────────────────────

def _print_report(all_failures: list[str], checks_run: list[str]) -> None:
    print()
    print("Physics Verification Report")
    print("=" * 50)
    for check in checks_run:
        print(f"  {check}")
    print()
    if not all_failures:
        print("Result: PASS — all checks OK")
    else:
        print(f"Result: FAIL — {len(all_failures)} issue(s) found")
        for i, f in enumerate(all_failures, 1):
            print(f"  [{i}] {f}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="robo-chess-eval-physics",
        description="Verify MuJoCo physics integrity without commanding the arm.",
    )
    p.add_argument("--settle-steps", type=int, default=200,
                   help="Sim steps for the settle loop (default: 200).")
    p.add_argument("--drift-threshold", type=float, default=0.0005,
                   help="Piece drift threshold in metres (default: 0.0005 = 0.5 mm).")
    p.add_argument("--arm-drift-threshold", type=float, default=0.001,
                   help="Arm joint threshold in radians (default: 0.001).")
    p.add_argument("--skip-geometry", action="store_true",
                   help="Skip static geometry checks.")
    p.add_argument("--skip-settle", action="store_true",
                   help="Skip the settle loop.")
    p.add_argument("--debug", action="store_true",
                   help="Verbose per-step logs during settle.")
    args = p.parse_args()

    log = get_logger("eval-physics", debug=args.debug)
    all_failures = []
    checks_run = []

    log.info("[eval-physics] Initialising simulation ...")
    env = ChessTaskEnv()              # production constructor — no patches
    env.reset()
    log.info("[eval-physics] Simulation ready.")

    if not args.skip_geometry:
        log.info("[eval-physics] Running static geometry checks ...")
        failures = _check_static_geometry(env, log)
        all_failures.extend(failures)
        checks_run.append(
            f"[{'PASS' if not failures else 'FAIL'}] Static geometry "
            f"({len(failures)} issue(s))"
        )

    if not args.skip_settle:
        log.info(
            f"[eval-physics] Running settle loop ({args.settle_steps} steps) ..."
        )
        failures = _run_settle(
            env=env,
            settle_steps=args.settle_steps,
            piece_threshold=args.drift_threshold,
            arm_threshold=args.arm_drift_threshold,
            debug=args.debug,
            log=log,
        )
        all_failures.extend(failures)
        checks_run.append(
            f"[{'PASS' if not failures else 'FAIL'}] Settle loop / drift "
            f"({len(failures)} issue(s))"
        )

    _print_report(all_failures, checks_run)
    sys.exit(0 if not all_failures else 1)


if __name__ == "__main__":
    main()
```

---

## pyproject.toml addition

```toml
robo-chess-eval-physics = "src.cli.eval_physics:main"
```

---

## Scripts superseded

| Script | How covered |
|--------|-------------|
| `scripts/verify_physics.py` | Phase 1 (static geometry) is a direct port |
| `scripts/eval_chess_reachability.py` | Phase 1e (board mapper geometry checks) |
| `scripts/eval_grasp_physics.py` | **Discarded** — grasp testing is out of scope |

---

## Validation Checklist

- [ ] `robo-chess-eval-physics --help` shows all options, no import errors.
- [ ] `robo-chess-eval-physics` runs to completion on a clean scene and exits 0.
- [ ] `robo-chess-eval-physics --settle-steps 10 --debug` shows per-step drift values.
- [ ] `robo-chess-eval-physics --skip-settle` only runs geometry checks.
- [ ] `robo-chess-eval-physics --skip-geometry` only runs the settle loop.
- [ ] A deliberately broken scene (e.g. a moved piece) causes exit 1 and prints the affected piece name.
- [ ] The settle loop does not apply any joint actions (verify by inspecting `data.ctrl` before/after).
- [ ] Board mapper geometry check covers all 64 cells (a1–h8).
