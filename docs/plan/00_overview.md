# Scripts Removal & CLI Migration — Plan Overview

## Goal

Remove the `scripts/` folder entirely. Migrate all evaluation logic worth keeping
into proper CLI entry points under `src/cli/` and a new training CLI under
`training/cli/`. Discard all one-off diagnostic and debug tooling that served its
purpose during development.

---

## Document Index

| File | Content |
|------|---------|
| `00_overview.md` | This file — audit, categorization, entry-point map |
| `01_training_cli.md` | Stage 1 — `training/cli/` + `robo-chess-train` entry point |
| `02_eval_stage.md` | Stage 2 — `robo-chess-eval-stage` command |
| `03_eval_physics.md` | Stage 3 — `robo-chess-eval-physics` command |
| `04_eval_flow.md` | Stage 4 — `robo-chess-eval-flow` command |
| `05_cli_updates.md` | Stage 5 — Impact on existing `src/cli/` commands |
| `06_cleanup.md` | Stage 6 — Delete `scripts/`, update `pyproject.toml` and docs |

---

## Scripts Audit

### Keep (migrate to CLI)

| Script | Destination | Rationale |
|--------|-------------|-----------|
| `scripts/train_rl.py` | `training/cli/main.py` → `robo-chess-train train` | Core training launcher; must survive removal |
| `scripts/eval_stages.py` | `src/cli/eval_stage.py` → `robo-chess-eval-stage` | Production-critical per-stage metrics |
| `scripts/verify_physics.py` | `src/cli/eval_physics.py` → `robo-chess-eval-physics` | Validates sim integrity before any run |
| `scripts/eval_chess_piece_move.py` | `src/cli/eval_flow.py` → `robo-chess-eval-flow --mode simple` | Single-move production-code path |
| `scripts/eval_targeted.py` | `src/cli/eval_flow.py` → `robo-chess-eval-flow --mode complex` | Known-hard-cell coverage |
| `scripts/eval_all_square_moves.py` | `src/cli/eval_flow.py` → `robo-chess-eval-flow --mode full` | Brute-force board sweep |
| `scripts/eval_all_cells_rl.py` | Folded into `robo-chess-eval-stage` (model path) | Redundant with eval_stage once model flag exists |
| `scripts/eval_stress.py` | Folded into `robo-chess-eval-flow --mode complex` | Corner/position stress ⊂ complex mode |
| `scripts/eval_sequence.py` | Folded into `robo-chess-eval-stage` (chain flag) | Stage chaining already handled by eval_stage |
| `scripts/eval_chess_reachability.py` | Folded into `robo-chess-eval-physics` (geometry checks) | Static geometry; belongs in physics verification |

### Discard (debug / dev tooling)

| Script | Reason |
|--------|--------|
| `scripts/diagnostics/` (all) | One-off development diagnostics; no production value |
| `scripts/analyze_debug_log.py` | JSONL log analyser for a specific debug run format; ephemeral |
| `scripts/validate_config.py` | Ad-hoc key presence check; superseded by proper config schema tests |
| `scripts/validate_deployed_models.py` | One-liner path check; superseded by model registry startup validation |
| `scripts/eval_draw_conditions.py` | Chess-logic unit test; belongs in `tests/`, not a CLI tool |
| `scripts/eval_game_logic.py` | Chess-logic unit test; belongs in `tests/`, not a CLI tool |
| `scripts/eval_special_moves.py` | Special-move planning test; belongs in `tests/`, not a CLI tool |
| `scripts/eval_chess_game_flow.py` | Integration test for logical+physical layers; belongs in `tests/` |
| `scripts/eval_grasp_physics.py` | Grasp-specific physics drill; superseded by `robo-chess-eval-physics` |

---

## New Entry Points

```
# Evaluation (src/cli/)
robo-chess-eval-stage    → src.cli.eval_stage:main
robo-chess-eval-physics  → src.cli.eval_physics:main
robo-chess-eval-flow     → src.cli.eval_flow:main

# Training (training/cli/)
robo-chess-train         → training.cli.main:main
```

Combined with the three existing entry points that are unchanged:

```
robo-chess-ui            → src.cli.run_chess_ui:main       (unchanged)
robo-chess-generate      → src.cli.generate_scene:main     (unchanged)
robo-chess-visualize     → src.cli.visualize:main          (unchanged — see Stage 5)
```

---

## Dependency Graph (what imports what)

```
robo-chess-eval-stage
  └── src.chess_env.task_runtime          (episode runner)
  └── src.chess_env.controller            (ScriptedController)
  └── src.chess_env.model_controller      (ModelEmbeddedController)
  └── src.utils.logger

robo-chess-eval-physics
  └── src.chess_env.task               (ChessTaskEnv — bare sim)
  └── src.chess_env.simulation         (step helpers)
  └── src.utils.logger

robo-chess-eval-flow
  └── src.physical.movement_executor   (production movement path)
  └── src.physical.plan_executor       (production plan path)
  └── src.chess_game.board_mapper      (cell → coordinate)
  └── src.chess_env.task_runtime
  └── src.chess_env.controller / model_controller
  └── src.utils.logger

robo-chess-train train
  └── training.trainer.SACTrainer

robo-chess-train eval
  └── src.cli.eval_stage               (reuses eval_stage logic directly)
```

---

## File Layout After Migration

```
src/cli/
├── __init__.py
├── generate_scene.py      (unchanged)
├── run_chess_ui.py        (unchanged)
├── visualize.py           (minor update — see Stage 5)
├── eval_stage.py          (NEW — Stage 2)
├── eval_physics.py        (NEW — Stage 3)
└── eval_flow.py           (NEW — Stage 4)

training/
├── __init__.py
├── trainer.py             (unchanged)
├── callbacks.py           (unchanged)
├── envs/                  (unchanged)
│   ├── transit_env.py
│   ├── descend_env.py
│   └── ascend_env.py
└── cli/                   (NEW — Stage 1)
    ├── __init__.py
    └── main.py
```

---

## Validation Strategy (per stage)

Each stage ends with a validation checklist. The overall acceptance criteria are:

1. `pip install -e .` completes without errors after `scripts/` is removed.
2. All seven entry points respond correctly to `--help`.
3. `robo-chess-eval-stage --stage transit --episodes 2 --controller scripted` runs
   end-to-end and prints a summary table.
4. `robo-chess-eval-physics` runs the settle loop and exits 0 on a clean sim.
5. `robo-chess-eval-flow --mode simple --episodes 1` runs one move end-to-end.
6. `robo-chess-train train --stage transit --timesteps 100` starts training without
   importing anything from `scripts/`.
7. The full test suite (`pytest`) passes.
