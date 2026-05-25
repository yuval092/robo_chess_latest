# Cleaning Refactor — Overview

**Goal**: Make RoboChess production-ready by removing dead code, eliminating magic numbers, enforcing PEP8 and SOLID principles, hardening configuration, standardising all scripts, and updating documentation to match. Additionally: cleanly separate training infrastructure from production code, and ensure the integrated RL models are the default execution path everywhere.

---

## Guiding Principles

1. **No magic numbers** — every constant must live in a YAML config or in a clearly-named class-level attribute loaded from config.
2. **No silent fallbacks** — `cfg.get("key", fallback)` with a non-None fallback is banned; use `cfg["key"]` so a misconfigured run fails loudly. *Exception*: multi-level scenario-specific lookups (e.g. `cfg.get("transit_braking_dist", cfg.get("braking_dist"))`) are permitted when the pattern is intentional and documented.
3. **Single responsibility** — each module/class does one thing; split if needed, merge only when it reduces duplication without blurring responsibility.
4. **Documented intent** — every file gets a module docstring; every function gets a one-line docstring; comments explain *why*, never *what*.
5. **No path hacks** — scripts must be runnable via `python -m scripts.X` or after `pip install -e .`; no `sys.path.append(os.getcwd())`.
6. **Training ≠ Production** — `src/` contains only production code. The `training/` package is a separate concern. Production code (including `model_controller.py`) must not import from `training/`.
7. **RL is the default** — all evaluation and production scripts default to the integrated RL pipeline. Scripted-only mode is an explicit opt-in flag (`--use-scripted-only`).
8. **Self-contained stages** — each stage below can be completed, reviewed, and its validation tests run independently before the next begins.

---

## Stage Index

| # | File | Title | Est. complexity |
|---|---|---|---|
| 0 | `00b_archive_cleanup.md` | Archive cleanup and base model relocation | Low |
| 1 | `01_file_cleanup.md` | Dead-file removal and small-file merges | Low |
| 2 | `02_package_structure.md` | Installable package and import cleanup | Low |
| 3 | `03_config_hardening.md` | Eliminate magic numbers; harden config access | High |
| 4 | `04_docstrings_and_comments.md` | File and function docstrings across all modules | Medium |
| 5 | `05_pep8_and_style.md` | PEP8, line length, import order, naming | Medium |
| 6 | `06_solid_and_duplication.md` | SOLID refactors; eliminate code duplication | High |
| 7 | `07_scripts_overhaul.md` | Scripts standardisation, RL-as-default, coverage gaps | Medium |
| 8 | `08_environment_generation.md` | Auto-generation hardening and documentation | Medium |
| 9 | `09_documentation_update.md` | Update `docs/current_status/` to match new code | Medium |
| 10 | `10_training_production_separation.md` | Break production→training dependency; clean `training/` | High |
| — | *(future)* `obs_builder_strategy.md` | Replace `_use_transfer_obs` flag with injected `ObsBuilder` | High (deferred) |

**Stages must be completed in order.** Stage 0 before Stage 1 (archive models are referenced). Stage 10 can be done alongside Stage 6. Stage 3.12 (`deployed_models.yaml` split) depends on Stage 10 being complete (both change `model_controller.py`'s config loading). The ObsBuilder stage is deferred until after Stage 10 validates the transfer observation boundary.

---

## Prerequisites

- All existing tests pass on `main` before any stage begins.
- Run `python -m pytest tests/ -v` to confirm baseline.
- Create and work on branch `chess_refactor_phase` (already exists).

---

## Validation Protocol (applies to every stage)

After completing each stage:

1. `python -m pytest tests/ -v` — must show ≥ existing test count passing, 0 failures.
2. `python -m pytest tests/ --tb=short -q` — clean summary.
3. Run the stage's specific validation commands listed at the end of its plan file.
4. Commit with message `[cleanup] stage N: <title>`.

---

## Pre-Refactor Characterization Tests (Run Before Stage 1)

Add these tests before any structural edits to lock down current behavior and catch regressions:

```bash
# tests/chess_env/test_characterization.py
```

Minimum coverage required before starting Stage 1:
1. `pick_and_place.xml` loads in MuJoCo without error.
2. `ModelEmbeddedController.load_model()` restores both wrapper and unwrapped env observation space after loading (and after a failed load).
3. `BoardMapper.from_configs()` returns the expected (x, y) for all 64 squares.
4. `ensure_environment_generated()` is idempotent and uses absolute paths.
5. `import src.chess_env` (without constructing an env) writes no files.

---

## Stale-Reference Grep (Run After Every Plan Edit)

Before implementing any stage, run this to catch plan contradictions:

```bash
rg "Assets are auto-generated on every \`import src.chess_env\`|robot\.xml.*no longer exists|shared\.xml.*no longer exists|parents\[3\]" docs/chess_refactor_plan/ -g '*.md' -g '!00_overview.md'
```

Must return nothing. If it returns any matches, fix them before implementing the stage.

---

## Import Graph Enforcement

After Stage 10, add a lightweight automated check (run in CI):

```bash
# scripts/check_import_graph.py
# Allowed directions:
#   src/ → (nothing outside src/ except gymnasium, numpy, etc.)
#   training/ → src/
#   scripts/ → src/ and (for training scripts) training/
# Forbidden:
#   src/ → training/
#   src/ → scripts/
```

The check uses `ast.walk` to scan all `import` statements and verifies no `src/` module imports `training/`. This catches future regressions automatically.

---

## Config Schema Validation

After Stage 3, add `scripts/validate_config.py`:
- Required keys present in each YAML file.
- Value types (float, int, list) match expectations.
- Path existence for deployed models.
- Board geometry consistency (`required_cell_size_m == cell_size_m`).

Run this in the test suite: `python -m pytest tests/test_config_schema.py`.

---

## CI Test Tiers

Document and enforce test tiers so the suite remains useful when models or displays are unavailable:

| Tier | Command | Requires |
|---|---|---|
| fast | `pytest tests/ -m "not mujoco and not rl"` | No MuJoCo, no checkpoints |
| sim | `pytest tests/ -m "mujoco"` | MuJoCo installed |
| rl | `pytest tests/ -m "rl"` | Checkpoints + MuJoCo |

Mark tests with `@pytest.mark.mujoco` and `@pytest.mark.rl` accordingly. The `fast` tier must always pass in CI.

---

## What Is Out of Scope

- Changing game logic or arm control algorithms.
- Rewriting the chess engine.
- Modifying the MuJoCo XML assets (other than removing dead ones).
- Adding new features.
- Changing reward functions or training hyperparameters (those live in dedicated training sessions).

---

## Current State Snapshot (as of 2026-05-25)

Active components not present in the original plan:
- `src/chess_env/model_controller.py` — production RL inference controller (new); currently imports `from training.envs import WRAPPER_MAP` (violation fixed in Stage 10)
- `src/chess_env/task.py` — uses `self._use_phase9_obs = False` and `_build_phase9_observation()` — renamed to `_use_transfer_obs` / `_build_transfer_observation()` in Stage 10
- `training/` — top-level training package: `trainer.py`, `callbacks.py`, `envs/` with three stage wrappers (new)
- `configs/training.yaml` — training hyperparameters and deployed model paths (not yet split; split done in Stage 3.12); still has `base_model: "archive/rl_system/models/..."` (fixed in Stage 0)
- `scripts/train_rl.py` — training launch script (new)
- `scripts/eval_all_cells_rl.py` — full-board RL evaluation (new)
- `scripts/eval_targeted.py` — targeted square RL evaluation for specific problem squares; has `main()` + argparse; sys.path hack present; loads from `training.yaml["deployed_models"]` (fix in Stage 7)
- `scripts/eval_rl_stages.py` — standalone RL stage evaluator, partially redundant with `eval_stages.py --use-rl-models` (new, to be moved to diagnostics in Stage 1)
- `scripts/diagnose_transit_full.py`, `diagnose_descend_afile.py`, `diagnose_transit_c2.py` — diagnostic scripts (new, to be organised in Stage 1)
- `game_orchestrator.py` — imports from `src.chess_game.models` (old shim) and `src.utils.config` (old shim); both fixed in Stage 1/2
- `model_controller.py` — has duplicate env-unwrapping loop in `transition()` method (fixed in Stage 6) and uses `.get("stability_vel_threshold", 0.02)` / `.get("eval_drift_limit", 0.010)` with unnecessary fallbacks (keys already exist in `env.yaml`; fixed in Stage 3)
