# Cleaning Refactor — Overview

**Goal**: Make RoboChess production-ready by removing dead code, eliminating magic numbers, enforcing PEP8 and SOLID principles, hardening configuration, standardising all scripts, and updating documentation to match.

---

## Guiding Principles

1. **No magic numbers** — every constant must live in a YAML config or in a clearly-named class-level attribute loaded from config.
2. **No fallback values** — `cfg.get("key", fallback)` is banned; use `cfg["key"]` so a misconfigured run fails loudly.
3. **Single responsibility** — each module/class does one thing; split if needed, merge only when it reduces duplication without blurring responsibility.
4. **Documented intent** — every file gets a module docstring; every function gets a one-line docstring; comments explain *why*, never *what*.
5. **No path hacks** — scripts must be runnable via `python -m scripts.X` or after `pip install -e .`; no `sys.path.append(os.getcwd())`.
6. **Self-contained stages** — each stage below can be completed, reviewed, and its validation tests run independently before the next begins.

---

## Stage Index

| # | File | Title | Est. complexity |
|---|---|---|---|
| 1 | `01_file_cleanup.md` | Dead-file removal and small-file merges | Low |
| 2 | `02_package_structure.md` | Installable package and import cleanup | Low |
| 3 | `03_config_hardening.md` | Eliminate magic numbers; harden config access | High |
| 4 | `04_docstrings_and_comments.md` | File and function docstrings across all modules | Medium |
| 5 | `05_pep8_and_style.md` | PEP8, line length, import order, naming | Medium |
| 6 | `06_solid_and_duplication.md` | SOLID refactors; eliminate code duplication | High |
| 7 | `07_scripts_overhaul.md` | Scripts standardisation and coverage gaps | Medium |
| 8 | `08_environment_generation.md` | Auto-generation hardening and documentation | Medium |
| 9 | `09_documentation_update.md` | Update `docs/current_status/` to match new code | Medium |

---

## Prerequisites

- All 81 existing tests pass on `main` before any stage begins.
- Run `python -m pytest tests/ -v` to confirm baseline.
- Create and work on branch `chess_refactor_phase` (already exists).

---

## Validation Protocol (applies to every stage)

After completing each stage:

1. `python -m pytest tests/ -v` — must show ≥ 81 tests passing, 0 failures.
2. `python -m pytest tests/ --tb=short -q` — clean summary.
3. Run the stage's specific validation commands listed at the end of its plan file.
4. Commit with message `[cleanup] stage N: <title>`.

---

## What Is Out of Scope

- Changing game logic or arm control algorithms.
- Rewriting the chess engine.
- Modifying the MuJoCo XML assets (other than removing dead ones).
- Adding new features.
