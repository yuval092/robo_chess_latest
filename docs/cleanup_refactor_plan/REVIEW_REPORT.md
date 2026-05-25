# Cleanup Refactor Plan Review Report

Review date: 2026-05-24

Scope reviewed:
- `docs/cleanup_refactor_plan/*.md`
- Current implementation under `src/`, `scripts/`, `training/`, `configs/`, `tests/`

This report focuses on plan defects that can break runtime behavior, invalid assumptions, and additions that would make the refactor safer and more useful.

## Executive Summary

The plan is directionally useful, but several items are unsafe as written. The highest-risk issues are:

1. Stage 1 would delete `shared.xml` and `robot.xml`, but `pick_and_place.xml` includes both. This would break every MuJoCo environment load.
2. Stage 8 would regenerate XML and STL assets on every `import src.chess_env`. That creates write side effects during imports, tests, packaging, and read-only deployments.
3. Stage 10's transfer observation model-load rewrite restores only the unwrapped env observation space, not the wrapper observation space that it mutates, leaving the wrapped env in the wrong observation schema.
4. Stage 2 defines console scripts pointing at `scripts.*` while excluding `scripts` from package discovery. The generated entry points will not be reliable outside editable-root execution.
5. Stage 3 moves validation constants into config but gives code snippets with wrong config paths and incomplete constructor changes.
6. Several "cleanup" merges remove useful module boundaries or dev tools, with little benefit and higher coupling.

## Blocking Findings

### 1. Stage 1 deletes XML files that are required by the active scene

Plan reference: `01_file_cleanup.md`, section 1.8.

The plan says these are unused and should be removed:

```bash
rm chess_env/assets/robot.xml
rm chess_env/assets/shared.xml
```

Current active scene references both:

- `chess_env/assets/pick_and_place.xml:8` includes `shared.xml`
- `chess_env/assets/pick_and_place.xml:12` includes `robot.xml`

Deleting them will make MuJoCo fail to load `pick_and_place.xml`, breaking `gym.make("ChessFetchTask-v0", ...)`, the UI, all eval scripts, and tests that create the environment.

Recommendation: keep `robot.xml` and `shared.xml`. Only `push.xml`, `reach.xml`, and `slide.xml` appear to be standalone demo scenes. Before deleting them, keep the current grep validation and explicitly exclude files included by `pick_and_place.xml`.

### 2. Stage 8 auto-generation on import introduces unsafe write side effects

Plan reference: `08_environment_generation.md`, section 8.1.

The plan adds:

```python
from src.chess_env.environment_generation import regenerate_environment
regenerate_environment()
```

to `src/chess_env/__init__.py`.

Problems:

- Importing `src.chess_env` currently only registers the Gym env. After this change, a plain import writes tracked files under `chess_env/assets/` and potentially `chess_env/stls/chess/`.
- `environment_generation.py` writes to `DEFAULT_SCENE_PATH = Path("chess_env/assets/pick_and_place.xml")`, which is relative to the current working directory. Imports from outside the repo can write to the wrong path or fail.
- Tests, package metadata tools, static analyzers, and `python -c "import src.chess_env"` would mutate the working tree.
- Read-only deployments or installed wheels may not be able to write package asset files.
- The plan claims this is fast `<100ms`, but STL generation can be much more expensive if included. The current code defaults `regenerate_environment(include_stls=False)`, while the plan text says XML and STL meshes regenerate every import.

Recommendation: do not write on import. Add an explicit startup guard in env construction instead, for example:

- `ensure_environment_generated()` called by `ChessSimulationEnv.__init__` before MuJoCo loads XML.
- Use absolute paths derived from `environment_generation.py`, not cwd-relative paths.
- Make it no-op when generated content already matches.
- Gate writes behind config/env var if running in read-only mode.
- Keep `scripts/generate_scene.py all` as the manual command.

### 3. Stage 10 does not restore wrapper observation space after enabling Phase-9

Plan reference: `10_training_production_separation.md`, sections 10.2 and 10.3.

The proposed `enable_transfer_obs(env)` mutates both:

```python
uw.observation_space = TRANSFER_OBS_SPACE
env.observation_space = TRANSFER_OBS_SPACE
```

The proposed `load_model()` restore only does:

```python
self._env._use_transfer_obs = prev_phase9
self._env.observation_space = prev_obs_space
```

It does not restore `self._wrapped_env.observation_space`. Current `ModelEmbeddedController.load_model()` already has a similar partial-restore problem via `WRAPPER_MAP[stage](self._wrapped_env)`, but the plan claims the new version is safe. It is not safe unless all mutated objects are restored.

Runtime impact: after loading a model, the wrapper may still advertise the 25-D transfer observation space while the unwrapped env has the normal observation dict. SB3 wrappers, validation, future resets, and any code inspecting `env.observation_space` can observe inconsistent state.

Recommendation: use a context manager that snapshots and restores all affected state:

```python
@contextmanager
def transfer_obs_enabled(env):
    wrapped_space = getattr(env, "observation_space", None)
    uw = env.unwrapped
    uw_phase9 = uw._use_transfer_obs
    uw_space = uw.observation_space
    try:
        enable_transfer_obs(env)
        yield env
    finally:
        uw._use_transfer_obs = uw_phase9
        uw.observation_space = uw_space
        if wrapped_space is not None:
            env.observation_space = wrapped_space
```

Then load with:

```python
with transfer_obs_enabled(self._wrapped_env):
    self._models[stage] = SAC.load(path, env=self._wrapped_env)
```

### 4. Stage 2 console scripts point to a package that is not installed

Plan reference: `02_package_structure.md`, `pyproject.toml` snippet.

The plan adds:

```toml
[tool.setuptools.packages.find]
include = ["src*", "chess_env*"]

[project.scripts]
robo-chess-ui = "scripts.run_chess_ui:main"
```

But `scripts` is excluded from package discovery. Console entry points import their target modules by name. This may work accidentally in editable mode from the repo root, but it is not a valid installed package contract.

Recommendation: choose one of these:

- Include `scripts*` in package discovery and keep `scripts/__init__.py`.
- Prefer a real package namespace, e.g. `src.cli.run_chess_ui:main`, and move thin CLI modules there.
- Remove `[project.scripts]` and document `python -m scripts.X` as development-only.

Also add `stable-baselines3` and `huggingface_hub` to `pyproject.toml` if training/RL scripts remain part of the project environment; they are in `requirements.txt` but missing from the proposed dependencies.

### 5. Stage 3 config paths for `BoardMapper` are wrong/incomplete

Plan reference: `03_config_hardening.md`, sections 3.2 and 3.9.

Section 3.2 says to add board validation constants to `configs/chess.yaml` at top level:

```yaml
required_cell_size_m: 0.08
```

Section 3.9 loads them from `board_cfg`:

```python
required_cell = float(board_cfg["required_cell_size_m"])
```

Current `configs/chess.yaml` has a `board:` section. The keys should be explicitly placed under `board:`. Otherwise, following section 3.2 literally and section 3.9 literally causes `KeyError`.

The constructor change is also incomplete. Current `BoardMapper.__init__` accepts only `geometry`. Tests or direct callers that instantiate `BoardMapper(geometry)` will break unless defaults/backward compatibility are added or every caller is updated.

Recommendation:

- Put the new keys under `board:`.
- Add `required_board_size: 8` if board size is also meant to be config-driven; otherwise keep `8` as a named chess rule constant, not config.
- Use a dedicated `BoardValidation` dataclass instead of positional constructor arguments.
- Update tests that instantiate `BoardMapper` directly.

### 6. Stage 8 documentation proposes marker names that do not match current markers

Plan reference: `08_environment_generation.md`, sections 8.2 and 8.5.

The plan says generated sections use:

```xml
<!-- BOARD_START --> ... <!-- BOARD_END -->
```

Current implementation uses:

- `<!-- generated board squares start -->`
- `<!-- generated board squares end -->`
- `<!-- generated chess pieces start -->`
- `<!-- generated chess pieces end -->`
- `<!-- generated zone markers start -->`
- `<!-- generated zone markers end -->`

If implementation follows only the docs and changes comments in XML without updating constants, generation will fail with `ValueError` from `text.index(...)`.

Recommendation: either keep existing marker strings in the plan or add an explicit migration step that changes both `pick_and_place.xml` and `environment_generation.py` constants atomically, with a test that `regenerate_scene()` succeeds.

## High-Risk Findings

### 7. Stage 1 deletes `eval_rl_stages.py` before preserving its training-specific use case

Plan reference: `01_file_cleanup.md`, section 1.5.

The plan says `eval_rl_stages.py` is redundant because `eval_stages.py --use-rl-models` exercises production inference. That is true for production integration, but not for training-debug isolation.

`eval_rl_stages.py` imports `training.envs.WRAPPER_MAP` and tests a model directly under the same wrapper shape used during training. That is useful when debugging whether a checkpoint itself is bad versus whether `ModelEmbeddedController` integration is bad.

Recommendation: do not delete this outright. Move it to `scripts/diagnostics/eval_rl_stages_direct.py` or `training/eval_stages_direct.py`, and mark it as a direct training-wrapper diagnostic. Keep production eval in `eval_stages.py`.

### 8. Stage 7 makes RL default everywhere without handling missing model files

Plan reference: `07_scripts_overhaul.md`, section 7.9.

The plan makes RL default and scripted-only opt-in. That matches the project direction, but current `configs/training.yaml` points to checkpoint paths that may not exist in a fresh clone or after cleanup. The repo also has generated checkpoint folders in the working tree.

Impact: basic eval scripts that used to work with scripted controller can fail immediately because deployed model files are missing.

Recommendation:

- After splitting `configs/deployed_models.yaml`, validate paths at startup and emit a clear error with `--use-scripted-only` remediation.
- Add `--allow-scripted-fallback` only for developer diagnostics; do not silently fall back in production.
- Add a small validation command: `python -m scripts.validate_deployed_models`.

### 9. Stage 0 ignores another model file in `archive/rl_system/models/`

Plan reference: `00b_archive_cleanup.md`, section 0.3.

The plan discusses moving `sac-FetchPickAndPlace-v4.zip`, then removing `archive/rl_system/models`. Current archive also contains:

```text
archive/rl_system/models/latest_model.zip
```

The plan says inspect if anything else exists, but it does not define how to classify or preserve it.

Recommendation: add an explicit inventory step:

```bash
find archive/rl_system/models -maxdepth 1 -type f -printf "%f %s\n"
```

Then either move `latest_model.zip` to a named legacy location, document why it is deleted, or checksum it and record the decision.

### 10. Stage 1.4 merge of `config.py` and `logger.py` creates churn and broad break risk for little gain

Plan reference: `01_file_cleanup.md`, section 1.4.

`src.utils.config` is imported throughout `src`, `scripts`, `tests`, and `training`. `src.utils.logger` is used by training callbacks. Merging these into `io.py` forces a broad import rewrite, makes config and logging less discoverable, and breaks external/dev snippets that import `src.utils.config`.

Recommendation: keep `config.py` and `logger.py`, or add backward-compatible shims:

```python
# src/utils/config.py
from src.utils.io import load_config
```

Do not delete the old modules until a later compatibility-removal stage.

### 11. Stage 6 proposes moving test helpers into production code

Plan reference: `06_solid_and_duplication.md`, section 6.1.4.

`NoOpPhysicalExecutor` is useful, but putting a testing stub in `src/physical/plan_executor.py` mixes production execution logic with test-only infrastructure.

Recommendation: place it in one of:

- `src/physical/noop_executor.py` if it is an officially supported headless executor.
- `tests/helpers.py` if it is only for tests.
- `scripts/support/headless_executor.py` if it is only for evaluation scripts.

If it becomes production-supported, define and name its contract explicitly as `HeadlessPhysicalExecutor`, not "NoOp" test vocabulary.

### 12. Stage 6 `GraspPlaceMixin` risks hiding required state dependencies

Plan reference: `06_solid_and_duplication.md`, section 6.2.

Extracting 400 lines of methods from `ChessTaskEnv` into a mixin can reduce file length, but the grasp/place code relies on many implicit attributes (`model`, `data`, `_utils`, constants, `grasp_mode`, active piece state, debug hooks). A mixin does not make those dependencies explicit and can make maintenance harder.

Recommendation: if extracting, prefer a collaborator object with explicit dependencies where practical, or at minimum:

- Keep extraction behavior-preserving only.
- Add focused tests for `execute_grasp`, `execute_place`, pick sequence, place sequence, and failure paths before extraction.
- Do not combine this extraction with observation-builder refactoring in the same stage.

### 13. Stage 6 observation-builder rewrite is too invasive for the cleanup plan

Plan reference: `06_solid_and_duplication.md`, section 6.5.

Replacing `_use_transfer_obs` with injected observation strategies is a legitimate design improvement, but it changes the env construction and training-wrapper contract. It overlaps heavily with Stage 10.

Recommendation: split it into a separate "observation architecture" phase after Stage 10. First create shared `transfer_obs.py` safely, then later replace `_use_transfer_obs` once tests cover:

- normal observation schema
- transfer observation schema
- SB3 model loading
- training wrapper reset/step
- restore behavior after model loading

## Medium-Risk Findings

### 14. Stage 3's "no magic numbers" rule is too broad

Plan reference: `00_overview.md` and `03_config_hardening.md`.

"Every constant must live in YAML config" will push fixed domain facts and schema values into mutable runtime config. Examples:

- Chess board size `8`
- vector dimensions `3`, action shape `4`, transfer observation dimension `25`
- unit conversion `1000.0`
- XML format dimensions that describe generated meshes

Recommendation: separate:

- tunable runtime parameters -> YAML
- domain invariants -> named module constants
- schema dimensions -> named constants near schema
- unit conversions -> named constants, not YAML

The plan already partially acknowledges this for `M_TO_MM`; extend that policy consistently.

### 15. Stage 4 "every function gets a docstring" will create noisy documentation

Plan reference: `04_docstrings_and_comments.md`.

One-line docstrings on every private helper and trivial method can become boilerplate that repeats function names. This adds churn and reduces signal.

Recommendation: require module/class/public API docstrings, and add private helper docstrings only when intent is not obvious or the function encodes domain behavior.

### 16. Stage 5 mypy guidance is informational but not integrated

Plan reference: `05_pep8_and_style.md`.

The plan mentions mypy but does not add mypy config, strictness level, ignored modules, or CI behavior.

Recommendation: either keep mypy out of scope, or add a `[tool.mypy]` config and define whether mypy failures are blocking.

### 17. Stage 7 says all eval scripts must support `--help`, but some diagnostics are intentionally positional/debug tools

Plan reference: `07_scripts_overhaul.md`, validation checklist.

This is fine for `eval_*.py`, but not necessarily for files moved under `scripts/diagnostics/`. The checklist should exclude diagnostics or require a weaker standard for them.

Recommendation: define script categories:

- `eval_*`: CI-capable, argparse, exit codes
- `diagnostics/*`: argparse preferred, not CI-blocking
- `generate_*`: deterministic asset generation
- `run_*`: app entry points

### 18. Documentation update stage misses `scripts/README.md`

Plan reference: `09_documentation_update.md`.

Current `scripts/README.md` still documents the split generation scripts, `visualize_chess_setup.py`, and `test_grasp_physics.py`. Stage 9 only covers `docs/current_status/`.

Recommendation: add `scripts/README.md` to Stage 9.

## Suggested Additions to Improve the Plan

### A. Add pre-refactor characterization tests

Before structural edits, add tests that lock down current behavior:

- `pick_and_place.xml` can be loaded by MuJoCo.
- `ModelEmbeddedController.load_model()` restores env observation state after loading or after a load failure.
- `BoardMapper.from_configs()` expected coordinates for all 64 squares.
- `regenerate_scene()` is idempotent and uses absolute repo paths.
- `run_chess_ui.build_orchestrator()` can construct with scripted and RL modes.

### B. Add import graph checks

Use a lightweight automated check:

- `src/` must not import `training/`
- `src/` must not import `scripts/`
- `training/` may import `src/`
- `scripts/` may import both `src/` and, for training scripts only, `training/`

### C. Add config schema validation

Introduce a small config validation module or typed dataclasses/Pydantic-free validators:

- required keys
- value types
- path existence for deployed models
- geometry consistency

Run this in tests and as `python -m scripts.validate_config`.

### D. Add CI-friendly command groups

Document tiers:

- fast: unit tests that do not create MuJoCo envs
- sim: tests requiring MuJoCo
- rl: model checkpoint presence and SB3
- docs: link/reference checks

This keeps validation useful when models or display are unavailable.

### E. Use compatibility shims for broad module moves

For `models.py`, `config.py`, `logger.py`, and moved UI backend modules, keep deprecation shims for one refactor phase. This reduces breakage for scripts, notebooks, and docs while still allowing the new structure.

### F. Split risky architecture work from cleanup work

Recommended sequencing:

1. Safety fixes and tests.
2. Packaging/import cleanup.
3. Config hardening.
4. Training/production boundary (`transfer_obs.py`) with restore tests.
5. Script standardization.
6. Documentation updates.
7. Optional architecture refactors: task execution extraction, observation strategy, model registry.

## Recommended Plan Edits Before Implementation

1. Amend Stage 1.8: do not delete `robot.xml` or `shared.xml`.
2. Amend Stage 2 `pyproject.toml`: include or move CLI modules; add missing RL dependencies if needed.
3. Amend Stage 3.9: put board validation config under `board:` and use a `BoardValidation` dataclass.
4. Amend Stage 8.1: replace import-time generation with explicit startup-time `ensure_environment_generated()`.
5. Amend Stage 10.3: restore both wrapped and unwrapped observation-space state via a context manager.
6. Move `eval_rl_stages.py` to diagnostics/training diagnostics instead of deleting it.
7. Add `scripts/README.md` to the docs update stage.
8. Add characterization tests before large refactors.

## Verification Performed

This was a static review. I inspected the plan documents and current source files, but did not run the full test suite or execute MuJoCo/RL flows.
