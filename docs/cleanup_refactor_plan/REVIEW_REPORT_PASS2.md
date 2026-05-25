# Cleanup Refactor Plan Review Report — Pass 2

Review date: 2026-05-24

Scope reviewed:
- Updated `docs/cleanup_refactor_plan/*.md`
- Current source references needed to validate plan claims

## Summary

The revised plan fixes most of the first-pass blockers:

- `robot.xml` and `shared.xml` are now preserved.
- Direct RL checkpoint evaluation is moved to diagnostics instead of deleted.
- `src.utils.config` / `src.utils.logger` are preserved as shims.
- Import-time scene regeneration was replaced with environment-construction generation.
- transfer observation model-load restoration now uses a context manager.
- Packaging now includes `scripts*` and missing RL dependencies.

Remaining issues are mostly consistency and implementation-detail problems. The highest-risk remaining items are:

1. Stage 8 uses the wrong repo-root path calculation: `Path(__file__).parents[3]` points above the repo.
2. Stage 6 still contains old unsafe `transfer_obs` / partial-restore code in the proposed `ModelRegistry`.
3. Stage 6 and Stage 7 still import `NoOpPhysicalExecutor` from `plan_executor.py` in snippets, while the updated design moves it to `src/physical/noop_executor.py`.
4. Stage 9 still documents the old first-pass behavior: import-time generation, deleting `robot.xml/shared.xml`, and `NoOpPhysicalExecutor` in `plan_executor.py`.
5. Packaging `scripts*` while excluding `training/` packages training-dependent scripts that will fail in a wheel install.
6. The plan still uses the name "transfer_obs"; a more descriptive name should replace it.

## Blocking Findings

### 1. Stage 8 computes the repo root incorrectly

Plan reference: `08_environment_generation.md`, section 8.1.

The proposed helper says:

```python
if not scene_path.is_absolute():
    scene_path = Path(__file__).parents[3] / scene_path
```

For the current file location:

```text
/home/user/projects/robo_chess_latest/src/chess_env/environment_generation.py
```

`Path(__file__).parents[2]` is the repo root:

```text
/home/user/projects/robo_chess_latest
```

`parents[3]` is:

```text
/home/user/projects
```

So the plan would look for or write:

```text
/home/user/projects/chess_env/assets/pick_and_place.xml
```

instead of:

```text
/home/user/projects/robo_chess_latest/chess_env/assets/pick_and_place.xml
```

Recommendation:

```python
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_PATH = _PROJECT_ROOT / "chess_env/assets/pick_and_place.xml"
DEFAULT_STL_DIR = _PROJECT_ROOT / "chess_env/stls/chess"
```

Then remove cwd-relative default paths entirely. Do the same for `configs/` access in `src/utils/io.py`.

### 2. Stage 6 `ModelRegistry` reintroduces the old partial-restore bug

Plan reference: `06_solid_and_duplication.md`, section 6.4.

The revised Stage 10 has the correct context manager:

```python
with transfer_obs_enabled(self._wrapped_env):
    self._models[stage] = SAC.load(path, env=self._wrapped_env)
```

But Stage 6's proposed `ModelRegistry` still does:

```python
prev_phase9 = self._inner_env._use_transfer_obs
prev_obs_space = self._inner_env.observation_space

enable_transfer_obs(self._wrapped_env)
self._models[stage] = SAC.load(str(path), env=self._wrapped_env)

self._inner_env._use_transfer_obs = prev_phase9
self._inner_env.observation_space = prev_obs_space
```

This again fails to restore `self._wrapped_env.observation_space`, and it fails to restore anything if `SAC.load` raises.

Recommendation: make Stage 6 depend on Stage 10 and use the same context manager in `ModelRegistry`. If the naming is changed as recommended below, this should become something like:

```python
from src.chess_env.transfer_obs import transfer_obs_enabled

with transfer_obs_enabled(self._wrapped_env):
    self._models[stage] = SAC.load(str(path), env=self._wrapped_env)
```

### 3. `NoOpPhysicalExecutor` import locations are inconsistent

Plan references:
- `06_solid_and_duplication.md`, sections 6.1.4 and 6.1.5
- `07_scripts_overhaul.md`, section 7.4
- `09_documentation_update.md`, section 9.2

The revised plan correctly creates:

```text
src/physical/noop_executor.py
```

But several snippets still import from the old location:

```python
from src.physical.plan_executor import NoOpPhysicalExecutor
```

Known stale snippets:

- `GameOrchestrator.create_headless()` in Stage 6
- `eval_special_moves.py` example in Stage 7
- `docs/current_status/06_physical_layer.md` instructions in Stage 9

Recommendation: all references should use:

```python
from src.physical.noop_executor import NoOpPhysicalExecutor
```

Also update Stage 6 checklist item 1, which says the only definition should be in `src/physical/plan_executor.py`; it should say `src/physical/noop_executor.py`.

### 4. Stage 9 still contains stale first-pass documentation instructions

Plan reference: `09_documentation_update.md`.

Stage 9 contradicts the revised plan in several places:

- Table says "`NoOpPhysicalExecutor` added to `plan_executor.py`", then later says it is added to `src/physical/noop_executor.py`.
- Overview "How to Run" says assets are auto-generated on every `import src.chess_env`.
- Scene generation section says `regenerate_environment()` is called automatically on import via `__init__.py`.
- XML asset list says remove `robot.xml` and `shared.xml`.
- Config docs list `board.required_cell_size_m`, but Stage 3 now puts those under `board.validation.required_cell_size_m`.

Recommendation: update Stage 9 to match the revised design:

- Scene generation happens in `ChessSimulationEnv.__init__` via `ensure_environment_generated()`.
- `import src.chess_env` must not write files.
- `robot.xml` and `shared.xml` stay in the asset list because `pick_and_place.xml` includes them.
- `NoOpPhysicalExecutor` is documented under `src/physical/noop_executor.py`.
- Board validation docs use `board.validation.*`.

### 5. Packaging `scripts*` conflicts with excluding `training/`

Plan reference: `02_package_structure.md`, section 2.3.

Including `scripts*` fixes console-script entry points, but it also packages:

- `scripts/train_rl.py`, which imports `training.trainer`
- `scripts/diagnostics/eval_rl_stages_direct.py`, which imports `training.envs`
- diagnostic scripts that may import `training` or expect repo-local checkpoint/log paths

At the same time, the plan excludes `training/` from package discovery. In an installed wheel, those packaged script modules can fail on import because their training dependencies are intentionally absent.

Recommendation: choose one clear packaging model:

1. Production wheel only:
   - Move only production CLI entry points to a package such as `src/cli/`.
   - Do not package `scripts*`.
   - Keep `scripts/` as repo-local development tools.

2. Development distribution:
   - Include `scripts*` and `training*`.
   - Mark training dependencies explicitly, possibly with extras like `[project.optional-dependencies] training = [...]`.

3. Hybrid:
   - Package `scripts.run_chess_ui`, `scripts.generate_scene`, and `scripts.visualize` only is awkward with setuptools package discovery; a `src/cli` package is cleaner.

The current plan says "training is excluded" but packages training-dependent script modules. That is an inconsistent install contract.

### 6. Stage 3 training config split omits active training keys and renames `num_envs`

Plan reference: `03_config_hardening.md`, section 3.12.

Current `training/trainer.py` and `training/callbacks.py` read these keys:

- `num_envs`
- `total_timesteps`
- `base_model`
- `eval_freq`
- `n_eval_episodes`
- `learning_rate`
- `batch_size`
- `target_entropy`
- `buffer_size`
- `learning_starts`
- `initial_ent_coef`
- `ent_coef_lr`
- `log_freq`
- `moving_avg_window`

The proposed new `configs/training.yaml` omits several of these and uses `n_envs` instead of the currently consumed `num_envs`.

Impact: training silently changes behavior because many reads still use `.get(..., fallback)` in `training/`, or it breaks once config-hardening is applied to training.

Recommendation:

- Keep the key name `num_envs`, or update `training/trainer.py` in the same stage.
- Preserve every currently consumed training key unless intentionally retired.
- If training config is also hardened, replace `.get()` fallbacks there and add schema validation for all keys.

## High-Risk Findings

### 7. Stage 8 validation command contains invalid Python

Plan reference: `08_environment_generation.md`, validation checklist.

This command is invalid Python:

```python
for k in list(sys.modules): 
    del sys.modules[k] if 'chess_env' in k else None
```

`del` cannot be used as a conditional expression.

Recommendation:

```python
for k in list(sys.modules):
    if "chess_env" in k:
        del sys.modules[k]
```

### 8. Stage 8 import-side-effect test will not reliably catch `Path.write_text`

Plan reference: `08_environment_generation.md`, section 8.6.

The proposed test monkeypatches `builtins.open`:

```python
monkeypatch.setattr('builtins.open', tracking_open)
```

But `pathlib.Path.write_text()` uses `Path.open()`, which delegates through `io.open`, not necessarily `builtins.open`. The test may miss exactly the writes it is trying to detect.

Recommendation: monkeypatch the generation functions or `Path.write_text` directly, or check filesystem mtimes/hashes before and after import:

```python
before = {p: p.stat().st_mtime_ns for p in tracked_paths}
import src.chess_env
after = {p: p.stat().st_mtime_ns for p in tracked_paths}
assert before == after
```

### 9. Stage 3 validation checklist still checks the old board config path

Plan reference: `03_config_hardening.md`, validation checklist.

Stage 3 now puts validation keys under:

```yaml
board:
  validation:
    required_cell_size_m: ...
```

But the checklist still does:

```python
board = cfg['board']
required = ['required_cell_size_m', ...]
for k in required:
    assert k in board
```

Recommendation:

```python
validation = cfg["board"]["validation"]
for k in required:
    assert k in validation
```

### 10. Stage 7 numbering skips 7.10

Plan reference: `07_scripts_overhaul.md`.

The plan goes from 7.9 to 7.11. This is minor but makes cross-references brittle.

Recommendation: renumber `7.11` to `7.10`, `7.12` to `7.11`, etc.

### 11. Stage 7 `eval_game_logic.py` imports unused common args

Plan reference: `07_scripts_overhaul.md`, section 7.5.

The proposed script includes:

```python
from src.utils.args import add_common_args
```

but does not use it. Ruff will flag this under the Stage 5 rules.

Recommendation: remove the import.

### 12. Stage 7 validation for deployed models should not be mandatory in non-RL CI

Plan reference: `07_scripts_overhaul.md`, validation checklist.

The checklist runs:

```bash
python scripts/validate_deployed_models.py
```

That should fail in a fresh checkout if checkpoints are not present. This conflicts with the new CI tiers where the fast tier should not require checkpoints.

Recommendation: put this under the `rl` tier only. For fast CI, verify the script's behavior with a temporary config or monkeypatch rather than requiring real checkpoint files.

## Naming Recommendation: Replace "transfer_obs"

The name "transfer_obs" is historical and not self-describing. It leaks implementation history into production code. The code and docs should use a name that describes what the observation schema does.

Recommended name:

```text
transfer_obs
```

Suggested symbols:

- file: `src/chess_env/transfer_obs.py`
- constant: `TRANSFER_OBS_SPACE`
- function: `enable_transfer_obs(env)`
- context manager: `transfer_obs_enabled(env)`
- flag, if retained temporarily: `_use_transfer_obs`
- future builder: `TransferObsBuilder`

Why this name fits:

- The schema exists to support transfer from FetchPickAndPlace/SB3 pretrained models.
- It is not tied to a project phase number.
- It stays short enough for function and file names.
- It can survive future cleanup better than "transfer_obs".

Alternative names, in order of preference:

1. `transfer_obs` — best balance of meaning and brevity.
2. `fetch_transfer_obs` — more explicit, useful if there will be other transfer schemas.
3. `carry_obs` — describes the "object follows gripper" trick, but less clear for model loading.
4. `grip_as_object_obs` — very precise, but too verbose for module names.

Recommended Stage 10 rewrite:

- Rename `transfer_obs.py` to `transfer_obs.py`.
- Rename `TRANSFER_OBS_SPACE` to `TRANSFER_OBS_SPACE`.
- Replace "transfer observation" in docs with "transfer observation schema".
- Replace `_use_transfer_obs` with `_use_transfer_obs` if you are touching the flag now. If that creates too much churn, keep the old flag temporarily but isolate it behind `transfer_obs.py` and plan the flag rename in the later observation-builder stage.

## Additional Recommendations

### A. Fix Stage 0 validation import path

Stage 0 validation still imports:

```python
from src.utils.config import load_config
```

That works because Stage 1 adds a shim later, but Stage 0 runs before Stage 1. It also works in the current code. No breakage, but if Stage 0 is edited later, keep this sequence dependency clear.

### B. Add a plan-level "stale reference" grep

After plan updates, run:

```bash
rg "phase9|Phase-9|PHASE9|plan_executor import NoOpPhysicalExecutor|on every `import src.chess_env`|robot.xml.*no longer exists|shared.xml.*no longer exists" docs/cleanup_refactor_plan
```

This would catch most of the remaining contradictions.

### C. Keep `GameStatus` shim for one stage

Stage 1 moves `GameStatus` from `models.py` to `chess_service.py` and deletes `models.py`. Since `GameStatus` is imported by `game_orchestrator.py` today, consider preserving `models.py` as a compatibility shim for one stage:

```python
"""Backward-compatible shim for GameStatus."""
from src.chess_game.chess_service import GameStatus
```

This mirrors the safer approach already adopted for `src.utils.config` and `src.utils.logger`.

## Overall Assessment

The revised plan is much safer than the first version. Before implementation, fix the remaining stale snippets and rename "transfer_obs" to a descriptive transfer-observation name. The most important technical correction is Stage 8's `parents[3]` path bug; if implemented as written, scene generation will target the wrong directory.
