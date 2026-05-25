# Cleanup Refactor Plan Review Report — Pass 3

Review date: 2026-05-24

Scope reviewed:
- Updated `docs/cleanup_refactor_plan/*.md`
- Targeted checks against current source behavior and file layout

## Summary

The pass-2 issues are mostly fixed:

- `transfer_obs` naming is now used in the active plan.
- Stage 8 now uses `parents[2]` instead of `parents[3]`.
- Stage 6 `ModelRegistry` now uses the restore context manager.
- Most `NoOpPhysicalExecutor` references now point to `src.physical.noop_executor`.
- Stage 7 model validation is marked as RL-tier only.
- Stage 3 validation now checks `board.validation.*`.

Remaining issues are narrower, but a few are still blocking if implemented literally.

## Blocking Findings

### 1. Stage 2 `src.cli` forwarding modules import excluded `scripts`

Plan reference: `02_package_structure.md`, section 2.3.

The plan excludes `scripts*` from package discovery:

```toml
include = ["src*", "chess_env*"]
```

but proposes packaged entry points:

```toml
robo-chess-ui = "src.cli.run_chess_ui:main"
```

with forwarding modules:

```python
# src/cli/run_chess_ui.py
from scripts.run_chess_ui import main
```

This still breaks in a wheel install because `scripts` is not packaged. The entry point imports `src.cli.run_chess_ui`, which imports `scripts.run_chess_ui`, which is unavailable.

Recommendation: do not make `src/cli` wrappers import `scripts`. Move the production CLI implementation into package modules, then optionally make repo-local scripts forward to those:

```text
src/cli/run_chess_ui.py      # real implementation
src/cli/generate_scene.py    # real implementation
src/cli/visualize.py         # real implementation
scripts/run_chess_ui.py      # thin dev wrapper importing src.cli.run_chess_ui
```

This preserves installable entry points and keeps `scripts/` development-only.

### 2. Stage 10 assumes `_use_transfer_obs` exists before it is introduced

Plan reference: `10_training_production_separation.md`.

Current code uses:

```python
self._use_phase9_obs = False
if self._use_phase9_obs:
    return self._build_phase9_observation()
```

Current training wrappers set:

```python
uw._use_phase9_obs = True
```

The updated plan now writes snippets as if the current code already uses:

```python
uw._use_transfer_obs = True
```

That attribute does not exist yet. Creating `transfer_obs.py` and changing wrappers is not enough; `ChessTaskEnv` must also be migrated, or `enable_transfer_obs()` will set an unused flag and `_get_obs()` will continue returning the standard observation schema.

Recommendation: Stage 10 must include an explicit compatibility/migration step:

Option A, full rename in Stage 10:

```python
self._use_transfer_obs = False

def _get_obs(self):
    if self._use_transfer_obs:
        return self._build_transfer_observation()
```

and rename `_build_phase9_observation()` to `_build_transfer_observation()`.

Option B, lower-churn compatibility:

```python
def enable_transfer_obs(env) -> None:
    uw = env.unwrapped
    uw._use_phase9_obs = True  # current internal flag, kept temporarily
    uw.observation_space = TRANSFER_OBS_SPACE
    env.observation_space = TRANSFER_OBS_SPACE
```

Then schedule the internal flag rename for the later observation-builder stage. Do not set `_use_transfer_obs` unless `ChessTaskEnv._get_obs()` reads it.

### 3. Stage 8 proposes invalid XML comments

Plan reference: `08_environment_generation.md`, section 8.5.

The proposed top-level XML comment contains literal nested XML comments:

```xml
<!--
  ...
    <!-- generated board squares start --> ... <!-- generated board squares end -->
  ...
-->
```

XML comments cannot contain `--`, and nested `<!-- ... -->` markers inside a comment make the XML invalid. This can break MuJoCo XML loading.

Recommendation: avoid literal comment syntax inside the explanatory comment. For example:

```xml
<!--
  Managed marker pairs:
    generated board squares start/end
    generated chess pieces start/end
    generated zone markers start/end

  Do not edit generated sections manually.
-->
```

The actual marker comments should remain only where they delimit generated content.

### 4. Stage 9 still says `robot.xml` and `shared.xml` no longer exist

Plan reference: `09_documentation_update.md`, `docs/current_status/04_scene_generation.md` instructions.

The plan still says:

```text
Remove `push.xml`, `reach.xml`, `slide.xml`, `robot.xml`, `shared.xml` from the file listing. They no longer exist.
```

This contradicts Stage 1, which correctly keeps `robot.xml` and `shared.xml` because `pick_and_place.xml` includes both.

Recommendation: Stage 9 should say:

- Remove `push.xml`, `reach.xml`, `slide.xml`.
- Keep `robot.xml` and `shared.xml` in the asset list and document that `pick_and_place.xml` includes them.

Also update the Stage 9 change table from "`5 unused XML assets deleted`" to "`3 unused XML demo scenes deleted`".

## High-Risk Findings

### 5. Stage 6 validation command for absence from `plan_executor.py` is wrong

Plan reference: `06_solid_and_duplication.md`, validation checklist.

The checklist says:

```bash
# Ensure it is NOT importable from plan_executor (wrong location):
python -c "from src.physical.noop_executor import NoOpPhysicalExecutor" 2>&1 | grep "ImportError" && echo "correctly absent from plan_executor"
```

This imports from `noop_executor`, not `plan_executor`, so it tests the opposite of what the comment says.

Recommendation:

```bash
python -c "from src.physical.plan_executor import NoOpPhysicalExecutor" 2>&1 | grep "ImportError" && echo "correctly absent from plan_executor"
```

Also update checklist item 1: it still says the only definition should be in `src/physical/plan_executor.py`; it should say `src/physical/noop_executor.py`.

### 6. Stage 8 STL constants comment still says regeneration happens on import

Plan reference: `08_environment_generation.md`, section 8.3.

The comment block says:

```python
# Changing these values requires re-running regenerate_stls() (done automatically
# on import via regenerate_environment()).
```

The revised plan explicitly avoids import-time generation. This comment should not be copied into code.

Recommendation:

```python
# Changing these values requires re-running regenerate_stls(); use:
#   python scripts/generate_scene.py stls
```

or note that XML generation is checked on environment construction, while STL regeneration is manual unless `include_stls=True` is explicitly used.

### 7. Stage 10 validation may fail before the migration initializes the new flag

Plan reference: `10_training_production_separation.md`, validation checklist.

The checklist reads:

```python
orig_uw_flag = uw._use_transfer_obs
```

This raises `AttributeError` unless Stage 10 also changes `ChessTaskEnv.__init__` to initialize `_use_transfer_obs`. This is the same root issue as finding #2, but it also affects validation.

Recommendation: either initialize `_use_transfer_obs` in `ChessTaskEnv`, or if using the compatibility approach, validate `_use_phase9_obs` until the later flag rename.

### 8. Stage 2 dependency guidance is internally ambiguous

Plan reference: `02_package_structure.md`, section 2.3.

The proposed `dependencies` includes:

```toml
"stable-baselines3",
"huggingface_hub",
```

and then also defines:

```toml
[project.optional-dependencies]
training = ["stable-baselines3", "huggingface_hub"]
```

The text says move them from optional to dependencies if inference is always expected; but the snippet already has them in both places.

Recommendation: choose one:

- If RL inference is production-default, keep SB3 in main dependencies and remove it from the `training` extra or add only truly training-only packages to the extra.
- If scripted-only install is supported, remove SB3 from main dependencies, guard RL imports, and define an `rl` or `training` extra.

Given the plan says RL is default, the simplest route is: keep `stable-baselines3` in main dependencies and drop the duplicate optional entry unless more training-only dependencies are added.

## Medium-Risk Findings

### 9. Stage 9 duplicates `NoOpPhysicalExecutor` in the change table

Plan reference: `09_documentation_update.md`, section 9.1.

The table lists:

```text
NoOpPhysicalExecutor added to src/physical/noop_executor.py
```

twice. This is harmless but noisy.

Recommendation: remove the duplicate row.

### 10. Stage 8 import-side-effect mtime test tracks a directory

Plan reference: `08_environment_generation.md`, section 8.6.

The test tracks:

```python
Path("chess_env/stls/chess")
```

Directory mtimes can change for reasons unrelated to file content and may not change when file contents are rewritten in place. Since the import must not write at all, it is better to track concrete files.

Recommendation:

```python
tracked = [
    Path("chess_env/assets/pick_and_place.xml"),
    *Path("chess_env/stls/chess").glob("*.stl"),
]
before = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in tracked}
```

### 11. Review reports now pollute stale-reference greps

Plan reference: `00_overview.md`, stale-reference grep.

The plan suggests grepping `docs/cleanup_refactor_plan/`, but this folder includes historical review reports that intentionally mention stale terms such as `Phase-9`, `parents[3]`, and old bad imports.

Recommendation: exclude review reports:

```bash
rg "..." docs/cleanup_refactor_plan -g '*.md' -g '!REVIEW_REPORT*.md'
```

## Overall Assessment

The plan is close. The biggest remaining implementation blocker is the Stage 10 flag mismatch: `transfer_obs.py` must either keep using the current internal `_use_phase9_obs` flag temporarily or migrate `ChessTaskEnv` in the same stage. The second important blocker is Stage 2's `src.cli` wrapper design; packaged modules cannot forward to excluded `scripts`.

Once those are fixed, the plan is structurally coherent enough to implement stage by stage.
