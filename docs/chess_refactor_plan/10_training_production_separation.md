# Stage 10 — Training / Production Separation

**Objective**: Break the one remaining production→training dependency (`model_controller.py` importing from `training/`). After this stage, `src/` contains zero imports from `training/`. The `training/` package remains a standalone development tool whose internal code may import from `src/` (which is correct — trainers use the production env) but is never imported by it.

**Can be done in parallel with Stage 6.** The code changes are localised to `src/chess_env/model_controller.py`, `src/chess_env/task.py`, `training/envs/`, and a new file `src/chess_env/transfer_obs.py`.

---

## 10.1 The Violation

In `src/chess_env/model_controller.py`, `load_model()` contains:

```python
from stable_baselines3 import SAC
from training.envs import WRAPPER_MAP

load_env = WRAPPER_MAP[stage](self._wrapped_env)
self._models[stage] = SAC.load(path, env=load_env)
```

`WRAPPER_MAP[stage]` is one of `TransitTrainEnv`, `DescendTrainEnv`, `AscendTrainEnv`. Each of these wrappers does only two things:

1. Sets `env.unwrapped._use_transfer_obs = True`  (currently named `_use_phase9_obs` — see 10.1b)
2. Replaces `env.unwrapped.observation_space` with the 25-D transfer obs dict space

The transfer observation space shape (`Dict{observation:(25,), achieved_goal:(3,), desired_goal:(3,)}`) is shared by all three stages — it is not stage-specific. Wrapping in a `gym.Wrapper` subclass just to do these two attribute mutations is excessive.

---

## 10.1b Rename Internal Flag in `ChessTaskEnv` (Do This Before 10.2)

**Current code uses the old name `_use_phase9_obs` and `_build_phase9_observation()`.** These must be renamed before `transfer_obs.py` sets `_use_transfer_obs`, otherwise the observation dispatch in `_get_obs()` will silently ignore the transfer flag and the model will receive the wrong observation space.

Perform these renames in `src/chess_env/task.py` **before** creating `transfer_obs.py`:

### Step 1: Rename the flag in `ChessTaskEnv.__init__`

```python
# Before
self._use_phase9_obs = False

# After
self._use_transfer_obs = False
```

### Step 2: Rename the flag check in `_get_obs()`

```python
# Before
if self._use_phase9_obs:
    return self._build_phase9_observation()

# After
if self._use_transfer_obs:
    return self._build_transfer_observation()
```

### Step 3: Rename the method

```python
# Before
def _build_phase9_observation(self) -> dict:
    ...

# After
def _build_transfer_observation(self) -> dict:
    ...
```

### Step 4: Update all internal callers of `_build_phase9_observation()`

```bash
grep -n "_build_phase9_observation\|_use_phase9_obs" src/chess_env/task.py
# Update every occurrence to the new name
```

### Step 5: Update `model_controller.py` (temporary — full fix in 10.3)

In `src/chess_env/model_controller.py`, `_run_stage()` directly sets the flag:
```python
# Before
previous_phase9 = env._use_phase9_obs
env._use_phase9_obs = True
# ... (stage execution) ...
env._use_phase9_obs = previous_phase9

# After (renamed, but still direct setting — context manager replaces this in 10.3)
previous_transfer = env._use_transfer_obs
env._use_transfer_obs = True
# ... (stage execution) ...
env._use_transfer_obs = previous_transfer
```

This temporary intermediate state is replaced entirely by the context manager in section 10.3.

### Step 6: Update all callers of the old flag — training wrappers AND diagnostic scripts

The grep in step 5 identifies every call site. At minimum these must change:

**In `training/envs/transit_env.py`, `descend_env.py`, `ascend_env.py`:**
```python
# Before
uw._use_phase9_obs = True

# After
uw._use_transfer_obs = True
```

**In any diagnostic scripts** (e.g., `scripts/diagnostics/diagnose_*.py`) that directly set `env._use_phase9_obs = True` to trigger transfer obs during debugging: update them to `env._use_transfer_obs = True`. Do not leave them on the old flag — they will silently have no effect after the rename.

```bash
# Find all diagnostic callers (run after grep in step 5 to get exact files):
grep -rn "_use_phase9_obs" scripts/
```

**Reason this rename is critical**: The `enable_transfer_obs()` function in the new `transfer_obs.py` (section 10.2) sets `uw._use_transfer_obs = True`. If `ChessTaskEnv._get_obs()` still checks `self._use_phase9_obs`, the transfer observation is silently ignored and the model gets the wrong observation space. The rename in task.py must happen before `transfer_obs.py` is integrated.

---

## 10.2 Create `src/chess_env/transfer_obs.py`

Create a new file in `src/chess_env/` with no `training/` imports. The key addition is a **context manager** that snapshots and restores all mutated state — both the unwrapped env and the wrapper — so there is no partial-restore bug:

**File: `src/chess_env/transfer_obs.py`**

```python
"""Transfer observation configuration for SAC specialist model inference."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import numpy as np
from gymnasium import spaces


TRANSFER_OBS_SPACE = spaces.Dict({
    "observation":   spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float64),
    "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,),  dtype=np.float64),
    "desired_goal":  spaces.Box(-np.inf, np.inf, shape=(3,),  dtype=np.float64),
})


def enable_transfer_obs(env) -> None:
    """Enable transfer observations on both the wrapper and unwrapped env.

    Transfer obs sets object_pos = grip_pos in the 25-D observation vector so the
    specialist models (trained with FetchPickAndPlace weights) receive an obs
    consistent with the pretrained distribution.
    """
    uw = env.unwrapped
    uw._use_transfer_obs = True
    uw.observation_space = TRANSFER_OBS_SPACE
    env.observation_space = TRANSFER_OBS_SPACE


@contextmanager
def transfer_obs_enabled(env) -> Generator[None, None, None]:
    """Context manager: enables transfer obs on entry, fully restores all state on exit.

    Snapshots and restores both the wrapper observation_space and the unwrapped
    env's observation_space and transfer-observation flag. Use this when loading an
    SB3 model so the production env is unaffected after loading.
    """
    uw = env.unwrapped
    saved_uw_flag = uw._use_transfer_obs       # valid because 10.1b renamed the attribute
    saved_uw_obs_space = uw.observation_space
    saved_wrapper_obs_space = env.observation_space
    try:
        enable_transfer_obs(env)
        yield
    finally:
        uw._use_transfer_obs = saved_uw_flag
        uw.observation_space = saved_uw_obs_space
        env.observation_space = saved_wrapper_obs_space
```

---

## 10.3 Rewrite `load_model()` Without the Training Import

**Current `load_model()` in `model_controller.py`:**

```python
def load_model(self, stage: str, path: str) -> None:
    from stable_baselines3 import SAC
    from training.envs import WRAPPER_MAP

    prev_flag = self._env._use_transfer_obs     # (after 10.1b rename)
    prev_obs_space = self._env.observation_space

    load_env = WRAPPER_MAP[stage](self._wrapped_env)
    self._models[stage] = SAC.load(path, env=load_env)

    self._env._use_transfer_obs = prev_flag
    self._env.observation_space = prev_obs_space
```

**Problem with the current restore**: It only restores `self._env` (unwrapped) but `WRAPPER_MAP[stage](self._wrapped_env)` also mutated `self._wrapped_env.observation_space`. After loading, the wrapper advertises the 25-D transfer obs space while the unwrapped env has the standard space — inconsistent state.

**After (no `training/` import, correct full restore via context manager):**

```python
def load_model(self, stage: str, path: str) -> None:
    """Load a specialist model for one stage."""
    if stage not in self._models:
        raise ValueError(f"Unknown stage: {stage!r}")
    if not path:
        raise ValueError(f"No model path provided for {stage!r}")

    from stable_baselines3 import SAC
    from src.chess_env.transfer_obs import transfer_obs_enabled

    with transfer_obs_enabled(self._wrapped_env):
        self._models[stage] = SAC.load(path, env=self._wrapped_env)

    print(f"[ModelEmbeddedController] Loaded {stage} model from {path}")
```

The context manager guarantees both the wrapper and the unwrapped env are restored to their original state after loading, even if `SAC.load` raises an exception.

Also update `_run_stage()` to remove the direct flag manipulation — after Stage 10 it should use `enable_transfer_obs()` during inference (not a bare attribute set), or the context manager if inference uses a temporary transfer obs window:

```python
# In _run_stage(), remove the old pattern:
# previous_transfer = env._use_transfer_obs
# env._use_transfer_obs = True
# ... step loop ...
# env._use_transfer_obs = previous_transfer

# Replace with enable_transfer_obs() called once at the start of inference
# (the model runs in transfer obs mode; the outer load_model context manager
# already restores state after loading, so _run_stage only needs to enable)
```

The exact pattern depends on whether inference also requires transfer obs to be active during `env.step()`. If yes, use `enable_transfer_obs(env)` before the step loop (without restoring after — the stage completes and the env is reset between stages). If _run_stage is always called within a transfer obs window already established by the caller, no change is needed.

---

## 10.4 Verify the Training Wrappers Still Work Independently

The training wrappers (`TransitTrainEnv`, `DescendTrainEnv`, `AscendTrainEnv`) in `training/envs/` are NOT removed by this change. They still exist and are used by `training/trainer.py` and `make_train_env()` during actual training. The only change is that `model_controller.py` no longer imports them.

After Stage 10, the import graph is:

```
training/ → src/chess_env/transfer_obs.py  (training uses the shared constant)
src/chess_env/model_controller.py → src/chess_env/transfer_obs.py  (production also uses it)
```

Instead of the previous violation:

```
src/chess_env/model_controller.py → training/envs/  ← VIOLATION (now removed)
training/envs/ → src/chess_env/transfer_obs.py        ← correct direction
```

Update `training/envs/transit_env.py` (and descend, ascend) to:
1. Import `TRANSFER_OBS_SPACE` from `src.chess_env.transfer_obs` (no more inline definition)
2. Use `uw._use_transfer_obs = True` (already done in 10.1b Step 6)

```python
# training/envs/transit_env.py — after
from src.chess_env.transfer_obs import TRANSFER_OBS_SPACE

class TransitTrainEnv(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        uw = env.unwrapped
        uw._use_transfer_obs = True
        uw.observation_space = TRANSFER_OBS_SPACE
        self.observation_space = TRANSFER_OBS_SPACE
```

This eliminates the duplication where all three training wrappers defined the same 25-D observation space inline.

---

## 10.5 Confirm No Remaining `training/` Imports in `src/`

```bash
grep -rn "from training\|import training" src/
# Must return nothing after Stage 10
```

Also verify that `_run_stage()` no longer directly sets `_use_transfer_obs` as a bare attribute assignment outside of `enable_transfer_obs()` / `transfer_obs_enabled()`. The flag is now managed exclusively through the `transfer_obs.py` API:

```bash
grep -n "_use_transfer_obs\s*=" src/chess_env/model_controller.py
# Must return nothing (all attribute sets go through transfer_obs.py)
```

---

## Stage 10 — Full Validation Checklist

```bash
# 1. No phase9 naming anywhere in src/ or training/ (rename complete)
grep -rn "_use_phase9_obs\|_build_phase9_observation" src/ training/
# Must return nothing

# 2. No training/ imports in src/
grep -rn "from training\|import training" src/
# Must return nothing

# 3. transfer_obs.py is importable
python -c "from src.chess_env.transfer_obs import enable_transfer_obs, transfer_obs_enabled, TRANSFER_OBS_SPACE; print('OK')"

# 4. _use_transfer_obs is properly initialized in ChessTaskEnv
python -c "
import gymnasium as gym
import src.chess_env
env = gym.make('ChessFetchTask-v0', force_scenario='transit')
env.reset()
uw = env.unwrapped
assert hasattr(uw, '_use_transfer_obs'), 'attribute not found'
assert uw._use_transfer_obs == False, f'expected False, got {uw._use_transfer_obs}'
env.close()
print('_use_transfer_obs initialized correctly')
"

# 5. ModelEmbeddedController loads a model without touching training/
PYTHONPATH=. python -c "
import src.chess_env
import gymnasium as gym
env = gym.make('ChessFetchTask-v0', force_scenario='transit')
from src.chess_env.model_controller import ModelEmbeddedController
ctrl = ModelEmbeddedController(env)
from src.utils.io import load_config
cfg = load_config('deployed_models')
ctrl.load_model('transit', cfg['transit'])
print('Load OK')
env.close()
"

# 6. Training envs still work independently (training/ untouched)
PYTHONPATH=. python -c "
from training.envs import WRAPPER_MAP, make_train_env
print('WRAPPER_MAP stages:', list(WRAPPER_MAP.keys()))
"

# 7. Training wrappers use _use_transfer_obs (not _use_phase9_obs)
grep -n "_use_transfer_obs\|_use_phase9_obs" training/envs/transit_env.py training/envs/descend_env.py training/envs/ascend_env.py
# All occurrences should use _use_transfer_obs

# 8. Both wrapper AND unwrapped obs spaces are fully restored after load
PYTHONPATH=. python -c "
import src.chess_env, gymnasium as gym
env = gym.make('ChessFetchTask-v0', force_scenario='transit')
uw = env.unwrapped
orig_uw_space   = uw.observation_space
orig_uw_flag    = uw._use_transfer_obs
orig_wrap_space = env.observation_space

from src.chess_env.model_controller import ModelEmbeddedController
from src.utils.io import load_config
ctrl = ModelEmbeddedController(env)
cfg = load_config('deployed_models')
ctrl.load_model('transit', cfg['transit'])

assert uw._use_transfer_obs == orig_uw_flag,   'transfer obs flag not restored on unwrapped env'
assert uw.observation_space   == orig_uw_space,   'obs_space not restored on unwrapped env'
assert env.observation_space  == orig_wrap_space,  'obs_space not restored on wrapper env'
print('Full restore OK')
env.close()
"

# 9. Restore works even if SAC.load raises (context manager exits cleanly)
PYTHONPATH=. python -c "
import src.chess_env, gymnasium as gym
env = gym.make('ChessFetchTask-v0', force_scenario='transit')
orig = env.observation_space
from src.chess_env.transfer_obs import transfer_obs_enabled
try:
    with transfer_obs_enabled(env):
        raise RuntimeError('simulated load failure')
except RuntimeError:
    pass
assert env.observation_space == orig, 'obs_space not restored after exception'
print('Exception-path restore OK')
env.close()
"

# 10. No _use_phase9_obs or _build_phase9_observation anywhere (old names fully gone)
grep -rn "_use_phase9_obs\|_build_phase9_observation\|phase9" src/ training/ scripts/ tests/
# Must return nothing. This catches:
#   - Any remaining references in src/ (task.py, model_controller.py)
#   - Training wrappers that were not updated (transit_env.py, descend_env.py, ascend_env.py)
#   - Diagnostic scripts (diagnose_*.py) that may still mutate the old flag

# 11. Full test suite
python -m pytest tests/ -v

# 12. Short RL eval to confirm inference still works end-to-end
PYTHONPATH=. python scripts/eval_stages.py --stages transit --n-episodes 3
```
