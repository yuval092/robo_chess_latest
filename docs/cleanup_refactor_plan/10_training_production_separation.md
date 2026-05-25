# Stage 10 — Training / Production Separation

**Objective**: Break the one remaining production→training dependency (`model_controller.py` importing from `training/`). After this stage, `src/` contains zero imports from `training/`. The `training/` package remains a standalone development tool whose internal code may import from `src/` (which is correct — trainers use the production env) but is never imported by it.

**Can be done in parallel with Stage 6.** The code changes are localised to `src/chess_env/model_controller.py` and a new file `src/chess_env/transfer_obs.py`.

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

1. Sets `env.unwrapped._use_transfer_obs = True`
2. Replaces `env.unwrapped.observation_space` with the 25-D transfer obs dict space

The transfer observation space shape (`Dict{observation:(25,), achieved_goal:(3,), desired_goal:(3,)}`) is shared by all three stages — it is not stage-specific. Wrapping in a `gym.Wrapper` subclass just to do these two attribute mutations is excessive.

---

## 10.2 Create `src/chess_env/transfer_obs.py`

Create a new file in `src/chess_env/` with no `training/` imports. The key addition is a **context manager** that snapshots and restores all mutated state — both the unwrapped env and the wrapper — so there is no partial-restore bug:

**File: `src/chess_env/transfer_obs.py`**

```python
"""transfer observation configuration for SAC specialist model inference."""
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
    saved_uw_flag = uw._use_transfer_obs
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

**Internal flag migration**: In the same stage, update `ChessTaskEnv` so the public API, internal flag, and method names all use transfer-observation terminology:

```python
# In ChessTaskEnv.__init__
self._use_transfer_obs = False

# In ChessTaskEnv._get_obs()
if self._use_transfer_obs:
    return self._build_transfer_observation()
```

Rename the existing transfer-observation builder method to `_build_transfer_observation()`, and update all training wrappers and model-loading helpers to set `_use_transfer_obs`.

---

## 10.3 Rewrite `load_model()` Without the Training Import

**Current `load_model()` in `model_controller.py`:**

```python
def load_model(self, stage: str, path: str) -> None:
    from stable_baselines3 import SAC
    from training.envs import WRAPPER_MAP

    prev_flag = self._env._use_transfer_obs
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

---

## 10.4 Verify the Training Wrappers Still Work Independently

The training wrappers (`TransitTrainEnv`, `DescendTrainEnv`, `AscendTrainEnv`) in `training/envs/` are NOT affected by this change. They still exist and are used by `training/trainer.py` and `make_train_env()` during actual training. The only change is that `model_controller.py` no longer imports them.

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

Update `training/envs/transit_env.py` (and descend, ascend) to import `TRANSFER_OBS_SPACE` from `src.chess_env.transfer_obs` instead of defining it inline:

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

---

## Stage 10 — Full Validation Checklist

```bash
# 1. No training/ imports in src/
grep -rn "from training\|import training" src/
# Must return nothing

# 2. transfer_obs.py is importable
python -c "from src.chess_env.transfer_obs import enable_transfer_obs, TRANSFER_OBS_SPACE; print('OK')"

# 3. ModelEmbeddedController loads a model without touching training/
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

# 4. Training envs still work independently (training/ untouched)
PYTHONPATH=. python -c "
from training.envs import WRAPPER_MAP, make_train_env
print('WRAPPER_MAP stages:', list(WRAPPER_MAP.keys()))
"

# 5. Both wrapper AND unwrapped obs spaces are fully restored after load
PYTHONPATH=. python -c "
import src.chess_env, gymnasium as gym
env = gym.make('ChessFetchTask-v0', force_scenario='transit')
uw = env.unwrapped
orig_uw_space   = uw.observation_space
orig_uw_flag  = uw._use_transfer_obs
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

# 5b. Restore works even if SAC.load raises (context manager exits cleanly)
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

# 6. Full test suite
python -m pytest tests/ -v

# 7. Short RL eval to confirm inference still works end-to-end
PYTHONPATH=. python scripts/eval_stages.py --stages transit --n-episodes 3
```
