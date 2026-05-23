# Observation Space & Environment Design

## 1. The Phase-9 Observation — Why It Works

Our environment (`ChessTaskEnv`) inherits from `MujocoFetchPickAndPlaceEnv` via `ChessSimulationEnv`.
The pretrained `sac-FetchPickAndPlace-v4.zip` was trained with a specific 25D flat observation and
a `MultiInputPolicy` over `{observation, achieved_goal, desired_goal}`. If we feed it a different
observation structure, the loaded weights produce garbage actions.

The **"Holding Object" trick**: FetchPickAndPlace always had a cube near the gripper. We fool the
model by setting `object_pos = grip_pos`. This tells the model "you're holding an object right now",
which activates the stable, momentum-aware transport dynamics learned during its original training.
Without this, the model acts as if the cube is elsewhere and behaves erratically.

The observation vector shape must be **exactly (25,)** to match the pretrained weight dimensions.

## 2. Changes to `ChessTaskEnv` — `_get_obs()` and `observation_space`

### 2a. New `_build_phase9_observation()` method

Add this private method to `src/chess_env/task.py`:

```python
def _build_phase9_observation(self):
    """
    Constructs the 25-dimensional Phase-9 observation vector.

    This format matches the FetchPickAndPlace-v4 pretrained model's expected input.
    The "Holding Object" trick (object_pos = grip_pos) activates its transport dynamics.

    Vector layout:
        0-2:   grip_pos           (current 3D gripper position)
        3-5:   object_pos         (TRICK: set to grip_pos, not real cube)
        6-8:   object-to-goal     (goal - grip_pos: the navigation signal)
        9-10:  gripper_state      (zeroed: masked out)
        11-13: object_rot         (zeroed: masked out, scenario_id removed)
        14-19: object_vel         (zeroed: masked out)
        20-22: grip_velp          (gripper XYZ velocity: critical for braking reward)
        23-24: finger_vel         (zeroed: masked out)
    """
    (
        grip_pos,
        object_pos,    # Real cube pos (not used directly — we override below)
        object_rel_pos,
        gripper_state,
        object_rot,
        object_velp,
        object_velr,
        grip_velp,
        gripper_vel,
    ) = self.generate_mujoco_observations()

    # "Holding Object" trick: tell the model it always has an object in its grip
    fake_object_pos = grip_pos.copy()

    # Goal-relative displacement: the primary navigation signal
    goal = self.goal if (hasattr(self, 'goal') and self.goal is not None) else np.zeros(3)
    rel_to_goal = goal.astype(np.float64) - grip_pos

    obs_vec = np.concatenate([
        grip_pos,          # 0-2
        fake_object_pos,   # 3-5  (Holding Object trick)
        rel_to_goal,       # 6-8  (goal - grip)
        np.zeros(2),       # 9-10 (fingers masked)
        np.zeros(3),       # 11-13 (object rot masked, scenario_id removed)
        np.zeros(3),       # 14-16 (object vel masked)
        np.zeros(3),       # 17-19 (object vel masked)
        grip_velp,         # 20-22 (velocity for braking)
        np.zeros(2),       # 23-24 (finger vel masked)
    ]).astype(np.float32)

    return {
        "observation":   obs_vec,
        "achieved_goal": grip_pos.astype(np.float32),
        "desired_goal":  goal.astype(np.float32),
    }
```

### 2b. Toggle between scripted and training observation mode

We need the training env to call `_build_phase9_observation()` while the production scripted
system can keep using the existing `_get_obs()` (which returns the richer dict for debugging).
We accomplish this with a flag on the env:

```python
# In ChessTaskEnv.__init__, add:
self._use_phase9_obs = False  # Set True by training wrappers

# Replace the existing _get_obs():
def _get_obs(self):
    if self._use_phase9_obs:
        return self._build_phase9_observation()
    # Original minimal dict for scripted controller (unchanged)
    grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy().astype(np.float32)
    grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip").copy().astype(np.float32)
    l_finger = np.float32(self._utils.get_joint_qpos(
        self.model, self.data, "robot0:l_gripper_finger_joint"
    ).item())
    goal_pos = self.goal_pos.copy().astype(np.float32) if self.goal_pos is not None else np.zeros(3, dtype=np.float32)
    obs_vec = np.concatenate([grip_pos, grip_vel, [l_finger]])
    scen_map = {None: 0, "transit": 1, "descend": 2, "ascend": 3}
    scenario_id = scen_map.get(self.current_scenario, 0)
    return {
        "observation":   obs_vec,
        "achieved_goal": grip_pos,
        "desired_goal":  goal_pos,
        "grip_pos":      grip_pos,
        "grip_vel":      grip_vel,
        "l_finger":      l_finger,
        "goal_pos":      goal_pos,
        "scenario_id":   scenario_id,
    }
```

> [!NOTE]
> The production scripted controller's observation dict (`grip_pos`, `grip_vel`, etc.) is still returned
> by default. Only training wrappers flip `_use_phase9_obs = True`. This means zero impact on
> all existing scripts and the `ScriptedController`.

### 2c. Update `observation_space` for training mode

The observation space must match the Phase-9 layout when training. The training wrappers
(see Section 3) will set this after flipping the flag:

```python
# In the training wrapper's __init__, after wrapping the env:
from gymnasium import spaces
unwrapped = env.unwrapped
unwrapped._use_phase9_obs = True
unwrapped.observation_space = spaces.Dict({
    "observation":   spaces.Box(-np.inf, np.inf, shape=(25,), dtype="float32"),
    "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,),  dtype="float32"),
    "desired_goal":  spaces.Box(-np.inf, np.inf, shape=(3,),  dtype="float32"),
})
# Also update the wrapper's observation_space to match
env.observation_space = unwrapped.observation_space
```

---

## 3. Training Environment Wrappers

We use **three specialist wrapper environments**, each locked to a single scenario.
They live in `training/envs/`. They are `gymnasium.Wrapper` subclasses around `ChessTaskEnv`.

Their responsibilities:
1. Set `_use_phase9_obs = True` and update `observation_space`
2. Call `env.reset()` normally (the base `_reset_sim` already handles scenario-specific init)
3. Optionally pass `drift_curriculum_steps` and `force_drift_limit` into `gym.make`
4. They do **not** override `step()` — the base `ChessTaskEnv.step()` handles everything

### Why wrappers at all, if `_reset_sim` already handles scenarios?

`gym.make("ChessFetchTask-v0", force_scenario="transit")` creates an env locked to transit.
The wrapper's job is to:
- Flip the observation mode (`_use_phase9_obs`)
- Update `observation_space` so SB3 sees the right space
- Provide a clean type name for isinstance checks (`TransitTrainEnv`, etc.)

### `training/envs/transit_env.py`

```python
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TransitTrainEnv(gym.Wrapper):
    """
    Training wrapper for the TRANSIT specialist model.
    Locks scenario to 'transit', enables Phase-9 observation.
    
    Episode structure (from ChessTaskEnv._reset_sim):
      - Arm starts at random [XY, SAFE_Z], fingers CLOSED
      - Goal is a random [XY, SAFE_Z] at MIN_GOAL_DIST away
      - Model must move horizontally to the goal without dropping below FLOOR_LIMIT
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Enable Phase-9 observation on the unwrapped env
        uw = env.unwrapped
        uw._use_phase9_obs = True
        phase9_space = spaces.Dict({
            "observation":   spaces.Box(-np.inf, np.inf, shape=(25,), dtype="float32"),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,),  dtype="float32"),
            "desired_goal":  spaces.Box(-np.inf, np.inf, shape=(3,),  dtype="float32"),
        })
        uw.observation_space = phase9_space
        self.observation_space = phase9_space
```

`AscendTrainEnv` and `DescendTrainEnv` are identical except for the name. The scenario lock
is applied via `force_scenario=...` in `gym.make()`, not inside the wrapper.

### Factory function for training script

```python
# training/envs/__init__.py

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from training.envs.transit_env import TransitTrainEnv
from training.envs.ascend_env import AscendTrainEnv
from training.envs.descend_env import DescendTrainEnv
import src.chess_env  # register ChessFetchTask-v0

WRAPPER_MAP = {
    "transit": TransitTrainEnv,
    "ascend":  AscendTrainEnv,
    "descend": DescendTrainEnv,
}

def make_train_env(stage: str, drift_curriculum_steps: int, debug: bool = False, fixed_drift: bool = False):
    """Creates a monitored, curriculum-enabled training env for the given stage."""
    def _init():
        base = gym.make(
            "ChessFetchTask-v0",
            force_scenario=stage,
            drift_curriculum_steps=drift_curriculum_steps,
            debug=debug,
            fixed_drift=fixed_drift,
        )
        wrapped = WRAPPER_MAP[stage](base)
        return Monitor(wrapped)
    return _init

def make_eval_env(stage: str, eval_drift_limit: float = 0.005, debug: bool = False):
    """Creates an evaluation env locked to a stage with a fixed drift limit."""
    def _init():
        base = gym.make(
            "ChessFetchTask-v0",
            force_scenario=stage,
            force_drift_limit=eval_drift_limit,
            debug=debug,
        )
        wrapped = WRAPPER_MAP[stage](base)
        return Monitor(wrapped)
    return _init
```

---

## 4. Required `ChessTaskEnv.__init__` Parameter Changes

The constructor currently strips these kwargs with `kwargs.pop(...)`. They need to become
**real parameters** with actual effect:

```python
def __init__(
    self,
    force_scenario=None,
    hide_object=True,
    show_chess_pieces=False,
    debug=False,
    drift_curriculum_steps=None,   # NEW: int — per-worker steps for curriculum
    force_drift_limit=None,        # NEW: float — override drift limit for eval
    fixed_drift=False,             # NEW: bool — skip curriculum, use DRIFT_LIMIT_END immediately
    **kwargs
):
    # Remove the kwargs.pop() calls for these three
    ...
    self.drift_curriculum_steps = drift_curriculum_steps
    self.force_drift_limit = force_drift_limit
    self.fixed_drift = fixed_drift
    
    # force_start_pos: optional XYZ override for the arm start in _reset_sim.
    # Referenced in the existing _reset_sim() but never initialised — set to None here.
    self.force_start_pos = None

    # Curriculum state (read in step())
    self.DRIFT_LIMIT_START = self.env_cfg.get("drift_limit_start", 0.100)
    self.DRIFT_LIMIT_END   = self.env_cfg.get("drift_limit_end", 0.010)
    # drift_curriculum_steps is per-worker (not total). With 8 workers,
    # total wall-clock drift steps = 8 × DRIFT_CURRICULUM_STEPS.
    self.DRIFT_CURRICULUM_STEPS = (
        drift_curriculum_steps or
        self.env_cfg.get("drift_curriculum_steps", 62_500)
    )
```

And in `step()`, the drift limit calculation becomes dynamic (see `02_reward_and_step.md`).
