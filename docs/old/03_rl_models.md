# Reinforcement Learning Models

## Overview

The arm's three movement stages — **transit**, **descend**, and **ascend** — are each controlled by a separate Soft Actor-Critic (SAC) specialist model. These models are fine-tuned from a pretrained `FetchPickAndPlace-v4` checkpoint using the **"Holding Object" transfer trick**. The grasp and place stages remain fully scripted.

---

## Why Three Specialist Models?

The pick-and-place task has three mechanically distinct sub-problems:

| Stage | Challenge | Key reward signal |
|---|---|---|
| Transit | Cover board-length distances (up to 0.9 m diagonal) at fixed altitude | Distance to goal + jitter penalty |
| Descend | Drop 70 mm vertically while staying within a 10 mm XY tube | Distance + Z accuracy + tube breach penalty |
| Ascend | Climb 70 mm vertically with the same tube constraint | Distance + Z accuracy + tube breach penalty |

A single model would need to handle all three regimes, requiring a more complex policy and longer training. Specialist models each see only their sub-problem, allowing focused reward shaping and shorter training runs.

---

## Transfer Learning Foundation

### Base Model

```
models/pretrained/sac-FetchPickAndPlace-v4.zip
```

This checkpoint is configured as the base model in `configs/training.yaml`. The policy's `MultiInputPolicy` expects a 25-dimensional observation.

### Why Transfer Works

RoboChess keeps the `FetchPickAndPlace-v4` observation layout expected by the base policy, but changes the observation contents so the object position equals the gripper position. The policy is therefore asked to solve a gripper-to-goal movement problem with the same input shape it was trained with.

---

## The "Holding Object" Trick

### Observation Layout (25-D)

```
Index  Content (FetchPickAndPlace-v4 standard)
[0:3]   grip_pos        — gripper site XYZ
[3:6]   object_pos      — object XYZ  ← MODIFIED
[6:9]   rel_to_goal     — goal - grip
[9:11]  gripper_state   — (zeroed)
[11:14] object_rot      — (zeroed)
[14:17] object_velp     — (zeroed)
[17:20] object_velr     — (zeroed)
[20:23] grip_velp       — gripper velocity
[23:25] gripper_vel     — (zeroed)
```

### The Trick

```python
fake_object_pos = grip_pos.copy()  # Set object = gripper
```

By setting `object_pos = grip_pos`, the policy perceives "the gripper is already holding the object at its current location." The pretrained weights then interpret this as a "carry the object to the goal" task, which maps directly to "move the gripper to the goal."

### Why It Works Mechanically

The policy learned to minimise `||object_pos - goal||` by moving the gripper. When `object_pos = grip_pos`, this becomes `||grip_pos - goal||` — exactly what we want for pure gripper movement. The policy's internal representation of "I'm holding something and need to move it to the goal" translates naturally to "I need to move myself to the goal."

### Implementation

```python
# src/chess_env/task.py : _build_transfer_observation()
fake_object_pos = grip_pos.copy()
goal = self.goal
rel_to_goal = goal.astype(np.float64) - grip_pos

obs_vec = np.concatenate([
    grip_pos, fake_object_pos, rel_to_goal,
    np.zeros(2), np.zeros(3), np.zeros(3), np.zeros(3),
    grip_velp, np.zeros(2)
]).astype(np.float64)
```

The flag `_use_transfer_obs` toggles between this 25-D observation and the native 7-D observation. Training wrappers enable it permanently; `ModelEmbeddedController` enables it temporarily for each inference step.

---

## Model Loading (`src/chess_env/model_controller.py`)

### Responsibility

`ModelEmbeddedController` stores SAC model objects keyed by stage name and loads them with the correct observation space active.

### Known Stages

```python
KNOWN_STAGES = frozenset({"transit", "descend", "ascend"})
```

### `load(stage, path)` — Model Loading

```python
with transfer_obs_enabled(self._wrapped_env):  # Switch to 25-D obs space
    with contextlib.redirect_stdout(io.StringIO()):  # Suppress SB3 verbosity
        self._models[stage] = SAC.load(path, env=self._wrapped_env)
```

`SAC.load` validates the loaded policy against the environment's current observation space. The `transfer_obs_enabled` context manager temporarily switches both the unwrapped env and wrapper to `TRANSFER_OBS_SPACE`, ensuring stable-baselines3 accepts the 25-D policy.

Movement stages fail fast if their model has not been loaded.

---

## `transfer_obs.py` (`src/chess_env/transfer_obs.py`)

### `TRANSFER_OBS_SPACE`

```python
TRANSFER_OBS_SPACE = spaces.Dict({
    "observation":   Box(-inf, inf, shape=(25,), dtype=float64),
    "achieved_goal": Box(-inf, inf, shape=(3,),  dtype=float64),
    "desired_goal":  Box(-inf, inf, shape=(3,),  dtype=float64),
})
```

### `transfer_obs_enabled(env)` — Context Manager

```python
@contextmanager
def transfer_obs_enabled(env):
    """Temporarily enables transfer obs on wrapper and unwrapped env."""
    saved_flag, saved_unwrapped_space, saved_wrapper_space = ...
    try:
        enable_transfer_obs(env)
        yield
    finally:
        unwrapped._use_transfer_obs = saved_flag
        unwrapped.observation_space = saved_unwrapped_space
        env.observation_space = saved_wrapper_space
```

This is used by `ModelEmbeddedController.load_model()` and by `ModelEmbeddedController._run_stage()`. It saves and restores all three affected state fields to ensure no cross-contamination between training and inference contexts.

---

## Deployed Models

### Configuration (`configs/deployed_models.yaml`)

```yaml
transit: "models/transit.zip"
descend: "models/descend.zip"
ascend:  "models/ascend.zip"
```

At startup, `build_controller()` in `main.py` reads this file and calls `controller.load_all(transit_path=..., descend_path=..., ascend_path=...)`.

### Active Checkpoints

The active model paths are code/config state, not hard-coded in the controller:

| Stage | Default path |
|---|---|
| `transit` | `models/transit.zip` |
| `descend` | `models/descend.zip` |
| `ascend` | `models/ascend.zip` |

### Model File Format

Standard stable-baselines3 ZIP archives. Each contains:
- Policy network weights (actor + critic)
- Normalisation statistics (if VecNormalize was used — not applicable here)
- Training configuration metadata

---

## Inference Pipeline (Model Stage Execution)

```
ModelEmbeddedController._run_stage("transit", target_pos):

1. Check model loaded → else: scripted_controller.run_transit(target_xy)

2. env.goal_pos = target_pos
   env.goal     = target_pos
   env.current_scenario = "transit"
   env.tube_center_xy   = None     (transit has no tube)

3. Set finger_target_joint based on stage (if not grasp_mode)

4. Finger precondition check (if not grasp_mode):
   |l_finger - expected_finger| > 3 mm → PRECONDITION_FINGER

5. env._use_transfer_obs = True

6. for step in range(rl_max_steps_per_stage=300):
     grip_pos = get_site_xpos("robot0:grip")
     grip_vel = get_site_xvelp("robot0:grip")

     if is_near(grip_pos, target_pos) and speed < stability_threshold (0.02 m/s):
         success = True; break

     obs = env._get_obs()           # 25-D transfer obs
     action, _ = model.predict(obs, deterministic=True)
     action = np.float32(action)
     action[3] = -1.0 if stage in {"transit", "ascend"} else 1.0

     env._set_action(action)
     if stage == "transit":
         env.data.mocap_quat[0][:] = env.VERTICAL_QUAT
     env._mujoco_step(action)

     crash = _check_crash(env, stage, grip_pos)
     if crash: break

7. env._use_transfer_obs = previous value    # restore (even on exception)

8. Return StageResult(success, steps, crash_reason, final_pos, error_mm)
```

---

## Training Environment Wrappers

Three thin `gym.Wrapper` subclasses in `training/envs/` switch the observation space to `TRANSFER_OBS_SPACE` at construction time. This is the only change needed because the base env already correctly produces 25-D observations when `_use_transfer_obs=True`.

```python
# training/envs/transit_env.py
class TransitTrainEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        uw = env.unwrapped
        uw._use_transfer_obs = True
        uw.observation_space = TRANSFER_OBS_SPACE
        self.observation_space = TRANSFER_OBS_SPACE
```

`DescendTrainEnv` and `AscendTrainEnv` are identical. They are applied via the `WRAPPER_MAP` in `training/envs/__init__.py`.

---

## Observation Space Notes

### During RL Training

The wrapper forces `_use_transfer_obs=True` permanently. Every `step()` call returns a 25-D dict. stable-baselines3's `MultiInputPolicy` processes each key separately through a shared feature extractor.

### During Inference (Model Stage)

`ModelEmbeddedController._run_stage` temporarily sets `_use_transfer_obs=True`, runs the inference loop, then restores the flag. Between model stages (in the scripted transition/grasp/place phases), the native 7-D observation is active.

### Stable-Baselines3 Compatibility

SB3's `SAC.load` validates observation space dimensions but does not re-construct the network. The loaded policy network weights match the 25-D layout from training, so inference is correct as long as `_use_transfer_obs=True` when `model.predict()` is called.

---

## Key Model Hyperparameters (`configs/training.yaml`)

| Parameter | Value | Notes |
|---|---|---|
| `base_model` | `sac-FetchPickAndPlace-v4.zip` | Starting checkpoint |
| `num_envs` | 4 | Parallel training environments |
| `total_timesteps` | 600,000 | Default training length |
| `learning_rate` | 5e-5 | Conservative; preserves pretrained features |
| `batch_size` | 512 | Large batch for stable SAC updates |
| `target_entropy` | -4.0 | Tuned for 4-D action space |
| `learning_starts` | 10,000 | Steps before any gradient updates |
| `buffer_size` | 300,000 | Replay buffer capacity |
| `initial_ent_coef` | 0.1 | Entropy coefficient reset on load |
| `ent_coef_lr` | 0.001 | Entropy coefficient learning rate |

The entropy coefficient is explicitly reset on model load because the pretrained model may have converged to near-zero entropy (deterministic behaviour), which would prevent exploration in the new task.
