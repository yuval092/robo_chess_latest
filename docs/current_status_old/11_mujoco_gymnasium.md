# MuJoCo & Gymnasium-Robotics

## What This Project Is Based On

The project is derived from **Gymnasium-Robotics FetchPickAndPlace-v4**, specifically the `MujocoFetchPickAndPlaceEnv` class. The Fetch environment was originally designed by OpenAI (now maintained by the Farama Foundation) to demonstrate robotic manipulation with a simulated 7-DOF Fetch arm.

Reference: https://robotics.farama.org/envs/fetch/pick_and_place/

The key properties of FetchPickAndPlace-v4:
- **Goal-conditioned**: `observation` + `achieved_goal` + `desired_goal` dictionary observation
- **HER compatible**: Supports Hindsight Experience Replay via GoalEnv
- **Dense or sparse reward**: Can use both
- **Default table**: Smaller board (52cm×50cm), lower torso

Our customization overrides most of these defaults while keeping the simulation backbone.

---

## Gymnasium-Robotics Source Reference

Located in `./tmp/fetch_code/` — extracted Python source of the Gymnasium-Robotics package, specifically the Fetch environment code. This was used for reference when understanding:
- The observation construction (`generate_mujoco_observations`)
- The mocap control mechanism (`mocap_set_action`)
- The reset and step lifecycle

Located in `./tmp/fetch_assets/` — the original MuJoCo XML files for the standard Fetch environment. These were referenced when building our custom XML.

---

## How We Override the Model XML

The standard `MujocoFetchPickAndPlaceEnv.__init__` references the model via `MODEL_XML_PATH`, a module-level variable. We monkey-patch it before calling `super().__init__()`:

```python
import gymnasium_robotics.envs.fetch.pick_and_place as _fpp_module
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv

_original_path = _fpp_module.MODEL_XML_PATH
_fpp_module.MODEL_XML_PATH = our_custom_asset_path
try:
    super().__init__(**kwargs)
finally:
    _fpp_module.MODEL_XML_PATH = _original_path
```

The `try/finally` block ensures the original path is always restored, even if `__init__` raises. This allows multiple environment instances with different XMLs.

---

## MuJoCo API Usage

### Key MuJoCo Data Structures
- `env.model` (`mujoco.MjModel`): Static model description (geometry, actuators, joints, bodies)
- `env.data` (`mujoco.MjData`): Dynamic simulation state (positions, velocities, forces)

### Position Queries
```python
# Grip site world position
grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip")

# Cube joint position (qpos slice)
obj_joint_id = env.model.joint("object0:joint").id
qpos_start = env.model.jnt_qposadr[obj_joint_id]
cube_pos = env.data.qpos[qpos_start : qpos_start + 3]
cube_quat = env.data.qpos[qpos_start + 3 : qpos_start + 7]

# Joint state
l_finger = env._utils.get_joint_qpos(env.model, env.data, "robot0:l_gripper_finger_joint").item()
```

### State Modification
```python
# Set joint position (teleport)
env._utils.set_joint_qpos(env.model, env.data, "robot0:torso_lift_joint", 0.4)

# Set mocap target
env.data.mocap_pos[0][:3] = target_pos
env.data.mocap_quat[0][:] = target_quat

# Force state recomputation without timestep
mujoco.mj_forward(env.model, env.data)

# Advance simulation
env._mujoco_step(action)  # internally calls mj_step2 or equivalent
```

### Physics Stepping
```python
# Internal to gymnasium-robotics:
mujoco.mj_step(model, data, nstep=n_substeps)  # n_substeps=20 by default

# Full step with callbacks:
env._mujoco_step(action)  # calls mj_step + _step_callback
```

---

## Gymnasium Interface

### Environment Registration
```python
# src/chess_env/__init__.py
from gymnasium.envs.registration import register

register(
    id="ChessFetchTask-v0",
    entry_point="src.chess_env.task:ChessTaskEnv",
    max_episode_steps=200,
)
```

`max_episode_steps=200` triggers a `TimeLimit` wrapper around the env, which adds `truncated=True` after 200 steps. The TimeLimit counter must be manually reset between scenarios in multi-scenario chains (via traversing `env.env._elapsed_steps = 0`).

### Observation Space

After customization, the observation is a `Dict` space:
```python
{
    "observation": Box(shape=(25,)),
    "achieved_goal": Box(shape=(3,)),    # grip position
    "desired_goal": Box(shape=(3,)),     # goal position
}
```

### Action Space
```python
Box([-1, -1, -1, -1], [1, 1, 1, 1], shape=(4,))
```

### Step Return
```python
obs, reward, terminated, truncated, info
```
Where `info` contains `{"is_success": float, "scenario": str, "crash_reason": str|None}`.

---

## Rendering

Three render modes:
- `render_mode="human"`: Opens a MuJoCo viewer window (requires display).
- `render_mode="rgb_array"`: Returns RGB frame as numpy array.
- `render_mode=None`: No rendering (fastest, for training).

```python
env = gym.make("ChessFetchTask-v0", render_mode="human")
env.render()  # Call after each step for real-time display
```

---

## What's Different From Standard Fetch

| Property | Standard FetchPickAndPlace | Our ChessFetchTask |
|----------|--------------------------|-------------------|
| Board size | 52×50cm | 56×56cm (code) |
| Torso height | Not forced | Forced to max (0.3861m) |
| Gripper orientation | Free | Locked vertical (crane mode) |
| Object placement | Uniform on board | Uniform on board, or hidden |
| Goal type | Table surface or lift | Three Z-levels: SAFE, HOVER, TABLE |
| Observation | Standard 25D | Phase 9 "Holding Object" trick |
| Reward | Sparse or dense | Dense multi-component |
| Scripted actions | None | Grasp, Place, soft_reset |
| Actuator Kp | 5000 | 150000 (for grasp holding) |
| Finger friction | Default | 10.0 (high for grasp) |
| Cube condim | Default (3) | 6 (torsional + tangential) |
| Cube solref | Default | `[0.002, 1]` (stiff) |

---

## Gymnasium-Robotics Helper: `_utils`

The `_utils` object (a `MujocoModelUtils` or similar class from gymnasium-robotics) provides convenient wrappers:

```python
env._utils.get_site_xpos(model, data, name)      # 3D world position of named site
env._utils.get_site_xvelp(model, data, name)     # Linear velocity of named site
env._utils.get_joint_qpos(model, data, name)     # Joint position scalar
env._utils.get_joint_qvel(model, data, name)     # Joint velocity scalar
env._utils.set_joint_qpos(model, data, name, v)  # Set joint position
env._utils.set_joint_qvel(model, data, name, v)  # Set joint velocity
env._utils.set_mocap_quat(model, data, name, q)  # Set mocap orientation
env._utils.mocap_set_action(model, data, action) # Apply mocap delta (resets first)
```

The `mocap_set_action` function is the critical one — it internally calls `reset_mocap2body_xpos` which copies the physical body position into the mocap position, then applies the delta. This is why re-asserting `mocap_pos` after `_set_action` is necessary.

---

## Generate MuJoCo Observations

```python
(
    grip_pos,         # grip site position
    object_pos,       # object site position
    object_rel_pos,   # object - grip
    gripper_state,    # [l_finger_pos, r_finger_pos]
    object_rot,       # object Euler angles
    object_velp,      # object linear velocity
    object_velr,      # object angular velocity
    grip_velp,        # grip site linear velocity
    gripper_vel,      # [l_finger_vel, r_finger_vel]
) = env.generate_mujoco_observations()
```

Our `_build_phase9_observation` calls this but then overrides specific indices with the "Holding Object" trick.
