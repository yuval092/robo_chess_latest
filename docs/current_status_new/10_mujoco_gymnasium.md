# MuJoCo & Gymnasium-Robotics

## What This Project Is Based On

The project is derived from **Gymnasium-Robotics `FetchPickAndPlace-v4`**, specifically the `MujocoFetchPickAndPlaceEnv` class. This environment was originally designed by OpenAI (now Farama Foundation) to demonstrate robotic manipulation with a simulated 7-DOF Fetch arm in MuJoCo.

Reference: https://robotics.farama.org/envs/fetch/pick_and_place/

---

## How We Override the Model XML

The standard `MujocoFetchPickAndPlaceEnv.__init__` loads a hardcoded XML path stored in `MODEL_XML_PATH` at the module level. We monkey-patch this before calling `super().__init__()`:

```python
# In ChessSimulationEnv.__init__:
import gymnasium_robotics.envs.fetch.pick_and_place as _fpp_module

our_xml_path = Path(__file__).parent.parent.parent / "chess_env" / "assets" / "pick_and_place.xml"
_original_path = _fpp_module.MODEL_XML_PATH
_fpp_module.MODEL_XML_PATH = str(our_xml_path)
try:
    super().__init__(**kwargs)
finally:
    _fpp_module.MODEL_XML_PATH = _original_path
```

The `try/finally` block ensures the original path is always restored, even if `super().__init__()` raises an exception. This allows multiple `ChessTaskEnv` instances to coexist with different XML files if needed.

---

## Gymnasium-Robotics Lifecycle

### `env.reset()` Flow

```
gym.make(...).reset()
  → TimeLimit.reset()
    → ChessTaskEnv.reset()
      → MujocoFetchPickAndPlaceEnv.reset()
        → ChessSimulationEnv._env_setup()     ← first reset only
        → ChessTaskEnv._reset_sim()           ← every reset
      → ChessTaskEnv._get_obs()
      → ChessTaskEnv.compute_info()
```

`_env_setup` is called once at environment creation. `_reset_sim` is called on every `reset()`.

### `env.step(action)` Flow

```
env.step(action)
  → TimeLimit.step(action)
    → ChessTaskEnv.step(action)
      → ChessSimulationEnv._set_action(action)    ← apply mocap delta
      → ChessSimulationEnv._mujoco_step(action)   ← advance physics
      → ChessTaskEnv._get_obs()
      → ChessTaskEnv.compute_reward()
      → ChessTaskEnv._check_crash()
      → info["crashed"], info["crash_reason"]
```

In the scripted-only pipeline, `env.step()` is rarely called directly. `ScriptedController` calls `env._set_action()` and `env._mujoco_step()` directly to avoid the Gymnasium wrapper overhead and step counter.

---

## MuJoCo API Usage

### Key Data Structures

```python
env.model  # mujoco.MjModel — static model (geometry, joints, actuators)
env.data   # mujoco.MjData — dynamic state (positions, velocities, forces)
```

### Position Queries

```python
# Grip site world position (used for all reach calculations)
grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip")

# Cube position via qpos slice
obj_joint_id = env.model.joint("object0:joint").id
qpos_start = env.model.jnt_qposadr[obj_joint_id]
cube_pos = env.data.qpos[qpos_start : qpos_start + 3]

# Joint position (e.g., finger)
l_finger = env._utils.get_joint_qpos(env.model, env.data,
                                      "robot0:l_gripper_finger_joint").item()

# Torso height
torso_h = env._utils.get_joint_qpos(env.model, env.data,
                                     "robot0:torso_lift_joint").item()
```

### State Modification

```python
# Teleport joint (sets qpos only — must also set ctrl if actuated)
env._utils.set_joint_qpos(env.model, env.data, "robot0:torso_lift_joint", 0.25)

# Set mocap target
env.data.mocap_pos[0][:3] = target_pos
env.data.mocap_quat[0][:] = target_quat

# Set actuator target (for position actuators)
act_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_joint")
env.data.ctrl[act_id] = target_value

# Recompute kinematics without advancing time
mujoco.mj_forward(env.model, env.data)

# Advance one physics step (20 substeps × 0.1ms = 2ms)
env._mujoco_step(None)
# Equivalent to: mujoco.mj_step(env.model, env.data, nstep=env.n_substeps)
```

### ID Lookups

```python
# Body ID
body_id = env.model.body("robot0:base_link").id

# Joint ID (for qposadr / dofadr lookups)
joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "robot0:torso_lift_joint")
# or equivalently:
joint_id = env.model.joint("robot0:torso_lift_joint").id

# Actuator ID
act_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_joint")

# Joint range
joint_max = env.model.jnt_range[joint_id][1]

# DOF address (for qvel/qacc access)
dof_adr = env.model.joint("robot0:l_gripper_finger_joint").dofadr[0]
velocity = env.data.qvel[dof_adr]
```

---

## Rendering

### Human Mode

```python
env = gym.make("ChessFetchTask-v0", render_mode="human")
env.reset()
env.render()   # Opens MuJoCo viewer on first call; updates existing viewer on subsequent calls
```

The MuJoCo passive viewer runs in a separate thread. **It only updates when `env.render()` is explicitly called.** Calling `mj_step` or `_mujoco_step` alone does NOT update the viewer.

For smooth animation during scripted movement:
- In `_move_mocap_to` (task.py): `if self.render_mode == "human": self.render()` after each `_mujoco_step`
- In `_run_movement_loop` (controller.py): `if self._render_fn is not None: self._render_fn()` after each `_mujoco_step`

The `ScriptedController` must be constructed with `render_fn=env.render` to enable per-step rendering:
```python
ctrl = ScriptedController(env, render_fn=env.render, render_delay=0.02)
```

### RGB/Offscreen Mode

```python
env = gym.make("ChessFetchTask-v0", render_mode="rgb_array")
frame = env.render()  # Returns numpy array HxWx3
```

Useful for recording.

---

## GoalEnv Observation Structure

`ChessTaskEnv` inherits from `GoalEnv` (via `MujocoFetchPickAndPlaceEnv`). Each `_get_obs()` returns:

```python
{
    "observation": np.array([
        grip_pos[0], grip_pos[1], grip_pos[2],   # Grip site XYZ
        gripper_vel[0], gripper_vel[1], gripper_vel[2],
        object_pos[0], object_pos[1], object_pos[2],  # Cube XYZ (or HIDDEN if hidden_object)
        object_rel_pos[0], object_rel_pos[1], object_rel_pos[2],
        object_velp[0], object_velp[1], object_velp[2],
        object_velr[0], object_velr[1], object_velr[2],
        grip_velp[0], grip_velp[1], grip_velp[2],
        finger_width,       # l_finger + r_finger joint sum
        scenario_id,        # 0=transit, 1=descend, 2=ascend
    ]),  # shape (22,)
    "achieved_goal": np.array([grip_pos[0], grip_pos[1], grip_pos[2]]),   # shape (3,)
    "desired_goal":  np.array([goal_pos[0], goal_pos[1], goal_pos[2]]),   # shape (3,)
    "grip_pos":      np.array([grip_pos[0], grip_pos[1], grip_pos[2]]),   # convenience
}
```

`achieved_goal` and `desired_goal` are used for `compute_reward()` (HER compatibility). In scripted-only mode, rewards are not used — only the `"observation"` key matters for future RL integration.

---

## Common Pitfalls

### 1. Calling `mj_forward` is not the same as `mj_step`

`mj_forward` recomputes kinematics from current qpos (no time advance, no contact forces). Use it after teleporting joints to ensure consistent state. Use `_mujoco_step` (which calls `mj_step`) to advance the simulation.

### 2. `_set_action(zeros)` resets mocap orientation

After calling `_set_action(np.zeros(4))`, the mocap quaternion is reset to the current body orientation. Always re-assert `VERTICAL_QUAT` immediately after in the same step.

### 3. Position actuators fight teleported joints

When you teleport a finger joint (`set_joint_qpos`), the position actuator still targets its previous `data.ctrl` value. On the next `mj_step`, the actuator applies force = Kp × (ctrl − qpos), which undoes the teleport. Always sync `data.ctrl[act_id] = target` when teleporting actuated joints.

### 4. `env.step()` advances the TimeLimit counter

The Gymnasium `TimeLimit` wrapper counts calls to `step()`. `ScriptedController._run_movement_loop` bypasses this by calling `_mujoco_step` directly. The `transition()` method manually resets `_elapsed_steps` on the TimeLimit wrapper after each `soft_reset`.

### 5. Grip site ≠ gripper body position

The grip site (`robot0:grip`) is 10.34cm below the gripper_link body. All Z-height logic (HOVER_Z, GRASP_Z, SAFE_Z checks) uses the site position. The mocap body and gripper_link body position is typically 1–3mm above the site due to constraint compliance.
