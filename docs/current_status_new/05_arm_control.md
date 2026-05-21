# Arm Control

## Control Paradigm: Mocap-Based

The Fetch arm is controlled entirely through the **mocap body** (`robot0:mocap`). A weld equality constraint (in `shared.xml`) forces `robot0:gripper_link` to follow the mocap body. MuJoCo's constraint solver applies joint forces to satisfy this weld at each timestep.

**We never set arm joint angles directly.** The entire arm configuration is driven by setting two values per timestep:
- `data.mocap_pos[0][:3]` — target end-effector position (3D)
- `data.mocap_quat[0][:]` — target end-effector orientation (quaternion)

The physics engine computes all joint configurations automatically via the equality constraint.

---

## The Mandatory 4-Step Pattern

Every scripted movement follows this exact sequence:

```python
# Step 1: Reset mocap to current body position
self._set_action(np.zeros(4))

# Step 2: Apply desired position delta
self.data.mocap_pos[0][:3] += delta_pos

# Step 3: Re-assert orientation (MUST come after _set_action)
self.data.mocap_quat[0][:] = self.VERTICAL_QUAT

# Step 4: Advance physics
self._mujoco_step(None)
```

**Why step 1 is mandatory**: `_set_action(np.zeros(4))` calls `mocap_set_action`, which calls `reset_mocap2body_xpos` — resetting `mocap_pos` to the current physical `gripper_link` body position. Skipping this step causes the mocap to drift since the body accumulates lag from the constraint. Without the reset, the position delta compounds into runaway behavior.

**Why step 3 must follow step 1**: `_set_action` resets the mocap quaternion to the body's current orientation. After heavy movements, the body's orientation may have drifted. Re-asserting `VERTICAL_QUAT` after the reset ensures the gripper always stays vertical.

---

## `_set_action(action)` — RL Interface

```python
def _set_action(self, action):
    # action: [dx, dy, dz, d_gripper] — all in meters/step
    pos_ctrl = action[:3] * POS_CTRL_SCALE   # 0.015 m/step max
    # ... finger enforcement ...
    mocap_action = np.concatenate([pos_ctrl, np.zeros(4)])
    self._utils.mocap_set_action(self.model, self.data, mocap_action)
```

When called with `np.zeros(4)` (as in all scripted paths), `pos_ctrl = [0, 0, 0]`, so `mocap_set_action` only performs the `reset_mocap2body_xpos` sync, leaving `mocap_pos = current_body_pos` and `mocap_quat = current_body_quat`.

---

## `_move_mocap_to` — Core Movement Primitive

All scripted movements within `ChessTaskEnv` (grasp, place, soft_reset alignment) use this method:

```python
def _move_mocap_to(self, target_pos, target_quat, max_steps=150, tolerance=0.001) -> bool:
    zero_action = np.zeros(4)
    should_render = (self.render_mode == "human")
    for _ in range(max_steps):
        grip_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        error = target_pos - grip_pos
        if np.linalg.norm(error) < tolerance:
            return True
        self._set_action(zero_action)           # Reset mocap → body
        self.data.mocap_pos[0][:3] += error     # Apply full error as delta
        self.data.mocap_quat[0][:] = target_quat
        self._mujoco_step(None)
        if should_render:
            self.render()                        # Per-step rendering when in human mode
    final_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
    return bool(np.linalg.norm(final_pos - target_pos) < tolerance)
```

**Control law**: This is a proportional (P) controller with full-error gain — `error` is applied directly as the mocap delta. Because the weld constraint is compliant and physics has inertia, the grip site converges over multiple steps rather than teleporting. The effective closed-loop gain is:

```
mocap_target = body_pos + error_from_site
new_body_pos ≈ body_pos + k × error_from_site    (where k < 1 due to constraint compliance)
```

**Convergence**: Typical convergence to 1mm tolerance is 40–120 steps for 10–30cm moves.

**Visualization**: When `render_mode="human"`, `self.render()` is called after each step. This provides smooth animation in the viewer. Without this, the viewer shows only the first and last frame.

---

## `_settle_arm_to_start` — Reset Helper

Used only during `_reset_sim` to position the arm at the scenario start:

```python
def _settle_arm_to_start(self, arm_start_pos):
    self._move_mocap_to(arm_start_pos, self.VERTICAL_QUAT, max_steps=100,
                        tolerance=self.SETTLE_TOLERANCE)  # 3mm
    self.data.qvel[:] = 0.0    # Zero all velocities
    self.data.qacc[:] = 0.0    # Zero all accelerations
```

The 3mm tolerance (vs 1mm for scripted movements) is intentional — exact positioning is not needed at episode start, only stability. Zeroing velocities prevents any carry-over momentum from the previous episode.

---

## ScriptedController — High-Level Movement

`ScriptedController` in `src/chess_env/controller.py` drives the arm through the three movement stages (transit, descend, ascend) using its own proportional movement loop:

```python
class ScriptedController:
    TRANSIT_TOLERANCE_M  = 0.004   # 4mm success threshold
    VERTICAL_TOLERANCE_M = 0.004   # 4mm success threshold for descend/ascend
    STEP_GAIN            = 1.0     # Full error applied per step
    MAX_STEP_SIZE_M      = 0.008   # 8mm per physics step cap (prevents overshoot)
    TRANSIT_MAX_STEPS    = 400     # Steps budget for long board diagonals
    VERTICAL_MAX_STEPS   = 200     # Steps budget for 90mm vertical travel
    FLOOR_LIMIT          = 0.400   # Abort transit if arm drops below this Z

    def __init__(self, env, drift_limit=0.010, render_fn=None, render_delay=0.0):
        ...
```

### `_run_movement_loop`

The core loop used for all three stage types:

```python
def _run_movement_loop(self, target_pos, *, tolerance, max_steps, abort_fn=None):
    import time
    env = self._env
    for step in range(max_steps):
        grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
        error = target_pos - grip_pos
        dist = float(np.linalg.norm(error))

        if dist < tolerance:
            return StageResult(success=True, ...)

        if abort_fn is not None:
            abort, reason = abort_fn(grip_pos)
            if abort:
                return StageResult(success=False, crash_reason=reason, ...)

        # Capped proportional step
        step_vec = self.STEP_GAIN * error
        if np.linalg.norm(step_vec) > self.MAX_STEP_SIZE_M:
            step_vec = step_vec / np.linalg.norm(step_vec) * self.MAX_STEP_SIZE_M

        env._set_action(np.zeros(4))          # Reset mocap → body
        env.data.mocap_pos[0][:3] += step_vec
        env.data.mocap_quat[0][:] = env.VERTICAL_QUAT
        env._mujoco_step(None)

        if self._render_fn is not None:       # Per-step rendering
            self._render_fn()
            if self._render_delay > 0:
                time.sleep(self._render_delay)
```

**Key difference from `_move_mocap_to`**: The `ScriptedController` caps the step size at 8mm per step (`MAX_STEP_SIZE_M`). `_move_mocap_to` applies the full error uncapped. The cap prevents large overshoots for long-distance moves.

### Per-Step Rendering

To enable smooth animation in the viewer, pass `render_fn`:
```python
ctrl = ScriptedController(env, drift_limit=0.010,
                           render_fn=env.render,
                           render_delay=0.02)  # 20ms delay per step
```

Without `render_fn`, no rendering occurs during movement. The viewer would only update when `env.render()` is called explicitly (e.g., after the episode).

---

## Finger State Management

Finger state is managed separately from arm state. The gripper has two modes:

| Mode | `finger_target_joint` | `data.ctrl` | Description |
|------|-----------------------|-------------|-------------|
| CLOSED | 0.0000 (0mm) | 0.0 | Fingers pressed together |
| OPEN | 0.0181 (18.1mm) | 0.0181 | Fingers spread for descent/approach |

### `_set_gripper_state()` — Teleport

Sets finger position instantly (bypasses physics):

```python
def _set_gripper_state(self):
    target = self.finger_target_joint
    set_joint_qpos(model, data, "robot0:l_gripper_finger_joint", target)
    set_joint_qpos(model, data, "robot0:r_gripper_finger_joint", target)
    data.qvel[l_dof] = 0.0
    data.qvel[r_dof] = 0.0
    data.ctrl[l_id] = target    # CRITICAL: sync actuator target
    data.ctrl[r_id] = target    # or next step undoes the teleport
    mujoco.mj_forward(model, data)
```

This is called at:
- Start of each `_reset_sim` episode (set to CLOSED)
- Start of descend scenario (open before descent)
- Start of ascend scenario (close after grasp/open after place)
- During grasp pipeline (various phases)

### Grasp-Mode Actuator Close

During `execute_grasp` Phase 4, the fingers are closed via actuator ramping (not teleport):

```python
for step in range(grasp_close_steps):  # 150 steps
    ramp = step / grasp_close_steps
    finger_target = FINGER_OPEN_JOINT * (1 - ramp)  # 0.0181 → 0.0
    self.finger_target_joint = finger_target
    self._set_gripper_state()   # Each step syncs qpos + ctrl
    _set_action(np.zeros(4))
    data.mocap_pos[0][:3] = grasp_pos  # Lock arm position during close
    data.mocap_quat[0][:] = VERTICAL_QUAT
    _mujoco_step(None)
```

This ramp approach (setting `ctrl` each step to match the desired position) means the actuator applies exactly zero steady-state error force — the finger simply tracks the script. The cube halts the finger at the physical contact point.

---

## Gripper Control Summary

| Scenario | Fingers | Controller | Notes |
|----------|---------|-----------|-------|
| Transit (no cube) | CLOSED (teleport) | `_set_gripper_state` each reset | No movement needed |
| Descend | OPEN (teleport) | `_set_gripper_state` once at reset | Opens for approach |
| Grasp Phase 3-4 | OPEN → ramp → stall | Actuator ramping | Physical close around cube |
| Ascend (grasp_mode) | Stalled at contact | Actuator at ctrl=0 | Cube forces maintain gap |
| Transit (grasp_mode) | Stalled at contact | Actuator at ctrl=0 | Cube held in transit |
| Place Phase | Released | `_set_gripper_state(OPEN)` | Cube released |
| Ascend (after place) | CLOSED (teleport) | `_set_gripper_state` | Empty gripper |
