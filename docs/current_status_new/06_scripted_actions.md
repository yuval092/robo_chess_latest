# Scripted Actions

Scripted actions are deterministic, physics-driven routines in `src/chess_env/task.py`. They handle the fine-grained manipulation that requires precise control: grasping, placing, and transitions between movement stages.

---

## `execute_grasp()` — 6-Phase Grasp Pipeline

**Preconditions** (checked at start, returns failure dict if violated):
- Arm is stationary: `||grip_vel|| < 0.005 m/s`
- Arm is at HOVER_Z: `|grip_z - HOVER_Z| < 0.025m`
- Fingers are open: `l_finger >= FINGER_OPEN_JOINT - 0.003`

### Phase 0: Halt & Settle (15 steps)

```python
data.qvel[:] = 0.0
data.qacc[:] = 0.0
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)

settle_pos = get_site_xpos("robot0:grip").copy()
settle_quat = data.mocap_quat[0].copy()
for _ in range(15):
    _set_action(zero_action)
    data.mocap_pos[0][:3] = settle_pos   # Lock current position
    data.mocap_quat[0][:] = settle_quat  # Lock current orientation (NOT VERTICAL_QUAT)
    _mujoco_step(None)
```

Zero all velocities, then hold position for 15 steps. Uses the **current orientation** (not VERTICAL_QUAT) to avoid a violent snap that could knock the cube. The arm gradually settles.

### Phase 1: Cube Rotation Check

Reads cube quaternion from `data.qpos`. Computes effective yaw from nearest 90° alignment. If `effective_yaw > 25°`, aborts with `CUBE_ROTATED`. This prevents grasping at an angle that would fail the contact geometry.

### Phase 2: Align & Verticalize

```python
align_target = np.array([cube_xy[0], cube_xy[1], HOVER_Z])
_move_mocap_to(align_target, VERTICAL_QUAT, max_steps=100, tolerance=0.001)
```

Simultaneously corrects any residual XY offset AND forces the gripper vertical. Combines two corrections in a single convergent motion.

### Phase 3: Fine Plunge (HOVER_Z → GRASP_Z)

1mm-per-step descent at constant XY:
```python
target_z = HOVER_Z
for _ in range(int((HOVER_Z - GRASP_Z) / 0.001) + 15):
    if grip_z <= GRASP_Z + 0.001: break
    target_z = max(GRASP_Z, target_z - 0.001)
    plunge_target = [cube_xy[0], cube_xy[1], target_z]
    
    _set_action(zero_action)
    error = plunge_target - grip_pos
    data.mocap_pos[0][:3] += error
    data.mocap_quat[0][:] = VERTICAL_QUAT
    _mujoco_step(None)
```

The 1mm/step rate (much slower than `MAX_STEP_SIZE_M=8mm`) is intentional — rapid descent would cause the fingers to push the cube into the table, generating contact instability. The slow descent allows the cube to settle into the gripper cavity.

Completes with `grasp_pos = grip_pos.copy()` — captured for use in Phase 6.

### Phase 4: Close Fingers (Actuator Ramp, `grasp_close_steps=150` steps)

```python
for step in range(grasp_close_steps):
    ramp = step / grasp_close_steps
    finger_target = FINGER_OPEN_JOINT * (1.0 - ramp)  # 0.0181 → 0.0
    self.finger_target_joint = finger_target
    self._set_gripper_state()    # Sets qpos + ctrl each step
    _set_action(zero_action)
    data.mocap_pos[0][:3] = grasp_pos  # Lock arm XY/Z during close
    data.mocap_quat[0][:] = VERTICAL_QUAT
    _mujoco_step(None)
    
    # Check for stall (cube contact)
    if l_finger < GRASP_VERIFY_FINGER_THRESHOLD:
        close_steps_used = step
        break
```

The finger closes linearly over 150 steps (300ms simulated). When the finger contacts the 30mm cube, it stalls at ~14mm. The `grasp_verify_finger_threshold=0.016` check detects whether fingers actually contacted something — if both fingers are still at ≥16mm after all steps, grasp failed.

### Phase 5: Hold (50 steps)

Lock all state for 50 steps to let the constraint system settle:
```python
for _ in range(grasp_hold_steps):
    _set_action(zero_action)
    data.mocap_pos[0][:3] = grasp_pos
    data.mocap_quat[0][:] = VERTICAL_QUAT
    _mujoco_step(None)
```

This ensures the cube-finger contact is stable before the arm starts moving.

### Phase 6: Verify

```python
grip_pos = get_site_xpos("robot0:grip")
cube_pos = get_cube_position()
xy_error = ||cube_xy - grip_xy||
z_error = |cube_z - (grip_z - 0.015)|   # ~15mm below grip when held

if xy_error > grasp_verify_xy_threshold (0.015):
    return {"success": False, "reason": "CUBE_NOT_CENTERED"}
if z_error > grasp_verify_z_threshold (0.020):
    return {"success": False, "reason": "CUBE_NOT_AT_HEIGHT"}
```

On success: sets `grasp_mode = True`. The cube is now considered held.

---

## `execute_place(dst_xy)` — Place Pipeline

**Preconditions**: `grasp_mode == True`, arm at HOVER_Z over `dst_xy`.

The place pipeline:

1. **Fine approach** (if not already at HOVER_Z + XY): `_move_mocap_to([dst_xy, HOVER_Z], VERTICAL_QUAT)`
2. **Lower to GRASP_Z**: 1mm-per-step plunge (same pattern as grasp Phase 3)
3. **Open fingers** (actuator ramp, ~80 steps): Ramps from stall position back to FINGER_OPEN_JOINT
4. **Retreat** (100 steps): Move arm up ~15mm while keeping XY locked
5. **Set grasp_mode = False**
6. **Verify** cube is on table: `|cube_z - TABLE_SURFACE_Z| < 0.030`

Returns `{"success": bool, "reason": str}`.

---

## `soft_reset(new_scenario, new_goal_pos, nominal_exit_pos, nominal_xy)` — Scenario Transition

Bridges consecutive scenarios without a full environment reset. Called by `ScriptedController.transition()` after each stage completes.

### Parameters

- `new_scenario`: The next scenario ("descend", "ascend", "transit")
- `new_goal_pos`: The goal for the next scenario
- `nominal_exit_pos`: Where the arm should ideally be at the current moment (end of previous stage)
- `nominal_xy`: The tube center XY for vertical stages (descend/ascend)

### The 5 Phases

**Phase 0: Halt**
```python
for _ in range(100):
    _set_action(zero_action)
    grip_vel = ||data.qvel[arm_dofs]||
    if grip_vel < halt_vel_threshold (0.0005 m/s): break
else:
    raise RuntimeError("HALT_FAILED")
```
Waits for the arm to fully stop before transitioning. Up to 100 steps (200ms).

**Phase 1: Open Fingers** (if transitioning to "descend")
```python
self.finger_target_joint = FINGER_OPEN_JOINT
self._set_gripper_state()
```
Opens fingers for descent. Syncs both qpos and ctrl.

**Phase 2: Align to Nominal Exit Position**
```python
_move_mocap_to(nominal_exit_pos, VERTICAL_QUAT, max_steps=200, tolerance=0.003)
```
Drives arm to where it should be (the exit of the previous stage). Max 200 steps; raises `ALIGN_FAILED` if not converged.

**Phase 3: State Update**
- `current_scenario = new_scenario`
- `goal_pos = new_goal_pos`
- `tube_center_xy = nominal_xy` (for vertical stages)
- `episode_steps = 0`

**Phase 4: Finger Validation** (after opening for descent)
```python
l_pos = get_joint_qpos("robot0:l_gripper_finger_joint")
if abs(l_pos - finger_target_joint) > 0.0005:
    raise RuntimeError("FINGER_VALIDATION_FAILED")
```
Validates that the ctrl sync in `_set_gripper_state()` actually holds — finger must be within 0.5mm of target. This would only fail if the position actuator had an unexpected override.

### Returns

`(obs, info)` where:
- `obs`: Current observation dict (via `_get_obs()`)
- `info["halt_steps"]`: Steps taken in Phase 0
- `info["align_steps"]`: Steps taken in Phase 2

---

## `transition_validate(nominal_exit_pos=None)` — Diagnostic

Returns a diagnostic dict about current arm state:
```python
{
    "grip_pos": [x, y, z],
    "grip_speed_mm_s": float,    # Current grip site speed
    "is_velocity_ok": bool,      # speed < halt_vel_threshold
    "error_from_nominal_mm": float or None,  # Distance to nominal_exit_pos
}
```
Used for debugging and transition pre-validation. Not part of the normal movement pipeline.

---

## `_check_cube_held(grip_pos)` — Continuous Hold Monitoring

Called every step in `_run_movement_loop` when `grasp_mode=True`:

```python
cube_pos = get_cube_position()
xy_error = ||cube_pos[:2] - grip_pos[:2]||
z_error  = |cube_pos[2] - (grip_pos[2] - 0.015)|

if xy_error > 0.030:
    return False, "CUBE_DROPPED_XY"
if z_error > 0.020:
    return False, "CUBE_DROPPED_Z"
return True, None
```

The 15mm offset in Z accounts for the cube CoM being ~15mm below the grip site when held. If the cube drifts more than 30mm in XY or 20mm in Z from expected position, the hold is declared lost and the movement stage aborts.
