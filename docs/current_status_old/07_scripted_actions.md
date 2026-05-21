# Scripted Actions

All scripted actions live in `src/chess_env/task.py` on the `ChessTaskEnv` class. They are the non-RL, deterministic part of the current hybrid system. After the refactor, these become the **entire** movement system.

---

## Core Primitive: `_move_mocap_to`

Everything scripted builds on this one method:

```python
def _move_mocap_to(self, target_pos, target_quat, max_steps=150, tolerance=0.001):
    zero_action = np.zeros(4)
    for _ in range(max_steps):
        grip_pos = get_site_xpos("robot0:grip")
        error = target_pos - grip_pos
        if norm(error) < tolerance:
            return True
        
        self._set_action(zero_action)          # (1) reset mocap → body (mandatory)
        self.data.mocap_pos[0][:3] += error    # (2) apply error as delta
        self.data.mocap_quat[0][:] = target_quat  # (3) enforce orientation
        self._mujoco_step(None)                # (4) advance physics
    
    return norm(get_site_xpos("robot0:grip") - target_pos) < tolerance
```

**The 4-step pattern** (call `_set_action` → set `mocap_pos` → set `mocap_quat` → `_mujoco_step`) is the invariant used throughout all scripted pipelines. This sequence must be followed exactly — swapping or omitting steps causes incorrect behavior.

---

## `execute_grasp()` — 6-Phase Grasp Pipeline

Preconditions (verified at start):
- Arm is stationary (`||grip_vel|| < 0.005 m/s`)
- Arm is at HOVER_Z (`|grip_z - HOVER_Z| < 0.025m`)
- Fingers are open (`l_finger ≥ FINGER_OPEN_JOINT - 0.003`)

### Phase 0: Halt & Settle (15 steps)

```python
self.data.qvel[:] = 0.0
self.data.qacc[:] = 0.0
self.data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)

settle_pos = get_site_xpos("robot0:grip").copy()
settle_quat = data.mocap_quat[0].copy()
for _ in range(15):
    _set_action(zero_action)
    data.mocap_pos[0][:3] = settle_pos   # Lock to current position
    data.mocap_quat[0][:] = settle_quat  # Lock to current orientation
    _mujoco_step(None)
```

Zeros all velocities (eliminates RL momentum), then runs 15 hold steps to let physics settle. **Critically uses current quat** (not VERTICAL_QUAT) — forcing vertical immediately while the arm has residual RL tilt can cause a violent snap that drops the cube.

### Phase 1+2: Rotation Abort + Align & Verticalize

Reads cube quaternion and computes effective yaw from nearest square axis. If `effective_yaw > 25°`, abort with `CUBE_ROTATED`.

Then `_move_mocap_to([cube_xy, HOVER_Z], VERTICAL_QUAT, max_steps=100, tolerance=0.001)`:
- Simultaneously corrects any wrist tilt AND aligns XY over the cube center.
- Achieves two goals in one motion by combining position and rotation correction.

### Phase 3: Plunge (HOVER_Z → GRASP_Z)

```python
for _ in range((HOVER_Z - GRASP_Z) / 0.001 + 15):
    grip_pos = get_site_xpos("robot0:grip")
    if grip_pos[2] <= GRASP_Z + 0.001: break
    
    target_z = max(GRASP_Z, target_z - 0.001)  # 1mm per step descent
    plunge_target = [cube_xy[0], cube_xy[1], target_z]
    
    _set_action(zero_action)              # Reset mocap
    error = plunge_target - grip_pos
    data.mocap_pos[0][:3] += error        # Re-assert XY lock + Z descent
    data.mocap_quat[0][:] = VERTICAL_QUAT
    _mujoco_step(None)
```

Descends 1mm per step from HOVER_Z (0.460) to GRASP_Z (0.425) = 35mm total. The XY position is re-locked every step by including the full error vector (which also contains XY components). The descent is very slow (1mm/step = 25mm/s in sim time at 25 Hz env), ensuring no physics shock.

The range `(HOVER_Z - GRASP_Z) / 0.001 + 15 = 35 + 15 = 50 steps` provides 15 extra steps for convergence below the exact GRASP_Z threshold.

### Phase 4: Grasp (Linear Finger Ramp, 150 steps)

```python
self.grasp_mode = True  # Switch to actuator-driven mode
ramp_start = FINGER_OPEN_JOINT   # 0.0181
ramp_end = FINGER_CLOSED_JOINT   # 0.0000
ramp_delta = (ramp_start - ramp_end) / GRASP_CLOSE_STEPS  # per step

for step in range(GRASP_CLOSE_STEPS):  # 150 steps
    self.finger_target_joint = max(ramp_end, ramp_start - ramp_delta * step)
    _set_action(zero_action)
    error = close_target - get_site_xpos("robot0:grip")
    data.mocap_pos[0][:3] += error   # Re-lock XY + Z
    data.mocap_quat[0][:] = VERTICAL_QUAT
    _mujoco_step(None)
    
    # Explosion guard every 10 steps
    if step % 10 == 0:
        if cube_pos[2] > GRASP_Z + 0.050: return CUBE_EXPLOSION
    
    # Early abort: fingers closed empty after 30 steps
    if step > 30 and l_finger < 0.003:
        return FINGER_CLOSED_EMPTY
```

At step 0: `finger_target_joint = 0.0181`. At step 150: commanded to `0.0`, but actual ≈ `0.0141` (stall against cube).

Early abort condition: If fingers go below j=0.003 after 30 steps, there's no cube between them (they closed completely empty). This aborts before a phantom "successful" grasp.

Explosion guard: If cube Z jumps >50mm above GRASP_Z, a physics explosion is happening.

### Phase 5: Hold & Verify (50 steps)

Run 50 hold steps (re-asserting position every step) to let contact impulses settle. Then check:
- `cube_xy_error < GRASP_VERIFY_XY_THRESHOLD` (15mm)
- `cube_z_error < GRASP_VERIFY_Z_THRESHOLD` (20mm)
- `l_finger > 0.003` (not fully closed = cube is there)

### Phase 6: Retract (GRASP_Z → HOVER_Z)

Mirror of Plunge: ascend 1mm per step while holding the cube. Each step checks for cube drop (cube Z must travel with grip Z). On drop detected: `CUBE_DROPPED_DURING_RETRACT`.

---

## `execute_place()` — 6-Phase Place Pipeline

Preconditions:
- Arm stationary at HOVER_Z over destination
- `grasp_mode = True` (cube is held)

The pipeline mirrors grasp but in reverse:

- **Phase 0**: Halt & settle (using CURRENT quat, not VERTICAL — avoids snap while holding cube)
- **Phase 1+2**: Vertical correction + XY align over `dst_xy` simultaneously
- **Phase 3**: Plunge HOVER_Z → GRASP_Z (1mm/step, maintaining cube grip)
- **Phase 4**: Release — linear ramp CLOSED → OPEN over 80 steps, then 30 full-open settle steps
  - Slow release prevents sudden opening force from launching cube sideways
  - Sets `grasp_mode = False` after release
- **Phase 5**: Verify placement:
  - `cube_xy_error < 20mm` from `dst_xy`
  - `cube_z_error < 10mm` from `TABLE_SURFACE_Z + CUBE_HEIGHT/2`
- **Phase 6**: Retract GRASP_Z → HOVER_Z

**Known bug**: Phase 6 uses `release_target[0], release_target[1]` but `release_target` is defined inside Phase 4 scope. This is a scoping issue that works incidentally because Phase 4's `release_target = plunge_target.copy()` which was the last plunge_target set.

---

## `soft_reset()` — Seamless Scenario Transition

Used between consecutive scenarios in a chain (no episode boundary, no arm teleport):

### Phase 1: Halt

Loop up to 100 steps sending zero actions until `||grip_vel|| < HALT_VEL_THRESHOLD (0.5 mm/s)`. Then zero `qvel[:ROBOT_DOF=15]` and `qacc[:ROBOT_DOF]` for robot DOFs only (preserves object physics).

### Phase 2: Waypoint Alignment

Proportional controller loop:
```python
ALIGN_GAIN = 0.8
ALIGN_MAX_STEP_M = 0.005  # 5mm per step max
for _ in range(200):
    error = nominal_exit_pos - grip_pos
    if dist < 0.003m: converged
    step_vec = clip(ALIGN_GAIN × error, max_norm=0.005)
    data.mocap_pos[0][:3] += step_vec
    _mujoco_step(None)
```

Moves arm to `nominal_exit_pos` (the canonical exit position for the just-completed scenario). This corrects any positional error from the RL model not stopping exactly at the target.

### Phase 3: State Update

Updates `current_scenario`, `goal_pos`, `goal`, `tube_center_xy`, `episode_steps = 0`.

### Phase 4: Gripper Transition

Without grasp_mode:
- Moving to descend from non-descend: Open fingers
- Moving from descend to ascend/transit: Close fingers
- Otherwise: No transition

Re-validates finger position within 0.5mm of target. Raises `RuntimeError` if validation fails.

---

## `_settle_arm_to_start()`

Simple wrapper:
1. `_move_mocap_to(arm_start_pos, VERTICAL_QUAT, max_steps=100, tolerance=SETTLE_TOLERANCE)`
2. Zero `qvel`, `qacc`, `ctrl`
3. `mj_forward()`

Used exclusively in `_reset_sim` Phase 1. The zero-velocity step is essential — without it, residual arm velocity from the previous episode's final RL action would persist into the next episode.

---

## `_set_gripper_state()`

Direct joint teleport for both fingers simultaneously:
```python
set_joint_qpos(model, data, "robot0:l_gripper_finger_joint", finger_target_joint)
set_joint_qpos(model, data, "robot0:r_gripper_finger_joint", finger_target_joint)
```

No physics step — purely kinematic. Used in `_reset_sim` Phase 1 to ensure fingers are in the correct state before settling. Without this, the actuator would take many steps to drive the fingers to the target from whatever state the previous episode left them in.

---

## Key Invariants Across All Scripted Actions

1. **Always call `_set_action(zero_action)` first**, then set `mocap_pos` and `mocap_quat`. Never skip `_set_action` when using the 4-step pattern — it resets the mocap coordinate to the current body position, which `mocap_pos += error` then applies correctly.

2. **Always re-assert `mocap_quat`** after every `_set_action`. Failing to do so allows the quaternion to drift away from vertical orientation.

3. **Check preconditions at the start of every scripted action**. The scripted pipelines assume specific entry conditions. Violating them (wrong Z, not stopped) can cause physics instability.

4. **`grasp_mode` must remain `True` throughout transit with cube**. Setting it to `False` would enable finger teleportation, which would release the cube instantly.

5. **Robot-only velocity zeroing**: Use `qvel[:ROBOT_DOF=15] = 0.0` (not `qvel[:] = 0.0`) during `soft_reset` to avoid destroying the cube's natural dynamics.
