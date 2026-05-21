# Arm Control

## Control Paradigm: Mocap-Based

The Fetch arm is controlled entirely through the **mocap body** (`robot0:mocap`). A weld equality constraint forces `robot0:gripper_link` to follow the mocap body. The physics solver uses inverse kinematics (via constraint forces) to drive the arm joints to satisfy this constraint each timestep.

This means:
- **We never set joint angles directly** for arm motion.
- **We set `data.mocap_pos` and `data.mocap_quat`** to specify end-effector target.
- The physics engine automatically computes the required joint torques.

---

## Key Control Methods

### `_set_action(action)` — RL Interface (4D)

```python
def _set_action(self, action):
    assert action.shape == (4,)
    pos_ctrl, gripper_ctrl = action[:3], action[3]
    pos_ctrl *= self.POS_CTRL_SCALE  # 0.015 m/step max
    rot_ctrl = np.zeros(4)            # Zero delta rotation
    
    # Finger enforcement (see below)
    ...
    
    mocap_action = np.concatenate([pos_ctrl, rot_ctrl])
    self._utils.mocap_set_action(self.model, self.data, mocap_action)
```

**Critically**, `mocap_set_action` calls `reset_mocap2body_xpos` first — this **resets** `mocap_pos` to the current physical `gripper_link` position before applying the delta `pos_ctrl`. After the call, `mocap_pos = current_gripper_pos + pos_ctrl` and `mocap_quat = current_gripper_quat + rot_ctrl`.

Since `rot_ctrl = zeros`, the rotation delta is zero — the quaternion change is zero, so `mocap_quat` remains unchanged from the body's current orientation.

This means after every `_set_action` call:
- `mocap_pos` reflects current position + RL delta
- `mocap_quat` reflects the current physical gripper orientation (may have drifted from `VERTICAL_QUAT`)

**Post-action invariant**: Any manually desired `mocap_quat` must be re-asserted AFTER every `_set_action` call.

---

### `_move_mocap_to(target_pos, target_quat, max_steps, tolerance)` — Core Primitive

```python
def _move_mocap_to(self, target_pos, target_quat, max_steps=150, tolerance=0.001):
    zero_action = np.zeros(4)
    for _ in range(max_steps):
        grip_pos = self._utils.get_site_xpos(model, data, "robot0:grip")
        error = target_pos - grip_pos
        if np.linalg.norm(error) < tolerance:
            return True
        
        self._set_action(zero_action)          # 1. reset mocap → body
        self.data.mocap_pos[0][:3] += error    # 2. apply error as delta
        self.data.mocap_quat[0][:] = target_quat  # 3. re-lock orientation
        self._mujoco_step(None)                # 4. advance physics
    
    final_pos = get_site_xpos(...)
    return np.linalg.norm(final_pos - target_pos) < tolerance
```

This is a **proportional controller** on position:
- Error = target - current grip site position
- Full error is applied as mocap delta each step
- With the weld constraint lag (~10ms), the grip site converges toward target over multiple steps
- Orientation is re-locked every step (compensates for `_set_action` resetting it)
- Uses the **grip site** (not gripper_link body) as the control reference point

**Why error as full delta (not scaled)?** The weld constraint is compliant — applying the full error as delta overshoots the body position but the physics constraint pulls the actual site back. After a few iterations, position converges. This is effectively a bang-bang controller with physics damping.

**Convergence**: For typical moves (10-30cm), convergence to 3mm tolerance typically takes 30-80 steps. For 1mm tolerance, 60-120 steps. With 150 max steps at 2ms each, this represents 300ms of simulated time.

---

### `_settle_arm_to_start(arm_start_pos)` — Reset Helper

```python
def _settle_arm_to_start(self, arm_start_pos):
    self._move_mocap_to(arm_start_pos, self.VERTICAL_QUAT, max_steps=100, tolerance=self.SETTLE_TOLERANCE)
    self.data.qvel[:] = 0.0  # Zero all velocities
    self.data.qacc[:] = 0.0  # Zero all accelerations
    self.data.ctrl[:] = 0.0  # Zero actuator control
    mujoco.mj_forward(model, data)  # Recompute state
```

After moving to start position, ALL velocities are zeroed. This prevents residual momentum from RL in the previous episode from carrying over. `mj_forward` recomputes contact forces and other derived quantities with the new zero-velocity state.

---

## Gripper Control

Two modes depending on `self.grasp_mode`:

### Teleport Mode (`grasp_mode = False`)
Used during transit, descend, and ascend when not holding a cube.

```python
self.data.ctrl[0] = target_qpos  # Set actuator target
self.data.ctrl[1] = target_qpos
self._utils.set_joint_qpos(model, data, "robot0:l_gripper_finger_joint", target_qpos)
self._utils.set_joint_qpos(model, data, "robot0:r_gripper_finger_joint", target_qpos)
self._utils.set_joint_qvel(model, data, "robot0:l_gripper_finger_joint", 0.0)
self._utils.set_joint_qvel(model, data, "robot0:r_gripper_finger_joint", 0.0)
```

Joint position is **directly overridden** every step. No physics for the fingers — they teleport to the target position. This prevents any unwanted contact during transit.

### Actuator Mode (`grasp_mode = True`)
Used during grasp execution and while carrying a cube.

```python
self.data.ctrl[0] = target_qpos  # Set actuator target only
self.data.ctrl[1] = target_qpos  # Do NOT override joint positions
```

Fingers are driven by the position actuator only. Contact physics are fully active — the cube can push back. This allows the grasp force to be computed by MuJoCo's constraint solver.

### Finger State Transitions

During `_reset_sim`:
- **Transit**: Fingers start CLOSED (`finger_target_joint = FINGER_CLOSED_JOINT = 0.0`)
- **Descend**: Fingers transition CLOSED → OPEN over `max_steps=150` via `_move_mocap_to` loop
- **Ascend**: Fingers transition OPEN → CLOSED (`grasp_mode=False` teleport, no cube)

The `finger_target_joint` attribute is set before calling `_move_mocap_to`. Because `_set_action` runs inside `_move_mocap_to`, and `_set_action` reads `self.finger_target_joint` to set `ctrl`, the finger position converges during the arm movement.

---

## Vertical Orientation (Crane Mode)

```python
raw_quat = np.array([1.0, 0.0, 1.0, 0.0])
self.VERTICAL_QUAT = raw_quat / np.linalg.norm(raw_quat)
# → [0.707, 0, 0.707, 0]  (w, x, y, z)
```

This quaternion is enforced at every opportunity:
1. In `_env_setup`: `set_mocap_quat(model, data, "robot0:mocap", VERTICAL_QUAT)`
2. In `_reset_sim`: Before settling the arm
3. In `_move_mocap_to`: Re-applied every step as `data.mocap_quat[0][:] = target_quat`
4. In `execute_grasp` / `execute_place`: Re-applied every step inside the pipelines
5. In `soft_reset`: After halt and alignment

Why enforce every step? The weld constraint has compliance — the actual `gripper_link` orientation can drift from `mocap_quat` due to contact forces (especially during grasp). Re-asserting `mocap_quat = VERTICAL_QUAT` every step prevents cumulative drift.

---

## Environment Setup (`_env_setup`)

Called once at the start of the environment lifetime (not per-episode):

```python
def _env_setup(self, initial_qpos):
    super()._env_setup(initial_qpos)          # Standard Fetch setup
    
    # Force torso to maximum height
    set_joint_qpos(model, data, "robot0:torso_lift_joint", 0.4)
    
    # Force vertical orientation
    set_mocap_quat(model, data, "robot0:mocap", VERTICAL_QUAT)
    
    # Run settle steps
    for _ in range(ENV_SETUP_STEPS):          # 10 steps
        mj_step(model, data, nstep=n_substeps)
```

**Why torso at 0.4 (not full 0.3861)?** The range is [0.0386, 0.3861]. Setting 0.4 exceeds the upper limit by 0.014m, which MuJoCo clamps to 0.3861. This ensures the torso is at maximum height regardless of floating-point precision. With the torso at maximum, the shoulder is approximately 1.11m above the floor, enabling the arm to operate in crane mode over the 0.40m table surface with ~71cm of vertical clearance.

---

## Reset Sequence (`_reset_sim` in ChessTaskEnv)

The full per-episode reset sequence:

1. **Scenario Selection**: Random choice of `{transit, descend, ascend}` (or forced via `force_scenario`).
2. **Position Sampling**: Sample `start_xy` from board. Derive `arm_start_pos` and `goal_pos` based on scenario:
   - Transit: start at random XY at SAFE_Z, goal at different random XY at SAFE_Z
   - Descend: start and goal at same XY, start at SAFE_Z, goal at HOVER_Z
   - Ascend: start and goal at same XY, start at HOVER_Z, goal at SAFE_Z
3. **Parent Reset**: `super()._reset_sim()` (ChessSimulationEnv's version places cube on board)
4. **Object Placement**: Either hide at `[2, 2, 0.015]`, use `force_cube_pos`, or place at `start_xy` on table.
5. **Torso Lift**: Force to 0.4 (max) again (super() may have changed it).
6. **Phase 1 - Settle CLOSED**: Set `finger_target_joint = FINGER_CLOSED_JOINT`, call `_settle_arm_to_start(arm_start_pos)`. Arm moves to start with fingers closed.
7. **Phase 2 - Scripted Transitions**:
   - Descend: Open fingers over 150 steps while staying at start position
   - Ascend: Open fingers → Close fingers (simulates coming out of a descend)
   - Transit: Stay closed (fingers already closed from Phase 1)
8. **Phase 3 - Validation**: Assert finger joint within 0.5mm of target. Return `False` if failed (episode will be retried).

---

## Observation Space (RL, 25D)

The "Phase 9" observation vector uses the **"Holding Object" trick**:

| Index | Dimension | Content | Note |
|-------|-----------|---------|------|
| 0-2 | 3 | Gripper position (world XYZ) | Actual grip site |
| 3-5 | 3 | **Object position** → **mapped to gripper position** | Trick: tricks model into thinking it holds an object |
| 6-8 | 3 | Goal relative position (goal - grip) | `goal_pos - grip_pos` |
| 9-10 | 2 | Gripper state → **zero masked** | Hides actual finger state |
| 11-13 | 3 | Scenario one-hot `[transit, descend, ascend]` | Tells model which scenario |
| 14-19 | 6 | Object velocity → **zero masked** | Hides object movement |
| 20-22 | 3 | Gripper velocity | Actual grip site velocity |
| 23-24 | 2 | Finger velocity → **zero masked** | Hides finger state |

**Why the trick?** The base FetchPickAndPlace model was trained with a cube in its gripper. The observation at indices 3-5 was "object position" — normally the cube in the gripper. By mapping this to the gripper position itself, the model sees `obj_pos == grip_pos`, which it interprets as "I am holding the cube at my grip site." This activates the model's "transport" policy, which moves the arm smoothly toward the goal. Without this trick, the model would see the hidden cube at `[2, 2, 0.015]` relative to the gripper and produce erratic behavior.

---

## Action Space (RL, 4D)

`Box([-1, -1, -1, -1], [1, 1, 1, 1])` — all dimensions clipped to [-1, 1].

- `action[0]`: Δx (scaled by `POS_CTRL_SCALE=0.015` → max 15mm/step)
- `action[1]`: Δy (same)
- `action[2]`: Δz (same)
- `action[3]`: Gripper (ignored — always overridden to -1 then replaced by `finger_target_joint`)

The gripper action is completely bypassed. The state machine in `step()` enforces:
- Descend: `finger_target_joint = FINGER_OPEN_JOINT`
- Other: `finger_target_joint = FINGER_CLOSED_JOINT`

---

## Key Invariants for Correct Physics

1. **Every step, re-assert mocap_quat** after `_set_action()` is called. Without this, vertical orientation drifts.
2. **Zero velocities before `mj_forward`** after teleporting any body. Residual velocity causes physics jumps.
3. **Never set joint positions while in grasp_mode**. The actuator must drive the fingers during grasp to allow contact physics.
4. **Torso at 0.4 after every reset** — parent class may reset it to default.
5. **`ROBOT_DOF=15`** — always be specific about which DOFs to zero to avoid affecting the cube's physics.
