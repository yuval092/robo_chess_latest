# Physics

## MuJoCo Simulation Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Timestep | 0.002s (2ms) | Per `n_substeps=20` at 0.1ms internal |
| Substeps per step | 20 | Each `mj_step` advances 20×0.1ms = 2ms |
| Gravity | −9.81 m/s² (Z) | Standard |
| Solver | Newton | MuJoCo default constraint solver |

---

## Table Physics

The table surface uses stiff contact parameters to prevent cube penetration and ensure stable placement:

```xml
solref="0.002 1" solimp="0.99 0.999 0.001"
```

- `solref="0.002 1"`: Contact time constant 2ms (very stiff) with critical damping.
- `solimp="0.99 0.999 0.001"`: Near-rigid — impedance range 0.99–0.999, width 0.001m.

The table body has `mass="2000"` kg. This makes it effectively immovable under any arm or cube force.

---

## Cube Physics

```xml
<geom name="object0" type="box" size="0.015 0.015 0.015"
      mass="0.05" condim="6" friction="1 0.005 0.0001"
      solimp="0.99 0.999 0.001" solref="0.002 1"/>
```

| Property | Value | Notes |
|----------|-------|-------|
| Size | 30mm cube (half 0.015m) | Chess piece scale |
| Mass | 50g | Light enough for Fetch actuators |
| `condim` | 6 | Full frictional contact: resists rotation |
| Friction | `1 0.005 0.0001` | Sliding/rolling/spinning |
| Contact | Stiff (same as table) | Stable resting |

The cube starts at rest with its bottom face on the table surface. Its center is at Z = TABLE_SURFACE_Z + CUBE_HEIGHT/2 = 0.400 + 0.015 = **0.415m**. The cube top is at Z = 0.430m = GRASP_Z.

---

## Gripper Physics

### Finger Joints

```
joint: robot0:l_gripper_finger_joint
       robot0:r_gripper_finger_joint
type:  slide (prismatic, symmetric)
range: [0, 0.05]   → 0 = closed, 0.05 = fully open
```

Both fingers move symmetrically. The joint value is the displacement of each finger from the centerline — a joint value of 0.018m means each finger is 18mm from center (36mm total gap).

### Finger Actuators

```xml
<position name="robot0:l_gripper_finger_joint"
          joint="robot0:l_gripper_finger_joint"
          ctrlrange="0 0.05" kp="20000"/>
```

- **Position actuator**: Each timestep applies torque = Kp × (`data.ctrl[act_id]` − `data.qpos[jnt_dofadr]`)
- **Kp = 20000** N/m: Very stiff. For a 1mm deviation, force ≈ 20N. This means when fingers are commanded to close (ctrl=0) but are held open by a cube (qpos≈0.014), they exert approximately 280N of grasp force.

### Critical: ctrl vs qpos Sync

When `_set_gripper_state()` teleports the finger joints (writes `data.qpos`), it **must also write `data.ctrl`** to the same value. If ctrl is not updated, the position actuator will drive fingers toward the old ctrl target on the next `mj_step`, undoing the teleport.

```python
def _set_gripper_state(self):
    target = self.finger_target_joint
    # Set physical position:
    set_joint_qpos(model, data, "robot0:l_gripper_finger_joint", target)
    set_joint_qpos(model, data, "robot0:r_gripper_finger_joint", target)
    # Zero velocity:
    data.qvel[l_dof] = 0.0
    data.qvel[r_dof] = 0.0
    # CRITICAL — sync actuator target:
    data.ctrl[l_id] = target
    data.ctrl[r_id] = target
    mujoco.mj_forward(model, data)
```

### Grasp Force Calculation

When closing on a 30mm cube:
- Cube half-width: 15mm
- Finger stall position: ~14mm (each finger)
- Finger error at stall: ctrl=0, qpos≈0.014 → force ≈ 20000 × 0.014 = **280N per finger**
- Two fingers: **560N total** — more than sufficient to hold a 50g cube under acceleration

The grasp verify threshold in `env.yaml` is `grasp_verify_finger_threshold: 0.016`. If each finger is at or above 16mm, the grasp is considered failed (fingers didn't close). The natural stall for a 30mm cube is ~14mm — this is comfortably below the 16mm threshold.

---

## Mocap/Weld Constraint Physics

### The Weld Mechanism

The equality weld forces `gripper_link` body to follow `mocap` body. MuJoCo implements this as a bilateral constraint: at each step, constraint forces are applied to make the body's position and orientation match the mocap target.

**Key properties**:
- The weld is compliant (not perfectly rigid): `solref="0.001 1" solimp="0.9 0.95 0.001"`
- The grip SITE position lags the mocap position by ~0.5–2mm at typical speeds due to compliance
- When `_set_action(np.zeros(4))` is called, `mocap_set_action` resets `mocap_pos` to the current `gripper_link` body position (not the grip site)

**Why the error-delta pattern works**:
```python
grip_pos = get_site_xpos("robot0:grip")  # Actual grip position
error = target - grip_pos
_set_action(np.zeros(4))                 # Resets mocap → current body position
data.mocap_pos[0][:3] += error           # Moves mocap toward target (overshoots body)
data.mocap_quat[0][:] = target_quat
_mujoco_step(None)
```
The mocap is set to `current_body_pos + error_from_site`. Because the site lags the body slightly, this effectively commands the mocap to a position slightly past the target — the constraint then pulls the body (and therefore the site) toward the target. This converges stably within 30–120 steps for typical distances.

---

## Arm Physics During Movement

### Transit (Horizontal)

During transit at SAFE_Z = 0.550m, the arm moves through free space. The main constraint is:
- `grip_z > FLOOR_LIMIT = 0.400m` — don't touch table (monitored by `run_transit`)
- When `grasp_mode=True` (cube held): cube proximity to grip checked each step

### Descend/Ascend (Vertical)

The arm moves vertically inside a cylindrical "tube":
- `tube_center_xy`: The XY position the arm is supposed to stay above
- `drift_limit = 10mm`: If `||grip_xy - tube_center_xy|| > 0.010`, abort with TUBE_BREACH
- `grip_z > TABLE_SURFACE_Z = 0.400m`: Don't crash into table

The tube constraint forces vertical-only movement and prevents the arm from sweeping across pieces during descent/ascent.

**Why XY drift occurs**: The proportional controller applies error as a delta to mocap_pos. As the arm descends, the changing joint configuration causes slight lateral coupling in the kinematic chain. Drift is typically 0–8mm in normal operation; the 10mm limit provides ~2mm headroom.

### Grasp Phases

During the 6-phase grasp pipeline, the arm is driven by `_move_mocap_to` with specific targets. The physics engine handles all force calculations — the script simply specifies targets. Key forces:
- Phase 3 (plunge): Gravity + constraint bring the arm straight down
- Phase 4/5 (close fingers): Kp=20000 actuator closes fingers; cube contact generates normal force; friction prevents cube sliding
- Phase 6 (verify): Geometry check (no additional forces)

---

## Contact Model

### Table-Arm Exclusion

```xml
<exclude body1="robot0:base_link" body2="table0"/>
```

The arm's base mesh (at x=0.60) partially overlaps the table geometry near its near edge. Without this exclusion, MuJoCo generates large interpenetration forces. The exclusion suppresses all contacts between these two bodies, allowing the base to geometrically overlap the table without physics consequences.

### Cube-Table Contact

When the cube rests on the table:
- Normal force ≈ mass × g = 0.05 × 9.81 ≈ 0.49N upward
- With `solimp="0.99 0.999 0.001"`: Near-rigid contact, negligible cube sinking
- Contact drift over 100 steps: < 0.001mm (confirmed by `test_static_stability`)

### Cube-Finger Contact

When fingers grip the cube:
- `condim=6`: Full frictional model (normal + 2 tangential + torsional + rolling + 2 conical)
- High Kp (20000) means the "intended position" of the finger is 0 (closed). Even as the cube pushes the finger open to 14mm, the actuator exerts ~280N per finger.
- The friction cone at the finger-cube interface is wide enough to prevent the cube from rotating or sliding out under gravity during transit/ascend.

---

## Stability Constants

From `configs/physics.yaml`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `settle_tolerance` | 0.003m (3mm) | Arm settle threshold in `_settle_arm_to_start` |
| `VERTICAL_QUAT` | [0, 1, 0, 0] | Gripper orientation (downward-facing) |

The VERTICAL_QUAT `[w=0, x=1, y=0, z=0]` is a 180° rotation around the X axis, which points the gripper straight down. This is the only gripper orientation used throughout the scripted pipeline.
