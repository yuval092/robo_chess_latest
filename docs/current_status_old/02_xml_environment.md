# XML Environment

## File Structure

The MuJoCo scene is split across three XML files:
- `chess_env/assets/pick_and_place.xml` — top-level scene (floor, table, cube, lights, actuators)
- `chess_env/assets/shared.xml` — asset declarations, equality constraints, contact rules, defaults
- `chess_env/assets/robot.xml` — full Fetch robot body hierarchy + mocap body

`pick_and_place.xml` includes the others via `<include file="shared.xml">` and `<include file="robot.xml">`.

---

## Compiler & Solver Options

```xml
<compiler angle="radian" meshdir="../stls/fetch" texturedir="../textures"/>
<option timestep="0.002">
    <flag warmstart="enable"/>
</option>
```

- **timestep**: 2ms. At `n_substeps=20` (default Fetch), one `env.step()` call advances physics by 40ms.
- **warmstart**: Enables warm-starting of contact constraints (improves convergence).
- **meshdir**: Points to `.stl` files for all Fetch arm links (relative to the XML file).
- Gravity is at MuJoCo default: `0 0 -9.81`.

---

## Floor

```xml
<geom name="floor0" pos="0.88 0.2641 0" size="1.0 1.0 1" type="plane" condim="3" material="floor_mat"/>
<body name="floor0" pos="0.88 0.2641 0">
    <site name="target0" pos="0 0 0.5" .../>
</body>
```

- Infinite plane geometry centered at `(0.88, 0.2641, 0)`.
- `condim=3`: 3D frictional contact (default).
- Material: `floor_mat` (dark grey, `rgba="0.2 0.2 0.2 1"`).
- The `target0` site is inherited from the Fetch framework and unused.

---

## Table (Board Substitute)

```xml
<body pos="0.88 0.2641 0.2" name="table0">
    <geom size="0.28 0.28 0.2" type="box" mass="2000" material="table_mat"/>
</body>
```

- **Type**: Single solid box (no legs).
- **Position**: Center at world `(0.88, 0.2641, 0.2)`.
- **Size (half-extents)**: `0.28 × 0.28 × 0.20` → actual dimensions: **56cm × 56cm × 40cm**.
- **Top surface Z**: `0.2 + 0.2 = 0.400m` = `TABLE_SURFACE_Z`.
- **Mass**: 2000 kg — effectively static (won't move under any robot force).
- **No explicit friction/solimp**: relies on MuJoCo defaults. The cube's own friction parameters dominate cube-table interaction.
- The table completely fills the space from floor to 0.40m — there is no opening underneath.

> **Note**: The Python config uses `table_half_x = 0.28` and `table_half_y = 0.28`, giving an effective "usable" range of `56cm × 56cm`. A 4cm edge margin reduces the sampling region to approximately `48cm × 48cm`.

---

## Cube (Chess Piece Placeholder)

```xml
<body name="object0" pos="0.01 0.01 0.01">
    <joint name="object0:joint" type="free" damping="0.1"/>
    <geom size="0.015 0.015 0.015" type="box" condim="6" name="object0"
          material="block_mat" mass="0.05"
          friction="2.0 0.005 0.0001"
          solref="0.002 1" solimp="0.99 0.999 0.001"/>
    <site name="object0" pos="0 0 0" size="0.015 0.015 0.015" rgba="1 0 0 1" type="sphere"/>
</body>
```

- **Size (half-extents)**: `0.015 × 0.015 × 0.015` → **30mm × 30mm × 30mm** cube.
- **Free joint**: 6-DOF (3 translation + 3 rotation) allows the cube to move freely.
- **Damping**: 0.1 on the free joint — slight damping to prevent perpetual oscillation.
- **Mass**: 0.05 kg = 50g. Light enough to be lifted, heavy enough to be stable.
- **condim=6**: Full cone friction (tangential + torsional resistance). Critical for grasp stability.
- **Friction**: `[2.0, 0.005, 0.0001]` = sliding friction coeff, torsional friction, rolling friction. High sliding friction (2.0) prevents cube from slipping during grasp.
- **solref**: `[0.002, 1]` — very stiff contact solver reference (2ms time constant). Prevents cube from sinking into surfaces.
- **solimp**: `[0.99, 0.999, 0.001]` — high impedance, very low penetration allowed.
- Initial position `(0.01, 0.01, 0.01)` is overridden in `_reset_sim`.

---

## Robot (Mocap Body + Fetch Arm)

### Mocap Body

```xml
<body mocap="true" name="robot0:mocap" pos="0 0 0">
    <geom conaffinity="0" contype="0" pos="0 0 0" rgba="0 0.5 0 0.7" size="0.005 0.005 0.005" type="box"/>
    <!-- 3 more visual-only axis indicator geoms -->
</body>
```

- **mocap=true**: Position/orientation is directly set by `data.mocap_pos[0]` / `data.mocap_quat[0]` from Python — it is not simulated by physics.
- All geoms have `contype=0, conaffinity=0` → no collision.
- Visually shows as a small green marker + axis lines.
- Connected to `robot0:gripper_link` via weld equality constraint in `shared.xml`.

### Equality Constraint (Weld)

```xml
<equality>
    <weld body1="robot0:mocap" body2="robot0:gripper_link" solimp="0.9 0.95 0.001" solref="0.01 1"/>
</equality>
```

This is the **core control mechanism**: the weld constraint drives `gripper_link` to follow `mocap`. Moving `mocap_pos` effectively moves the end-effector. The arm's joints are solved by the physics engine to satisfy the kinematic constraint.

- **solimp**: `[0.9, 0.95, 0.001]` — soft weld, allows some compliance.
- **solref**: `[0.01, 1]` — 10ms time constant for constraint convergence.

### Robot Base

```xml
<body childclass="robot0:fetch" name="robot0:base_link" pos="0.2869 0.2641 0">
    <joint axis="1 0 0" name="robot0:slide0" type="slide" damping="1e+11" armature="0.0001"/>
    <joint axis="0 1 0" name="robot0:slide1" type="slide" damping="1e+11" armature="0.0001"/>
    <joint axis="0 0 1" name="robot0:slide2" type="slide" damping="1e+11" armature="0.0001"/>
```

- **Position**: World `(0.2869, 0.2641, 0)`.
- **3 slide joints** (DOFs 0-2 of robot): allow the base to translate in XYZ but with extreme damping (`1e+11`) — effectively locked unless explicitly driven. Their initial positions are set via `initial_qpos` in Python.
- Mass: 70 kg body inertia.

### Torso Lift

```xml
<body name="robot0:torso_lift_link" pos="-0.0869 0 0.3774">
    <joint axis="0 0 1" name="robot0:torso_lift_joint" range="0.0386 0.3861" type="slide" damping="1e+07"/>
```

- Slides vertically from 0.0386m to 0.3861m of lift.
- Forced to maximum (0.3861 = ~0.386m) during `_env_setup` and every `_reset_sim` to maximize arm reach height.
- Body local offset: `(-0.0869, 0, 0.3774)` from base_link.
- At max lift, torso base height = `0.3774 + 0.3861 = 0.7635m` above robot base.

### Arm Links (shoulder → wrist)

Seven revolute/continuous joints form the arm chain. Key joint names and their axes:

| Joint | Axis | Range |
|-------|------|-------|
| `robot0:shoulder_pan_joint` | Z (yaw) | ±1.6056 rad |
| `robot0:shoulder_lift_joint` | Y | -1.221 to 1.518 rad |
| `robot0:upperarm_roll_joint` | X | ±6.28 rad (continuous) |
| `robot0:elbow_flex_joint` | Y | ±2.251 rad |
| `robot0:forearm_roll_joint` | X | ±6.28 rad (continuous) |
| `robot0:wrist_flex_joint` | Y | ±2.16 rad |
| `robot0:wrist_roll_joint` | X | ±6.28 rad (continuous) |

All joints use class `robot0:fetch` defaults: `armature=1, damping=50, frictionloss=0, stiffness=0`. The forearm roll joint is an exception with `armature=2.7538, damping=3.5247, stiffness=10`.

Link lengths along the arm kinematic chain (from `pos` attributes):
- shoulder_lift offset from shoulder_pan: `(0.117, 0, 0.06)`
- upperarm from shoulder_lift: `(0.219, 0, 0)`
- elbow from upperarm: `(0.133, 0, 0)`
- forearm from elbow: `(0.197, 0, 0)`
- wrist_flex from forearm: `(0.1245, 0, 0)`
- wrist_roll from wrist_flex: `(0.1385, 0, 0)`
- gripper_link from wrist_roll: `(0.1664, 0, 0)`

Total kinematic chain length (extended): ~1.10m.

### Gripper

```xml
<body childclass="robot0:fetchGripper" name="robot0:r_gripper_finger_link" pos="0 0.0159 0">
    <joint axis="0 1 0" name="robot0:r_gripper_finger_joint" range="0 0.05" damping="5000"/>
    <geom pos="0 -0.008 0" size="0.0385 0.007 0.0135" type="box"
          friction="10.0 0.5 0.01" condim="4" .../>
</body>
<body childclass="robot0:fetchGripper" name="robot0:l_gripper_finger_link" pos="0 -0.0159 0">
    <joint axis="0 -1 0" name="robot0:l_gripper_finger_joint" range="0 0.05" damping="5000"/>
    <geom pos="0 0.008 0" size="0.0385 0.007 0.0135" type="box"
          friction="10.0 0.5 0.01" condim="4" .../>
</body>
<site name="robot0:grip" pos="0.02 0 0" rgba="0 0 0 0" size="0.02 0.02 0.02"/>
```

- **Right finger**: translates in +Y direction (joint axis=`0 1 0`).
- **Left finger**: translates in -Y direction (joint axis=`0 -1 0`).
- Both fingers range from 0 (closed) to 0.05m (fully open) from gripper center.
- **Damping**: 5000 Ns/m — very high to prevent finger oscillation.
- **Finger mass**: 4 kg each — heavy to prevent physics explosions during contact.
- **Finger geom size**: `0.0385 × 0.007 × 0.0135` (half-extents) → pads are ~77mm long, 14mm wide, 27mm tall.
- **Friction**: `10.0 0.5 0.01` — very high friction for secure grasping.
- **condim=4**: 4D contact (includes torsional friction, important for grasping).
- **grip site**: Located 20mm forward of gripper_link origin; serves as the control reference point for all position calculations in Python code.

### Contact Exclusions

```xml
<contact>
    <exclude body1="robot0:r_gripper_finger_link" body2="robot0:l_gripper_finger_link"/>
    <exclude body1="robot0:torso_lift_link" body2="robot0:torso_fixed_link"/>
    <exclude body1="robot0:torso_lift_link" body2="robot0:shoulder_pan_link"/>
</contact>
```

- Fingers don't collide with each other (prevents phantom forces during grasp).
- Torso lift doesn't collide with fixed torso or shoulder (prevents spurious joint forces during lift).

---

## Actuators

```xml
<actuator>
    <position ctrllimited="true" ctrlrange="0 0.05"
              joint="robot0:l_gripper_finger_joint" kp="150000"
              name="robot0:l_gripper_finger_joint" user="1"/>
    <position ctrllimited="true" ctrlrange="0 0.05"
              joint="robot0:r_gripper_finger_joint" kp="150000"
              name="robot0:r_gripper_finger_joint" user="1"/>
</actuator>
```

- Only the two gripper finger joints have actuators (indices 0 and 1 in `data.ctrl`).
- **Type**: Position actuator — `ctrl` sets the target joint position.
- **Kp**: 150,000 N/m — extremely stiff. This produces forces up to ~22,500 N for a 0.15m error. Was specifically tuned to be strong enough to hold the cube against gravity.
- **ctrlrange**: [0, 0.05] matching joint range.
- The arm's 7 revolute joints are **not actuated** in the traditional sense — they are driven entirely by the weld constraint (mocap body) via the physics solver.

---

## Materials and Visual Assets

All materials defined in `shared.xml`:
- `floor_mat`: Dark grey `(0.2, 0.2, 0.2)`
- `table_mat`: Near-white `(0.93, 0.93, 0.93)`
- `block_mat`: Dark grey `(0.2, 0.2, 0.2)` — cube color
- `robot0:*mat` variants: Grey/blue arm aesthetic

Two textures:
- `texture_block`: Cube texture (loaded from `block.png`)
- Skybox: Green gradient

---

## Key Coordinate Facts

| Object | World Position | Notes |
|--------|---------------|-------|
| Floor | `(0.88, 0.2641, 0)` | Infinite plane |
| Table center | `(0.88, 0.2641, 0.2)` | Half-size 0.28×0.28×0.20 |
| Table top (Z) | `0.400m` | `TABLE_SURFACE_Z` |
| Cube CoM (on table) | `(x, y, 0.415)` | `TABLE_SURFACE_Z + CUBE_HEIGHT/2` |
| Robot base | `(0.2869, 0.2641, 0)` | Fixed in world |
| Table near edge (X) | `0.60m` | `0.88 - 0.28` |
| Table far edge (X) | `1.16m` | `0.88 + 0.28` |
| Table near edge (Y) | `-0.0159m` | `0.2641 - 0.28` |
| Table far edge (Y) | `0.5441m` | `0.2641 + 0.28` |
