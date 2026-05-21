# XML Environment

## File Layout

MuJoCo scenes are split across three XML files in `chess_env/assets/`:

| File | Role |
|------|------|
| `pick_and_place.xml` | Root scene. Includes robot.xml and shared.xml. Defines table, object, actuators, sensors. |
| `shared.xml` | Assets only: materials, mesh paths, equality constraints, contact exclusions, compiler options. |
| `robot.xml` | Full Fetch robot body tree: torso, arm links, wrist, gripper. Placed via `<body pos="0.56 0.2641 0">`. |

These files are NOT loaded directly by MuJoCo API calls in our code. Instead, `pick_and_place.xml` is monkey-patched into `MujocoFetchPickAndPlaceEnv.MODEL_XML_PATH` before calling `super().__init__()`. See `src/chess_env/simulation.py` and the MuJoCo & Gymnasium-Robotics doc for the patching mechanism.

---

## Scene Structure (pick_and_place.xml)

### Floor

```xml
<geom name="floor0" type="plane" pos="0 0 0" size="0 0 1" material="floor_mat"
      condim="3" friction="1 0.005 0.0001" solref="0.002 1"/>
```

Infinite plane at Z=0. No collision with the arm base (contact exclusion is on `robot0:base_link` ↔ `table0` in `shared.xml`).

### Table

The table is a 70×70cm surface at Z=0.400m with four legs, centered at world `(0.88, 0.2641, 0)`.

```xml
<body pos="0.88 0.2641 0" name="table0">
    <!-- Surface slab: 70cm × 70cm × 5cm; top face at z=0.400m -->
    <geom name="table0_surface" type="box" size="0.35 0.35 0.025"
          pos="0 0 0.375" mass="2000" material="table_mat"
          solref="0.002 1" solimp="0.99 0.999 0.001"/>

    <!-- Four legs: positioned at ±0.31m from center (within surface footprint) -->
    <geom name="table0_leg_far_plus"  type="box" size="0.04 0.04 0.1875"
          pos="0.31  0.31  0.1875" mass="0" material="table_mat"/>
    <geom name="table0_leg_far_minus" type="box" size="0.04 0.04 0.1875"
          pos="0.31 -0.31  0.1875" mass="0" material="table_mat"/>
    <geom name="table0_leg_near_plus" type="box" size="0.04 0.04 0.1875"
          pos="-0.31  0.31  0.1875" mass="0" material="table_mat"/>
    <geom name="table0_leg_near_minus" type="box" size="0.04 0.04 0.1875"
          pos="-0.31 -0.31  0.1875" mass="0" material="table_mat"/>
</body>
```

Table geometry in world frame:
- Surface top: Z = 0 + 0.375 + 0.025 = **0.400m**
- Near edge (X): 0.88 − 0.35 = **0.53m** (usable with 4cm margin: 0.57m)
- Far edge (X): 0.88 + 0.35 = **1.23m** (usable with 4cm margin: 1.19m)
- Right edge (Y): 0.2641 − 0.35 = **−0.0859m** (usable with 4cm margin: −0.046m)
- Left edge (Y): 0.2641 + 0.35 = **0.6141m** (usable with 4cm margin: 0.574m)

The table has `mass="2000"` to prevent movement when grasped pieces fall or the arm applies side force. Legs have `mass="0"` since they're purely visual structure.

### Cube (Manipulation Object)

```xml
<body name="object0" pos="0.01 0.01 0.01">
    <freejoint name="object0:joint"/>
    <geom name="object0" type="box" size="0.015 0.015 0.015"
          mass="0.05" condim="6" friction="1 0.005 0.0001"
          solimp="0.99 0.999 0.001" solref="0.002 1"/>
</body>
```

30mm cube (half-size 0.015m), mass 50g, `condim=6` for full frictional contact (resists rotation). Position is managed at runtime by `_reset_sim` via `data.qpos` manipulation, not by XML `pos` attribute.

Key cube geometry constants:
- `CUBE_HEIGHT = 0.030m` (full height)
- Bottom at table surface: Z = 0.400m
- Center (default rest): Z = 0.415m
- Top: Z = 0.430m = `GRASP_Z`

### Actuators

Two position actuators drive the gripper fingers:

```xml
<position name="robot0:l_gripper_finger_joint"
          joint="robot0:l_gripper_finger_joint"
          ctrlrange="0 0.05" kp="20000"/>
<position name="robot0:r_gripper_finger_joint"
          joint="robot0:r_gripper_finger_joint"
          ctrlrange="0 0.05" kp="20000"/>
```

- **Kp = 20000**: High gain ensures near-rigid position tracking. Each finger generates force = 20000 × (ctrl − qpos).
- **ctrlrange = [0, 0.05]**: 0 = fully closed, 0.05 = fully open (50mm finger separation is ~100mm total gripper width).
- `data.ctrl[l_id]` and `data.ctrl[r_id]` must be set to match the current finger target any time fingers are teleported (see Scripted Actions doc).

### Sensors

```xml
<sensor>
    <touch name="robot0:l_finger_touch" site="robot0:l_finger_touch_site"/>
    <touch name="robot0:r_finger_touch" site="robot0:r_finger_touch_site"/>
</sensor>
```

Touch sensors on finger pads. Available via `data.sensor` but not currently used in scripted pipeline (grasp success is determined geometrically, not by touch sensor readings).

---

## Shared Assets (shared.xml)

### Key Materials

- `table_mat`: Table surface appearance (brown)
- `floor_mat`: Floor tile appearance
- `fetchGripper`: Gripper geometry class with `solimp="0.99 0.999 0.001"` for stiff contact

### Equality Constraints

```xml
<equality>
    <weld name="robot0:mocap_weld" body1="robot0:mocap" body2="robot0:gripper_link"
          solref="0.001 1" solimp="0.9 0.95 0.001"/>
</equality>
```

This weld is the foundation of arm control. It forces `robot0:gripper_link` to follow the position and orientation of the **mocap body** (`robot0:mocap`) at each timestep. Moving the mocap body causes the physics solver to apply forces and torques on all arm joints to satisfy the constraint.

The `solref` and `solimp` values are tuned for stiffness without causing instability. A tighter weld (e.g., `solref="0.0001 1"`) would cause constraint explosions; the current values give ~1mm compliance.

### Contact Exclusion

```xml
<contact>
    <exclude name="base_table" body1="robot0:base_link" body2="table0"/>
</contact>
```

Prevents MuJoCo from generating contact forces between the robot base mesh and the table geometry. Without this exclusion, the arm base (which is at x=0.56, just outside the near table edge at 0.53m) would generate large interpenetration forces that destabilize the simulation.

---

## Robot XML (robot.xml)

### Arm Placement

```xml
<body childclass="robot0:fetch" name="robot0:base_link" pos="0.56 0.2641 0">
```

The entire arm tree is rooted here. Changing this `pos` attribute moves the arm in world space.

- x=0.56 places the arm 3cm outside the near table edge (table near edge = 0.53m; usable board edge = 0.57m)
- y=0.2641 aligns the arm with the table center (table cy = 0.2641m)

**Why x=0.56 (not the original x=0.60):** Moving the arm 4cm further back allows it to escape the kinematic dead zone at close X distances. At x=0.60, the nearest chess row (x≈0.61) is only 1cm in front of the arm base, making it unreachable. At x=0.56, combined with torso_height=0.3661m, all 64 chess squares are reachable within 5mm.

### Torso Joint

```xml
<joint name="robot0:torso_lift_joint" type="slide" axis="0 0 1"
       range="0.0386 0.3861" damping="1e+11"/>
```

- **Type**: Slide (prismatic, vertical Z movement)
- **Range**: 3.86cm to 38.61cm above base
- **Damping**: 1e+11 N·s/m — extremely high, effectively overdamped. The torso stays wherever it's positioned.
- **Operating height**: Set to **0.3661m** at each `_reset_sim` via config `torso_height: 0.3661`. This is 94.8% of the joint maximum (0.3861m). The high torso position raises the shoulder so the arm can "look down" to reach near-row chess squares (x≈0.61) that would fall in a kinematic dead zone at lower torso heights. See Known Kinematic Dead Zone below.

### Gripper Body and Mocap Body

```xml
<body name="robot0:mocap" mocap="true">
    <!-- Invisible body; position is set by data.mocap_pos[0] -->
</body>
```

The mocap body has no mass or geometry. It is the control target: scripts set `data.mocap_pos[0]` and `data.mocap_quat[0]`, and the weld constraint forces the arm's `gripper_link` body to follow.

### Grip Site

```xml
<site name="robot0:grip" pos="0 0 0.1034" size="0.005"/>
```

A virtual measurement point 10.34cm below the gripper link body. This is the reference point for all reach calculations, grasp Z-height checks, and position errors. All scripts query `get_site_xpos(model, data, "robot0:grip")` rather than the body position.

---

## Kinematic Reachability

All **64 chess square centers** are reachable at both `SAFE_Z` (0.550m) and `GRASP_Z` (0.430m) with maximum error < 5mm (confirmed: 2.9mm max, verified by `scripts/verify_physics.py`).

**Why this required tuning**: The near chess row at x≈0.61 is close to the arm's own centerline. Without careful placement, the arm folds back on itself and cannot reach these positions. The solution requires both:
1. **arm_x = 0.56** (4cm behind original 0.60) — increases horizontal clearance from 1cm to 5cm for the near row
2. **torso_height = 0.3661m** (raised from original 0.25m) — lifts the shoulder so the arm can descend to reach near-row targets

**Near table edge (x=0.57)**: NOT a chess square. This position (the absolute board edge) remains unreachable at y=0.2641 (arm centerline). This is acceptable: no chess square is placed at the absolute edge.

**Chess square positions (70×70cm board)**:
- Near row (row 0) center X: 0.609m (4.9cm in front of arm)  
- Far row (row 7) center X: 1.151m
- Max reachability error (definitive test): 2.9mm at row 0, col 3 (near center), SAFE_Z
