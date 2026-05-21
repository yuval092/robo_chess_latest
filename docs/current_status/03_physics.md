# Physics

## Simulation Parameters

- **Timestep**: 2ms (`option timestep="0.002"`)
- **Substeps**: 20 (from `MujocoFetchPickAndPlaceEnv` default), so each `env.step()` advances 40ms of simulation time.
- **Warmstart**: Enabled — solver reuses contact forces from previous step as initial guess. Improves convergence for repeated contacts.
- **Gravity**: Default `(0, 0, -9.81)` m/s².
- **Integrator**: Implicit Euler (MuJoCo default for `mujoco >= 3.x`).

---

## Solver Settings

No custom `<option solver="...">` is set, so MuJoCo defaults apply:
- **Constraint solver**: Newton (iterative)
- **Iterations**: 100 (default)
- **Tolerance**: 1e-8 (default)
- **Cone**: Pyramidal (default)

For rigid, high-stiffness contacts (cube on table), this converges quickly.

---

## Object (Cube) Physics

The cube is the most physically sensitive body in the simulation. All parameters were carefully tuned to ensure grasp stability:

### Mass & Inertia
```xml
mass="0.05"
```
50 grams. Light enough for the gripper to lift, heavy enough to rest stably on the table.

### Contact Model
```xml
condim="6"  friction="2.0 0.005 0.0001"  solref="0.002 1"  solimp="0.99 0.999 0.001"
```

- **condim=6**: Full contact dimensionality — includes tangential friction in 2D and torsional friction. Critical for preventing the cube from spinning out of the gripper.
- **friction `[μ_slide, μ_torsional, μ_rolling]`**:
  - `μ_slide = 2.0`: Very high sliding friction — cube grips the table and gripper fingers.
  - `μ_torsional = 0.005`: Low torsional friction — allows the cube to rotate slightly.
  - `μ_rolling = 0.0001`: Nearly zero rolling friction.
- **solref `[timeconst, dampratio]`**: `[0.002, 1]` — constraint time constant of 2ms = 1 simulation step. Extremely stiff contact response. Prevents cube from sinking into surfaces.
- **solimp `[dmin, dmax, width]`**: `[0.99, 0.999, 0.001]` — nearly 100% contact force transfer, nearly zero penetration depth.

### Free Joint Damping
```xml
<joint name="object0:joint" type="free" damping="0.1"/>
```
0.1 Ns/m on the 6-DOF free joint. Small but non-zero — damps oscillations after impact.

---

## Gripper Physics

### Finger Actuator Stiffness
```xml
kp="150000"
```
Position actuator with Kp=150,000 N/m. This is extremely stiff — produces up to ~22,500 N of contact force when the cube (at j≈0.0141) resists closure beyond `FINGER_CLOSED_JOINT=0.0`. In practice, the cube "stalls" the finger at j≈0.0141, and the finger contacts apply ~300-500 N of force to the cube, enough to hold it against gravity.

**Why 150,000?** This was specifically tuned:
- Too low (e.g., 5,000 from reference XML): Fingers can't overcome gravity during lift — cube drops.
- Too high: Physics explosion on first step of grasp.
- 150,000 with a linear ramp over 150 steps (~6 seconds of simulation) produces stable ~300 N grip force.

### Finger Mass
```xml
mass="4"
```
4 kg per finger. Heavy fingers reduce high-frequency oscillations in contact dynamics. The large inertia smooths out impulse spikes during the grasp ramp.

### Finger Damping
```xml
<joint damping="5000"/>
```
5000 Ns/m joint damping. Prevents finger "bouncing" during grasp closure.

### Finger Friction (vs Cube)
```xml
friction="10.0 0.5 0.01"
```
Very high sliding friction (10.0) — finger surface grips the cube surface strongly.

---

## Weld Constraint Physics

```xml
<weld body1="robot0:mocap" body2="robot0:gripper_link" solimp="0.9 0.95 0.001" solref="0.01 1"/>
```

- **solimp `[0.9, 0.95, 0.001]`**: Soft weld — allows ~5-10% compliance. This means the gripper doesn't instantly jump to the mocap position but converges over a few steps.
- **solref `[0.01, 1]`**: 10ms time constant. The gripper "chases" the mocap body with ~10ms lag.
- This softness is intentional: hard welds can cause instability when the mocap is moved too fast.

---

## Static Stability

### Arm Stability
The arm's 7 joints all have `damping=50` (except forearm roll: 3.5247) and zero stiffness. The mocap weld holds the gripper in place, and the PD-like constraint prevents the arm from sagging under gravity.

The 3 slide joints have `damping=1e+11` — effectively infinite. The arm base doesn't drift.

**Verify_physics test result**: After 100 no-action steps, the arm should not move more than ~0.001m from its initial position.

### Cube Stability
With `friction=2.0` and `solref="0.002 1"` (stiff contact), the cube does not drift on the table surface during idle steps. The high friction coefficient and stiff contact model prevent any sliding.

**Known issue**: If the cube is placed at the very edge of the table, it may slowly drift due to numerical integration. The `EDGE_MARGIN=0.04` in config prevents placement within 4cm of any edge.

---

## Grasp Physics: Why the Ramp Is Necessary

A direct jump from `FINGER_OPEN_JOINT=0.0181` to `FINGER_CLOSED_JOINT=0.0` in one step would produce an **instantaneous impulse** of:

```
F = Kp × (j_target - j_actual) = 150,000 × 0.0181 ≈ 2,715 N
```

This 2,715 N impulse in a single 2ms timestep produces an acceleration of:
```
a = F / m_cube = 2715 / 0.05 = 54,300 m/s²
```

...which launches the cube off the table at ~2.2 m/s in one step — a "physics explosion."

**Solution**: Ramp finger position from OPEN to CLOSED over 150 steps:
```
ramp_delta = (0.0181 - 0.0) / 150 ≈ 0.000121 per step
F_per_step = 150,000 × 0.000121 ≈ 18 N per step
```

18 N per step is well within the cube's stable operating range. The cube pushes back, finger stalls at j≈0.0141 (when contact force equals Kp × (0.0141 - 0.0) = 2,115 N in steady state), and the grasp stabilizes.

---

## Physics Coordinate System

MuJoCo uses a right-handed coordinate system:
- **+X**: Forward (toward table, away from arm)
- **+Y**: Left (when facing the table from the arm)
- **+Z**: Up

The `VERTICAL_QUAT = [1, 0, 1, 0]` (normalized to `[0.707, 0, 0.707, 0]`) rotates the gripper to point straight down (+Z becomes the gripper's pointing axis → negated = pointing down). This is the "crane mode" orientation.

---

## Known Physics Quirks & Patches

### 1. Gravity Sag in `_move_mocap_to`
The gripper experiences gravity sag — the weld constraint has finite stiffness, so the actual grip site position lags behind `mocap_pos` by a few mm due to gravity. `_move_mocap_to` addresses this by computing the error as:
```python
error = target_pos - grip_pos  # grip_pos is actual site position
self.data.mocap_pos[0][:3] += error
```
This feedback loop corrects for any systematic offset, including gravity sag.

### 2. Mocap Reset by `_set_action`
Every call to `_set_action` internally calls `self._utils.mocap_set_action(...)`, which first calls `reset_mocap2body_xpos` — this **resets mocap_pos to the current physical gripper_link position**, then applies the delta. After any `_set_action` call, any manually set `mocap_pos` is lost.

**Critical invariant**: After every `_set_action()` call, you must re-assert `data.mocap_pos` AND `data.mocap_quat` before calling `_mujoco_step`. Failure to do this allows the mocap to drift back to the body position.

### 3. Cube Rotation Abort
A cube rotated >25° diagonally (equivalent yaw from nearest square axis) has an effective corner-to-corner width of:
```
30mm × √2 × cos(45° - 25°) ≈ 42.4mm
```
This exceeds the maximum finger opening of ~38mm (2 × `FINGER_OPEN_JOINT=0.0181` × 2 = 36mm net, roughly). The cube corners would catch on the outside of the fingers instead of entering the gripper gap. The grasp pipeline aborts with `CUBE_ROTATED` reason.

### 4. ROBOT_DOF = 15
The Fetch robot has exactly 15 DOFs:
- 3 slide joints (base X, Y, Z)
- 1 torso lift joint
- 2 head joints (pan, tilt) — passive in our setup
- 7 arm joints (shoulder pan, shoulder lift, upperarm roll, elbow flex, forearm roll, wrist flex, wrist roll)
- 2 gripper finger joints

When zeroing robot velocities (e.g., in `soft_reset`), only `qvel[:15]` is zeroed — preserving object velocity physics.

### 5. Cube CoM Offset During Hold
When gripped, the cube's center of mass hangs ~15mm **below** the grip site:
```
grip_site_z - cube_CoM_z ≈ 0.015m
```
This is accounted for in `_check_cube_held` and `evaluate_grasp_quality`. The offset arises because the grip site is at the finger tips, but the cube center is at the cube's geometric center, which is inside the finger gap, not at the finger tips.
