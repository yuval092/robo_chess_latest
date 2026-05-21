# Stage 1: Environment Geometry

## Goal

Replace the current solid-box table with a physically accurate 4-legged table (60×60cm surface), reposition the arm base closer to the board to improve reachability at all positions, and update the associated config values. The `TABLE_SURFACE_Z = 0.400` constraint is preserved exactly so that all existing Z-level constants remain valid.

---

## Preconditions

- The project runs correctly in its current state (the RL model is still present; this stage makes no Python changes).
- `python scripts/verify_physics.py` passes all 5 tests before starting.

---

## Coordinate Reference Frame

All coordinates are in world space:

| Landmark | World X | World Y | World Z |
|----------|---------|---------|---------|
| Table center (XY) | 0.88 | 0.2641 | — |
| Table surface top | 0.88 | 0.2641 | **0.400** |
| Table near edge (−X) | **0.58** | 0.2641 | 0.400 |
| Table far edge (+X) | **1.18** | 0.2641 | 0.400 |
| Table left edge (+Y) | 0.88 | **0.5641** | 0.400 |
| Table right edge (−Y) | 0.88 | **−0.0359** | 0.400 |
| Current arm base | 0.2869 | 0.2641 | 0 |
| **New arm base** | **0.60** | 0.2641 | 0 |

---

## 1.1 — New Table Design: Geometry Derivation

### Surface Slab

- Surface top must stay at Z = 0.400 (matches `TABLE_SURFACE_Z`).
- Slab thickness: 5cm (half-thickness = 0.025m) — same as reference.
- Slab occupies: Z ∈ [0.375, 0.400].
- Half-size in XY: 0.30 × 0.30 (60cm × 60cm board).
- Geom position relative to table body origin at (0.88, 0.2641, 0): `pos="0 0 0.375"`.
- Geom size: `size="0.30 0.30 0.025"`.

### Legs

- Leg top must meet slab bottom at Z = 0.375.
- Leg half-height = 0.1875m → leg occupies Z ∈ [0, 0.375].
- Leg center Z = 0.1875m.
- Leg cross-section: 8cm × 8cm (half = 0.04m each side) — same as reference.
- Leg XY positions: flush with table corners.
  - Leg center offset from table center: 0.30 − 0.04 = 0.26m (so outer leg face aligns with board edge).
  - 4 positions (relative to table body): `(±0.26, ±0.26)`.

| Leg Name | Relative pos (X, Y) | World pos (X, Y) |
|----------|---------------------|------------------|
| far_plus | (0.26, 0.26) | (1.14, 0.5241) |
| far_minus | (0.26, −0.26) | (1.14, 0.0041) |
| near_plus | (−0.26, 0.26) | (0.62, 0.5241) |
| near_minus | (−0.26, −0.26) | (0.62, 0.0041) |

All legs: `size="0.04 0.04 0.1875"`, relative `pos="±0.26 ±0.26 0.1875"`.

### Arm Base Positioning

New arm base: `pos="0.60 0.2641 0"`.

- The near table edge is at world X = 0.58. The arm base at 0.60 is **2cm inside** the table boundary → "partly under the table". ✓
- The arm base sits between the two near legs in Y: legs at Y = 0.0041 and Y = 0.5241; arm at Y = 0.2641 (midpoint). ✓
- No physical overlap: the near legs are at X = 0.62, ±0.04 (occupying X ∈ [0.58, 0.66]). The arm base is at X = 0.60, Y = 0.2641. The legs are at Y = 0.0041 and Y = 0.5241 — the arm base Y = 0.2641 is 0.26m from each leg center in Y. No collision. ✓

### Reachability Verification

| Target position | Horizontal distance from new base (0.60, 0.2641) |
|----------------|--------------------------------------------------|
| Near corner (0.58, −0.0359) | √(0.02² + 0.30²) = **0.301m** |
| Near corner (0.58, 0.5641) | √(0.02² + 0.30²) = **0.301m** |
| Far corner (1.18, −0.0359) | √(0.58² + 0.30²) = **0.654m** |
| Far corner (1.18, 0.5641) | √(0.58² + 0.30²) = **0.654m** |
| Center (0.88, 0.2641) | 0.28m |

Fetch arm reach ≈ 1.0m. All targets are within 0.654m horizontal distance. ✓

Note: The near corners are 0.301m from the arm base and 2cm behind it in X. If crane mode causes singularity at this position, increase arm base X to 0.58 (at the exact near edge). This should be verified in Stage 1 validation.

---

## 1.2 — File Changes

### File: `chess_env/assets/pick_and_place.xml`

**Current `table0` body (lines 18–20):**
```xml
<body pos="0.88 0.2641 0.2" name="table0">
    <geom size="0.28 0.28 0.2" type="box" mass="2000" material="table_mat"></geom>
</body>
```

**Replace with:**
```xml
<body pos="0.88 0.2641 0" name="table0">
    <geom name="table0_surface" type="box" size="0.30 0.30 0.025"
          pos="0 0 0.375" mass="2000" material="table_mat"/>
    <geom name="table0_leg_far_plus" type="box" size="0.04 0.04 0.1875"
          pos="0.26 0.26 0.1875" mass="0" material="table_mat"/>
    <geom name="table0_leg_far_minus" type="box" size="0.04 0.04 0.1875"
          pos="0.26 -0.26 0.1875" mass="0" material="table_mat"/>
    <geom name="table0_leg_near_plus" type="box" size="0.04 0.04 0.1875"
          pos="-0.26 0.26 0.1875" mass="0" material="table_mat"/>
    <geom name="table0_leg_near_minus" type="box" size="0.04 0.04 0.1875"
          pos="-0.26 -0.26 0.1875" mass="0" material="table_mat"/>
</body>
```

Notes on the replacement:
- Body anchor is now at world Z = 0 (floor level) instead of Z = 0.2. All geometry is expressed relative to this floor-level anchor.
- `mass="2000"` stays on the surface slab (provides the physics mass for a stable table). The legs have `mass="0"` — they contribute only geometry and collision, not inertia. This prevents instability from multiple massive bodies.
- All 5 geoms get explicit `name` attributes for clarity and for potential contact exclusion in `shared.xml` if needed.
- Surface slab top = 0 + 0.375 + 0.025 = 0.400. ✓

### File: `chess_env/assets/robot.xml`

**Current arm base line (line 8):**
```xml
<body childclass="robot0:fetch" name="robot0:base_link" pos="0.2869 0.2641 0">
```

**Replace with:**
```xml
<body childclass="robot0:fetch" name="robot0:base_link" pos="0.60 0.2641 0">
```

Only the X coordinate changes. Y (0.2641) and Z (0) remain the same.

### File: `configs/env.yaml`

**Current:**
```yaml
table_half_x: 0.28
table_half_y: 0.28
```

**Replace with:**
```yaml
table_half_x: 0.30
table_half_y: 0.30
```

No other config values change. `table_surface_z: 0.400`, `cube_height: 0.030`, `grasp_z: 0.425`, `hover_z: 0.460`, `safe_z: 0.550` are all preserved unchanged.

---

## 1.3 — Downstream Impact Analysis

### `task.py`: Board Boundary Sampling

The board boundary is computed from config:
```python
board_min_x = table_center_xy[0] - table_half_x  # 0.88 - 0.28 = 0.60 → now 0.88 - 0.30 = 0.58
board_max_x = table_center_xy[0] + table_half_x  # 0.88 + 0.28 = 1.16 → now 0.88 + 0.30 = 1.18
board_min_y = table_center_xy[1] - table_half_y  # now 0.2641 - 0.30 = -0.0359
board_max_y = table_center_xy[1] + table_half_y  # now 0.2641 + 0.30 = 0.5641
```

After the config change, random cube placement will use the new 60×60cm boundary automatically. No code change needed.

### `task.py`: `hidden_object_pos`

The hidden object position (used to hide the cube during transit) places the cube below the table. This position is likely hardcoded or computed from table coords. Search for `hidden_object_pos` in task.py and verify it remains valid (below floor level, outside the new leg positions).

### `scripts/test_corners.py`

This script has hardcoded corner coordinates:
```python
# Old hardcoded corners (approximate, using old table size)
[0.6, -0.3], [0.6, 0.3], [1.2, -0.3], [1.2, 0.3]
```

After Stage 1, the actual corners are:
```
[0.58, -0.0359], [0.58, 0.5641], [1.18, -0.0359], [1.18, 0.5641]
```

The hardcoded values will be slightly outside the new board. This is acceptable for Stage 1 (the script is already known to use approximate values). It will be properly fixed in Stage 4 when eval scripts are rewritten.

### `verify_physics.py`: `GRASP_Z` assertion

The test `verify_grasp_xml_changes` checks:
> `GRASP_Z=0.425`

This test reads from the config (or asserts based on a hardcoded constant). It does not test table dimensions. It should still pass after Stage 1.

---

## 1.4 — Validation Steps

After making all three file changes (XML, robot.xml, env.yaml), run the following checks in order. Each must pass before proceeding.

### Check 1: Environment Loads Without Error

```bash
python -c "
import gymnasium as gym
import src.chess_env
env = gym.make('ChessFetchTask-v0')
obs, info = env.reset()
print('Load OK. Obs keys:', list(obs.keys()))
env.close()
"
```

**Expected**: Prints `Load OK` with obs keys. No MuJoCo errors about invalid geometry.

### Check 2: Table Surface Z is Still 0.400

```bash
python -c "
import gymnasium as gym
import src.chess_env
import mujoco
env = gym.make('ChessFetchTask-v0')
env.reset()
# Find table surface geom Z position
model = env.unwrapped.model
geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'table0_surface')
geom_pos = model.geom_pos[geom_id]
# Table body is at (0.88, 0.2641, 0); geom pos is relative
# body_pos + geom_pos_z + geom_size_z = surface top
body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'table0')
body_z = model.body_pos[body_id][2]
geom_half_z = model.geom_size[geom_id][2]
surface_top = body_z + geom_pos[2] + geom_half_z
print(f'Surface top Z = {surface_top:.4f}  (expected 0.4000)')
assert abs(surface_top - 0.400) < 0.0001, 'FAIL: surface Z wrong'
print('PASS')
env.close()
"
```

**Expected**: `Surface top Z = 0.4000` and `PASS`.

### Check 3: Arm Base Position Updated

```bash
python -c "
import gymnasium as gym
import src.chess_env
import mujoco
env = gym.make('ChessFetchTask-v0')
env.reset()
model = env.unwrapped.model
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot0:base_link')
base_x = model.body_pos[base_id][0]
print(f'Arm base X = {base_x:.4f}  (expected 0.6000)')
assert abs(base_x - 0.60) < 0.001, 'FAIL: arm base X wrong'
print('PASS')
env.close()
"
```

**Expected**: `Arm base X = 0.6000` and `PASS`.

### Check 4: Static Stability Test

Cube should not drift under gravity with the new table:

```bash
python -c "
import gymnasium as gym
import numpy as np
import src.chess_env
env = gym.make('ChessFetchTask-v0')
obs, info = env.reset()
inner = env.unwrapped

# Get initial cube position
from src.chess_env.config_loader import load_config
cfg = load_config()
import mujoco
obj_joint_id = inner.model.joint('object0:joint').id
qpos_start = inner.model.jnt_qposadr[obj_joint_id]
pos0 = inner.data.qpos[qpos_start:qpos_start+3].copy()

# Run 100 zero-action steps
action = np.zeros(4)
for _ in range(100):
    obs, _, _, _, _ = env.step(action)

pos1 = inner.data.qpos[qpos_start:qpos_start+3].copy()
drift = np.linalg.norm(pos1 - pos0)
print(f'Cube drift after 100 steps: {drift*1000:.2f}mm  (expected < 1mm)')
assert drift < 0.001, f'FAIL: drift={drift:.4f}'
print('PASS')
env.close()
"
```

**Expected**: Drift < 1mm (0.001m). `PASS`.

### Check 5: Reachability at All Corners

Run the existing kinematic reachability test modified to hit all 4 corners plus center:

```bash
python -c "
import gymnasium as gym
import numpy as np
import src.chess_env
env = gym.make('ChessFetchTask-v0')
env.reset()
inner = env.unwrapped

# Table corners in new 60x60 board
corners = [
    [0.58, -0.0359],  # near-right
    [0.58,  0.5641],  # near-left
    [1.18, -0.0359],  # far-right
    [1.18,  0.5641],  # far-left
    [0.88,  0.2641],  # center
]

from src.chess_env.waypoints import SAFE_Z, HOVER_Z
import mujoco

results = []
for xy in corners:
    target = np.array([xy[0], xy[1], SAFE_Z])
    success = inner._settle_arm_to_start(target)
    grip = inner._utils.get_site_xpos(inner.model, inner.data, 'robot0:grip')
    err = np.linalg.norm(grip - target)
    results.append((xy, err, success))
    print(f'  Corner {xy}: err={err*1000:.1f}mm  {\"PASS\" if err < 0.005 else \"FAIL\"}')

env.close()
all_pass = all(e < 0.005 for _, e, _ in results)
print('ALL PASS' if all_pass else 'SOME CORNERS FAILED')
"
```

**Expected**: All 5 positions show error < 5mm. If near corners fail, see the note in 1.1 about adjusting arm base X.

### Check 6: Full `verify_physics.py` Suite

```bash
python scripts/verify_physics.py
```

**Expected**: All 5 tests pass. The test `test_kinematic_reachability` samples random board positions — with the updated config these will use the new ±0.30m board limits.

---

## 1.5 — Rollback Procedure

If any validation check fails:

1. Revert `chess_env/assets/pick_and_place.xml` to the original solid-box table.
2. Revert `chess_env/assets/robot.xml` arm base to `pos="0.2869 0.2641 0"`.
3. Revert `configs/env.yaml` table_half_x/y to `0.28`.
4. Diagnose the failure before re-attempting.

Common failure modes:
- **MuJoCo geometry error on load**: Check that the geom names are unique and the `size` parameters are positive.
- **Surface Z wrong**: Recalculate body_pos + geom_pos_z + geom_half_z = 0.400.
- **Near corner reachability failure**: Increase arm base X from 0.60 to 0.62 or 0.65, then re-verify.
- **Cube falls through table**: Leg geometry is not touching the surface slab, leaving a gap. Verify leg top Z = slab bottom Z = 0.375.

---

## 1.6 — Summary of Changes

| File | Change |
|------|--------|
| `chess_env/assets/pick_and_place.xml` | Replace solid-box `table0` body with 5-geom body (1 slab + 4 legs) |
| `chess_env/assets/robot.xml` | Arm base X: 0.2869 → 0.60 |
| `configs/env.yaml` | `table_half_x/y`: 0.28 → 0.30 |

No Python source files change in Stage 1.

**Stage 1 is complete when `python scripts/verify_physics.py` passes all 5 tests with the new geometry.**
