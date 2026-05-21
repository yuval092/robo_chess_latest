# Waypoint Algorithm

## Concept

The waypoint algorithm decomposes arm movement into discrete **scenarios** (movement phases). Each scenario has a well-defined entry state, a well-defined exit state, and a specific movement pattern with safety constraints. The arm moves through scenarios sequentially, with `soft_reset()` bridging the gap between consecutive scenarios.

This structure enables:
1. **Modular training**: Each scenario can be trained independently
2. **Predictable behavior**: Movement stays within a clearly-defined envelope
3. **Safe transitions**: The arm halts and verifies state before switching modes

---

## Z-Level Architecture

All movement is at one of three heights:

| Level | Z (meters) | Name | Purpose |
|-------|-----------|------|---------|
| `SAFE_Z` | 0.550 | Transit height | 15cm above table; clears all pieces during horizontal movement |
| `HOVER_Z` | 0.460 | Pre-grasp height | 60mm above cube top; safe approach height before grasping |
| `GRASP_Z` | 0.430 | Grasp height | Exactly at cube top; finger overlap zone |
| `TABLE_SURFACE_Z` | 0.400 | Table top | Physical floor; absolute minimum Z during normal operation |

The arm only exists at SAFE_Z during transit, at HOVER_Z to GRASP_Z during grasp operations, and between HOVER_Z and SAFE_Z during descend/ascend.

---

## The Three Scenarios

### Transit

**Entry**: Arm at any XY, at SAFE_Z  
**Goal**: `[target_x, target_y, SAFE_Z]`  
**Exit**: Arm at `[target_x, target_y]` within 4mm, at SAFE_Z  
**Movement**: Horizontal — proportional controller moves to goal in 3D but Z is already at SAFE_Z  
**Constraints**:
- `grip_z > FLOOR_LIMIT (0.400)` — don't sink into table
- When `grasp_mode=True`: `_check_cube_held()` each step  
**Gripper**: CLOSED (`finger_target_joint = FINGER_CLOSED_JOINT = 0.0`)

Transit is unconstrained in XY — the arm takes the most direct path. Start and goal positions are sampled uniformly from the usable board area (within 4cm of all edges).

### Descend

**Entry**: Arm at `[target_x, target_y, SAFE_Z]` (directly above target square)  
**Goal**: `[target_x, target_y, HOVER_Z]`  
**Exit**: Arm at `[target_x, target_y]` within 4mm, at HOVER_Z  
**Movement**: Vertical — arm descends straight down 90mm  
**Constraints**:
- `||grip_xy - tube_center_xy|| < drift_limit (10mm)` — XY tube  
- `grip_z > TABLE_SURFACE_Z (0.400)` — don't crash into table  
**Gripper**: OPEN (`finger_target_joint = FINGER_OPEN_JOINT = 0.0181`)

The "tube" constraint is a cylinder of radius 10mm centered on the target XY. The arm must stay within this cylinder during descent. This constraint forces vertical movement and prevents the arm from sweeping laterally while descending through piece clearance heights.

### Ascend

**Entry**: Arm at `[target_x, target_y, HOVER_Z]`  
**Goal**: `[target_x, target_y, SAFE_Z]`  
**Exit**: Arm at `[target_x, target_y]` within 4mm, at SAFE_Z  
**Movement**: Vertical — arm ascends 90mm  
**Constraints**:
- `||grip_xy - tube_center_xy|| < drift_limit (10mm)` — same XY tube  
- When `grasp_mode=True`: `_check_cube_held()` each step  
**Gripper**:
- When NOT holding cube: CLOSED (teleport)
- When holding cube (`grasp_mode=True`): Actuator at ctrl=0 (fingers stalled against cube)

---

## Valid Transitions

Defined in `src/chess_env/waypoints.py`:

```python
VALID_TRANSITIONS = {
    ("transit", "transit"): True,   # Reposition at SAFE_Z without descending
    ("transit", "descend"): True,   # Over target square → start descent
    ("descend", "ascend"):  True,   # Grasp at bottom → ascend with cube
    ("ascend",  "transit"): True,   # Lifted cube → move to destination
    ("ascend",  "descend"): True,   # After placing → descend again (e.g., consecutive picks)
}
```

**Invalid transitions** (raise `ValueError`):
- `("descend", "transit")`: Can't transit from HOVER_Z — would sweep across pieces
- `("transit", "ascend")`: SAFE_Z is already the top; ascending from there makes no sense
- `("descend", "descend")`: Double descent — already at HOVER_Z
- `("ascend", "ascend")`: Double ascent — already at SAFE_Z

---

## Chain Definitions

```python
CHAIN_SHORTCUTS = {
    "full_move": ["transit", "descend", "ascend", "transit", "descend", "ascend"],
    "pick":      ["transit", "descend", "ascend"],
    "place":     ["transit", "descend", "ascend"],
    "vertical":  ["descend", "ascend"],
}
```

A "full_move" is two complete `pick`/`place` sub-chains. The chain shortcut names are used in training configuration to specify which scenario sequences to train.

---

## `exit_waypoint(scenario)` — Nominal Exit Position

Returns the expected arm position at the end of each scenario:

```python
def exit_waypoint(scenario: str, tube_center_xy=None) -> np.ndarray:
    if scenario == "transit":
        return np.array([goal_pos[0], goal_pos[1], SAFE_Z])
    elif scenario == "descend":
        return np.array([tube_center_xy[0], tube_center_xy[1], HOVER_Z])
    elif scenario == "ascend":
        return np.array([tube_center_xy[0], tube_center_xy[1], SAFE_Z])
```

Used by `ScriptedController.transition()` to call `soft_reset(nominal_exit_pos=exit_waypoint(...))`.

---

## `derive_goal_pos(scenario, xy)` — Goal Position

Returns the scenario's goal as a 3D position:

```python
def derive_goal_pos(scenario: str, xy: np.ndarray) -> np.ndarray:
    if scenario == "transit":
        return np.array([xy[0], xy[1], SAFE_Z])
    elif scenario == "descend":
        return np.array([xy[0], xy[1], HOVER_Z])
    elif scenario == "ascend":
        return np.array([xy[0], xy[1], SAFE_Z])
```

---

## Full Pick-Place Sequence (Detailed)

### `ScriptedController.run_pick_sequence(src_xy)`:

```
1. run_transit(src_xy)
   → arm moves to [src_xy, SAFE_Z]

2. transition("descend",
              new_goal_pos=[src_xy, HOVER_Z],
              nominal_exit_pos=[src_xy, SAFE_Z],
              nominal_xy=src_xy)
   → soft_reset: halt, open fingers, align to [src_xy, SAFE_Z]

3. run_descend(src_xy)
   → arm descends to [src_xy, HOVER_Z] (inside 10mm tube)

4. run_grasp()
   → calls env.execute_grasp(): 6-phase grasp pipeline
   → grasp_mode = True on success

5. transition("ascend",
              new_goal_pos=[src_xy, SAFE_Z],
              nominal_exit_pos=[src_xy, HOVER_Z],
              nominal_xy=src_xy)
   → soft_reset: halt, close fingers (grasp_mode preserved)

6. run_ascend(src_xy)
   → arm ascends to [src_xy, SAFE_Z] (tube, cube-drop monitoring)
```

### `ScriptedController.run_place_sequence(dst_xy)`:

```
1. run_transit(dst_xy)
   → arm moves to [dst_xy, SAFE_Z] (cube held throughout)

2. transition("descend", new_goal_pos=[dst_xy, HOVER_Z], ...)
   → soft_reset: halt, fingers already closed (grasp_mode=True)

3. run_descend(dst_xy)
   → descend to HOVER_Z (tube + cube-drop monitoring)

4. run_place(dst_xy)
   → calls env.execute_place(dst_xy): place pipeline
   → grasp_mode = False after placing

5. transition("ascend", new_goal_pos=[dst_xy, SAFE_Z], ...)
   → soft_reset: halt, close empty gripper

6. run_ascend(dst_xy)
   → ascend back to SAFE_Z (empty gripper, tube monitoring)
```

---

## Scenario Observation Encoding

Each scenario has a numeric ID in the observation space:

| Scenario | `scenario_id` |
|----------|--------------|
| transit | 0 |
| descend | 1 |
| ascend | 2 |

This is included in every `_get_obs()` output under the `observation` key. Future RL agents can use this to learn scenario-specific policies.

---

## Position Sampling

New positions are sampled uniformly from the usable board area:

```python
def _sample_board_position(self) -> np.ndarray:
    cx, cy = TABLE_CENTER_XY
    hx, hy = TABLE_HALF_X, TABLE_HALF_Y
    margin = EDGE_MARGIN  # 0.04m
    
    x = np.random.uniform(cx - hx + margin, cx + hx - margin)  # [0.57, 1.19]
    y = np.random.uniform(cy - hy + margin, cy + hy - margin)  # [-0.046, 0.574]
    z = TABLE_SURFACE_Z
    return np.array([x, y, z])
```

For transit: start and goal are independently sampled, with `MIN_GOAL_DIST = 0.10m` enforced (resample if too close).
