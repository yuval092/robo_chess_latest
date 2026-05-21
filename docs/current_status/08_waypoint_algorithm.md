# Waypoint Algorithm

## Concept

The waypoint algorithm decomposes arm movement into discrete **scenarios** (movement phases). Each scenario has a well-defined entry state, a well-defined exit state, and a specific movement pattern. The arm moves through scenarios sequentially, with `soft_reset()` bridging the gap between consecutive scenarios.

---

## Z-Level Architecture

All movement happens at one of three Z-levels:

| Level | Z (meters) | Name | Purpose |
|-------|-----------|------|---------|
| `SAFE_Z` | 0.550 | Transit height | High enough to clear all pieces (~15cm above table) |
| `HOVER_Z` | 0.460 | Pre-grasp height | 60mm above cube top; safe RL stop zone |
| `GRASP_Z` | 0.425 | Grasp height | 25mm above table; 78% finger-cube overlap |
| `TABLE_SURFACE_Z` | 0.400 | Table top | Physical floor for all cube operations |

---

## The Three Scenarios

### Transit
**Entry**: Anywhere at `SAFE_Z`  
**Goal**: `[target_x, target_y, SAFE_Z]`  
**Exit**: `[target_x, target_y, SAFE_Z]`  
**RL behavior**: Gripper moves horizontally at `SAFE_Z` to reach target XY.  
**Constraint**: `gripper_z > FLOOR_LIMIT (0.400)` — don't touch the table surface.  
**Gripper state**: CLOSED (fingers teleport-locked at `FINGER_CLOSED_JOINT`).

### Descend
**Entry**: `[target_x, target_y, SAFE_Z]` (arm hovering over target square)  
**Goal**: `[target_x, target_y, HOVER_Z]`  
**Exit**: `[target_x, target_y, HOVER_Z]`  
**RL behavior**: Gripper descends vertically from `SAFE_Z` to `HOVER_Z`.  
**Constraint**: `||gripper_xy - tube_center_xy|| < drift_limit` — stay within a cylinder.  
**Constraint**: `gripper_z > TABLE_SURFACE_Z` — don't crash into table.  
**Gripper state**: OPEN (teleport-locked at `FINGER_OPEN_JOINT`).

### Ascend
**Entry**: `[target_x, target_y, HOVER_Z]` (arm at hover height)  
**Goal**: `[target_x, target_y, SAFE_Z]`  
**Exit**: `[target_x, target_y, SAFE_Z]`  
**RL behavior**: Gripper ascends vertically from `HOVER_Z` to `SAFE_Z`.  
**Constraint**: `||gripper_xy - tube_center_xy|| < drift_limit` — stay within the cylinder.  
**Gripper state**: CLOSED (teleport-locked, or actuator-driven in grasp_mode).

---

## Valid Transitions

Defined in `waypoints.py`:

```python
VALID_TRANSITIONS = {
    ("transit", "transit"): True,   # XY repositioning at SAFE_Z
    ("transit", "descend"): True,   # Over target → descend
    ("descend", "ascend"):  True,   # Descend → grasp → ascend
    ("ascend",  "transit"): True,   # Ascend → move to next square
    ("ascend",  "descend"): True,   # Consecutive vertical (descend → place → ascend → descend)
}
```

Invalid transitions (not in the dict):
- `("descend", "transit")`: Can't transit at HOVER_Z level — below SAFE_Z, risk hitting pieces
- `("transit", "ascend")`: Can't ascend from SAFE_Z — that's already at transit height
- `("descend", "descend")`: Can't descend twice
- `("ascend", "ascend")`: Can't ascend twice

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

A "pick" chain moves the arm to the source square and grasps. A "place" chain moves to the destination and releases. A "full_move" combines both.

In `eval_sequence.py` and `eval_grasp.py`, waypoints are pre-assigned:
- `full_move`: waypoints = `[src_xy, src_xy, src_xy, dst_xy, dst_xy, dst_xy]`
- `pick`: waypoints = `[src_xy, src_xy, src_xy]`
- `place`: waypoints = `[dst_xy, dst_xy, dst_xy]`
- `vertical`: waypoints = `[target_xy, target_xy]`

---

## Transition Mechanics (`soft_reset`)

Between scenarios, `soft_reset` is called with:
- `new_scenario`: The next scenario type
- `new_goal_pos`: The 3D goal for the next scenario
- `nominal_exit_pos`: Where the arm "should" be after the completed scenario
- `nominal_xy`: The tube center for next scenario (if descend/ascend)

The `nominal_exit_pos` is computed by `exit_waypoint(scenario, cell_xy)`:

```python
SCENARIO_EXIT_Z = {
    "transit": SAFE_Z,    # 0.550m
    "descend": HOVER_Z,   # 0.460m
    "ascend":  SAFE_Z,    # 0.550m
}
exit_waypoint = [cell_xy[0], cell_xy[1], SCENARIO_EXIT_Z[scenario]]
```

The alignment step in `soft_reset` moves the arm to `nominal_exit_pos` with 3mm tolerance. This corrects any positional error accumulated during the RL episode (RL models stop within a few mm of the goal, not exactly at it).

---

## Full Pick-and-Place Flow

Implemented in `eval_grasp.py::run_one_pick_sequence`:

```
1. Initialize:
   - force_start_pos = home_pos (e.g., [0.68, 0.2641, 0.550])
   - force_cube_pos = [src_xy[0], src_xy[1], TABLE_SURFACE_Z + CUBE_HEIGHT/2]
   - force_scenario = "transit"
   - env.reset()

2. Transit: home_pos → [src_xy, SAFE_Z]
   → RL runs transit scenario
   → success triggers exit at [src_xy, SAFE_Z]

3. soft_reset("descend", [src_xy, HOVER_Z], [src_xy, SAFE_Z], src_xy)
   → Arm aligned to [src_xy, SAFE_Z]
   → Fingers open

4. Descend: [src_xy, SAFE_Z] → [src_xy, HOVER_Z]
   → RL runs descend scenario
   → success triggers exit at [src_xy, HOVER_Z]

5. execute_grasp()
   → Scripted: HOVER_Z → GRASP_Z → close fingers → verify → retract to HOVER_Z
   → grasp_mode = True

6. soft_reset("ascend", [src_xy, SAFE_Z], [src_xy, HOVER_Z], src_xy)
   → Arm aligned to [src_xy, HOVER_Z]
   → grasp_mode=True preserved (no finger transition)

7. Ascend: [src_xy, HOVER_Z] → [src_xy, SAFE_Z]
   → RL runs ascend with grasp_mode=True (cube held)
   → cube hold monitored: drop = crash
   → success triggers exit at [src_xy, SAFE_Z]

8. soft_reset("transit", [home_pos_xy, SAFE_Z], [src_xy, SAFE_Z])
   → Arm aligned to [src_xy, SAFE_Z]
   → grasp_mode=True preserved (still carrying cube)

9. Transit: [src_xy, SAFE_Z] → [home_pos, SAFE_Z]
   → RL runs transit while holding cube
   → cube hold monitored throughout

(Full place = add steps 10-14 with dst_xy and execute_place)
```

---

## Tube Constraint (Descend/Ascend)

During descend and ascend, the arm must stay within a radial cylinder:

```python
drift = ||gripper_pos[:2] - tube_center_xy||
if drift > current_drift_limit:
    crash("TUBE_BREACH")
```

`tube_center_xy` is set to the target square's XY in `_reset_sim`. The drift limit tightens from 10cm to 1cm over training. At evaluation, 1cm (or 0.5cm) is enforced.

The tube constraint exists because in a full chess game, adjacent squares are separated by 3.5-4cm. An arm that drifts >2cm from center could collide with adjacent pieces. The 1cm constraint provides a 2cm clearance from the closest piece edge.

---

## `tube_center_xy` vs. Nominal XY

There's a subtle distinction in the codebase:
- `tube_center_xy`: Used during the RL step for drift detection. Set at reset time.
- `nominal_exit_pos`: Used in `soft_reset` for alignment. The canonical exit position.

These should be consistent: `tube_center_xy == nominal_exit_pos[:2]`. The code enforces this via the `nominal_xy` parameter to `soft_reset`.

---

## Known Issues

### Test Assertion Bug
In `tests/chess_env/test_waypoints.py`, the tests assert:
```python
derive_goal_pos("descend", cell) == [x, y, 0.430]  # WRONG
exit_waypoint("descend", cell) == [x, y, 0.430]    # WRONG
```

But the actual values use `HOVER_Z = 0.460`, not `0.430`. These tests **fail** with the current configuration. The comment says `# GRASP_Z` which is also wrong (GRASP_Z is not used for descend goals). These appear to have been written when `hover_z` was 0.430 and were not updated when `hover_z` was changed to 0.460.

### `exit_waypoint("descend")` = HOVER_Z or GRASP_Z?
`SCENARIO_EXIT_Z["descend"] = HOVER_Z`. The descend scenario ends at HOVER_Z (0.460), not GRASP_Z (0.425). The grasp stage then goes further down to GRASP_Z, but that's a scripted action, not the scenario exit point.
