# Waypoints and Scenario System

## Purpose

The waypoint system (`src/chess_env/waypoints.py`) defines the three discrete Z-levels the arm operates at, the valid transitions between movement phases, and utility functions for computing 3D target positions. It is the bridge between the abstract "move a piece from A to B" intent and the concrete sequence of physical stages.

---

## Z-Level Constants

All heights are in metres above the table surface (which sits at Z = 0.400 m in world frame):

| Constant | Value | Description |
|----------|-------|-------------|
| `SAFE_Z` | 0.530 m | Cruise altitude for horizontal transit. Keeps the arm lower for better reachability while still clearing standing chess pieces. |
| `HOVER_Z` | 0.460 m | Stop height after descend; start height for grasp. Arm is above the piece but close enough to begin contact approach. |
| `GRASP_Z` | 0.430 m | Finger contact height. Grip reaches 18.5 mm into the piece (cube top is at 0.415 m; GRASP_Z = 0.430 m). |

These are loaded from `configs/env.yaml` at module import time and used as module-level constants throughout the codebase.

---

## Scenario Definitions

A *scenario* is one of three named movement phases:

| Scenario | Entry Z | Exit Z | Purpose |
|----------|---------|--------|---------|
| `transit` | SAFE_Z | SAFE_Z | Horizontal arm travel at cruise altitude |
| `descend` | SAFE_Z | HOVER_Z | Vertical drop from SAFE_Z to HOVER_Z above a square |
| `ascend` | HOVER_Z | SAFE_Z | Vertical rise from HOVER_Z back to SAFE_Z |

---

## Valid Transition Graph

Not all scenario sequences are physically valid. A descend must be followed by an ascend before any further transit. The `VALID_TRANSITIONS` dict enforces this:

```
transit ──→ transit   (re-entry: arm was already at SAFE_Z, transits again)
transit ──→ descend   (approach a square for grasp or place)
descend ──→ ascend    (after grasp or place, return to SAFE_Z)
ascend  ──→ transit   (carry piece to destination)
ascend  ──→ descend   (immediately descend at new square — used in eval scripts)
```

Invalid sequences (e.g., `descend → transit`, `ascend → ascend`) raise `ValueError` from `validate_chain()`.

---

## Pre-Defined Chain Shortcuts

```python
# waypoints.py:34
CHAIN_SHORTCUTS = {
    "full_move": ["transit", "descend", "ascend", "transit", "descend", "ascend"],
    "pick":      ["transit", "descend", "ascend"],
    "place":     ["transit", "descend", "ascend"],
    "vertical":  ["descend", "ascend"],
}
```

`full_move` is the standard chess piece move sequence: transit to source → descend → grasp+ascend → transit to destination → descend → place+ascend.

---

## `derive_goal_pos(scenario, cell_xy)`

Returns the 3D target `[X, Y, Z]` for the arm at the end of a given scenario:
- `transit` → Z = SAFE_Z
- `descend` → Z = HOVER_Z
- `ascend` → Z = SAFE_Z

---

## `exit_waypoint(scenario, cell_xy)`

Returns the 3D position the arm is expected to be at when leaving a scenario. Used by `ScriptedController.transition()` to position the next scenario's initial state and to pass to `ChessTaskEnv.soft_reset()`.

---

## Soft Reset and Scenario Transitions

`ChessTaskEnv.soft_reset()` handles the transition between consecutive scenarios during a full move. It:
1. **Halts** the arm (zeroes qvel, qacc).
2. **Aligns** the arm to the nominal exit position from the previous scenario.
3. **Transitions** the gripper state (open for transit/descend, closed for ascend-with-piece).
4. Updates `current_scenario`, `goal_pos`, `tube_center_xy`.

This is called by `ScriptedController.transition()` between every stage.

---

## Tube Constraint (Descend / Ascend)

During descend and ascend, the arm is not allowed to drift horizontally beyond `drift_limit` (default 10 mm) from the target square centre. This constraint exists because:
- The chess board is densely packed (80 mm cells, 30 mm pieces).
- A drifting descent could clip an adjacent piece.
- The tube constraint is enforced by the `descend_abort` and `ascend_abort` functions in `controller.py`, which check `norm(grip_pos[:2] - tube_center)`.

If the tube is breached, the stage returns `crash_reason="TUBE_BREACH"` and the move fails.
