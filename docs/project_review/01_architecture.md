# RoboChess — Architecture Review

## Layer Overview

### 1. Simulation Layer (`src/chess_env/`)

**Files:** `simulation.py`, `task.py`, `controller.py`, `waypoints.py`, `environment_generation.py`

The simulation is built on top of `gymnasium-robotics`'s `MujocoFetchPickAndPlaceEnv`. Two classes extend the base:

```
MujocoFetchPickAndPlaceEnv
    └── ChessSimulationEnv   (simulation.py)  — geometry, action scaling, model injection
            └── ChessTaskEnv  (task.py)        — scenario logic, grasp/place pipelines, soft_reset
```

**Good design decisions:**
- `ChessSimulationEnv` handles all MuJoCo-level concerns (XML injection, board sampling, action scaling).
- `ChessTaskEnv` handles task-level state: scenario (transit/descend/ascend), grasp mode, finger targets.
- The global XML path is patched and restored in a `try/finally` block (`simulation.py:61-70`), preventing state leakage on exception.

**Design concern:** The global module-level monkey-patch of `_fpp_module.MODEL_XML_PATH` is not thread-safe (`simulation.py:61-70`). If two environments are constructed simultaneously in different threads (parallel training), both see each other's path changes. Currently not triggered in practice (single environment per process), but should be documented.

---

### 2. Scripted Controller (`src/chess_env/controller.py`)

`ScriptedController` drives the arm deterministically. It operates via `run_transit`, `run_descend`, `run_ascend`, `run_grasp`, `run_place`, composed into `run_pick_sequence`, `run_place_sequence`, `run_full_move`.

**Core loop (`_run_movement_loop`):** Proportional control with clamped step size:
- STEP_GAIN = 1.0 (full error applied)
- MIN_STEP_SIZE_M = 2mm (prevents final-approach creep)
- MAX_STEP_SIZE_M = 24mm (keeps board coverage within timeout budget)
- Tolerance: 4mm for transit, 4mm for vertical

**Measured performance:**
- Transit: avg 1.6mm final error, P95 2.2mm
- Descend: avg 0.4mm final error, P95 0.5mm
- Ascend: avg 2.5mm final error, P95 3.6mm (consistent with tolerance = 4mm)

All within the 4mm tolerance. This is well-tuned.

**Design concern — `transition()` ignores `soft_reset` errors:**  
`controller.py:267-283` calls `self._env.soft_reset(...)` but `soft_reset` can raise `RuntimeError` (HALT_FAILED or ALIGN_FAILED). The `RuntimeError` propagates up through `run_pick_sequence` / `run_full_move`, but `move_piece_xy` in `movement_executor.py` does not catch it. It propagates all the way to `PhysicalPlanExecutor.execute()` which has a blanket `except Exception` — so it IS caught eventually, but the error path skips the `StageResult` return value entirely, making failure analysis harder.

**Fix:** Have `transition()` return the result dict and check the `halt_steps` / `align_steps` for anomalies, or wrap in try/except with a more informative failure path.

---

### 3. Waypoints (`src/chess_env/waypoints.py`)

Clean, minimal module. Defines Z-levels, valid scenario transitions, and chain shortcuts.

**Z-level hierarchy:**
```
TABLE_SURFACE_Z = 0.400m
cube top        = 0.430m  (TABLE_Z + CUBE_HEIGHT = 0.400 + 0.030)
GRASP_Z         = 0.430m  (at cube top — 18.5mm finger overlap)
HOVER_Z         = 0.460m  (safe RL stop, finger clearance)
SAFE_Z          = 0.550m  (transit height, 90mm above hover)
```

**Valid chain transitions:**
```
transit → transit  (horizontal repositioning)
transit → descend  (approach a square)
descend → ascend   (vertical pair)
ascend  → transit  (carry to destination)
ascend  → descend  (place at destination)
```

All chain shortcuts (`full_move`, `pick`, `place`, `vertical`) are validated by `test_chain_shortcuts_validity`.

---

### 4. Environment Generation (`src/chess_env/environment_generation.py`)

Generates the MuJoCo XML scene (board squares, piece bodies, zone markers) and STL meshes at runtime. Uses marker-delimited text replacement rather than XML DOM manipulation — this is brittle but works because the markers are unique.

**Piece geometry pipeline:**  
Each piece type has a procedural mesh generator built from `frustum_triangles`, `sphere_triangles`, `cone_triangles`, `box_triangles`. Max height ≤ 0.0605m (verified by test), which is below HOVER_Z (0.460m) — important for arm clearance.

**Note on `_replace_marked_fragment`:** Uses `str.index()` which raises `ValueError` if markers are absent. If the XML file is corrupted or markers accidentally deleted, `regenerate_scene()` crashes without a clear error message. Consider checking marker presence before replacement.

---

### 5. Chess Game Layer (`src/chess_game/`)

**Files:** `chess_service.py`, `game_orchestrator.py`, `move_planner.py`, `board_mapper.py`, `models.py`

Clean, well-tested layer. Full delegation to `python-chess` for rules; no custom rule implementation.

**`ChessService.choose_engine_move()`** is a greedy one-ply selector, not a minimax engine. It picks: checkmate > capture value > promotion > check > lexicographic UCI. This is documented nowhere — callers may expect a stronger opponent. See `04_chess_logic.md` for details.

**`BoardMapper`** validates geometry against hard constraints (cell_size = 0.08m, board = 0.64m, margin ≥ 0.03m). Any config mismatch raises `ValueError` at construction time — excellent defensive design.

**`GameOrchestrator._submit_move()`** correctly:
1. Plans the move (pure logic, no side effects)
2. Executes physically
3. Calls `return_to_home()`
4. Commits to chess state only on success

The separation of physical execution from logical commit prevents state desynchronisation on failure.

---

### 6. Physical Execution Layer (`src/physical/`)

**Files:** `movement_executor.py`, `plan_executor.py`, `piece_teleport.py`, `piece_registry.py`, `occupancy.py`

**`PieceRegistry`** — deterministic registry of 32 active + 64 reserve pieces. Piece IDs are stable across games (`white_pawn_e`, `white_reserve_queen_1`, etc.). Good.

**`PhysicalOccupancy`** — expected-state mirror of where pieces should be. Guards against double-occupancy on a square. Used to prevent physically invalid moves before arm motion starts.

**Teleportation design:** After a successful arm move, `movement_executor.py:75` teleports the piece body to the exact square center:
```python
self.teleporter.teleport_piece_to_xyz(piece_id, placed_xyz, quat=IDENTITY_QUAT)
```
This corrects any placement drift from the arm (typically <20mm). The teleport happens after the 20mm XY reconciliation check, so it's always to a valid position. Smart design.

---

### 7. UI Layer (`src/ui/`)

Flask backend with a single-page JavaScript frontend. The `QueuedUIBackend` in `run_chess_ui.py` serializes all UI requests into a queue processed on the simulation thread. This avoids thread-safety issues with MuJoCo, which is not thread-safe.

**Concern:** Queue serialization means if the arm is moving (which can take 1-2 seconds), all UI requests queue up. If the user rapidly clicks, requests accumulate. Adding a `max_size` to the queue would prevent unbounded accumulation.

---

### 8. Configuration System

Three YAML configs, loaded by `src/utils/config.py`:

| File | Domain |
|------|--------|
| `configs/env.yaml` | Table geometry, Z-levels, grasp thresholds, logging |
| `configs/chess.yaml` | Board geometry, piece types, graveyard/reserve zones, game settings |
| `configs/physics.yaml` | MuJoCo simulation parameters, quaternion |

**Concern — Duplicated home position:**  
`env.yaml:home_position_xy = [0.88, 0.2641]` and `chess.yaml:game.arm_home_xy = [0.88, 0.2641]` are the same value, read by different modules. If they diverge (e.g., after a table repositioning), the arm will home to different positions depending on context. These should reference a single value.

**Fix:** Remove `game.arm_home_xy` from `chess.yaml` and have `plan_executor.py` read `env.yaml:home_position_xy`.
