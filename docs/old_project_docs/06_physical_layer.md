# Physical Layer

This document covers the physical execution layer: the `PhysicalPlanExecutor`, `MovementExecutor`, `PhysicalOccupancy`, `PieceTeleporter`, and `PieceRegistry`.

---

## 1. Overview

The physical layer is responsible for translating abstract `PhysicalPlan` objects (see `05_chess_logic.md`) into concrete robot actions: arm movements and instantaneous teleports. It operates on the MuJoCo simulation, not the real world.

```
PhysicalPlan
    │
    ▼
PhysicalPlanExecutor          ← top-level executor; routes commands
    ├── MovementExecutor       ← runs arm moves (grasp/transit/place)
    ├── PieceTeleporter        ← instantaneous free-joint position writes
    └── PhysicalOccupancy      ← expected square occupancy state
```

All components receive the same `env` and `controller` references at construction time.

---

## 2. PhysicalPlanExecutor (`src/physical/plan_executor.py`)

`PhysicalPlanExecutor` iterates the command list in a `PhysicalPlan` and dispatches each command to the appropriate handler:

### Command dispatch

| Command type | Handler | Side effects |
|---|---|---|
| `RemoveFromBoardCommand` | `piece_teleporter.teleport_piece_to_graveyard()` | Teleports piece, clears occupancy |
| `ArmMoveCommand` | `movement_executor.move_piece_between_squares()` | Full arm grasp/transit/place |
| `TeleportCommand("promotion_reserve", ...)` | `piece_teleporter.teleport_piece_to_promotion_reserve()` | Teleports piece, clears occupancy |
| `TeleportCommand("graveyard", ...)` | `piece_teleporter.teleport_piece_to_graveyard()` | Teleports piece, clears occupancy |
| `TeleportCommand("square", ...)` | `piece_teleporter.teleport_piece_to_square()` | Teleports piece, sets occupancy |

Execution stops at the first failing `ArmMoveCommand` and returns `PhysicalExecutionResult(success=False, ...)` with the error. Teleport commands cannot fail (they are direct qpos writes).

### return_to_home()

After every move, the orchestrator calls `return_to_home()`. This runs `controller.run_transit(home_xy)` where `home_xy` is loaded from `configs/env.yaml: home_position_xy` (board center: `[0.88, 0.2641]`). If that succeeds and the environment exposes `reset_arm_to_home_posture()`, the executor then restores the captured reset-time torso/arm/wrist/finger joint posture and mocap pose. The reset refuses to run while a piece is held, so it does not move fingers during carry. If `controller` or `env` is None (headless/test mode), returns success immediately.

### reset_board_state()

Resets all 32 pieces to their starting squares. Calls `env._reset_chess_piece_bodies()` if available (fast internal reset), otherwise teleports each piece individually. Also resets `PhysicalOccupancy`.

---

## 3. MovementExecutor (`src/physical/movement_executor.py`)

`MovementExecutor` handles a single board-to-board arm move. It is the bridge between the command layer and the `ScriptedController`.

### move_piece_between_squares()

```
1. Assert: piece_id is at src_square (PhysicalOccupancy check)
2. Assert: dst_square is empty (PhysicalOccupancy check)
3. Compute src_xy, dst_xy from BoardMapper
4. Call move_piece_xy(piece_id, src_xy, dst_xy, ...)
```

### move_piece_xy()

```
1. env.set_active_piece(piece_id)     ← tells task env which piece contact applies to
2. controller.run_full_move(src_xy, dst_xy)  ← full grasp/transit/place pipeline
3. Check result.success; on failure: extract crash_reason from stage_results
4. Reconciliation check:
   - env.get_active_piece_position() → actual piece position after place
   - Assert XY error < 20mm
   - Assert Z error < 10mm
5. If dst_square is provided: teleport piece to exact grid position
   (removes physics drift; ensures clean starting state for next move)
6. occupancy.set_piece_square(piece_id, dst_square)
```

The post-place teleport (step 5) is critical: after the arm releases, the piece may have drifted slightly due to physics. Teleporting it to the exact grid XYZ ensures the next move can reliably find it.

---

## 4. PhysicalOccupancy (`src/physical/occupancy.py`)

`PhysicalOccupancy` tracks the **expected** (not simulated) square occupancy. It is the physical layer's ground truth for "what piece is where."

### Data structures

```python
_piece_to_square: dict[str, str | None]  # piece_id → square
_square_to_piece: dict[str, str]         # square → piece_id (O(1) inverse)
```

### Key methods

| Method | Description |
|---|---|
| `piece_at_square(square)` | Returns piece_id at square, or None |
| `square_of_piece(piece_id)` | Returns square of piece, or None |
| `set_piece_square(piece_id, square)` | Moves piece; raises if destination is already occupied by a different piece |
| `assert_piece_at(piece_id, square)` | Raises `ValueError` if expected location doesn't match |
| `assert_square_empty(square)` | Raises `ValueError` if square is occupied |
| `reset(starting_square_map)` | Full reset from a piece_id → square map |

### Double-occupancy protection

`set_piece_square()` checks for double-occupancy before committing:
```python
current = self._square_to_piece.get(square)
if current is not None and current != piece_id:
    raise ValueError(f"Square {square} is already occupied by {current}")
```
This catches logical errors in move planning (e.g., trying to place on an occupied square).

---

## 5. PieceTeleporter (`src/physical/piece_teleport.py`)

`PieceTeleporter` performs instantaneous position writes directly into MuJoCo's `qpos` array. It is used for:
- Captured pieces → graveyard slots
- Promoted pawns → promotion reserve zone
- Reserve pieces → board squares (during promotion)
- Post-place reconciliation (called by `MovementExecutor`)

### How teleportation works

Each chess piece has a **freejoint** in the MuJoCo XML. A freejoint has 7 degrees of freedom in `qpos`: `[x, y, z, qw, qx, qy, qz]`.

```python
joint_id = model.joint(f"piece_{piece_id}:joint").id
qpos_start = model.jnt_qposadr[joint_id]
dof_start  = model.jnt_dofadr[joint_id]
data.qpos[qpos_start:qpos_start+3] = xyz
data.qpos[qpos_start+3:qpos_start+7] = quat
data.qvel[dof_start:dof_start+6] = 0.0   # zero velocity
data.qacc[dof_start:dof_start+6] = 0.0   # zero acceleration
mujoco.mj_forward(model, data)            # recompute derived state
```

The identity quaternion `[1, 0, 0, 0]` is used by default (upright orientation).

### Destination helpers

**Board squares** (`teleport_piece_to_square`):
- Calls `BoardMapper.square_to_piece_xyz(square)` to get exact world position.

**Graveyard** (`teleport_piece_to_graveyard`):
- Looks up `chess_cfg["graveyards"][color]` (origin_xyz, rows, cols).
- Computes slot XYZ: `origin + [row*spacing, col*spacing, 0]` where `slot = slot_index`, `row = slot // cols`, `col = slot % cols`.
- Spacing: `configs/chess.yaml: reserves.graveyard_slot_spacing_m = 0.045`.

**Promotion reserve** (`teleport_piece_to_promotion_reserve`):
- Same logic with `chess_cfg["promotion_reserve"][color]` and `promotion_slot_spacing_m = 0.045`.

**Slot parsing** (`_slot_index`):
- Slot IDs use format `"slot_NN"` (e.g., `"slot_00"`, `"slot_07"`).
- `re.fullmatch(r"slot_(\d+)", slot_id)` — strict format validation.

---

## 6. PieceRegistry (`src/physical/piece_registry.py`)

`PieceRegistry` is the authoritative list of the 32 active chess pieces. It is deterministic and stateless — it builds the same list every time.

### Piece ID naming convention

| Category | Format | Examples |
|---|---|---|
| King / Queen | `{color}_{type}` | `white_king`, `black_queen` |
| Other back-rank | `{color}_{type}_{file}` | `white_rook_a`, `black_knight_g` |
| Pawns | `{color}_pawn_{file}` | `white_pawn_e`, `black_pawn_d` |
| Reserve pieces | `{color}_reserve_{type}_{n}` | `white_reserve_queen_1` |

### PhysicalPiece dataclass

Each piece has:
- `piece_id`: unique stable identifier
- `color`: `"white"` or `"black"`
- `piece_type`: `"pawn"`, `"rook"`, `"knight"`, `"bishop"`, `"queen"`, `"king"`
- `body_name`: `"piece_{piece_id}"` — MuJoCo body name
- `joint_name`: `"piece_{piece_id}:joint"` — MuJoCo joint name
- `cube_geom_name`: `"piece_{piece_id}_cube"` — collision body geom name
- `visual_geom_name`: `"piece_{piece_id}_visual"` — visual mesh geom name
- `initial_square`: starting square (e.g., `"e1"` for white king), or None

### Reserve pieces

`reserve_piece_ids()` generates 4 colors × 4 types × 8 copies = 64 reserve piece entries (4 types × 8 per type = 32 per color, × 2 colors = 64 total). These are pre-placed in the promotion reserve zone in the XML and teleported onto the board during promotion.

### starting_square_map()

Returns `{piece_id: initial_square}` for all 32 active pieces. Used by `LogicalPieceTracker`, `PhysicalOccupancy`, and `reset_board_state()`.

---

## 7. Data Flow: Capture Move

To illustrate how the layers interact, here is the full flow for a capture move (e.g., `d5xc6` en passant):

```
GameOrchestrator.submit_human_move("d5", "c6")
  │
  ├─ ChessService.validate_square_move("d5", "c6")
  │    └─ chess.Move(d5, c6) validated as legal en passant
  │
  ├─ MovePlanner.plan(move)
  │    ├─ _captured_square → c5 (pawn behind, not c6)
  │    ├─ tracker.piece_id_at("c5") → "black_pawn_c"
  │    ├─ tracker.next_graveyard_slot("black") → "slot_01"
  │    └─ PhysicalPlan: [
  │         RemoveFromBoardCommand("black_pawn_c", "slot_01"),
  │         ArmMoveCommand("white_pawn_d", "d5", "c6")
  │       ]
  │
  ├─ PhysicalPlanExecutor.execute(plan)
  │    ├─ RemoveFromBoardCommand:
  │    │    teleporter.teleport_piece_to_graveyard("black_pawn_c", "slot_01")
  │    │    occupancy.set_piece_square("black_pawn_c", None)
  │    │
  │    └─ ArmMoveCommand:
  │         movement_executor.move_piece_between_squares(
  │             "white_pawn_d", "d5", "c6")
  │         → occupancy.assert_piece_at("white_pawn_d", "d5")  ✓
  │         → occupancy.assert_square_empty("c6")              ✓
  │         → controller.run_full_move(d5_xy, c6_xy)
  │         → post-place teleport to exact grid pos
  │         → occupancy.set_piece_square("white_pawn_d", "c6")
  │
  ├─ physical_executor.return_to_home()
  │
  ├─ chess_service.push(move)         ← commit chess state
  └─ piece_tracker.apply_committed_move(move, plan)
       ├─ RemoveFromBoardCommand: _remove_piece("black_pawn_c")
       └─ ArmMoveCommand: set_piece_at("c6", "white_pawn_d")
```

## NoOpPhysicalExecutor

`NoOpPhysicalExecutor` in `src/physical/noop_executor.py` accepts all physical plans and reports each command as successful without touching MuJoCo. It is used by logic-only eval scripts and tests.

`PhysicalPlanExecutor` now dispatches commands through a handler registry, so new command types can be added by registering a handler instead of expanding a long conditional chain.
