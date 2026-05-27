# Physical Execution Layer

## Overview

The physical layer translates high-level `PhysicalPlan` command objects into concrete arm movements and MuJoCo freejoint teleports. It consists of four classes that form a pipeline:

```
PhysicalPlanExecutor
  ├── MovementExecutor      ← arm moves (ArmMoveCommand)
  ├── PieceTeleporter       ← instant repositioning (TeleportCommand, RemoveFromBoardCommand)
  └── PhysicalOccupancy     ← expected-state validation
```

---

## `PhysicalPlanExecutor` (`src/physical/plan_executor.py`)

### Responsibility

Iterates the commands in a `PhysicalPlan` and dispatches each to the appropriate handler. Stops on the first arm-move failure.

### Constructor

```python
PhysicalPlanExecutor(
    movement_executor: MovementExecutor,
    piece_teleporter: PieceTeleporter,
    occupancy: PhysicalOccupancy,
    controller=None,    # for return_to_home
    env=None,           # for reset_board_state
)
```

### `execute(plan: PhysicalPlan) → PhysicalExecutionResult`

```python
for command in plan.commands:
    handler = self._handlers[type(command)]
    result = handler(command)
    if hasattr(result, "success") and not result.success:
        return PhysicalExecutionResult(False, results, result.error)
return PhysicalExecutionResult(True, results)
```

Handler dispatch table:
- `RemoveFromBoardCommand` → `_handle_remove`
- `ArmMoveCommand` → `_handle_arm_move`
- `TeleportCommand` → `_handle_teleport`

### `_handle_remove(command)` — Piece Removal

```python
piece_teleporter.teleport_piece_to_graveyard(command.piece_id, command.graveyard_slot)
occupancy.set_piece_square(command.piece_id, None)
```

A removed piece is teleported to its graveyard slot and marked as off-board in the occupancy map.

### `_handle_arm_move(command)` — Physical Arm Move

```python
return movement_executor.move_piece_between_squares(
    command.piece_id, command.src_square, command.dst_square
)
```

Returns a `PhysicalMoveResult` whose `success` field determines whether execution continues.

### `_handle_teleport(command)` — Instant Repositioning

Dispatches based on `destination_kind`:
- `"square"`: teleport piece to board square + update occupancy
- `"graveyard"`: teleport to graveyard slot + clear occupancy
- `"promotion_reserve"`: teleport to off-board reserve + clear occupancy

### `return_to_home() → PhysicalExecutionResult`

```python
home_xy = load_config("env")["home_position_xy"]  # [0.88, 0.2641]
result = controller.run_transit(home_xy)

if result.success and hasattr(env, "reset_arm_to_home_posture"):
    posture_result = env.reset_arm_to_home_posture()
    if not posture_result["success"]:
        return PhysicalExecutionResult(False, ..., posture_result["reason"])

return PhysicalExecutionResult(True, ...)
```

### `reset_board_state()` — Full Board Reset

Called on `new_game()`. Either calls `env._reset_chess_piece_bodies()` (fast: resets all freejoint poses in MuJoCo) or falls back to teleporting each piece individually.

### `PhysicalExecutionResult`

```python
@dataclass(frozen=True)
class PhysicalExecutionResult:
    success:         bool
    command_results: list   # [(command, result), ...]
    error:           str | None
```

---

## `MovementExecutor` (`src/physical/movement_executor.py`)

### Responsibility

Executes a single board-to-board arm move: validates expected occupancy, runs the controller, and performs post-move reconciliation.

### Constructor

```python
MovementExecutor(
    env,
    controller,           # ScriptedController or ModelEmbeddedController
    board_mapper: BoardMapper,
    occupancy: PhysicalOccupancy,
)
```

### `move_piece_between_squares(piece_id, src_square, dst_square) → PhysicalMoveResult`

1. **Occupancy validation:**
   ```python
   occupancy.assert_piece_at(piece_id, src_square)    # expect piece here
   occupancy.assert_square_empty(dst_square)           # expect nothing there
   ```
2. Resolve world XY: `board_mapper.square_name_to_xy(src_square/dst_square)`
3. Delegate to `move_piece_xy()`

### `move_piece_xy(piece_id, src_xy, dst_xy, ...)` — Core Move

```python
env.set_active_piece(piece_id)         # tell env which piece is active
result = controller.run_full_move(src_xy, dst_xy)

if not result.success:
    # Extract crash reason from stage_results
    return PhysicalMoveResult(False, ..., error=f"{failed_stage}: {reason}")

# Post-move reconciliation
piece_pos = env.get_active_piece_position()
expected_z = TABLE_Z + CUBE_HEIGHT / 2
xy_error = ||piece_xy - dst_xy||
z_error  = |piece_z - expected_z|

if xy_error > reconcile_xy_tolerance_m (20 mm): → XY_RECONCILE_FAILED
if z_error  > reconcile_z_tolerance_m  (10 mm): → Z_RECONCILE_FAILED

# Snap to exact position
placed_xyz = board_mapper.square_to_piece_xyz(dst_square)
teleporter.teleport_piece_to_xyz(piece_id, placed_xyz, quat=IDENTITY_QUAT)
occupancy.set_piece_square(piece_id, dst_square)

return PhysicalMoveResult(True, ...)
```

### `PhysicalMoveResult`

```python
@dataclass
class PhysicalMoveResult:
    success:       bool
    piece_id:      str
    src_square:    str | None
    dst_square:    str | None
    stage_results: list         # from controller
    error:         str | None
```

---

## `PhysicalOccupancy` (`src/physical/occupancy.py`)

### Responsibility

Maintains the **expected** physical occupancy: which piece is supposed to be on which square according to the game plan. This is separate from the `LogicalPieceTracker` (which tracks the logical game state) and from MuJoCo's actual physics state.

This class acts as a sanity check. Before moving a piece, the executor verifies that the planned occupancy matches expectations. If a previous arm move failed and left a piece in the wrong place, these assertions catch the inconsistency.

### Internal State

```python
_piece_to_square: dict[str, str | None]   # piece_id → square or None
_square_to_piece: dict[str, str]          # square → piece_id (inverse)
```

### Methods

```python
piece_at_square(square) → str | None
square_of_piece(piece_id) → str | None

set_piece_square(piece_id, square):
    # Validates no collision before updating
    if square is not None and already_occupied:
        raise ValueError(f"Square {square} already occupied by {current}")
    ...

assert_square_empty(square):
    if occupied: raise ValueError(...)

assert_piece_at(piece_id, square):
    if actual != square: raise ValueError(...)

reset(starting_square_map):
    # Rebuild from scratch (used on new_game)
```

### Difference from `LogicalPieceTracker`

| Aspect | `PhysicalOccupancy` | `LogicalPieceTracker` |
|---|---|---|
| Tracks | Expected physical state | Logical game state |
| Updated by | `PhysicalPlanExecutor` | `GameOrchestrator` (after move commits) |
| Updated when | During plan execution | After successful execution |
| Contains | Active pieces only | Active + reserve + captured |
| Purpose | Validate arm preconditions | Map squares for planning |

---

## `PieceRegistry` (`src/physical/piece_registry.py`)

### Responsibility

Canonical source of truth for all physical chess piece identifiers. Deterministically generates the 32 active pieces and describes the 64 reserve pieces.

### `PhysicalPiece`

```python
@dataclass(frozen=True)
class PhysicalPiece:
    piece_id:         str   # e.g., "white_pawn_e"
    color:            str   # "white" or "black"
    piece_type:       str   # "pawn", "rook", "knight", "bishop", "queen", "king"
    body_name:        str   # MuJoCo body: "piece_white_pawn_e"
    joint_name:       str   # MuJoCo joint: "piece_white_pawn_e:joint"
    cube_geom_name:   str   # "piece_white_pawn_e_cube"
    visual_geom_name: str   # "piece_white_pawn_e_visual"
    initial_square:   str | None  # "e2" for white e-pawn
```

### Piece ID Convention

| Piece type | ID format | Example |
|---|---|---|
| King | `{color}_king` | `white_king` |
| Queen | `{color}_queen` | `black_queen` |
| Rook | `{color}_rook_{file}` | `white_rook_a` |
| Knight | `{color}_knight_{file}` | `black_knight_b` |
| Bishop | `{color}_bishop_{file}` | `white_bishop_c` |
| Pawn | `{color}_pawn_{file}` | `white_pawn_e` |

### Back Rank Order

```python
BACK_RANK = [
    ("a", "rook"), ("b", "knight"), ("c", "bishop"), ("d", "queen"),
    ("e", "king"), ("f", "bishop"), ("g", "knight"), ("h", "rook"),
]
```

### `starting_square_map() → dict[str, str]`

Returns `{piece_id: initial_square}` for all 32 active pieces. Used to initialise `LogicalPieceTracker` and `PhysicalOccupancy`.

### `reserve_piece_ids() → list[tuple[str, str, str]]`

Returns `(piece_id, color, piece_type)` for all 64 reserve pieces:
- 8 queens, 8 rooks, 8 bishops, 8 knights per colour
- IDs: `white_reserve_queen_1` through `white_reserve_knight_8`, same for black

---

## `PieceTeleporter` (`src/physical/piece_teleport.py`)

### Responsibility

Instantaneous repositioning of chess piece bodies in MuJoCo by directly writing to the free-joint `qpos` array. No physics is simulated — the piece simply appears at the new location after `mj_forward`.

### How MuJoCo Freejoints Work

Each chess piece body has a `type="free"` joint, which gives it 6 DOF (3 translation + 3 rotation as quaternion). The joint's state lives in `data.qpos[qpos_start : qpos_start+7]`:
```
qpos[0:3] = xyz position
qpos[3:7] = quaternion (w, x, y, z)
```

Teleport sets these directly and zeros `qvel` and `qacc`:
```python
joint_id = env.model.joint(joint_name).id
qpos_start = env.model.jnt_qposadr[joint_id]
dof_start  = env.model.jnt_dofadr[joint_id]
data.qpos[qpos_start:qpos_start+3] = xyz
data.qpos[qpos_start+3:qpos_start+7] = quat
data.qvel[dof_start:dof_start+6] = 0.0
data.qacc[dof_start:dof_start+6] = 0.0
mujoco.mj_forward(model, data)   # propagate kinematics
```

### `teleport_piece_to_xyz(piece_id, xyz, quat=None)`

Direct teleport to a world XYZ position with optional quaternion (defaults to `IDENTITY_QUAT = [1,0,0,0]`).

### `teleport_piece_to_square(piece_id, square)`

Converts square name to XYZ via `board_mapper.square_to_piece_xyz` and calls `teleport_piece_to_xyz`.

### `teleport_piece_to_graveyard(piece_id, slot_id)`

Computes graveyard slot XYZ from config and calls `teleport_piece_to_xyz`:

```python
# configs/chess.yaml:graveyards
white:
  origin_xyz: [0.640, -0.300, 0.015]
  rows: 4
  cols: 4
black:
  origin_xyz: [0.640, 0.680, 0.015]
  rows: 4
  cols: 4
```

Slot indexing: `row = slot_index // cols`, `col = slot_index % cols`.

### `teleport_piece_to_promotion_reserve(piece_id, slot_id)`

Same logic, using `configs/chess.yaml:promotion_reserve`:

```python
white:
  origin_xyz: [0.820, -0.300, 0.015]
  rows: 8
  cols: 4
black:
  origin_xyz: [0.820, 0.680, 0.015]
  rows: 8
  cols: 4
```

### Slot ID Format

Slot IDs are `"slot_NN"` (zero-padded two-digit integers). `_slot_index("slot_03")` = 3. Maximum slot is `rows * cols - 1`.

---

## Graveyard and Reserve Zone Geometry

### Graveyard

The graveyard holds captured pieces. Each colour has a separate zone:

| Zone | Origin XYZ | Grid | Spacing |
|---|---|---|---|
| White graveyard | `[0.640, -0.300, 0.015]` | 4×4 = 16 slots | 45mm |
| Black graveyard | `[0.640, 0.680, 0.015]`  | 4×4 = 16 slots | 45mm |

16 slots per colour supports full captures: at most 15 pieces can be captured in a game (both sides' 15 non-king pieces), plus 1 buffer.

### Promotion Reserve

The promotion reserve holds extra pieces for pawn promotion:

| Zone | Origin XYZ | Grid | Spacing |
|---|---|---|---|
| White reserve | `[0.820, -0.300, 0.015]` | 8×4 = 32 slots | 45mm |
| Black reserve | `[0.820, 0.680, 0.015]`  | 8×4 = 32 slots | 45mm |

32 slots: 8 queens × 1 + 8 rooks × 1 + 8 bishops × 1 + 8 knights × 1 per colour. Pieces are pre-placed here at game start in scene generation, then teleported to the board on promotion.

---

## Full Physical Execution Pipeline for a Capture

Example: White pawn on e5 captures Black pawn on d6 (d5→d6 en passant for clarity).

```
plan = PhysicalPlan("e5d6", commands=[
    RemoveFromBoardCommand("black_pawn_d", "slot_00"),   ← captured pawn
    ArmMoveCommand("white_pawn_e", "e5", "d6"),          ← moving pawn
])

execute(plan):
  1. _handle_remove(RemoveFromBoardCommand("black_pawn_d", "slot_00"))
     → PieceTeleporter.teleport_piece_to_graveyard("black_pawn_d", "slot_00")
        Writes black_pawn_d freejoint to [0.640, 0.635, 0.015]  (black graveyard slot 0)
     → occupancy.set_piece_square("black_pawn_d", None)

  2. _handle_arm_move(ArmMoveCommand("white_pawn_e", "e5", "d6"))
     → movement_executor.move_piece_between_squares("white_pawn_e", "e5", "d6")
        occupancy.assert_piece_at("white_pawn_e", "e5")  ✓
        occupancy.assert_square_empty("d6")              ✓  (black pawn already removed)
        src_xy = board_mapper.square_name_to_xy("e5")    → [0.760, 0.3641]
        dst_xy = board_mapper.square_name_to_xy("d6")    → [0.680, 0.4441]
        env.set_active_piece("white_pawn_e")
        controller.run_full_move(src_xy, dst_xy)
          ... 8 stages of arm movement ...
        Post-snap: teleport_piece_to_xyz("white_pawn_e", board_mapper.square_to_piece_xyz("d6"))
        occupancy.set_piece_square("white_pawn_e", "d6")
     → PhysicalMoveResult(success=True, ...)
     → results.append(...)

  return PhysicalExecutionResult(success=True, ...)
```

After `execute()` succeeds, `GameOrchestrator` calls:
```python
chess_service.push(move)
piece_tracker.apply_committed_move(move, plan)  # updates LogicalPieceTracker
```
