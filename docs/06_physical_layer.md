# Physical Layer

## PieceRegistry (`piece_registry.py`)

Deterministic registry of all 32 active chess pieces plus 64 reserve pieces.

Each piece is a frozen `PhysicalPiece` dataclass with:
- `piece_id` — e.g. `"white_rook_a"`, `"black_pawn_e"`, `"white_queen"`
- `body_name` — MuJoCo body name (`f"piece_{piece_id}"`)
- `joint_name` — free-joint name (`f"piece_{piece_id}:joint"`)
- `cube_geom_name` / `visual_geom_name` — collision and visual geom names
- `initial_square` — starting board square (e.g. `"a1"`)

Naming rules:
- King and queen: `"{color}_{type}"` (singular, no file letter)
- Other back-rank pieces: `"{color}_{type}_{file}"` (e.g. `"white_rook_a"`, `"white_rook_h"`)
- Pawns: `"{color}_pawn_{file}"`
- Reserve pieces: `"{color}_reserve_{type}_{1..8}"`

`reserve_piece_ids()` returns all 64 reserve pieces as `(piece_id, color, piece_type)` tuples.

---

## PhysicalOccupancy (`occupancy.py`)

Tracks the *expected* physical square occupancy. Updated after every move completes; used to guard against invalid move requests.

| Method | Description |
|---|---|
| `piece_at_square(square)` | Physical piece ID at a square, or `None` |
| `square_of_piece(piece_id)` | Square a piece is on, or `None` |
| `set_piece_square(piece_id, square)` | Move a piece; raises if target square is occupied by another piece |
| `assert_square_empty(square)` | Raise `ValueError` if occupied |
| `assert_piece_at(piece_id, square)` | Raise `ValueError` if piece is not there |
| `reset(starting_square_map)` | Reset to a given layout |

Maintains bidirectional maps for O(1) lookups in both directions.

---

## PieceTeleporter (`piece_teleport.py`)

Instant repositioning by directly writing to MuJoCo free-joint `qpos` and zeroing velocities, followed by `mj_forward`.

| Method | Description |
|---|---|
| `teleport_piece_to_xyz(piece_id, xyz)` | Move piece to absolute world XYZ |
| `teleport_piece_to_square(piece_id, square)` | Move piece to board square centre |
| `teleport_piece_to_graveyard(piece_id, slot_id)` | Move piece to a graveyard slot |
| `teleport_piece_to_promotion_reserve(piece_id, slot_id)` | Move to promotion reserve slot |

Slot IDs are formatted as `"slot_NN"` (e.g. `"slot_00"`, `"slot_03"`). Graveyard and reserve positions come from `configs/chess.yaml`.

---

## MovementExecutor (`movement_executor.py`)

Executes physical board-to-board arm moves. The primary entry point is:

```python
move_piece_between_squares(piece_id, src_square, dst_square) -> PhysicalMoveResult
```

Flow:
1. Assert occupancy (piece is at src, dst is empty).
2. Convert squares to world XY via `BoardMapper`.
3. `set_active_piece(piece_id)` on the env.
4. `controller.run_full_move(src_xy, dst_xy)` — eight-stage SAC + scripted pipeline.
5. `_check_landing_tolerance` — verify piece landed within `reconcile_xy_tolerance_m` and `reconcile_z_tolerance_m` of destination.
6. `_reconcile_placement` — teleport piece to exact destination centre and update occupancy.

Landing tolerance check is a second line of defence after `_verify_placement` inside the place pipeline. It catches cases where the physical simulation ended up slightly off.

`PhysicalMoveResult` contains `success`, `piece_id`, `src_square`, `dst_square`, `stage_results` (list of `(stage_name, StageResult)` tuples), and `error`.

---

## PhysicalPlanExecutor (`plan_executor.py`)

Executes a list of physical commands produced by `MovePlanner`.

### `execute(plan)`

Iterates the command list. For each command:
- `RemoveFromBoardCommand` → teleport to graveyard, clear occupancy.
- `ArmMoveCommand` → `movement_executor.move_piece_between_squares`.
- `TeleportCommand` → teleport to square / graveyard / reserve, update occupancy.

Stops on the first `ArmMoveCommand` failure. Teleport commands never fail.

### `return_to_home()`

1. `controller.run_transit(home_xy)` — SAC transit to `home_position_xy`.
2. `env.reset_arm_to_home_posture()` — snap wrist/roll joints back to canonical posture.

`home_xy` is cached at construction from `configs/env.yaml` to avoid repeated disk reads.

Returns `PhysicalExecutionResult(success, command_results, error)`.
