# Physical Layer

## Purpose

The physical layer converts `PhysicalPlan` commands into MuJoCo actions and
freejoint teleports. It is responsible for expected occupancy validation,
arm-executed board moves, captured-piece removal, promotion reserve movement, and
board resets.

## Piece Registry

`PieceRegistry` deterministically defines the 32 active physical pieces.

Piece ID examples:

| Piece | ID |
|---|---|
| White king | `white_king` |
| Black queen | `black_queen` |
| White a-rook | `white_rook_a` |
| Black b-knight | `black_knight_b` |
| White e-pawn | `white_pawn_e` |

Each `PhysicalPiece` contains:

```python
piece_id: str
color: str
piece_type: str
body_name: str
joint_name: str
cube_geom_name: str
visual_geom_name: str
initial_square: str | None
```

`reserve_piece_ids()` returns 64 promotion reserve IDs:

```text
{color}_reserve_{queen|rook|bishop|knight}_{1..8}
```

## Physical Occupancy

`PhysicalOccupancy` tracks expected board occupancy for active physical pieces:

- `piece_at_square(square)`
- `square_of_piece(piece_id)`
- `set_piece_square(piece_id, square | None)`
- `assert_square_empty(square)`
- `assert_piece_at(piece_id, square)`
- `reset(starting_square_map)`

This is separate from:

- `ChessService`, which owns legal chess state.
- `LogicalPieceTracker`, which maps logical chess pieces to physical IDs.
- MuJoCo, which owns actual simulated positions.

Occupancy assertions catch stale or inconsistent expected state before the arm
starts a move.

## Piece Teleporting

`PieceTeleporter` directly edits MuJoCo freejoint `qpos`, `qvel`, and `qacc`.

Methods:

| Method | Destination |
|---|---|
| `teleport_piece_to_xyz(piece_id, xyz, quat=None)` | Arbitrary pose |
| `teleport_piece_to_square(piece_id, square)` | Board square center |
| `teleport_piece_to_graveyard(piece_id, slot_id)` | Captured-piece grid |
| `teleport_piece_to_promotion_reserve(piece_id, slot_id)` | Promotion reserve grid |

All teleports use identity quaternion unless another quaternion is passed.

Slot IDs must match `slot_NN`, for example `slot_00`.

## Movement Executor

`MovementExecutor.move_piece_between_squares(piece_id, src_square, dst_square)`:

1. Verifies `piece_id` is expected at `src_square`.
2. Verifies `dst_square` is expected empty.
3. Converts both squares to XY with `BoardMapper`.
4. Selects the active piece in the environment.
5. Runs `controller.run_full_move(src_xy, dst_xy)`.
6. Checks final active piece position.
7. Fails if XY error exceeds `reconcile_xy_tolerance_m`.
8. Fails if Z error exceeds `reconcile_z_tolerance_m`.
9. Snaps the piece exactly to destination XYZ.
10. Updates `PhysicalOccupancy`.

Result type:

```python
@dataclass
class PhysicalMoveResult:
    success: bool
    piece_id: str
    src_square: str | None
    dst_square: str | None
    stage_results: list
    error: str | None = None
```

## Plan Executor

`PhysicalPlanExecutor.execute(plan)` dispatches commands:

| Command | Handler |
|---|---|
| `RemoveFromBoardCommand` | Teleport to graveyard and clear occupancy |
| `ArmMoveCommand` | Run `MovementExecutor` |
| `TeleportCommand(destination_kind="promotion_reserve")` | Teleport to promotion reserve and clear occupancy |
| `TeleportCommand(destination_kind="graveyard")` | Teleport to graveyard and clear occupancy |
| `TeleportCommand(destination_kind="square")` | Teleport to square and set occupancy |

It stops on the first command result with `success=False`. Exceptions are caught
and returned as `PhysicalExecutionResult(success=False, error=str(exc))`.

## Return Home and Reset

`return_to_home()`:

- No-ops successfully if no controller/env was supplied.
- Otherwise runs transit to `env.home_position_xy`.
- Then calls `env.reset_arm_to_home_posture()` when available.

`reset_board_state()`:

- Rebuilds the starting square map from `PieceRegistry`.
- Uses `env._reset_chess_piece_bodies()` when available.
- Clears active piece selection.
- Resets physical occupancy.

