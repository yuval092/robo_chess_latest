# Stage 5: Move Planning And Special Moves

## Goal

Translate a validated chess move into an ordered physical execution plan.

The planner is the integration boundary between chess rules and physical movement. It knows chess move semantics and physical location categories, but it must not directly manipulate MuJoCo data. It emits commands that a physical executor can run.

## Command Model

Create command dataclasses in `src/chess_game/move_planner.py`.

```python
@dataclass(frozen=True)
class ArmMoveCommand:
    piece_id: str
    src_square: str
    dst_square: str

@dataclass(frozen=True)
class TeleportCommand:
    piece_id: str
    destination_kind: str  # graveyard | promotion_reserve | square
    destination_id: str

@dataclass(frozen=True)
class RemoveFromBoardCommand:
    piece_id: str
    graveyard_slot: str

@dataclass(frozen=True)
class PromoteCommand:
    pawn_piece_id: str
    promoted_piece_id: str
    square: str
    reserve_slot: str

@dataclass(frozen=True)
class PhysicalPlan:
    chess_move_uci: str
    commands: list
```

The executor must process commands in order.

## Piece Identity Mapping

The chess board stores only color and piece type. The physical layer needs stable piece IDs.

Create a `LogicalPieceTracker`:

```python
class LogicalPieceTracker:
    def piece_id_at(self, square: str) -> str | None: ...
    def set_piece_at(self, square: str, piece_id: str | None) -> None: ...
    def captured_pieces(self, color: str) -> list[str]: ...
    def reserve_piece_for(self, color: str, piece_type: str) -> str: ...
    def apply_committed_move(self, move, plan) -> None: ...
```

This tracker is updated only after the physical plan succeeds and the chess board push is committed.

## Normal Quiet Move

Example: `e2e4`

Plan:

```text
ArmMoveCommand(piece_id=piece_at_e2, src_square=e2, dst_square=e4)
```

No teleportation.

## Capture

Example: `e4d5`

Plan:

```text
RemoveFromBoardCommand(captured_piece_id, next_opponent_graveyard_slot)
ArmMoveCommand(attacker_piece_id, src_square=e4, dst_square=d5)
```

Capture order matters:

1. Teleport captured piece to graveyard first.
2. Then move attacker to destination with the arm.

Reason:

- The destination square must be empty before the arm descends.
- The captured piece goes to an unreachable floor graveyard, so teleportation is required.

White graveyard contains captured white pieces. Black graveyard contains captured black pieces.

## Castling

Examples:

- White kingside: `e1g1`, rook `h1f1`
- White queenside: `e1c1`, rook `a1d1`
- Black kingside: `e8g8`, rook `h8f8`
- Black queenside: `e8c8`, rook `a8d8`

Plan:

```text
ArmMoveCommand(king_id, src_square=e1, dst_square=g1)
ArmMoveCommand(rook_id, src_square=h1, dst_square=f1)
```

Validate with `python-chess` before planning. The planner should use `board.is_castling(move)`.

Implementation detail:

- It is physically acceptable to move king first then rook because the move has already been legally validated.
- If king move succeeds and rook move fails, enter recovery-required state. Do not push the chess board until both physical moves succeed.

## En Passant

Example:

- White pawn on `e5`, black pawn moved `d7d5`, white plays `e5d6`.
- Captured pawn is on `d5`, not destination `d6`.

Plan:

```text
RemoveFromBoardCommand(captured_pawn_id_at_d5, black_graveyard_slot)
ArmMoveCommand(white_pawn_id, src_square=e5, dst_square=d6)
```

Use:

```python
board.is_en_passant(move)
```

Captured square calculation:

```python
if board.turn == chess.WHITE:
    captured_square = move.to_square - 8
else:
    captured_square = move.to_square + 8
```

## Promotion

Example: `e7e8q`

Physical representation:

- The pawn cube is moved by the arm to the promotion square.
- Then the pawn piece is teleported to the promotion reserve.
- A reserve visual piece of the promoted type and color is teleported to the promotion square.

Plan without capture:

```text
ArmMoveCommand(pawn_id, src_square=e7, dst_square=e8)
TeleportCommand(pawn_id, destination_kind=promotion_reserve, destination_id=next_reserve_slot)
TeleportCommand(promoted_piece_id, destination_kind=square, destination_id=e8)
```

Plan with capture:

```text
RemoveFromBoardCommand(captured_piece_id_at_e8, graveyard_slot)
ArmMoveCommand(pawn_id, src_square=e7, dst_square=e8)
TeleportCommand(pawn_id, destination_kind=promotion_reserve, destination_id=next_reserve_slot)
TeleportCommand(promoted_piece_id, destination_kind=square, destination_id=e8)
```

Reserve pieces:

- Pre-create extra visual/physical queen, rook, bishop, and knight pieces for each color.
- They may use the same cube physical shape.
- Park them in promotion reserve slots until needed.
- When promoted, the reserve piece becomes the active logical piece on the board.

Why not change the pawn STL in place:

- MuJoCo mesh/material swapping at runtime is awkward.
- Keeping reserve pieces makes identity and visuals explicit.
- Teleporting to/from reserves matches the user requirement.

## Plan Transaction Rules

The integration layer must follow:

1. Validate chess move.
2. Build physical plan against current logical piece tracker.
3. Execute physical plan.
4. If all commands succeed:
   - Push move to `python-chess` board.
   - Commit tracker update.
   - Emit new status to UI.
5. If any command fails:
   - Do not push chess board.
   - Do not commit tracker update.
   - Return recovery-required state.

## Tests

Add `tests/chess_game/test_move_planner.py`.

Required tests:

- Normal move emits one arm command.
- Capture emits remove-then-arm.
- Castling emits king arm move then rook arm move.
- En passant removes the pawn from the passed-over square.
- Promotion emits pawn arm move and two teleport commands.
- Capture-promotion emits capture removal before promotion sequence.
- Illegal moves are rejected before planning.
- Planner never emits arm commands to graveyard or promotion reserve coordinates.

## Validation

Run:

```bash
pytest tests/chess_game/test_move_planner.py -v
pytest tests/chess_game/test_chess_service.py -v
```

Pass criteria:

- All special moves produce exact expected command sequences.
- Physical plans are deterministic.
- Chess board state is not mutated by planning.
