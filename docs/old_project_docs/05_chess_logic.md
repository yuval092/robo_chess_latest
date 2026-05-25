# Chess Logic

This document covers the full chess logic stack: the python-chess engine integration, the service layer, move planning, piece tracking, and the game orchestrator that ties it all together.

---

## 1. python-chess Library

[python-chess](https://python-chess.readthedocs.io/) is the core rules engine for the project. The project uses it for:

- **Board state** (`chess.Board`): holds piece positions, side to move, castling rights, en-passant square, half-move clock, full-move number.
- **Legal move generation** (`board.legal_moves`): fully correct, including en passant, castling, promotion, and check evasion.
- **Move validation** (`move in board.legal_moves`): single authoritative check — if python-chess says it's legal, it is.
- **Special move detection**: `board.is_castling(move)`, `board.is_en_passant(move)`, `board.is_capture(move)`, `move.promotion` attribute.
- **Move representation** (`chess.Move`): from/to squares plus optional promotion piece type.
- **UCI strings**: `move.uci()` → `"e2e4"`, `"e7e8q"` (promotion), `"e1g1"` (kingside castle).
- **SAN notation**: `board.san(move)` → `"Nf3"`, `"O-O"`, `"e8=Q"`.
- **Game-over detection**: `board.is_game_over(claim_draw=True)`, plus individual terminal conditions.
- **Square utilities**: `chess.parse_square("e4")` → integer 0–63; `chess.square_name(sq)` → `"e4"`; `chess.SQUARE_NAMES` list.

The library is **never subclassed or monkey-patched** — the project wraps it in `ChessService`.

---

## 2. ChessService (`src/chess_game/chess_service.py`)

`ChessService` is a thin facade over `chess.Board`. It adds:

- **SAN history tracking**: every pushed move is recorded in `_san_history` via `board.san(move)` before `board.push(move)`.
- **Square-based move input**: `validate_square_move(src, dst, promotion)` constructs a `chess.Move` from algebraic square names and validates it against legal moves.
- **UCI input**: `validate_uci(uci)` parses a UCI string and validates it.
- **Promotion parsing**: `_parse_promotion("q")` → `chess.QUEEN`. Rejects unknown promotion letters with `IllegalMoveError`.
- **Game-over status** (`status()`): returns a `GameStatus` dataclass with all terminal and non-terminal flags.
- **Save/load**: `save_to_file(path)` / `load_from_file(path)` — serialises FEN + SAN history to JSON.
- **Built-in chess engine**: `choose_engine_move()` — picks the best legal move using a hand-written heuristic scorer (see Section 6).

### Key design choices

- `push()` always re-validates the move before committing — defensive against callers passing stale moves.
- `_san_history` is maintained in sync with `board` so `pop()` removes the last SAN entry too.
- `IllegalMoveError` is a `ValueError` subclass; callers catch it to return clean error messages.

---

## 3. GameStatus Model (`src/chess_game/chess_service.py`)

`GameStatus` is a frozen dataclass returned by `ChessService.status()` and embedded in every `GameSnapshot`. Fields:

| Field | Type | Meaning |
|---|---|---|
| `turn` | `str` | `"white"` or `"black"` |
| `is_check` | `bool` | King in check |
| `is_game_over` | `bool` | Any terminal condition |
| `is_checkmate` | `bool` | Checkmate |
| `is_stalemate` | `bool` | Stalemate |
| `is_insufficient_material` | `bool` | Draw by material |
| `is_seventyfive_moves` | `bool` | 75-move rule (automatic draw) |
| `is_fivefold_repetition` | `bool` | 5-fold repetition (automatic draw) |
| `can_claim_fifty_moves` | `bool` | 50-move rule (claimable) |
| `can_claim_threefold_repetition` | `bool` | 3-fold repetition (claimable) |
| `outcome` | `str \| None` | PGN result string: `"1-0"`, `"0-1"`, `"1/2-1/2"` |
| `fen` | `str` | Current FEN string |
| `legal_moves` | `list[str]` | All legal moves in UCI format |

---

## 4. MovePlanner (`src/chess_game/move_planner.py`)

`MovePlanner` translates a validated `chess.Move` into a `PhysicalPlan` — a list of physical commands that the robot arm and teleporter must execute. It bridges the logical chess world and the physical simulation world.

### PhysicalPlan structure

```
PhysicalPlan
  .chess_move_uci  str           # e.g. "e2e4"
  .commands        list          # ordered list of commands
```

Three command types exist:

```
ArmMoveCommand(piece_id, src_square, dst_square)
    → Robot arm physically picks up and places piece

TeleportCommand(piece_id, destination_kind, destination_id)
    → Instantaneous teleport; no arm involvement
    → destination_kind: "square" | "promotion_reserve" | "graveyard"

RemoveFromBoardCommand(piece_id, graveyard_slot)
    → Teleports a captured piece to the graveyard
    → No arm movement needed (piece was already there)
```

### Normal move

For a quiet move or standard capture:

1. If capture: `RemoveFromBoardCommand(captured_piece_id, next_graveyard_slot)`
2. `ArmMoveCommand(moving_piece_id, src, dst)`

The captured piece is teleported away **before** the arm tries to move to the destination, ensuring the destination square is empty when the arm places the piece.

### Castling

`MovePlanner._castle_commands()` is called when `board.is_castling(move)` is true.

python-chess encodes castling as a king move to its destination square:
- Kingside: `e1g1` (white) / `e8g8` (black)
- Queenside: `e1c1` (white) / `e8c8` (black)

The rook's source and destination are computed from the rank and direction:

```python
if move.to_square > move.from_square:   # kingside
    rook_src = (file=7, rank=rank)       # h-file
    rook_dst = (file=5, rank=rank)       # f-file
else:                                    # queenside
    rook_src = (file=0, rank=rank)       # a-file
    rook_dst = (file=3, rank=rank)       # d-file
```

Output: two `ArmMoveCommand`s — king first, then rook. The arm executes them sequentially.

### En passant

En passant is detected with `board.is_en_passant(move)`. The captured pawn is on a **different square** than the move destination:

```python
def _captured_square(self, move):
    if board.is_en_passant(move):
        return move.to_square - 8   # white captures upward
               or move.to_square + 8  # black captures downward
    return move.to_square
```

Output: `RemoveFromBoardCommand(captured_pawn_id, slot)` + `ArmMoveCommand(moving_pawn, src, dst)`.

### Pawn promotion

When `move.promotion is not None`:

1. `RemoveFromBoardCommand` for any captured piece (if capture-promotion)
2. `ArmMoveCommand(pawn_id, src, dst)` — arm physically moves the pawn
3. `TeleportCommand(pawn_id, "promotion_reserve", slot)` — teleport pawn off to reserve zone
4. `TeleportCommand(reserve_piece_id, "square", dst)` — teleport the promoted piece onto the board

The promoted piece is pulled from the reserve pool (e.g., `white_reserve_queen_1`) via `tracker.reserve_piece_for(color, piece_type)`.

---

## 5. LogicalPieceTracker (`src/chess_game/move_planner.py`)

`LogicalPieceTracker` maintains a bidirectional map between **stable physical piece IDs** (like `"white_pawn_e"`) and **chess squares** (like `"e4"`).

### Internal data structures

```python
_piece_to_square: dict[str, str | None]     # piece_id → square (None = off-board)
_reserve_to_square: dict[str, str | None]   # reserve piece_id → square (None = in reserve)
_square_to_piece: dict[str, str]            # square → piece_id  (O(1) inverse map)
_captured: dict[str, list[str]]             # "white"/"black" → list of captured piece IDs
```

The `_square_to_piece` inverse map was added to bring `piece_id_at(square)` from O(n) to O(1). All mutations keep both maps in sync.

### Key methods

| Method | Description |
|---|---|
| `piece_id_at(square)` | Returns piece ID at square, or None |
| `set_piece_at(square, piece_id)` | Move a piece to a square; evicts occupant; updates both maps |
| `reserve_piece_for(color, piece_type)` | Returns first free reserve piece of given type |
| `next_graveyard_slot(color)` | Returns slot ID for next captured piece: `"slot_00"`, `"slot_01"`, … |
| `next_promotion_reserve_slot(color)` | Returns slot ID for next promoted pawn in promotion reserve zone |
| `apply_committed_move(move, plan)` | Applies all commands in a PhysicalPlan to update internal maps |

### apply_committed_move

This is called **after** physical execution succeeds, to commit the logical state:

- `RemoveFromBoardCommand`: calls `_remove_piece()` → sets piece off-board, adds to `_captured`
- `ArmMoveCommand`: calls `set_piece_at(dst, piece_id)` → moves piece to destination
- `TeleportCommand` with kind `"square"`: calls `set_piece_at(destination_id, piece_id)`
- `TeleportCommand` otherwise: calls `_set_off_board(piece_id)` → piece leaves the board

---

## 6. Built-in Chess Engine (`ChessService.choose_engine_move`)

The project does not use Stockfish or any external engine. It uses a greedy heuristic scorer:

```python
score(move) = (
    1 if is_checkmate else 0,   # checkmate > everything
    captured_value,              # MVV (most valuable victim)
    promotion_value,             # queen promotion = 9
    1 if gives_check else 0,     # check is a bonus
    move.uci()                   # deterministic tiebreak
)
```

`PIECE_VALUES`: pawn=1, knight=3, bishop=3, rook=5, queen=9, king=100.

The engine always picks the **single best move** by this metric. It is deterministic (same position → same move). This is intentional: the project is about physical chess execution, not AI strength.

---

## 7. GameOrchestrator (`src/chess_game/game_orchestrator.py`)

`GameOrchestrator` is the top-level coordinator. It owns:
- `chess_service` — the chess rules engine
- `physical_executor` — the physical plan executor (arm + teleporter)
- `piece_tracker` — the logical piece tracker
- `is_busy` — global lock preventing concurrent operations
- `error` — last error message surfaced to the UI
- `last_move` — UCI string of the last completed move

### GameSnapshot

Every public method returns a `GameSnapshot` (or `MoveExecutionResult` which embeds one):

```python
@dataclass(frozen=True)
class GameSnapshot:
    fen: str
    turn: str
    board: dict[str, str | None]       # square → piece symbol ("P", "k", etc.)
    physical_piece_ids: dict[str, str | None]  # square → piece_id
    legal_moves: list[str]
    status: GameStatus
    last_move: str | None
    move_history_san: list[str]
    is_busy: bool
    error: str | None
```

### Move submission flow

`submit_human_move(src, dst, promotion)`:
1. Guards: rejects if not human's turn, or if `is_busy`.
2. Validates move via `chess_service.validate_square_move()`.
3. Builds `PhysicalPlan` via `MovePlanner(board, tracker).plan(move)`.
4. Sets `is_busy = True`.
5. Calls `physical_executor.execute(plan)`.
6. If physical execution fails: clears `is_busy`, returns failure result.
7. Calls `physical_executor.return_to_home()`.
8. Calls `chess_service.push(move)` to commit chess state.
9. Calls `piece_tracker.apply_committed_move(move, plan)` to commit tracker state.
10. Clears `is_busy`, returns success result.
11. If `auto_computer_reply` is enabled and game is not over, calls `let_computer_play_current_turn()`.

### new_game guard

`new_game()` checks `is_busy` first. If the arm is mid-move, it returns an error snapshot rather than resetting game state mid-execution. This prevents tracker/board state divergence.

### Error handling

- Physical executor exceptions propagate as `INTERNAL_ERROR: ...` in the error field.
- `is_busy` is always cleared in the `finally` block even on exception.
- Home return failure is non-fatal: logged as a warning, move is still committed.

---

## 8. BoardMapper (`src/chess_game/board_mapper.py`)

`BoardMapper` converts between chess square names (`"e4"`) and 3D world coordinates (`[x, y, z]`).

Configuration is loaded from `configs/chess.yaml`:
- `board.center_xy`: board center in world XY
- `board.cell_size_m`: 80mm per cell
- `board.orientation.white_side`: which direction is the white side (`low_y`)
- `board.orientation.file_axis`: which world axis files run along (`x`)
- `board.orientation.rank_axis`: which world axis ranks run along (`y`)

### Conversion logic

`square_to_piece_xyz(sq: chess.Square) → np.ndarray`:
1. Get file (0–7) and rank (0–7) from `chess.square_file(sq)` / `chess.square_rank(sq)`.
2. With `white_side=low_y`: rank 1 = low Y, rank 8 = high Y; file a = low X.
3. Compute XY from center and cell size: `center_xy + (file - 3.5) * cell_size * [1, 0]` etc.
4. Z = `table_z + cube_height / 2` (piece center is at half-height above table).

`square_name_to_xy(square: str) → np.ndarray`:
Convenience wrapper calling `square_to_piece_xyz` and returning XY only.

---

## 9. Special Move Summary

| Move type | Detected by | Physical commands |
|---|---|---|
| Normal quiet | default | ArmMove(src → dst) |
| Normal capture | `board.is_capture` | Remove(captured) + ArmMove |
| Castling | `board.is_castling` | ArmMove(king) + ArmMove(rook) |
| En passant | `board.is_en_passant` | Remove(pawn behind) + ArmMove |
| Promotion | `move.promotion is not None` | [Remove(captured)?] + ArmMove(pawn) + Teleport(pawn→reserve) + Teleport(new_piece→dst) |
| Promotion+capture | both | Remove(captured) + ArmMove(pawn) + Teleport(pawn→reserve) + Teleport(new_piece→dst) |

## Headless Orchestrator Factory

`GameOrchestrator.create_headless()` creates an orchestrator backed by `NoOpPhysicalExecutor` for tests and logic-only evaluation scripts. `GameStatus` is defined in `chess_service.py`.
