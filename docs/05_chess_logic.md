# Chess Logic

## Overview

The chess logic layer translates human input and Stockfish decisions into `PhysicalPlan` objects that the physical execution layer can execute. It uses the `python-chess` library for move validation and board state, and the `stockfish` UCI engine for computer moves.

---

## `ChessService` (`src/chess_game/chess_service.py`)

### Responsibility

Thin wrapper around `python-chess`. Maintains the game board, validates moves, queries the Stockfish engine, and handles game persistence.

### Constructor

```python
ChessService(
    starting_fen: str | None = None,  # None → standard starting position
    engine_cfg: dict[str, Any] | None = None,
)
```

`engine_cfg` keys:

| Key | Default | Description |
|---|---|---|
| `stockfish_path` | `"stockfish"` | Executable name or full path |
| `skill_level` | 5 | Stockfish skill 0–20 |
| `think_time_s` | 0.5 | Seconds Stockfish may think per move |

If `engine_cfg` is `None`, no engine is started. This is used for headless/test instances.

### Key Methods

#### `validate_uci(uci: str) → chess.Move`

Parses a UCI string (e.g., `"e2e4"`) and validates it against the current legal moves. Raises `IllegalMoveError` if invalid.

#### `validate_square_move(src, dst, promotion) → chess.Move`

Constructs a move from source/destination square names (e.g., `"e2"`, `"e4"`) and an optional promotion piece character (`"q"`, `"r"`, `"b"`, `"n"`). Validates legality.

#### `push(move: chess.Move)`

Commits a validated move to the board. Records the SAN notation in `_san_history`. The move must have passed `validate_uci` or `validate_square_move` first.

#### `pop() → chess.Move`

Undoes the last committed move. Removes from SAN history.

#### `status() → GameStatus`

Returns a complete frozen snapshot of the current game state:

```python
@dataclass(frozen=True)
class GameStatus:
    turn:                     str       # "white" or "black"
    is_check:                 bool
    is_game_over:             bool
    is_checkmate:             bool
    is_stalemate:             bool
    is_insufficient_material: bool
    is_seventyfive_moves:     bool
    is_fivefold_repetition:   bool
    can_claim_fifty_moves:    bool
    can_claim_threefold_repetition: bool
    outcome:                  str | None  # "1-0", "0-1", "1/2-1/2"
    fen:                      str
    legal_moves:              list[str]   # UCI strings
```

#### `choose_engine_move() → chess.Move`

Queries Stockfish for the best move at the current position. Uses `chess.engine.Limit(time=think_time_s)`. Raises `RuntimeError` if no engine is configured.

#### `save_to_file(path)` / `load_from_file(path)`

JSON persistence of `{fen, san_history, half_move_clock, full_move_number}`.

#### `close()`

Shuts down the Stockfish process. Called on `new_game()` before creating a fresh service.

### `IllegalMoveError`

A `ValueError` subclass raised whenever an invalid or illegal move is submitted. The `GameOrchestrator` catches this and returns a rejected `MoveExecutionResult`.

---

## `GameOrchestrator` (`src/chess_game/game_orchestrator.py`)

### Responsibility

Top-level coordinator for a single chess game session. Combines chess rules, physical execution, and piece tracking. Manages the `is_busy` flag that prevents concurrent move execution.

### Constructor

```python
GameOrchestrator(
    chess_service: ChessService,
    physical_executor,              # PhysicalPlanExecutor or NoOpPhysicalExecutor
    piece_tracker: LogicalPieceTracker,
    human_color: str | None = None, # "white", "black", or "both"
    auto_computer_reply: bool | None = None,
    engine_cfg: dict | None = None,
)
```

Reads defaults from `configs/chess.yaml:game`:
- `human_color: "white"` — default player colour
- `auto_computer_reply: true` — computer plays immediately after human move

### `create_headless()` — Factory for Testing

```python
orchestrator = GameOrchestrator.create_headless()
```

Creates an orchestrator with `NoOpPhysicalExecutor` (no arm, instant moves) and no engine. Used in tests and CLI evaluation tools that only need the chess logic.

### Data Structures

#### `GameSnapshot`

```python
@dataclass(frozen=True)
class GameSnapshot:
    fen:                str
    turn:               str          # "white" or "black"
    board:              dict[str, str | None]  # square → piece symbol
    physical_piece_ids: dict[str, str | None]  # square → piece_id
    legal_moves:        list[str]    # UCI strings
    status:             GameStatus
    last_move:          str | None   # UCI of most recent move
    move_history_san:   list[str]
    is_busy:            bool
    error:              str | None
```

Returned by `snapshot()` and embedded in `MoveExecutionResult`. The browser UI uses this to render the board.

#### `MoveExecutionResult`

```python
@dataclass(frozen=True)
class MoveExecutionResult:
    accepted:            bool     # was the move logically valid?
    physical_success:    bool     # did the arm execute successfully?
    move_uci:            str | None
    error:               str | None
    snapshot:            GameSnapshot
    awaiting_promotion:  bool = False
    promotion_square:    str | None = None
```

### Move Execution Flow

`_submit_move(move_factory)` is the core internal method:

```python
def _submit_move(self, move_factory):
    if self.is_busy:
        return self._rejected("Game is busy.")

    try:
        move = move_factory()
        plan = MovePlanner(self.chess_service.board, self.piece_tracker).plan(move)
    except (IllegalMoveError, ValueError) as exc:
        return self._rejected(str(exc))

    self.is_busy = True
    try:
        physical_result = self.physical_executor.execute(plan)
        if not physical_result.success:
            self.error = physical_result.error
            return MoveExecutionResult(True, False, move.uci(), ...)

        home_result = self.physical_executor.return_to_home()
        if not home_result.success:
            self.error = home_result.error  # non-fatal warning

        self.chess_service.push(move)
        self.piece_tracker.apply_committed_move(move, plan)
        self.last_move = move.uci()
        self.error = None
        return MoveExecutionResult(True, True, move.uci(), ...)
    finally:
        self.is_busy = False
```

Note: `chess_service.push()` is called **after** physical execution. The board state only advances once the arm has physically completed the move. This prevents the chess engine from planning moves for a position the arm hasn't reached yet.

### `submit_human_move(src, dst, promotion)`

1. Validates it is the human's turn.
2. Calls `_submit_move` with `chess_service.validate_square_move(src, dst, promotion)`.
3. If `auto_computer_reply=True` and game not over, chains to `let_computer_play_current_turn()`.

### `let_computer_play_current_turn()`

1. Checks for game over.
2. Calls `_submit_move` with `chess_service.choose_engine_move`.
3. If `auto_computer_reply=True` and it's still the computer's turn (e.g., both sides are computer-controlled), chains recursively.

### `new_game()`

1. Refuses if `is_busy`.
2. Closes the old Stockfish process.
3. Creates a new `ChessService` with the engine config.
4. Resets `LogicalPieceTracker`.
5. Clears `error` and `last_move`.
6. Returns a fresh `GameSnapshot`.

---

## `BoardMapper` (`src/chess_game/board_mapper.py`)

### Responsibility

Authoritative translation between python-chess square indices (`chess.Square`, 0–63) and world-frame XY coordinates used by the arm.

### `BoardGeometry`

```python
@dataclass(frozen=True)
class BoardGeometry:
    center_xy:       tuple[float, float]  # [0.88, 0.2641]
    cell_size_m:     float                # 0.08 m
    board_size:      int                  # 8
    table_surface_z: float                # 0.400 m
    cube_height_m:   float                # 0.030 m
    table_half_x:    float                # 0.35 m
    table_half_y:    float                # 0.35 m
```

### Board Coordinate Convention

```
World X axis → chess file axis (a=0.60m, h=1.16m)
World Y axis → chess rank axis (rank1=−0.016m, rank8=0.544m)
White plays from low Y (rank 1–2); Black from high Y (rank 7–8)
```

The board centre is at world XY `[0.88, 0.2641]`. Each cell is 80mm × 80mm. The full board occupies `[0.60, −0.016]` to `[1.16, 0.544]`.

### `square_to_xy(square: chess.Square) → np.ndarray`

```python
rank_index = chess.square_rank(square)
file_index = chess.square_file(square)
min_xy = board_min_xy   # lower-left corner
return [min_xy[0] + (file_index + 0.5) * cell_size_m,
        min_xy[1] + (rank_index + 0.5) * cell_size_m]
```

### `square_to_piece_xyz(square) → np.ndarray`

Returns the 3D piece centre (XY on board + Z at piece CoM height):
```python
z = table_surface_z + cube_height_m / 2   # 0.400 + 0.015 = 0.415 m
```

### `nearest_square(xy) → chess.Square`

Finds the closest square to a world XY coordinate:
```python
offset = (xy - board_min_xy) / cell_size_m
file_index = clamp(floor(offset[0]), 0, 7)
rank_index = clamp(floor(offset[1]), 0, 7)
return chess.square(file_index, rank_index)
```

### `assert_on_board(xy)` / `_validate_geometry()`

`assert_on_board` raises `ValueError` if the coordinate is outside the 8×8 board extent.

`_validate_geometry` checks that `cell_size_m == 0.08`, `board_size == 8`, board width = 0.64 m, and that the board fits within the table with at least 30mm margins on all sides.

---

## `MovePlanner` (`src/chess_game/move_planner.py`)

### Responsibility

Translates a validated `chess.Move` into a `PhysicalPlan` — an ordered list of physical commands the arm can execute.

### `PhysicalPlan`

```python
@dataclass(frozen=True)
class PhysicalPlan:
    chess_move_uci: str
    commands: list   # ordered list of commands
```

Three command types:

```python
@dataclass(frozen=True)
class ArmMoveCommand:
    piece_id:   str   # physical piece identifier
    src_square: str   # e.g., "e2"
    dst_square: str   # e.g., "e4"

@dataclass(frozen=True)
class TeleportCommand:
    piece_id:          str
    destination_kind:  str   # "square", "promotion_reserve", "graveyard"
    destination_id:    str   # square name or slot id

@dataclass(frozen=True)
class RemoveFromBoardCommand:
    piece_id:       str
    graveyard_slot: str   # e.g., "slot_00"
```

### `plan(move: chess.Move) → PhysicalPlan`

The planner handles four move types:

#### Normal Move

```
commands = [ArmMoveCommand(moving_piece_id, src, dst)]
```

#### Capture

```
captured_square = move.to_square (or en passant square)
commands = [
    RemoveFromBoardCommand(captured_piece_id, next_graveyard_slot),
    ArmMoveCommand(moving_piece_id, src, dst),
]
```

The captured piece is teleported to the graveyard before the moving piece arrives at the destination square.

#### Castling

```
commands = [
    ArmMoveCommand(king_id, king_src, king_dst),
    ArmMoveCommand(rook_id, rook_src, rook_dst),
]
```

Both the king and rook are moved physically. The rook source/destination squares are computed from the move's to_square relative to the from_square:
- Kingside: rook h→f
- Queenside: rook a→d

#### Promotion

```
commands = [
    ArmMoveCommand(pawn_id, src, dst),
    TeleportCommand(pawn_id, "promotion_reserve", promotion_slot),
    TeleportCommand(promoted_piece_id, "square", dst),
]
```

The pawn physically moves to the promotion square, is then teleported to the promotion reserve area (off-board), and the reserve piece of the promoted type is teleported onto the promotion square.

### En Passant Square Calculation

```python
if self.board.is_en_passant(move):
    captured_square = (
        move.to_square - 8   # White captures Black pawn one rank behind
        if self.board.turn == chess.WHITE
        else move.to_square + 8
    )
```

---

## `LogicalPieceTracker` (`src/chess_game/move_planner.py`)

### Responsibility

Maintains a bijective mapping between piece IDs (physical identifiers like `"white_pawn_e"`) and board squares. Updated after each committed move.

### Internal State

```python
_piece_to_square:  dict[str, str | None]  # piece_id → square name or None (off-board)
_reserve_to_square: dict[str, str | None] # reserve_piece_id → square name or None
_square_to_piece:  dict[str, str]         # square → piece_id (inverse map)
_captured:         dict[str, list[str]]   # {"white": [...], "black": [...]}
```

The inverse map `_square_to_piece` enables O(1) square→piece_id lookups.

### Initialisation

At game start, `PieceRegistry().starting_square_map()` provides the initial `piece_id → square` mapping for all 32 active pieces. Reserve pieces (for promotion) start with `None` square.

### `piece_id_at(square) → str | None`

Returns the physical piece ID at a square, or `None` if empty.

### `set_piece_at(square, piece_id)` — Direct Update

Updates both the forward and inverse maps consistently. Evicts any existing occupant of the target square.

### `apply_committed_move(move, plan)` — Plan Application

Applies each command in the plan to update logical occupancy:
- `RemoveFromBoardCommand` → marks piece off-board, adds to `_captured`
- `ArmMoveCommand` → calls `set_piece_at(dst_square, piece_id)`
- `TeleportCommand` with `destination_kind="square"` → calls `set_piece_at`
- `TeleportCommand` with other destinations → marks piece off-board

### `reserve_piece_for(color, piece_type) → str`

Finds the first available reserve piece of the requested colour and type:
```python
prefix = f"{color}_reserve_{piece_type}_"
# Returns first piece_id with that prefix and square=None
```

### `next_graveyard_slot(color) → str`

Returns `f"slot_{len(_captured[color]):02d}"` — the next sequential graveyard slot.

---

## `NoOpPhysicalExecutor` (`src/physical/noop_executor.py`)

A stub physical executor that immediately returns `success=True` for any plan. Used in:
- `GameOrchestrator.create_headless()`
- Tests that only test chess logic

---

## Stockfish Engine Configuration

Stockfish is configured via `configs/chess.yaml:engine`:

```yaml
engine:
  stockfish_path: stockfish  # name on PATH, or full path
  skill_level: 5             # Stockfish Skill Level option, 0-20
  think_time_s: 0.5          # seconds per move
```

Stockfish is started via `chess.engine.SimpleEngine.popen_uci(path)` and configured with `{"Skill Level": skill}`. It is shut down with `engine.quit()` on `ChessService.close()`.

---

## Special Move Handling Summary

| Move Type | Physical Commands |
|---|---|
| Normal | 1 `ArmMoveCommand` |
| Capture | 1 `RemoveFromBoardCommand` + 1 `ArmMoveCommand` |
| En Passant | 1 `RemoveFromBoardCommand` (captured pawn square) + 1 `ArmMoveCommand` |
| Castling | 2 `ArmMoveCommand` (king + rook) |
| Promotion | 1 `ArmMoveCommand` + 1 `TeleportCommand` (pawn → reserve) + 1 `TeleportCommand` (reserve piece → board) |
| Promotion with Capture | 1 `RemoveFromBoardCommand` + 1 `ArmMoveCommand` + 2 `TeleportCommand` |
