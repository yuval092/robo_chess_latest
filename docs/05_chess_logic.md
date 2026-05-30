# Chess Logic

## ChessService (`chess_service.py`)

Wraps a `python-chess` board and an optional Stockfish UCI engine.

### Key Methods

| Method | Description |
|---|---|
| `push(move)` | Commit a validated move to the board and SAN history |
| `pop()` | Undo the latest move |
| `status()` | Return a frozen `GameStatus` snapshot (turn, check, game-over, legal moves, FEN) |
| `legal_moves()` | Return all legal moves as UCI strings |
| `parse_uci(uci)` | Parse and validate a UCI string, raise `IllegalMoveError` if illegal |
| `construct_move_from_squares(src, dst, promotion)` | Build and validate a move from square names |
| `choose_engine_move()` | Ask Stockfish for a move; raises if no engine configured |
| `san_history()` | Return the move history in Standard Algebraic Notation |
| `fen()` | Return the current board FEN |
| `close()` | Shut down the Stockfish subprocess |

### UciEngine

Minimal synchronous UCI wrapper. Uses `select`-based non-blocking reads with a configurable timeout. Sends `uci` / `isready` handshake on startup. `choose_move` sends `position fen` + `go movetime` and reads until `bestmove`.

---

## LogicalPieceTracker (`move_planner.py`)

Tracks which physical piece ID occupies which board square. Separate from `PhysicalOccupancy` — this is the *logical* view derived from game moves, not the physical simulation state.

| Method | Description |
|---|---|
| `piece_id_at(square)` | Physical piece ID at a square, or `None` |
| `place_piece_at(square, piece_id)` | Move a piece to a square (evicts any occupant) |
| `find_reserve_piece(color, piece_type)` | Reserve a promotion piece by type |
| `next_graveyard_slot(color)` | Next available graveyard slot ID |
| `next_promotion_reserve_slot(color)` | Next promotion reserve slot ID |
| `apply_plan(plan)` | Update occupancy from a committed physical plan |

Maintains two maps: `_piece_to_square` and `_square_to_piece` for O(1) lookups in both directions. Also tracks captured pieces and promoted-out pawns for slot assignment.

---

## MovePlanner (`move_planner.py`)

Translates a legal `chess.Move` into a list of physical commands.

### Command Types

| Command | Description |
|---|---|
| `ArmMoveCommand(piece_id, src, dst)` | Physical arm move between squares |
| `TeleportCommand(piece_id, kind, id)` | Instant repositioning to square / graveyard / promotion_reserve |
| `RemoveFromBoardCommand(piece_id, slot)` | Remove a captured piece to graveyard |

### Move Translation

- **Normal move** → `[ArmMoveCommand]` (capture remove command prepended if captures).
- **Castle** → `[ArmMoveCommand(king), ArmMoveCommand(rook)]`.
- **Promotion** → `[ArmMoveCommand(pawn), TeleportCommand(pawn→reserve), TeleportCommand(new_piece→square)]`.
- **En passant** — the captured pawn square is computed from `to_square ± 8`.

---

## GameOrchestrator (`game_orchestrator.py`)

Top-level coordinator that ties chess logic, physical execution, and the engine together.

### Public Methods

| Method | Description |
|---|---|
| `new_game()` | Reset chess board, piece tracker, and physical state |
| `snapshot()` | Return a frozen `GameSnapshot` |
| `submit_human_move(src, dst, promotion)` | Validate and execute a human move |
| `let_computer_play_current_turn()` | Execute one engine move |

### Move Execution Flow (`_execute_move`)

1. `MovePlanner.plan(move)` — build the physical command list.
2. `_run_plan(move, plan)` — execute physically, commit on success.
3. `_auto_play_if_computer_turn()` — if `auto_computer_reply` and it's now the engine's turn, play immediately.

### `_run_plan`

```python
physical_result = physical_executor.execute(plan)
if not physical_result.success:
    return False, physical_result.error
home_result = physical_executor.return_to_home()
chess_service.push(move)
piece_tracker.apply_plan(plan)
last_move = move.uci()
return True, home_result.error   # None on full success; error string if home return failed
```

The move is committed to the chess board even if `return_to_home` fails (the physical move succeeded). The caller gets `physical_success=True` with a non-`None` error string in that case.

### GameSnapshot

Returned by `snapshot()` and included in every `MoveExecutionResult`. Contains:
- `fen`, `turn`, `legal_moves`, `status` — chess state
- `board` — dict of square → piece symbol (or `None`)
- `physical_piece_ids` — dict of square → physical piece ID
- `last_move`, `move_history_san` — history
- `error` — last error string, or `None`
