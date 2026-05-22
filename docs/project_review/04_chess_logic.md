# RoboChess — Chess Logic Review

## ChessService (`src/chess_game/chess_service.py`)

### Correctness

All chess rules are delegated to `python-chess`, which is a mature, well-tested library. The service correctly wraps:

- Legal move validation (`validate_uci`, `validate_square_move`)
- Game state (`status()` — check, checkmate, stalemate, draws, FEN)
- History (`san_history()`)
- Persistence (`save_to_file`, `load_from_file`)

**Tested scenarios (all pass):**
- Fool's mate (checkmate detection)
- Stalemate detection
- Castling availability
- En passant availability
- Promotion requires promotion piece
- Draw conditions: 50-move, 75-move, threefold repetition, insufficient material

### `choose_engine_move()` — Greedy Heuristic, Not an Engine

```python
def score(move: chess.Move) -> tuple[int, int, int, int, str]:
    ...
    return (
        1 if is_checkmate else 0,      # priority 1: checkmate
        captured_value,                 # priority 2: capture by value
        promotion_value,                # priority 3: promote
        1 if gives_check else 0,        # priority 4: give check
        self._stable_move_tiebreak(move), # priority 5: lexicographic UCI
    )
```

**Issue:** This is a greedy one-ply selector. It will:
1. Not see tactics more than 1 move deep (e.g., discovered attacks, forks)
2. Walk into hanging pieces if no immediate capture is available
3. Not evaluate positional factors (development, king safety, pawn structure)
4. Use lexicographic UCI as tiebreaker — e.g., `a2a3` beats `b2b4` regardless of chess strength

**This is fine for a demo.** However, the method name `choose_engine_move` implies more sophistication than it provides. Rename to `choose_greedy_move` or add a docstring that clearly states "single-ply greedy heuristic."

**Known weakness — `push/pop` inside `score()`:**  
The score function calls `board.push(move)` and `board.pop()` for each legal move to check for checkmate. With 20+ legal moves, this is 40+ push/pop operations per engine turn. For a greedy one-ply search this is unavoidable, but it's O(moves) not O(1).

---

## MovePlanner (`src/chess_game/move_planner.py`)

### Physical Plan Generation

For each legal chess move, `MovePlanner.plan()` generates a `PhysicalPlan` containing a sequence of `RemoveFromBoardCommand`, `ArmMoveCommand`, and `TeleportCommand`.

**Verified move types:**

| Move | Commands Generated |
|------|-------------------|
| Normal move (e2e4) | `ArmMove(white_pawn_e, e2, e4)` |
| Capture (e4d5) | `Remove(black_pawn_d, slot_00)` + `ArmMove(white_pawn_e, e4, d5)` |
| Castling (e1g1) | `ArmMove(white_king, e1, g1)` + `ArmMove(white_rook_h, h1, f1)` |
| En passant (e5d6) | `Remove(black_pawn_d, slot_00)` + `ArmMove(white_pawn_e, e5, d6)` |
| Promotion (a7a8q) | `ArmMove(white_pawn_a, a7, a8)` + `Teleport(white_pawn_a, promotion_reserve, slot_00)` + `Teleport(white_reserve_queen_1, square, a8)` |

All verified by unit tests. The ordering is correct:
- Capture removes the defending piece before the arm moves in (prevents collision)
- Promotion uses two teleports (pawn off, new queen on) after the arm places the pawn

### Castling Implementation

```python
def _castle_commands(self, move, king_id, king_src, king_dst):
    rank = chess.square_rank(move.from_square)
    if move.to_square > move.from_square:   # kingside
        rook_src = chess.square(7, rank)
        rook_dst = chess.square(5, rank)
    else:                                   # queenside
        rook_src = chess.square(0, rank)
        rook_dst = chess.square(3, rank)
```

Correctly handles both kingside (O-O) and queenside (O-O-O) for both colors. The rank is derived from the king's starting square, so it works for both white (rank 0) and black (rank 7).

**Note:** Castling moves both king and rook with `ArmMoveCommand`. This means the arm physically moves both pieces sequentially. There is no occupancy check between the two moves — if a piece somehow occupied the rook's destination square (f1 for kingside), the second move would fail with an occupancy error. In practice this cannot happen since `ChessService.validate_square_move()` would have rejected the castling move before planning.

### En Passant — Capture Square Calculation

```python
def _captured_square(self, move: chess.Move) -> chess.Square:
    if self.board.is_en_passant(move):
        return move.to_square - 8 if self.board.turn == chess.WHITE else move.to_square + 8
    return move.to_square
```

Verified correct:
- White en passant: e5→d6, captured pawn on d5. d6 - 8 = d5. Correct.
- Black en passant: e4→d3, captured pawn on d4. d3 + 8 = d4. Correct.

---

## LogicalPieceTracker (`src/chess_game/move_planner.py:44-126`)

Tracks physical piece IDs against logical squares. Maintained in parallel with `python-chess` board state.

### Promotion Tracking

After promotion, `apply_committed_move` processes:
1. `ArmMoveCommand(white_pawn_a, a7, a8)` → `set_piece_at("a8", "white_pawn_a")`
2. `TeleportCommand(white_pawn_a, promotion_reserve, slot_00)` → `_set_off_board("white_pawn_a")`
3. `TeleportCommand(white_reserve_queen_1, square, a8)` → `set_piece_at("a8", "white_reserve_queen_1")`

After step 1, pawn is at a8. After step 2, a8 is vacated. After step 3, queen is at a8. The final state is correct.

**Gap:** No test directly verifies the tracker state after promotion (only the plan commands are tested). See `06_test_coverage.md`.

### `next_graveyard_slot()` — Slot Indexing

```python
def next_graveyard_slot(self, color: str) -> str:
    return f"slot_{len(self._captured[color]):02d}"
```

Returns `slot_00` through `slot_15` (at most 15 non-king pieces per color can be captured). The graveyard has 4×4 = 16 slots, so this is safe. However, the slot index is a count of captures so far, not the actual grid layout index used by `PieceTeleporter._slot_xyz()`. The teleporter uses `_slot_index(slot_id)` which parses the trailing integer. For `slot_00` through `slot_15`, this correctly maps to rows 0-3 and cols 0-3.

---

## GameOrchestrator (`src/chess_game/game_orchestrator.py`)

### Move Commit Safety

The orchestrator only commits the chess move to the logical state after physical execution succeeds:

```python
physical_result = self.physical_executor.execute(plan)
if not physical_result.success:
    # NOT committed: FEN and tracker unchanged
    return MoveExecutionResult(True, False, move.uci(), ...)

home_result = self.physical_executor.return_to_home()
self.chess_service.push(move)          # committed here
self.piece_tracker.apply_committed_move(move, plan)
```

This is the correct approach — physical and logical states stay synchronised.

### Internal Exception Handling

```python
try:
    physical_result = self.physical_executor.execute(plan)
    ...
except Exception as exc:
    self.error = f"INTERNAL_ERROR: {exc}"
    raise  # re-raised
finally:
    self.is_busy = False  # always reset
```

`is_busy` is always cleared via `finally`. The exception is re-raised to the caller, which for `QueuedUIBackend._call()` means the exception is captured and re-raised in the UI thread:
```python
ok, result = response.get()
if not ok:
    raise result
```

This propagates the exception to the HTTP handler, which would return a 500 error to the UI. The UI's `postJson()` doesn't handle non-OK status codes gracefully — see `05_ui.md`.

---

## Draw Condition Evaluation Results

All draw conditions verified (from `eval_draw_conditions.py`):

| Condition | Result | Outcome |
|-----------|--------|---------|
| Stalemate | OK | `1/2-1/2`, 0 legal moves |
| Insufficient material (K vs K+N) | OK | `1/2-1/2` |
| 50-move claim | OK | `1/2-1/2` |
| Threefold repetition claim | OK | `1/2-1/2` |

The `claim_draw=True` parameter is passed to `outcome()` and `is_game_over()`, enabling draw claims automatically. This means the game ends when a draw can be *claimed*, not only when it is *forced* (e.g., 50-move vs 75-move). This is a design choice that should be documented.
