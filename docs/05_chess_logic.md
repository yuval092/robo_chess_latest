# Chess Logic

## `ChessService`

`src/chess_game/chess_service.py` wraps `python-chess` and optional UCI engine
access.

Constructor:

```python
ChessService(starting_fen: str | None = None, engine_cfg: dict | None = None)
```

If `starting_fen` is omitted, the standard starting position is used. If
`engine_cfg` is provided and contains a truthy `stockfish_path`, a UCI engine
process is started.

Supported engine config keys:

| Key | Default | Meaning |
|---|---:|---|
| `stockfish_path` | `stockfish` | Executable name or full path |
| `skill_level` | `5` | UCI Skill Level option |
| `think_time_s` | `0.5` | Move time sent as `go movetime` |

`configs/chess.yaml` also contains `fallback_depth`, but the current
implementation does not use a fallback minimax engine.

## UCI Engine Wrapper

`UciEngine` is a minimal synchronous Stockfish-style wrapper. It:

- Starts the engine with `subprocess.Popen([command])`.
- Sends `uci`, waits for `uciok`.
- Optionally sets `Skill Level`.
- Sends `isready`, waits for `readyok`.
- For move choice, sends `position fen ...` and `go movetime ...`.
- Parses the final `bestmove`.
- Validates the returned move against `board.legal_moves`.

The wrapper is intentionally small and synchronous. It is not a general UCI
management library.

## Game Status

`GameStatus` is a frozen dataclass returned by `ChessService.status()`:

```python
turn: str
is_check: bool
is_game_over: bool
is_checkmate: bool
is_stalemate: bool
is_insufficient_material: bool
is_seventyfive_moves: bool
is_fivefold_repetition: bool
can_claim_fifty_moves: bool
can_claim_threefold_repetition: bool
outcome: str | None
fen: str
legal_moves: list[str]
```

Draw claims are evaluated with `claim_draw=True`.

## Move Validation and Persistence

Important methods:

| Method | Purpose |
|---|---|
| `parse_uci(uci)` | Parse and validate UCI string |
| `construct_move_from_squares(src, dst, promotion)` | Build and validate from square names |
| `push(move)` | Revalidate, record SAN, and push to board |
| `pop()` | Undo latest move and SAN entry |
| `legal_moves()` | Return UCI strings |
| `choose_engine_move()` | Ask the configured UCI engine for a legal move |
| `close()` | Shut down the engine process |

Invalid moves raise `IllegalMoveError`.

## `GameOrchestrator`

`GameOrchestrator` coordinates turn ownership, planning, physical execution, and
commit order.

Constructor:

```python
GameOrchestrator(
    chess_service,
    physical_executor,
    piece_tracker,
    human_color=None,
    auto_computer_reply=None,
    engine_cfg=None,
)
```

Defaults come from `configs/chess.yaml:game`:

```yaml
human_color: white
auto_computer_reply: true
```

`human_color` may be `"white"`, `"black"`, or `"both"`.

## Snapshots and Results

`GameSnapshot` contains the full UI-facing state:

```python
fen: str
turn: str
board: dict[str, str | None]
physical_piece_ids: dict[str, str | None]
legal_moves: list[str]
status: GameStatus
last_move: str | None
move_history_san: list[str]
error: str | None
```

`MoveExecutionResult` contains:

```python
accepted: bool
physical_success: bool
move_uci: str | None
error: str | None
snapshot: GameSnapshot
```


## Commit Ordering

`_submit_move()` plans the move then execute it. Once physical execution
starts:

1. `physical_executor.execute(plan)` must succeed.
2. `physical_executor.return_to_home()` runs.
3. `chess_service.push(move)` commits the legal board.
4. `piece_tracker.apply_committed_move(plan)` commits logical piece IDs.
5. `last_move` and `error` are updated.

If physical execution fails, the chess board is left unchanged and the move is
reported as `accepted=True, physical_success=False`.

