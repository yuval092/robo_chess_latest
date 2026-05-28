# Web UI

## Overview

The UI is a single-page Flask app:

```text
src/ui/app.py
src/ui/queued_backend.py
src/ui/templates/index.html
src/ui/static/app.js
src/ui/static/styles.css
```

The browser renders the board from JSON snapshots and sends REST requests for
new games, moves, and computer turns.

## Flask App Factory

`create_app(backend)` creates an isolated Flask app bound to a `UIBackend`
protocol:

```python
snapshot()
new_game()
submit_human_move(src, dst, promotion=None)
let_computer_play_current_turn()
```

Production passes `QueuedUIBackend`. Tests can pass `GameOrchestrator` or a fake
object directly.

`to_jsonable()` recursively converts dataclasses, dicts, lists, and tuples into
plain JSON-compatible data for `jsonify()`.

## REST Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Serve the browser app |
| `GET` | `/api/snapshot` | Return current `GameSnapshot` |
| `POST` | `/api/new-game` | Reset game and physical board |
| `POST` | `/api/move` | Submit a human move |
| `POST` | `/api/promote` | Placeholder, returns 501 |
| `POST` | `/api/let-computer-play` | Ask engine to play current turn |
| `POST` | `/api/undo` | Placeholder, returns 501 |

`POST /api/move` body:

```json
{"src": "e2", "dst": "e4", "promotion": null}
```

Promotion uses `"q"`, `"r"`, `"b"`, or `"n"`.

Missing `src` or `dst` returns:

```json
{"accepted": false, "error": "src and dst are required"}
```

with HTTP 400.

## Queued Backend

`QueuedUIBackend` exists because MuJoCo is not thread-safe.

Data members:

| Field | Purpose |
|---|---|
| `_orchestrator` | Real game backend |
| `_requests` | `queue.Queue[UIRequest]` |
| `_snapshot` | Cached latest snapshot |
| `_lock` | Protects `_snapshot` |

Mutating methods call `_call()`, which enqueues a request and waits on a
single-slot response queue. The main runtime loop calls:

```python
backend.process_one(env=env, timeout=0.05)
```

For `new_game`, `process_one()` first resets MuJoCo and physical board state:

1. `env.reset()`
2. `physical_executor.reset_board_state()` or `reset_occupancy()`
3. `orchestrator.new_game()`

Then it updates the cached snapshot.

## Browser Behavior

`app.js`:

- Loads `/api/snapshot` on page load.
- Renders board squares and unicode chess pieces.
- Tracks selected square and legal destinations.
- Shows last move, turn, legal move count, FEN, move history, and errors.
- Supports board flipping.
- Detects promotion by pawn destination rank and opens a dialog.
- Disables controls while a request is in flight or the game is over.
- Polls snapshots with exponential backoff while `snapshot.is_busy` is true.

Busy polling uses the documented config values:

| Config | Value |
|---|---:|
| `busy_poll_initial_ms` | `200` |
| `busy_poll_max_ms` | `1500` |

The JavaScript currently hard-codes those same values.

