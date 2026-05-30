# Web UI

## Flask Application (`src/ui/app.py`)

Created by `create_app(backend: UIBackend)`. All routes delegate to the `UIBackend` protocol — the concrete implementation is always `QueuedUIBackend`.

### REST Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves `index.html` (the browser chess UI) |
| `GET` | `/api/snapshot` | Returns the current `GameSnapshot` as JSON |
| `POST` | `/api/new-game` | Resets the game and physical state |
| `POST` | `/api/move` | Submit a human move; body: `{"src": "e2", "dst": "e4", "promotion": null}` |
| `POST` | `/api/let-computer-play` | Trigger one engine move |

### Response Format

All `/api/*` responses return JSON serialised from dataclasses via `to_jsonable` (recursive `asdict` for dataclasses, passthrough for primitives). The `/api/move` and `/api/let-computer-play` endpoints return `MoveExecutionResult`:

```json
{
  "accepted": true,
  "physical_success": true,
  "move_uci": "e2e4",
  "error": null,
  "snapshot": { ... }
}
```

`accepted=false` → 400 Bad Request (illegal move, wrong turn). `accepted=true, physical_success=false` → 200 OK (move accepted by rules but physical execution failed).

---

## QueuedUIBackend (`src/ui/queued_backend.py`)

Thread-safe wrapper that serialises all `GameOrchestrator` calls onto the main thread via a queue.

### Why a queue?

MuJoCo is not thread-safe. Flask runs on a daemon thread. The queue ensures all simulation state changes happen on the main thread inside the game loop.

### Request Flow

```
Flask thread                    Main thread
-----------                     -----------
_call("submit_human_move", ...) 
  → put UIRequest on queue
  → block on response.get()
                                process_request()
                                  → pop UIRequest
                                  → call orchestrator.submit_human_move(...)
                                  → put (True, result) on response
response.get() unblocks
returns result to Flask handler
```

### `process_request(env, timeout)`

Called in the main game loop:

```python
while True:
    backend.process_request(env=env, timeout=0.05)
    env.render()
    time.sleep(0.01)
```

Pops one request with a `timeout` second wait. If `new_game` is requested, calls `env.reset()` before the orchestrator call (resets MuJoCo simulation state). Updates `_snapshot` under a lock after each call so `snapshot()` always returns the latest state without blocking.

### UIBackend Protocol

Any object implementing these four methods can replace `QueuedUIBackend` (e.g. for testing):

```python
def snapshot(self) -> GameSnapshot: ...
def new_game(self) -> GameSnapshot: ...
def submit_human_move(self, src, dst, promotion) -> MoveExecutionResult: ...
def let_computer_play_current_turn(self) -> MoveExecutionResult: ...
```
