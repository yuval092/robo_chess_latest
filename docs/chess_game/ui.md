# UI

The chess UI is a Flask app backed by `GameOrchestrator`.

## Run

```bash
python scripts/run_chess_ui.py --host 127.0.0.1 --port 8000
```

Add `--visualize` to show the MuJoCo renderer while moves execute.

## Endpoints

- `GET /api/snapshot`: current FEN, turn, board, physical piece ids, legal moves, status, busy state, and error.
- `POST /api/new-game`: reset logical game and piece tracker.
- `POST /api/move`: submit `{ "src": "e2", "dst": "e4", "promotion": "q" }`.
- `POST /api/let-computer-play`: ask the deterministic move selector to play the current side.

Undo and deferred promotion continuation endpoints are present but intentionally return `501` until those workflows are implemented.
