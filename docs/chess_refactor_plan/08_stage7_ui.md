# Stage 7: UI

## Goal

Provide a human-facing UI alongside the MuJoCo simulator.

The UI shows board state, lets the human select source and destination squares, displays game status and errors, and provides "Let computer play for me".

## Recommended Implementation

Use a lightweight local web UI:

- Backend: Flask or FastAPI.
- Frontend: simple HTML/CSS/JavaScript.
- API backed by `GameOrchestrator`.

This keeps the UI independent of MuJoCo rendering and avoids coupling chess board display to the simulator viewer.

## Backend API

Create `src/ui/app.py`.

Required endpoints:

```text
GET  /                 -> board UI
GET  /api/snapshot     -> GameSnapshot JSON
POST /api/new-game     -> reset game
POST /api/move         -> {src, dst, promotion?}
POST /api/let-computer-play
```

Optional:

```text
GET /api/events        -> server-sent events for move progress
```

If no event stream is implemented, the frontend can poll `/api/snapshot` every 500ms while busy.

## Frontend Behavior

Board display:

- 8x8 grid.
- Show piece symbols from snapshot.
- Highlight selected source square.
- Highlight legal destinations for selected source if provided by API.
- Highlight last move.
- Render from White's perspective by default.

Move input:

1. User clicks source square.
2. User clicks destination square.
3. If move is promotion candidate, show promotion choice: queen, rook, bishop, knight.
4. POST `/api/move`.
5. Disable board while `is_busy=true`.

Status panel:

- Side to move.
- Check/checkmate/stalemate/draw status.
- Last move.
- Error text.
- Move history.

Controls:

- New game.
- Let computer play for me.
- Optional: flip board.
- Optional: resync physical board to logical board recovery command.

Do not add explanatory marketing text. The first screen should be the playable board and controls.

## UI State Contract

The frontend must not infer rules by itself. It may ask the backend for legal moves or submit a proposed move.

Recommended snapshot board shape:

```json
{
  "board": {
    "e2": "P",
    "e4": null,
    "e7": "p"
  },
  "turn": "white",
  "legal_moves": ["e2e4", "g1f3"],
  "is_busy": false,
  "error": null
}
```

Use uppercase piece letters for White and lowercase for Black, matching FEN convention.

## Running

Add script:

```text
scripts/run_chess_ui.py
```

Command:

```bash
python scripts/run_chess_ui.py --host 127.0.0.1 --port 8000
```

If MuJoCo viewer is desired simultaneously, add a `--visualize` flag that starts the env with `render_mode="human"` and passes render callbacks into `ScriptedController`.

## Tests

Backend tests:

- `GET /api/snapshot` returns initial board.
- Illegal move returns 400 or accepted false.
- Legal move returns accepted true in mocked physical mode.
- Let-computer-play returns a legal move.
- Busy state is represented.

Frontend smoke tests if Playwright is available:

- Page loads.
- Board has 64 squares.
- Clicking `e2` then `e4` sends move.
- Promotion dialog appears for a promotion scenario.

## Validation

Run:

```bash
pytest tests/ui -v
python scripts/run_chess_ui.py --host 127.0.0.1 --port 8000
```

Manual validation:

- Board renders correct starting position.
- Human can click a legal move.
- Illegal move produces visible error.
- "Let computer play for me" performs a legal move.
- MuJoCo simulator visibly performs physical moves when not mocked.
