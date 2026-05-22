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

## Server Architecture

The UI server runs as a background daemon thread in the same process as MuJoCo. All MuJoCo env calls stay in the main thread. The Flask thread must never call `env.render()`, `env.step()`, or any physical layer method directly.

Startup sequence in `scripts/run_chess_ui.py`:

```python
# 1. Create env in main thread
env = gym.make("ChessFetchTask-v0", render_mode="human" if args.visualize else None)

# 2. Create orchestrator in main thread
orchestrator = GameOrchestrator(...)
orchestrator.new_game()

# 3. Start Flask in background daemon thread
server = threading.Thread(target=run_flask_app, args=(orchestrator,), daemon=True)
server.start()

# 4. Main thread: run game loop — dequeues requests, runs moves, calls env.render()
game_loop(orchestrator, env, args.delay)
```

The `game_loop` blocks on the request queue and calls `env.render()` after each simulation step. Flask runs fully independently in the background.

If `--visualize` is not provided, `game_loop` still runs but without rendering. This allows headless operation (e.g., SSH) where only the UI browser is the interface.

## Backend API

Create `src/ui/app.py`.

Required endpoints:

```text
GET  /                 -> board UI
GET  /api/snapshot     -> GameSnapshot JSON
POST /api/new-game     -> reset game
POST /api/move         -> {src, dst, promotion?}
POST /api/promote      -> {piece, square}   -- used after awaiting_promotion=True
POST /api/let-computer-play
POST /api/undo         -> (optional) take back last move via resync
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
3. POST `/api/move` (no promotion field yet).
4. Disable board while `is_busy=true`.
5. Server responds with `awaiting_promotion=true` if pawn reached back rank.
6. Show promotion dialog: queen, rook, bishop, knight.
7. POST `/api/promote` with the chosen piece type.
8. Board re-enables when `is_busy=false`.

**Promotion dialog timing**: The arm has already moved the pawn to the back rank and retracted to home before `awaiting_promotion=true` is returned. The dialog can wait as long as the human needs. There is no arm holding state during promotion dialog.

**Board label display**: Show rank numbers (1–8) and file letters (a–h) as labels along the board edges. Orientation: White's rank 1 at the bottom of the board from White's perspective. File 'a' at the right from White's perspective (consistent with physical mapping where White's a-file is near the arm's right side).

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
