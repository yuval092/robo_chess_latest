# Runtime Usage

## Starting the Game

```bash
python main.py
```

Opens the MuJoCo viewer and starts a Flask server at `http://127.0.0.1:8000`.

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--host` | `127.0.0.1` | Flask bind address |
| `--port` | `8000` | Flask bind port |
| `--no-visualize` | — | Run without the MuJoCo viewer window |
| `--delay SECS` | `0.0` | Per-step render delay (slows down playback for debugging) |
| `--debug` | — | Enable verbose per-step environment logs |
| `--transit-model PATH` | — | Override the deployed transit SAC model |
| `--descend-model PATH` | — | Override the deployed descend SAC model |
| `--ascend-model PATH` | — | Override the deployed ascend SAC model |

### Model Paths

By default, `main.py` reads model paths from `configs/deployed_models.yaml`:

```yaml
transit: models/transit/model.zip
descend: models/descend/model.zip
ascend:  models/ascend/model.zip
```

Override any path with `--transit-model`, `--descend-model`, or `--ascend-model`.

## Game Configuration

Chess game settings live in `configs/chess.yaml`:

- **`game.human_color`** — `"white"`, `"black"`, or `"both"`. Controls which side the human controls.
- **`game.auto_computer_reply`** — if `true`, the engine automatically plays after each human move.
- **`engine.stockfish_path`** — path to the Stockfish executable on `PATH`. Set to `null` to disable the engine.
- **`engine.skill_level`** — UCI Skill Level (0 = weakest, 20 = strongest).
- **`engine.think_time_s`** — seconds Stockfish may search per move.

## Web UI

Open `http://127.0.0.1:8000` in any browser. The UI allows:

- Clicking a piece then a destination square to submit a human move.
- Clicking "Let Computer Play" to trigger the engine.
- Clicking "New Game" to reset board and arm.

## Running Tests

```bash
pytest tests/                              # all unit and integration tests
pytest tests/ -m "not slow"               # skip slow integration tests
RUN_EXHAUSTIVE_PHYSICAL_MOVES=1 pytest tests/integration/test_all_square_moves.py
```

The exhaustive all-square test sweeps all 64×63 source→destination combinations and is opt-in via the environment variable.
