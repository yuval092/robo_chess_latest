# Runtime Usage

## Starting the Game

```bash
robo-chess-play
```

Opens the MuJoCo viewer and starts a Flask server at `http://127.0.0.1:9999`.
Play mode does not accept CLI options; runtime settings come from configuration files.

### Model Paths

By default, `main.py` reads model paths from `configs/deployed_models.yaml`:

```yaml
transit: models/transit/model.zip
descend: models/descend/model.zip
ascend:  models/ascend/model.zip
```

Update `configs/deployed_models.yaml` to change the deployed model files.

## Game Configuration

Chess game settings live in `configs/chess.yaml`:

- **`game.human_color`** — `"white"`, `"black"`, or `"both"`. Controls which side the human controls.
- **`game.auto_computer_reply`** — if `true`, the engine automatically plays after each human move.
- **`engine.stockfish_path`** — path to the Stockfish executable on `PATH`. Set to `null` to disable the engine.
- **`engine.skill_level`** — UCI Skill Level (0 = weakest, 20 = strongest).
- **`engine.think_time_s`** — seconds Stockfish may search per move.

## Web UI

Open `http://127.0.0.1:9999` in any browser. The UI allows:

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
