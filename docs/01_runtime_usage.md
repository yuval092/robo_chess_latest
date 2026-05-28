# Runtime Usage

## Installation

Install the project in editable mode from the repository root:

```bash
pip install -e .
```

The Python package requires MuJoCo, Gymnasium Robotics, Flask, `python-chess`,
Stable-Baselines3, NumPy, and PyYAML. These are declared in `pyproject.toml`;
`requirements.txt` contains the same practical runtime dependency set.

Computer moves require a Stockfish-compatible executable. By default the config
uses:

```yaml
engine:
  stockfish_path: stockfish
```

Set that path to a full executable path or put `stockfish` on `PATH`. Set it to
`null` or construct `ChessService(engine_cfg=None)` in tests if no engine should
be started.

## Start Play Mode

```bash
python main.py
```

Default behavior:

- Creates `ChessFetchTask-v0`.
- Shows the MuJoCo viewer.
- Shows all chess pieces in the scene.
- Hides the legacy Fetch `object0`.
- Loads model paths from `configs/deployed_models.yaml`.
- Starts Flask at `http://127.0.0.1:8000`.

Headless mode:

```bash
python main.py --no-visualize
```

Useful flags:

| Flag | Default | Meaning |
|---|---:|---|
| `--host` | `127.0.0.1` | Flask bind host |
| `--port` | `8000` | Flask bind port |
| `--no-visualize` | off | Disable MuJoCo viewer |
| `--delay` | `0.0` | Delay after render calls during controller inference |
| `--debug` | off | Enable verbose environment logging |
| `--transit-model PATH` | config | Override transit checkpoint |
| `--descend-model PATH` | config | Override descend checkpoint |
| `--ascend-model PATH` | config | Override ascend checkpoint |

Example:

```bash
python main.py --no-visualize --port 8010 --transit-model checkpoints/transit/final_transit.zip
```

## Playing From the Browser

Open the runtime URL in a browser. The UI renders the legal board state returned
by `/api/snapshot`.

Normal interaction:

1. Click a source square.
2. Legal destination hints appear.
3. Click a destination square.
4. For promotion, choose queen, rook, bishop, or knight in the dialog.
5. The HTTP request blocks until the arm has completed or failed the physical
   move.

If `configs/chess.yaml:game.auto_computer_reply` is true and the human is
configured as one side, the orchestrator can immediately execute the engine reply
after a successful human move.

## Training

The installed console script is:

```bash
robo-chess-train train --stage {transit,descend,ascend}
```

Common options:

| Option | Meaning |
|---|---|
| `--envs N` | Override parallel env count |
| `--model PATH` | Resume or fine-tune from a checkpoint |
| `--timesteps N` | Override total training timesteps |
| `--save-dir PATH` | Output checkpoint directory |
| `--fixed-drift` | Use final drift limit immediately |
| `--debug` | Verbose training/env logs |

Example:

```bash
robo-chess-train train --stage descend --model checkpoints/descend_prev/best_model_descend.zip
```

After training, either update `configs/deployed_models.yaml` or pass model
overrides to `python main.py`.

## Verification Commands

Run the default suite:

```bash
pytest
```

Run focused static/physics checks:

```bash
pytest tests/chess_env tests/physical tests/test_config_schema.py
```

Run the long all-square physical sweep:

```bash
RUN_EXHAUSTIVE_PHYSICAL_MOVES=1 pytest tests/integration/test_all_square_moves.py
```

