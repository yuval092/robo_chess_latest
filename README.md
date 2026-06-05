# RoboChess

RoboChess is a MuJoCo simulation of a robotic chess player. A simulated Fetch
arm picks up and places chess pieces on an 8x8 board while game state is managed
with `python-chess` and an optional Stockfish UCI engine.

The project is simulation-only. There is no real hardware integration.

## Install

```bash
pip install -e .
```

Install Stockfish separately and make sure the configured executable is on
`PATH` if you want computer moves.

## Play

Start the web UI and live MuJoCo simulation:

```bash
robo-chess-play
```

Open `http://127.0.0.1:9999` in your browser. The MuJoCo viewer is always shown.
Model paths are read from `configs/deployed_models.yaml`.

## Full Board Sweep

The exhaustive 64x63 physical move sweep is test-only and is skipped unless
explicitly requested:

```bash
RUN_EXHAUSTIVE_PHYSICAL_MOVES=1 pytest tests/integration/test_all_square_moves.py -s
```

The sweep uses `black_rook_a` by default and is not part of normal runtime.

## Training

Only training is installed as a console script:

```bash
robo-chess-train --stage transit
robo-chess-train --stage descend
robo-chess-train --stage ascend
```

After training, update `configs/deployed_models.yaml`.

## Scene Assets

`chess_env/assets/pick_and_place.xml` and the STL files under
`chess_env/stls/` are checked-in source-of-truth assets. Runtime startup does
not regenerate XML fragments or meshes.

## Tests

```bash
pytest
```

Some tests require MuJoCo. Tests that use Stockfish require the `stockfish`
executable to be available on `PATH`.

The exhaustive physical move sweep remains opt-in for pytest:

```bash
RUN_EXHAUSTIVE_PHYSICAL_MOVES=1 pytest tests/integration/test_all_square_moves.py -s
```

## Project Structure

```text
robo_chess_latest/
├── main.py             # Play-mode runtime entry point
├── src/
│   ├── chess_env/      # MuJoCo environment and model controller infrastructure
│   ├── chess_game/     # Chess rules, Stockfish wrapper, move planning
│   ├── physical/       # Physical execution pipeline and piece teleports
│   ├── ui/             # Flask app, queued backend, browser assets
│   └── utils/          # Config loading, validation, shared helpers
├── training/           # SAC training pipeline and training CLI
├── chess_env/          # MuJoCo XML, textures, STL assets
├── configs/            # YAML configuration files
├── docs/               # Detailed project documentation
├── models/             # Deployed and pretrained model checkpoints
└── tests/              # pytest suite
```
