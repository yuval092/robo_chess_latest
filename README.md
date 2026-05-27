# RoboChess

RoboChess is a MuJoCo simulation of a robotic chess player. A simulated Fetch
arm picks up and places chess pieces on an 8x8 board, while the game state is
managed with `python-chess` and an optional Stockfish UCI engine.

The project is simulation-only. There is no real hardware integration.

The arm can run with trained SAC specialist models for the movement stages
(`transit`, `descend`, `ascend`) or with the deterministic scripted controller.
Grasp and place are scripted in both modes.

---

## Install

```bash
pip install -e .
```

The main runtime dependencies are declared in `pyproject.toml`: MuJoCo,
Gymnasium Robotics, Stable-Baselines3, Flask, NumPy, PyYAML, and python-chess.
Install Stockfish separately and make sure the configured executable is on
`PATH` if you want computer moves.

---

## Play

Start the web UI and live MuJoCo simulation:

```bash
robo-chess-play
```

Open `http://127.0.0.1:8000` in your browser. By default, the MuJoCo viewer is
shown. For headless operation:

```bash
robo-chess-play --no-visualize
```

To use the scripted controller instead of configured RL models:

```bash
robo-chess-play --use-scripted-controller
```

Model paths are read from `configs/deployed_models.yaml` and can be overridden:

```bash
robo-chess-play --transit-model PATH --descend-model PATH --ascend-model PATH
```

---

## Configuration

Configuration lives in `configs/`:

| File | Purpose |
|---|---|
| `env.yaml` | Z levels, controller tuning, reward weights, grasp thresholds |
| `chess.yaml` | Board geometry, piece/zones layout, game settings, Stockfish config |
| `physics.yaml` | MuJoCo setup, action scale, vertical gripper quaternion |
| `training.yaml` | SAC training hyperparameters and schedules |
| `deployed_models.yaml` | Default model checkpoint paths |

Stockfish config is under `configs/chess.yaml:engine`:

```yaml
engine:
  stockfish_path: stockfish
  skill_level: 5
  think_time_s: 0.5
```

Set `stockfish_path: null` to disable engine startup.

---

## Scene Assets

The MuJoCo scene contains generated sections for board squares, pieces, and
graveyard/reserve zones. Startup regenerates XML fragments when needed. STL
meshes are regenerated manually.

```bash
# Print generated XML fragments
robo-chess-generate board
robo-chess-generate pieces
robo-chess-generate zones

# Write generated XML fragments into chess_env/assets/pick_and_place.xml
robo-chess-generate board --write
robo-chess-generate pieces --write
robo-chess-generate zones --write

# Regenerate chess STL meshes
robo-chess-generate stls

# Regenerate XML fragments
robo-chess-generate all
```

---

## Evaluation

Installed evaluation commands:

```bash
# Per-stage waypoint accuracy
robo-chess-eval-stage --stage all --controller model --episodes 50
robo-chess-eval-stage --stage all --controller scripted --episodes 20

# Physics and scene integrity
robo-chess-eval-physics

# Full piece-move flow
robo-chess-eval-flow --mode simple --src e2 --dst e4 --controller scripted
robo-chess-eval-flow --mode complex --controller model
```

`robo-chess-eval-flow --mode full` runs all 4,032 distinct source/destination
board pairs and can take hours.

---

## Training

Train specialist SAC models with:

```bash
robo-chess-train train --stage transit
robo-chess-train train --stage descend
robo-chess-train train --stage ascend
```

Evaluate a checkpoint:

```bash
robo-chess-train eval --stage ascend --model models/ascend.zip --episodes 50
```

After training, update `configs/deployed_models.yaml`, then validate with
`robo-chess-eval-stage` and `robo-chess-eval-flow`.

---

## Tests

```bash
pytest
```

Some tests require MuJoCo. Tests that use Stockfish require the `stockfish`
executable to be available on `PATH`.

The exhaustive physical move sweep is opt-in:

```bash
RUN_EXHAUSTIVE_PHYSICAL_MOVES=1 pytest tests/integration/test_all_square_moves.py
```

---

## Project Structure

```text
robo_chess_latest/
├── src/
│   ├── chess_env/       # MuJoCo environment and RL/model controller infrastructure
│   ├── chess_game/      # Chess rules, Stockfish wrapper, move planning
│   ├── physical/        # Physical execution pipeline and piece teleports
│   ├── ui/              # Flask app, queued backend, browser assets
│   ├── cli/             # Installed runtime/evaluation entry points
│   └── utils/           # Config loading, validation, shared helpers
├── training/            # SAC training pipeline and training CLI
├── chess_env/           # MuJoCo XML, textures, STL assets
├── configs/             # YAML configuration files
├── docs/                # Detailed project documentation
├── models/              # Deployed and pretrained model checkpoints
└── tests/               # pytest suite
```

See `docs/00_overview.md` for the detailed architecture and `docs/10_scripts.md`
for the full CLI reference.
