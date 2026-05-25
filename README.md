# RoboChess

A MuJoCo-based robot arm that plays physical chess. A Fetch robot arm picks and
places chess pieces on a real board, driven by a scripted waypoint controller or
by trained SAC (Soft Actor-Critic) specialist RL models.

---

## How to run

### 1. Install

```bash
pip install -e .
```

### 2. Play chess against the computer

```bash
robo-chess-ui
```

This starts a Flask web server. Open **http://127.0.0.1:8000** in your browser.
You play as White; the computer (Stockfish) plays as Black. When you make a move
in the browser, the robot arm physically executes it in the MuJoCo simulation
running in the background.

To also open the MuJoCo window and watch the arm move in real time:

```bash
robo-chess-ui --visualize
```

To use the scripted controller instead of RL models (no trained checkpoints needed):

```bash
robo-chess-ui --use-scripted-controller
```

### 3. Watch the arm move without playing chess

`robo-chess-visualize` is a **debugging tool** for arm movement — not a chess
game. It runs the scripted controller in a loop, moving the arm to random board
positions to check mechanics. The board starts empty by default.

```bash
# Watch the arm execute transit movements (empty board)
robo-chess-visualize

# Show all pieces in starting positions, then watch the arm move
robo-chess-visualize --setup

# Run a full pick-and-place cycle between two squares
robo-chess-visualize --scenario full_move --src-xy "0.88 0.2641" --dst-xy "1.00 0.40"
```

### 4. Regenerate scene assets

Only needed if you change piece geometry or board layout:

```bash
robo-chess-generate all
```

---

## Chess engine difficulty

The computer opponent uses **Stockfish 16**, configurable in `configs/chess.yaml`:

```yaml
engine:
  skill_level: 5       # 0 = very weak, 20 = full Stockfish strength
  think_time_s: 0.5    # seconds Stockfish thinks per move
```

| `skill_level` | Approx. ELO | Description |
|---|---|---|
| 0–2 | ~800 | Blunders pieces, makes obvious mistakes |
| 5 | ~1100 | Makes some tactical errors — default |
| 10 | ~1600 | Solid club player, hard to beat casually |
| 15 | ~2200 | Master level |
| 20 | ~3400 | Full Stockfish — effectively unbeatable |

---

## RL Training

The arm's movement is split into three specialist models (transit, descend, ascend)
trained independently with SAC:

```bash
python scripts/train_rl.py --stage transit
python scripts/train_rl.py --stage descend
python scripts/train_rl.py --stage ascend
```

After training, update `configs/deployed_models.yaml` with the checkpoint paths.
The scripted controller (`--use-scripted-controller`) is always available as a
fallback that requires no trained models.

See `scripts/README.md` for the full list of evaluation and diagnostic scripts.

---

## Project structure

```
robo_chess_latest/
├── src/
│   ├── chess_env/       # MuJoCo environment and RL infrastructure
│   ├── chess_game/      # Chess logic — game rules, Stockfish, move planning
│   ├── physical/        # Physical execution pipeline (plan → arm movement)
│   ├── ui/              # Flask web UI backend
│   ├── cli/             # Installed entry points (robo-chess-ui, etc.)
│   └── utils/           # Config loading, logging, shared argparse helpers
├── training/            # SAC training infrastructure (not imported by src/)
├── scripts/             # Evaluation, validation, and analysis tools
├── chess_env/           # MuJoCo assets — XML scenes and STL meshes
├── configs/             # YAML config files (env, chess, training, deployed models)
├── models/pretrained/   # Base pretrained FetchPickAndPlace SAC weights
└── tests/               # pytest test suite
```
