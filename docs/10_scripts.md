# CLI Reference

The former `scripts/` directory has been removed. All evaluation logic is now
available as proper installed CLI entry points. Run `pip install -e .` once to
register all commands.

---

## Production Commands

### `robo-chess-play`

Start the RoboChess web UI with live MuJoCo simulation. Opens a browser-accessible
chess interface and drives the robot arm through each move. The MuJoCo viewer is
shown by default; pass `--no-visualize` for headless operation.

```bash
robo-chess-play [--host HOST] [--port PORT] [--no-visualize] [--delay DELAY]
                [--transit-model PATH] [--descend-model PATH] [--ascend-model PATH]
```

### `robo-chess-generate`

Regenerate MuJoCo XML scene assets and STL meshes.

```bash
robo-chess-generate {board|pieces|zones|stls|all}
robo-chess-generate board --write
robo-chess-generate pieces --write
robo-chess-generate zones --write
```

`board`, `pieces`, and `zones` print their generated XML by default; pass `--write`
to update `chess_env/assets/pick_and_place.xml`. `all` regenerates the XML
fragments only. `stls` regenerates the chess STL meshes.

---

## Evaluation Commands

All evaluation commands accept `--visualize` to open the MuJoCo viewer while the
command runs, and `--delay SECS` to slow down rendering for closer inspection.

### `robo-chess-eval-stage`

Evaluate waypoint-stage accuracy. Runs N episodes per stage and reports success
rate, crash rate, timeout rate, average/P95 positional error, average step count,
and a per-crash-reason breakdown. A live progress bar shows running OK% during
the run.

```bash
robo-chess-eval-stage --stage STAGE [options]

STAGE choices:
  transit     Evaluate transit stage only
  descend     Evaluate descend stage only
  ascend      Evaluate ascend stage only
  all         Evaluate all three stages (default)
  full_move   Alias for all  (transit + descend + ascend)
  pick        transit + descend
  vertical    descend + ascend

Options:
  --episodes N          Episodes per stage (default: 50)
  --transit-model PATH  Override transit model ZIP
  --descend-model PATH  Override descend model ZIP
  --ascend-model PATH   Override ascend model ZIP
  --drift-limit METRES  Override tube constraint radius
  --visualize           Open the MuJoCo viewer while episodes run
  --delay SECS          Per-step sleep for slow-motion visualization (default: 0.0)
  --debug               Verbose per-step environment logs
```

**Examples:**

```bash
# Evaluate deployed model on transit, 100 episodes
robo-chess-eval-stage --stage transit --episodes 100

# Evaluate a newly trained model
robo-chess-eval-stage --stage ascend --ascend-model checkpoints/ascend_run/model.zip

# Full pipeline (transit + descend + ascend)
robo-chess-eval-stage --stage full_move --episodes 50
```

---

### `robo-chess-eval-physics`

Verify MuJoCo physics assets, scene geometry, and simulation stability without
commanding the arm. Exits 0 if all checks pass, 1 if any fail.

**Checks performed:**
- XML integrity: required sites present, legacy sites absent
- Table geometry: 70×70 cm, 4 legs, surface Z = 0.400 m
- Board visual geometry: 64 square geoms, 8 cm spacing, correct positions
- Zone visual geometry: graveyard and reserve markers (visual-only, world-space)
- Chess piece modeling: 32 active + 64 reserve bodies, joints, damping, geom sizes
- Grasp XML parameters: cube mass, actuator Kp, ctrlrange, GRASP_Z
- Teleport verification: object placed correctly when `hide_object=True`
- Chess piece idle stability: ≤1 mm drift over settle loop steps
- Static stability: object drift < 1 mm over 100 zero-action steps
- Kinematic reachability: all 64 squares reachable within 5 mm at SAFE_Z and GRASP_Z

```bash
robo-chess-eval-physics [options]

Options:
  --settle-steps N      Steps for idle-stability loop (default: 2500 ≈ 5 sim-seconds)
  --skip-geometry       Skip all static geometry checks
  --skip-stability      Skip piece-idle and static stability loops
  --skip-reachability   Skip kinematic reachability (fastest to skip)
  --visualize           Open the MuJoCo viewer during each check
  --debug               Verbose environment logging
```

**Examples:**

```bash
# Full check (takes ~2 minutes)
robo-chess-eval-physics

# Geometry only — no simulation steps
robo-chess-eval-physics --skip-stability --skip-reachability

# Reachability only
robo-chess-eval-physics --skip-geometry --skip-stability
```

---

### `robo-chess-eval-flow`

Evaluate the full arm piece-move flow using **production code** — no mocks, no
patches. Moves one physical chess piece through the complete
transit → descend → grasp → ascend → transit → descend → place → ascend path.

Three modes:

| Mode | What it runs | Duration |
|------|-------------|----------|
| `simple` | One move; optional `--src`/`--dst` (random if unspecified) | Seconds |
| `complex` | Corners, near-corners, edge midpoints, known-hard cells | Minutes |
| `full` | All 4 032 (src, dst) board pairs — brute-force sweep | Hours |

Reports per-stage breakdown, failure reasons, and worst destination cells.
Complex and full modes display a live progress bar with running OK%.

```bash
robo-chess-eval-flow --mode {simple,complex,full} [options]

Options:
  --src CELL            Source cell (simple mode only, e.g. e2). Default: random.
  --dst CELL            Destination cell (simple mode only, e.g. e4). Default: random.
  --piece ID            Physical piece ID to move (default: black_rook_a)
  --episodes N          Repetitions per (src,dst) pair (default: 1; complex default: 3)
  --transit-model PATH
  --descend-model PATH
  --ascend-model PATH
  --drift-limit METRES
  --visualize           Open the MuJoCo viewer while moves run
  --delay SECS          Per-step sleep for slow-motion visualization (default: 0.0)
  --debug               Verbose per-step logs
```

**Examples:**

```bash
# Quick sanity: move e2 → e4
robo-chess-eval-flow --mode simple --src e2 --dst e4

# Random single move with RL models
robo-chess-eval-flow --mode simple

# Targeted hard-cell coverage
robo-chess-eval-flow --mode complex

# Full board sweep (run overnight)
robo-chess-eval-flow --mode full
```

---

## Training Command

### `robo-chess-train`

Train or evaluate a specialist SAC model. Two sub-commands: `train` and `eval`.

#### `robo-chess-train train`

```bash
robo-chess-train train --stage {transit,descend,ascend} [options]

Options:
  --stage STAGE         REQUIRED. Stage to train.
  --envs N              Parallel environments (default: from configs/training.yaml)
  --model PATH          Resume from checkpoint / base model
  --timesteps N         Total training timesteps (default: from config)
  --save-dir PATH       Output directory (default: checkpoints/<stage>_<timestamp>/)
  --fixed-drift         Skip drift curriculum; use final limit immediately
  --debug               Verbose training and environment logs
```

**Examples:**

```bash
robo-chess-train train --stage transit
robo-chess-train train --stage descend --model checkpoints/descend_prev/model.zip
robo-chess-train train --stage ascend --timesteps 100   # quick test run
```

#### `robo-chess-train eval`

Evaluate a trained model on a movement stage. Delegates entirely to
`robo-chess-eval-stage` — same output format and metrics.

```bash
robo-chess-train eval --stage STAGE --model PATH [options]

Options:
  --stage STAGE         REQUIRED. Stage to evaluate.
  --model PATH          REQUIRED. Path to the model ZIP.
  --episodes N          Episodes to run (default: 50)
  --drift-limit METRES  Override drift tolerance
  --debug               Verbose debug logs
```

**Examples:**

```bash
robo-chess-train eval --stage transit --model checkpoints/transit_run/model.zip
robo-chess-train eval --stage ascend  --model models/ascend.zip --episodes 10
```

---

## Typical Workflows

### After training a new model

```bash
# 1. Evaluate stage accuracy
robo-chess-train eval --stage transit --model <path> --episodes 100

# 2. Run targeted flow evaluation
robo-chess-eval-flow --mode complex --transit-model <path>

# 3. Update configs/deployed_models.yaml, then run a physics sanity check
robo-chess-eval-physics --skip-stability
```

### Pre-deployment checklist

```bash
robo-chess-eval-physics                               # full physics check
robo-chess-eval-stage --stage all                    # all-stage accuracy
robo-chess-eval-flow --mode complex                   # hard-cell coverage
```

### Debugging a failing destination cell

```bash
# Investigate transit failures on a5 with verbose logs
robo-chess-eval-flow --mode simple --dst a5 --episodes 10 --debug
```
