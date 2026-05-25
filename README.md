# RoboChess

A MuJoCo-based robot arm that plays physical chess. A Fetch robot arm picks and places chess pieces on a real board geometry, driven either by scripted waypoint controllers or by SAC (Soft Actor-Critic) specialist RL models trained per movement stage.

## What it does

The system integrates two layers:

- **Physical layer** — A Gymnasium/MuJoCo environment (`ChessFetchTask-v0`) simulates the Fetch arm performing piece pick-and-place. Movement is split into three RL stages: *transit* (horizontal travel at safe height), *descend* (precision lowering over the target square), and *ascend* (raising the piece after grasp).
- **Game layer** — A `python-chess`-backed game engine validates moves, drives the computer opponent (via Stockfish or random play), tracks piece positions, and orchestrates the physical execution pipeline.

A Flask web UI ties them together: the human moves pieces via the browser, the computer responds, and the arm executes each move physically.

---

## Project Structure

```
robo_chess_latest/
├── src/                        # Production source code (installable package)
│   ├── chess_env/              # MuJoCo environment and RL infrastructure
│   │   ├── task.py             # ChessTaskEnv — observations, rewards, reset
│   │   ├── task_execution.py   # Scripted grasp/place pipeline (GraspPlaceMixin)
│   │   ├── task_runtime.py     # Runtime helpers (TaskRuntimeMixin)
│   │   ├── task_state.py       # State management (TaskStateMixin)
│   │   ├── simulation.py       # ChessSimulationEnv — MuJoCo setup & scene loading
│   │   ├── controller.py       # ScriptedController — waypoint-based movement
│   │   ├── model_controller.py # ModelEmbeddedController — RL model inference
│   │   ├── model_registry.py   # ModelRegistry — load/cache SAC models per stage
│   │   ├── waypoints.py        # SAFE_Z, HOVER_Z, waypoint helpers
│   │   ├── transfer_obs.py     # 25-D transfer observation space definition
│   │   └── environment_generation.py  # Procedural XML/STL scene generation
│   ├── chess_game/             # Chess logic (no MuJoCo dependency)
│   │   ├── chess_service.py    # Chess engine wrapper (python-chess + Stockfish)
│   │   ├── game_orchestrator.py # Top-level game coordinator
│   │   ├── move_planner.py     # Physical command planning from chess moves
│   │   └── board_mapper.py     # Chess square ↔ XY coordinate mapping
│   ├── physical/               # Physical execution pipeline
│   │   ├── plan_executor.py    # PhysicalPlanExecutor — executes arm command plans
│   │   ├── movement_executor.py # MovementExecutor — coordinates full pick-and-place
│   │   ├── piece_teleport.py   # PieceTeleporter — MuJoCo body position control
│   │   ├── piece_registry.py   # PieceRegistry — piece ID ↔ MuJoCo body mapping
│   │   ├── occupancy.py        # PhysicalOccupancy — board square occupancy tracking
│   │   └── noop_executor.py    # NoOpPhysicalExecutor — headless stub for testing
│   ├── ui/
│   │   └── queued_backend.py   # QueuedUIBackend — thread-safe UI/game bridge
│   ├── cli/                    # Installable CLI entry points
│   │   ├── run_chess_ui.py     # `robo-chess-ui` — start the Flask web UI
│   │   ├── visualize.py        # `robo-chess-visualize` — open MuJoCo viewer
│   │   └── generate_scene.py   # `robo-chess-generate` — regenerate scene assets
│   └── utils/
│       ├── io.py               # load_config(), setup_logger()
│       └── args.py             # Shared argparse helpers for eval scripts
│
├── training/                   # RL training infrastructure (not imported by src/)
│   ├── trainer.py              # SACTrainer — SB3 SAC training loop with curriculum
│   ├── callbacks.py            # Custom SB3 callbacks (drift curriculum, checkpointing)
│   └── envs/                   # Per-stage Gymnasium wrappers for training
│       ├── transit_env.py
│       ├── descend_env.py
│       └── ascend_env.py
│
├── scripts/                    # Evaluation, validation, training, and analysis tools
│   ├── eval_stages.py          # Per-stage accuracy (transit / descend / ascend)
│   ├── eval_sequence.py        # Full scenario chain (pick / full_move / vertical)
│   ├── eval_stress.py          # Corner + grid position stress test
│   ├── eval_targeted.py        # Known-difficult squares targeted test
│   ├── eval_all_cells_rl.py    # All-64-squares RL coverage test
│   ├── eval_all_square_moves.py # Exhaustive board reachability sweep
│   ├── eval_chess_piece_move.py # Single piece move evaluation
│   ├── eval_chess_game_flow.py # Full game flow (logical + physical layers)
│   ├── eval_chess_reachability.py # All 64 squares reachability check
│   ├── eval_draw_conditions.py # Chess draw condition detection
│   ├── eval_special_moves.py   # Castling, en passant, promotion
│   ├── eval_game_logic.py      # Game logic sequences (no MuJoCo)
│   ├── eval_grasp_physics.py   # Grasp physics verification
│   ├── train_rl.py             # Launch SAC training for one stage
│   ├── validate_config.py      # Validate all config keys and model paths
│   ├── validate_deployed_models.py  # Check deployed model files exist
│   ├── analyze_debug_log.py    # Parse debug_one_move JSONL logs → Markdown report
│   ├── verify_physics.py       # MuJoCo physics asset integrity check
│   └── diagnostics/            # Debug and developer tools (not CI-blocking)
│
├── chess_env/                  # MuJoCo asset directory (XML scenes + STL meshes)
│   ├── assets/
│   │   ├── pick_and_place.xml  # Main scene (board, pieces, robot)
│   │   ├── robot.xml           # Fetch robot MJCF
│   │   └── shared.xml          # Shared actuators and sensors
│   └── stls/chess/             # Procedurally generated STL meshes per piece type
│
├── configs/                    # YAML configuration files
│   ├── env.yaml                # Environment constants (heights, tolerances, thresholds)
│   ├── chess.yaml              # Board geometry, square mapping, game settings
│   ├── training.yaml           # SAC hyperparameters, curriculum schedule, stage config
│   ├── deployed_models.yaml    # Paths to trained model checkpoints for production
│   └── physics.yaml            # MuJoCo physics parameters
│
├── models/
│   └── pretrained/             # Base pretrained weights (FetchPickAndPlace-v4 SAC)
│
├── checkpoints/                # Training checkpoints (git-ignored)
├── logs/                       # Training and debug logs (git-ignored)
└── tests/                      # pytest test suite
```

---

## Quickstart

```bash
pip install -e .
```

**Run the chess UI:**
```bash
robo-chess-ui
# or: python -m src.cli.run_chess_ui
```

**Open the MuJoCo visualizer:**
```bash
robo-chess-visualize
robo-chess-visualize --setup   # teleport pieces to starting positions first
```

**Regenerate scene assets (XML + STLs):**
```bash
robo-chess-generate all
```

---

## RL Training

Each movement stage has its own specialist SAC model trained independently:

```bash
python scripts/train_rl.py --stage transit
python scripts/train_rl.py --stage descend
python scripts/train_rl.py --stage ascend
```

Update `configs/deployed_models.yaml` with the checkpoint paths after training, then run:

```bash
python scripts/validate_deployed_models.py
```

---

## Evaluation

```bash
# Per-stage accuracy (uses deployed RL models by default)
python scripts/eval_stages.py --stages transit,descend,ascend --n-episodes 50

# Scripted-only fallback (no trained models required)
python scripts/eval_stages.py --stages transit --use-scripted-only

# Full pick-and-place sequence
python scripts/eval_sequence.py --chain full_move --n-episodes 20

# All 64 squares RL coverage
python scripts/eval_all_cells_rl.py --mode key

# Chess logic (no MuJoCo needed)
python scripts/eval_game_logic.py --verbose
python scripts/eval_draw_conditions.py --verbose
python scripts/eval_special_moves.py --verbose
```

---

## Architecture Notes

**Strict layering**: `src/` never imports from `training/`. The training wrappers import from `src/` (to reuse the environment), but not the other way around.

**Three-stage RL**: Each specialist model handles one stage (transit/descend/ascend). The `ModelEmbeddedController` orchestrates them, falling back to `ScriptedController` for any stage without a loaded model.

**Transfer observation**: A 25-dimensional observation space (`transfer_obs.py`) is shared between training wrappers and inference time — enabling direct model deployment without wrapper overhead.

**Scripted fallback**: `ScriptedController` provides a deterministic waypoint-based baseline. It is production-ready and used when RL models are unavailable.
