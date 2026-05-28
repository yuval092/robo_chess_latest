# RoboChess Overview

RoboChess is a simulation-only robotic chess system. A Fetch robot arm in MuJoCo
plays chess on a physical 8x8 board by picking up and placing simulated chess
pieces. Game legality is handled by `python-chess`, optional computer moves are
chosen by a Stockfish-compatible UCI engine, and the browser UI communicates with
the runtime through a Flask API.

The project has no hardware integration layer. The "physical" layer means
physical execution inside MuJoCo: free-joint chess piece bodies, collision cubes,
arm movement, grasp/place checks, and post-move reconciliation.

## Main Runtime Flow

```text
Browser UI
  -> Flask REST API
  -> QueuedUIBackend
  -> GameOrchestrator
  -> ChessService + MovePlanner
  -> PhysicalPlanExecutor
  -> MovementExecutor / PieceTeleporter
  -> ModelEmbeddedController
  -> ChessTaskEnv / MuJoCo
```

The important threading rule is that MuJoCo is only touched from the main Python
thread. Flask handles HTTP requests in a server thread, while `QueuedUIBackend`
serializes requests onto the main simulation loop.

## Component Map

| Area | Files | Responsibility |
|---|---|---|
| Runtime entry point | `main.py` | Builds the env, controller, game stack, Flask app, and main loop |
| MuJoCo env | `src/chess_env/` | Fetch-derived chess task environment, observations, resets, rewards, scripted grasp/place |
| Arm controller | `src/chess_env/model_controller.py` | Runs SAC specialists for transit/descend/ascend and scripted grasp/place sequences |
| Chess rules | `src/chess_game/chess_service.py` | `python-chess` wrapper plus a minimal UCI engine wrapper |
| Game orchestration | `src/chess_game/game_orchestrator.py` | Validates turns, plans moves, commits board state after physical success |
| Move planning | `src/chess_game/move_planner.py` | Converts legal chess moves into physical commands |
| Board mapping | `src/chess_game/board_mapper.py` | Converts chess squares to world coordinates |
| Physical execution | `src/physical/` | Piece registry, occupancy, teleports, arm command dispatch |
| Web UI | `src/ui/` | Flask routes, queued backend, static browser app |
| Training | `training/` | SAC specialist fine-tuning CLI and env wrappers |
| Config | `configs/` | Geometry, task constants, physics, model paths, training params |
| Assets | `chess_env/` | Static MuJoCo XML, textures, and STL meshes |
| Tests | `tests/` | Unit, physics, UI, and integration coverage |

## Entry Points

| Command | Purpose |
|---|---|
| `pip install -e .` | Install package and `robo-chess-train` CLI |
| `python main.py` | Start the web UI and MuJoCo viewer |
| `python main.py --no-visualize` | Start the web UI headless |
| `robo-chess-train train --stage transit` | Train one movement specialist |
| `pytest` | Run the default test suite |
| `RUN_EXHAUSTIVE_PHYSICAL_MOVES=1 pytest tests/integration/test_all_square_moves.py` | Run the opt-in full board move sweep |

## Documentation Index

| Document | Contents |
|---|---|
| [01_runtime_usage.md](01_runtime_usage.md) | Install, run, CLI flags, normal workflows |
| [02_architecture.md](02_architecture.md) | End-to-end architecture and state boundaries |
| [03_mujoco_environment.md](03_mujoco_environment.md) | MuJoCo env, observations, resets, rewards |
| [04_arm_control.md](04_arm_control.md) | Model controller, movement stages, grasp/place |
| [05_chess_logic.md](05_chess_logic.md) | Chess service, UCI engine, orchestration |
| [06_physical_layer.md](06_physical_layer.md) | Physical commands, occupancy, teleports |
| [07_scene_assets.md](07_scene_assets.md) | XML/STL ownership and scene invariants |
| [08_web_ui.md](08_web_ui.md) | Flask API, queued backend, browser behavior |
| [09_training.md](09_training.md) | SAC training pipeline and deployment of checkpoints |
| [10_configuration.md](10_configuration.md) | YAML config reference |
| [11_testing.md](11_testing.md) | Test structure and verification commands |
| [12_development_notes.md](12_development_notes.md) | Implementation invariants and common extension points |

