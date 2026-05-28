# RoboChess Overview

RoboChess is a MuJoCo simulation of a Fetch robot arm that plays chess on an
8x8 board. A browser UI sends moves to Flask, `python-chess` validates the game
state, and the physical layer executes legal moves in the simulated scene.

## Main Layers

| Layer | Package / File | Responsibility |
|---|---|---|
| Browser UI | `src/ui/static/app.js`, `src/ui/templates/index.html` | Render board and call REST API |
| Flask API | `src/ui/app.py` | HTTP routes and JSON responses |
| Queue bridge | `src/ui/queued_backend.py` | Serialize UI requests onto the MuJoCo thread |
| Game orchestration | `src/chess_game/game_orchestrator.py` | Coordinate chess rules and physical execution |
| Physical execution | `src/physical/` | Move execution, occupancy, teleports |
| Controller | `src/chess_env/model_controller.py` | SAC stage inference plus scripted grasp/place |
| MuJoCo env | `src/chess_env/` | Fetch-based chess task environment |
| Runtime | `main.py` | Play mode |
| Training | `training/` | SAC training pipeline and `robo-chess-train train` |
| Tests | `tests/` | Static asset, physics, game, UI, and integration coverage |

## Entry Points

| Command | Description |
|---|---|
| `python main.py` | Start the web UI and MuJoCo simulation |
| `python main.py --no-visualize` | Start play mode headless |
| `robo-chess-train train` | Train one SAC specialist model |

Scene XML and STL assets are static checked-in source files. Diagnostics that
used to be installed evaluation commands now live in pytest.
