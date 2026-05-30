# RoboChess Overview

RoboChess is a simulation-only robotic chess system. A Fetch robot arm in MuJoCo plays chess on a physical 8×8 board by picking up and placing simulated chess pieces. Game legality is enforced by `python-chess`, optional computer moves are chosen by a Stockfish-compatible UCI engine, and a browser UI communicates with the runtime through a Flask REST API.

There is no hardware integration layer. "Physical execution" means execution inside MuJoCo: free-joint chess piece bodies, collision geometry, arm movement, scripted grasp/place pipelines, and post-move reconciliation.

## Runtime Flow

```
Browser UI
  └─> Flask REST API          (src/ui/app.py)
        └─> QueuedUIBackend   (src/ui/queued_backend.py)   serialises onto main thread
              └─> GameOrchestrator (src/chess_game/game_orchestrator.py)
                    ├─> ChessService + MovePlanner          (src/chess_game/)
                    └─> PhysicalPlanExecutor                (src/physical/plan_executor.py)
                          ├─> MovementExecutor              (src/physical/movement_executor.py)
                          │     └─> ModelEmbeddedController (src/chess_env/model_controller.py)
                          │           └─> ChessProductionEnv / MuJoCo
                          └─> PieceTeleporter               (src/physical/piece_teleport.py)
```

**Threading rule**: MuJoCo is only touched from the main Python thread. Flask handles HTTP on a daemon thread. `QueuedUIBackend.process_one()` drains the request queue on the main thread inside the game loop in `main.py`.

## Component Map

| Area | File(s) | Responsibility |
|---|---|---|
| Entry point | `main.py` | Builds env, controller, game stack, Flask app, and main loop |
| MuJoCo env | `src/chess_env/` | Fetch-derived chess task environment |
| Arm controller | `src/chess_env/model_controller.py` | SAC inference + scripted grasp/place sequencing |
| Chess rules | `src/chess_game/chess_service.py` | python-chess board + Stockfish UCI wrapper |
| Move planning | `src/chess_game/move_planner.py` | Translates chess moves to physical command lists |
| Orchestration | `src/chess_game/game_orchestrator.py` | Coordinates logical and physical move execution |
| Physical exec | `src/physical/plan_executor.py` | Executes physical command plans |
| Movement | `src/physical/movement_executor.py` | Board-to-board arm moves with reconciliation |
| Occupancy | `src/physical/occupancy.py` | Tracks which piece is on which square |
| Teleport | `src/physical/piece_teleport.py` | Instant piece repositioning via freejoint |
| Piece registry | `src/physical/piece_registry.py` | Deterministic registry of all 32 + reserve pieces |
| Board mapping | `src/chess_game/board_mapper.py` | Square names ↔ world XY coordinates |
| Web UI | `src/ui/` | Flask app and thread-safe backend wrapper |
| Training | `training/` | SAC specialist training scripts |
| Config | `configs/` | YAML configuration files |

## Key Design Decisions

**Three specialist SAC models** — transit, descend, and ascend are each trained separately. The scripted grasp and place pipelines bridge the model-driven stages.

**`str | None` return convention** — scripted pipeline functions return `None` on success or a reason string on failure. Callers use `if reason := self.method(): return reason` for early-exit chains.

**Walrus-operator pipeline** — each phase in `execute_grasp` / `execute_place` short-circuits on the first failure, making the happy path read as a flat sequence.

**grasp_mode flag** — controls whether finger joints are actuator-driven (contact forces active, piece can push back) or teleported (direct joint assignment, no physics). Set to `True` when a piece is held.
