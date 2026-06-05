# Architecture

## Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Browser / HTTP                                          │
├─────────────────────────────────────────────────────────┤
│  Flask REST API          src/ui/app.py                   │
│  QueuedUIBackend         src/ui/queued_backend.py        │
├─────────────────────────────────────────────────────────┤
│  Game Logic                                              │
│    GameOrchestrator      src/chess_game/game_orchestrator│
│    ChessService          src/chess_game/chess_service    │
│    MovePlanner           src/chess_game/move_planner     │
│    LogicalPieceTracker   src/chess_game/move_planner     │
├─────────────────────────────────────────────────────────┤
│  Physical Execution                                      │
│    PhysicalPlanExecutor  src/physical/plan_executor      │
│    MovementExecutor      src/physical/movement_executor  │
│    PhysicalOccupancy     src/physical/occupancy          │
│    PieceTeleporter       src/physical/piece_teleport     │
│    PieceRegistry         src/physical/piece_registry     │
│    BoardMapper           src/chess_game/board_mapper     │
├─────────────────────────────────────────────────────────┤
│  Arm Control                                             │
│    ModelEmbeddedController  src/chess_env/model_controller│
├─────────────────────────────────────────────────────────┤
│  MuJoCo Environment                                      │
│    ChessProductionEnv    src/chess_env/production_env    │
│    ChessBaseEnv          src/chess_env/base_env          │
│    ChessSimulationEnv    src/chess_env/simulation        │
│    MujocoFetchPickAndPlaceEnv  (gymnasium-robotics)      │
└─────────────────────────────────────────────────────────┘
```

## Source Tree

```
src/
├── chess_env/
│   ├── __init__.py          # Gymnasium env registration
│   ├── simulation.py        # ChessSimulationEnv — MuJoCo XML loading, action scaling
│   ├── base_env.py          # ChessBaseEnv — config, obs space, reset, soft_reset, helpers
│   ├── production_env.py    # ChessProductionEnv — piece registry, grasp/place pipelines
│   ├── training_env.py      # ChessTrainingEnv — RL reward, crash checks, curriculum
│   └── model_controller.py  # ModelEmbeddedController — SAC inference + scripted pipelines
│
├── chess_game/
│   ├── board_mapper.py      # Square name ↔ world XY
│   ├── chess_service.py     # python-chess board + UCI engine wrapper
│   ├── move_planner.py      # LogicalPieceTracker + MovePlanner (chess move → command list)
│   └── game_orchestrator.py # Top-level coordinator: logic + physical + engine
│
├── physical/
│   ├── piece_registry.py    # Deterministic list of all 32+reserve pieces
│   ├── piece_teleport.py    # Free-joint instant repositioning
│   ├── occupancy.py         # Physical square occupancy tracker
│   ├── movement_executor.py # Board-to-board arm move with reconciliation
│   └── plan_executor.py     # Plan execution, return_to_home
│
├── ui/
│   ├── app.py               # Flask factory + REST routes
│   └── queued_backend.py    # Thread-safe UIBackend wrapping GameOrchestrator
│
└── utils/
    ├── io.py                # load_config, resolve_model_paths, setup_logger
    └── config_validation.py # Startup config/model checks
```

## Threading Model

The game loop in `main.py` runs on the **main thread**:

```python
while True:
    backend.process_request(env=env, timeout=0.05)  # drains one queued request
    env.render()
    time.sleep(0.01)
```

Flask runs on a **daemon thread**. All API handlers call into `QueuedUIBackend`, which puts a `UIRequest` on a queue and blocks waiting for the result. `process_request` on the main thread pops the request, calls the orchestrator, and puts the result back.

MuJoCo is **never touched from the Flask thread**. All simulation state changes happen inside `process_request` on the main thread.

## Environment Class Hierarchy

```
MujocoFetchPickAndPlaceEnv  (gymnasium-robotics)
  └── ChessSimulationEnv    — XML swap, action scaling, board sampling
        └── ChessBaseEnv    — config, obs, reset, soft_reset, low-level helpers
              ├── ChessProductionEnv  — piece registry, scripted grasp/place
              └── ChessTrainingEnv   — RL reward, crash checks, drift curriculum
```

`ChessProductionEnv` is used at runtime (registered as `ChessFetchTask-Play-v0`).  
`ChessTrainingEnv` is used during specialist model training.
