# RoboChess — Project Overview

## What Is RoboChess?

RoboChess is a simulation of a physical robotic chess player. A Fetch robotic arm, running inside a MuJoCo physics simulation, plays chess against a human player via a web browser interface. The human selects moves by clicking squares on a rendered board; the arm physically picks up the chess piece from its source square, carries it across the board, and places it on the destination square. Captured pieces are teleported to off-board "graveyard" zones, and promoted pawns are swapped for reserve pieces.

The project is purely simulation-based — there is no connection to real hardware. All physics are computed by MuJoCo through the `gymnasium-robotics` `FetchPickAndPlace` environment family.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (HTML/CSS/JS)                                          │
│  Click squares → POST /api/move → display result               │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP (Flask)
┌────────────────────▼────────────────────────────────────────────┐
│  src/ui/app.py   Flask routes                                   │
│  UIBackend protocol → QueuedUIBackend (serialises requests)     │
└────────────────────┬────────────────────────────────────────────┘
                     │ method calls (serialised through queue)
┌────────────────────▼────────────────────────────────────────────┐
│  src/chess_game/game_orchestrator.py  GameOrchestrator          │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐  │
│  │  ChessService   │  │  PhysicalPlanExecutor               │  │
│  │  (python-chess) │  │  ┌──────────────┐ ┌─────────────┐  │  │
│  │                 │  │  │MovementExec. │ │PieceTeleport│  │  │
│  │                 │  │  │ScriptedCtrl  │ │             │  │  │
│  │  MovePlanner    │  │  └──────┬───────┘ └─────────────┘  │  │
│  │  LogicalPiece   │  │         │                           │  │
│  │  Tracker        │  │  ┌──────▼───────────────────────┐  │  │
│  └─────────────────┘  │  │  ChessTaskEnv (MuJoCo)       │  │  │
│                        │  │  ChessSimulationEnv          │  │  │
│                        │  │  MujocoFetchPickAndPlaceEnv  │  │  │
│                        │  └──────────────────────────────┘  │  │
│                        └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

The system has five distinct layers:

| Layer | Files | Responsibility |
|-------|-------|----------------|
| **UI** | `src/ui/`, `src/ui/queued_backend.py` | Browser frontend + Flask HTTP server |
| **Chess logic** | `src/chess_game/` | Rules, move planning, piece tracking |
| **Physical executor** | `src/physical/` | Translates chess commands to arm motions |
| **Arm controller** | `src/chess_env/controller.py` | Scripted proportional movement |
| **Simulation** | `src/chess_env/` | MuJoCo physics, scene generation, waypoints |

---

## Project File Map

```
robo_chess_latest/
├── chess_env/                  ← Root-level package (legacy Gymnasium registration stub)
│   ├── __init__.py             ← Comment-only; active registration is in src/chess_env
│   ├── assets/
│   │   └── pick_and_place.xml  ← MuJoCo scene XML (auto-generated sections)
│   └── stls/chess/             ← Procedurally generated STL meshes (6 piece types)
│
├── configs/
│   ├── env.yaml                ← Physics constants, z-levels, grasp thresholds
│   ├── chess.yaml              ← Board geometry, piece config, graveyard/reserve zones
│   └── physics.yaml            ← MuJoCo solver settings, quaternion, setup steps
│
├── scripts/
│   ├── run_chess_ui.py         ← Entry point: starts Flask + MuJoCo loop
│   ├── debug_one_move.py       ← Record per-step debug log for one move
│   ├── eval_stages.py          ← Unit eval: transit / descend / ascend
│   ├── eval_sequence.py        ← Integration eval: full pick-and-place
│   ├── eval_stress.py          ← Corner/grid stress test
│   ├── eval_special_moves.py   ← Castling, en passant, promotion
│   ├── eval_chess_reachability.py ← 6 chess square centres reachable?
│   ├── eval_chess_game_flow.py ← Multi-move game with mock executor
│   ├── eval_draw_conditions.py ← Stalemate / draw detection
│   └── analyze_debug_log.py    ← Parse JSONL debug log → markdown report
│
├── src/
│   ├── chess_env/
│   │   ├── __init__.py         ← Registers ChessFetchTask-v0 with Gymnasium
│   │   ├── controller.py       ← ScriptedController: all movement stages
│   │   ├── environment_generation.py ← XML fragment injection, STL builders
│   │   ├── simulation.py       ← ChessSimulationEnv (extends Fetch)
│   │   ├── task.py             ← ChessTaskEnv: grasp/place pipeline, rewards
│   │   └── waypoints.py        ← Z-level constants, valid transitions
│   │
│   ├── chess_game/
│   │   ├── board_mapper.py     ← Square names ↔ world XY coordinates
│   │   ├── chess_service.py    ← python-chess wrapper + engine heuristic
│   │   ├── game_orchestrator.py ← Orchestrates chess + physical execution
│   │   ├── chess_service.py    ← ChessService and GameStatus dataclass
│   │   └── move_planner.py     ← Command list generation, LogicalPieceTracker
│   │
│   ├── physical/
│   │   ├── movement_executor.py ← Board-to-board arm move
│   │   ├── occupancy.py        ← Expected physical occupancy map
│   │   ├── piece_registry.py   ← Canonical 32 + 64 reserve piece descriptors
│   │   ├── piece_teleport.py   ← MuJoCo freejoint teleport
│   │   └── plan_executor.py    ← Executes PhysicalPlan command list
│   │
│   ├── ui/
│   │   ├── app.py              ← Flask routes (create_app factory)
│   │   ├── templates/index.html ← Single-page HTML
│   │   └── static/
│   │       ├── app.js          ← Board interaction, promotion dialog
│   │       └── styles.css      ← Green-themed UI stylesheet
│   │
│   └── utils/
│       ├── args.py             ← Shared argparse helpers for scripts
│       └── io.py               ← canonical load_config() and setup_logger()
│
└── tests/                      ← pytest suite (81 tests)
    ├── chess_env/              ← Waypoint, task chaining, scene generation
    ├── chess_game/             ← Service, planner, orchestrator, mapper
    ├── integration/            ← Multi-layer chess move flow
    ├── physical/               ← Registry, teleport, XML, zone alignment
    └── ui/                     ← Flask routes, QueuedUIBackend
```

---

## Data Flow: Human Move

1. **Browser** — user clicks e2, then e4. JS sends `POST /api/move { src:"e2", dst:"e4" }`.
2. **Flask** (`app.py`) — routes to `backend.submit_human_move("e2", "e4", None)`.
3. **QueuedUIBackend** — enqueues the request; main thread dequeues and calls `orchestrator.submit_human_move`.
4. **GameOrchestrator** — validates move via `ChessService.validate_square_move`, builds `PhysicalPlan` via `MovePlanner`, sets `is_busy=True`, calls `physical_executor.execute(plan)`.
5. **PhysicalPlanExecutor** — iterates commands: `ArmMoveCommand` → `MovementExecutor.move_piece_between_squares`.
6. **MovementExecutor** — validates occupancy, resolves XY from `BoardMapper`, calls `controller.run_full_move(src_xy, dst_xy)`.
7. **ScriptedController** — executes transit → descend → grasp pipeline → ascend → transit → descend → place pipeline → ascend sequence.
8. **ChessTaskEnv** — each step calls `_set_action`, advances MuJoCo physics, monitors abort conditions.
9. On success, `MovementExecutor` snaps the piece to exact square centre via teleport.
10. **GameOrchestrator** — commits: `chess_service.push(move)`, `piece_tracker.apply_committed_move(move, plan)`, clears `is_busy`.
11. **Browser** — receives `MoveExecutionResult` with embedded `GameSnapshot`; renders new board state.

## Cleanup Refactor Notes

- The project is installable with `pip install -e .`; scripts no longer mutate `sys.path`.
- `src/utils/io.py` is the canonical home for config loading and logger setup. Legacy shims remain for compatibility.
- Production CLI entry points live under `src/cli/`; repo-local `scripts/` wrappers call those packaged modules.
- Training code remains in `training/` and is not imported by `src/` production modules.
