# RoboChess — Project Overview

## What Is RoboChess?

RoboChess is a physics simulation of a robotic chess player. A **Fetch robot arm** simulated inside MuJoCo physically picks up chess pieces and places them on a board, controlled by either a deterministic scripted controller or three specialist **Soft Actor-Critic (SAC) reinforcement-learning models**. A human plays against the **Stockfish** chess engine through a browser-based web UI; every legal move is executed both logically (python-chess rules) and physically (the arm moves in the MuJoCo scene).

The project is **simulation-only** — no connection to real hardware. All physics are computed by MuJoCo through the `gymnasium-robotics` `FetchPickAndPlace-v4` environment family.

---

## System-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Browser  (HTML / CSS / JS)                                              │
│  Click squares → POST /api/move → poll /api/snapshot → update board      │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  HTTP  (Flask, port 8000)
┌───────────────────────────────▼──────────────────────────────────────────┐
│  src/ui/app.py   Flask REST API                                           │
│  UIBackend protocol  →  QueuedUIBackend (serialises onto main thread)     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  method calls via queue
┌───────────────────────────────▼──────────────────────────────────────────┐
│  src/chess_game/game_orchestrator.py   GameOrchestrator                  │
│  ┌──────────────────────┐   ┌─────────────────────────────────────────┐  │
│  │  ChessService        │   │  PhysicalPlanExecutor                   │  │
│  │  (python-chess       │   │  ┌──────────────────┐ ┌─────────────┐  │  │
│  │   + Stockfish)       │   │  │ MovementExecutor  │ │PieceTelep.  │  │  │
│  │                      │   │  │ ┌──────────────┐  │ │             │  │  │
│  │  MovePlanner         │   │  │ │Controller    │  │ └─────────────┘  │  │
│  │  LogicalPieceTracker │   │  │ │ Scripted or  │  │                  │  │
│  └──────────────────────┘   │  │ │ ModelEmbed.  │  │                  │  │
│                              │  │ └──────┬───────┘  │                  │  │
│                              │  └────────│───────────┘                  │  │
│                              └───────────│─────────────────────────────┘  │
└───────────────────────────────────────── │──────────────────────────────┘
                                           │  low-level arm commands
┌──────────────────────────────────────────▼──────────────────────────────┐
│  src/chess_env/   MuJoCo simulation environment                          │
│  ChessTaskEnv  ⊃  ChessSimulationEnv  ⊃  MujocoFetchPickAndPlaceEnv     │
│  Observation / Reward / Reset / Step / Grasp / Place pipeline           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Summary

| Layer | Package / File | Responsibility |
|---|---|---|
| **Browser UI** | `src/ui/static/app.js`, `templates/index.html` | Render board, capture clicks, call REST API |
| **Flask API** | `src/ui/app.py` | HTTP routes, JSON serialisation |
| **Concurrency bridge** | `src/ui/queued_backend.py` | Thread-safe request queue between Flask thread and main sim thread |
| **Game orchestration** | `src/chess_game/game_orchestrator.py` | Coordinate chess rules + physical execution for one move |
| **Chess rules** | `src/chess_game/chess_service.py` | python-chess wrapper, Stockfish engine, game status |
| **Move planning** | `src/chess_game/move_planner.py` | Translate chess move → `PhysicalPlan` command list |
| **Piece tracking** | `src/chess_game/move_planner.py:LogicalPieceTracker` | Maintain square→piece_id map across moves |
| **Coordinate mapping** | `src/chess_game/board_mapper.py` | Square name ↔ world XY coordinates |
| **Physical execution** | `src/physical/plan_executor.py` | Execute arm moves, teleports, and removes |
| **Board-to-board move** | `src/physical/movement_executor.py` | Single piece: src→dst with occupancy checks |
| **Occupancy** | `src/physical/occupancy.py` | Expected physical square occupancy map |
| **Piece teleport** | `src/physical/piece_teleport.py` | Instantaneous MuJoCo free-joint repositioning |
| **Piece registry** | `src/physical/piece_registry.py` | Canonical 32+64 physical piece descriptors |
| **Scripted controller** | `src/chess_env/controller.py` | Deterministic proportional-control arm driver |
| **RL controller** | `src/chess_env/model_controller.py` | SAC model inference with scripted fallback |
| **Model registry** | `src/chess_env/model_registry.py` | Load and retrieve specialist SAC models |
| **RL environment** | `src/chess_env/task.py` | Observation, reward, reset, done logic |
| **MuJoCo base** | `src/chess_env/simulation.py` | Extends FetchPickAndPlace-v4 |
| **Scene generation** | `src/chess_env/environment_generation.py` | Inject XML fragments, generate STL meshes |
| **Waypoints** | `src/chess_env/waypoints.py` | Z-level constants, valid stage transitions |
| **Config loader** | `src/utils/io.py` | YAML config loading, logger factory |
| **Training** | `training/` | SAC fine-tuning pipeline (not imported by `src/`) |
| **CLI Evaluation** | `src/cli/eval_*.py` | Stage accuracy, physics verification, full flow evaluation |
| **Training CLI** | `training/cli/main.py` | SAC training entry point and model evaluation |
| **Tests** | `tests/` | pytest suite |

---

## Repository Layout

```
robo_chess_latest/
├── src/
│   ├── chess_env/          # MuJoCo environment & RL infrastructure
│   │   ├── __init__.py     # Gymnasium registration: ChessFetchTask-v0
│   │   ├── controller.py   # ScriptedController
│   │   ├── model_controller.py  # ModelEmbeddedController
│   │   ├── model_registry.py    # SAC model loader
│   │   ├── simulation.py   # ChessSimulationEnv
│   │   ├── task.py         # ChessTaskEnv (RL logic)
│   │   ├── task_execution.py    # GraspPlaceMixin
│   │   ├── task_runtime.py      # TaskRuntimeMixin (reset, step, reward)
│   │   ├── task_state.py        # TaskStateMixin (active piece, home posture)
│   │   ├── transfer_obs.py      # 25-D transfer observation space
│   │   ├── environment_generation.py  # XML + STL builders
│   │   └── waypoints.py    # Z-levels, transition validation
│   │
│   ├── chess_game/         # Chess rules & move planning
│   │   ├── board_mapper.py
│   │   ├── chess_service.py
│   │   ├── game_orchestrator.py
│   │   ├── models.py       # Compat shim → GameStatus
│   │   └── move_planner.py
│   │
│   ├── physical/           # Physical execution pipeline
│   │   ├── movement_executor.py
│   │   ├── noop_executor.py
│   │   ├── occupancy.py
│   │   ├── piece_registry.py
│   │   ├── piece_teleport.py
│   │   └── plan_executor.py
│   │
│   ├── ui/                 # Flask web UI
│   │   ├── app.py
│   │   ├── queued_backend.py
│   │   ├── static/app.js
│   │   ├── static/styles.css
│   │   └── templates/index.html
│   │
│   ├── cli/                # Installed entry points
│   │   ├── play.py             → robo-chess-play
│   │   ├── generate_scene.py   → robo-chess-generate
│   │   ├── eval_stage.py       → robo-chess-eval-stage
│   │   ├── eval_physics.py     → robo-chess-eval-physics
│   │   └── eval_flow.py        → robo-chess-eval-flow
│   │
│   └── utils/
│       ├── io.py               # load_config(), setup_logger()
│       ├── args.py             # Shared argparse helpers
│       ├── config.py           # Backward-compat shim
│       └── config_validation.py # validate_config() for test suite
│
├── training/               # SAC training (not imported by src/)
│   ├── trainer.py          # SACTrainer
│   ├── callbacks.py        # DetailedLoggingCallback, SuccessRateEvalCallback
│   ├── cli/
│   │   └── main.py         → robo-chess-train
│   └── envs/               # Per-stage gym.Wrapper wrappers + factory
│       ├── __init__.py     # make_train_env(), make_eval_env()
│       ├── transit_env.py
│       ├── descend_env.py
│       └── ascend_env.py
│
├── chess_env/              # MuJoCo assets
│   ├── assets/
│   │   ├── pick_and_place.xml  # Main scene (auto-generated sections)
│   │   ├── robot.xml
│   │   └── shared.xml
│   └── stls/
│       ├── chess/          # 6 chess piece STL meshes (procedurally generated)
│       └── fetch/          # Fetch robot collision meshes
│
├── configs/
│   ├── env.yaml            # Physics constants, Z-levels, grasp thresholds
│   ├── chess.yaml          # Board geometry, graveyard/reserve layout, engine config
│   ├── physics.yaml        # MuJoCo solver, quaternion, initial qpos
│   ├── training.yaml       # SAC hyperparameters, eval schedule
│   └── deployed_models.yaml # Active model checkpoint paths
│
├── models/
│   ├── transit.zip         # Active transit specialist
│   ├── descend.zip         # Active descend specialist
│   ├── ascend.zip          # Active ascend specialist
│   └── pretrained/
│       └── sac-FetchPickAndPlace-v4.zip  # Pretrained base model
│
└── tests/                  # pytest suite
    ├── chess_env/
    ├── chess_game/
    ├── integration/
    ├── physical/
    └── ui/
```

---

## Data Flow: Human Move (End-to-End)

The following trace follows a single move (e.g., e2→e4) from browser click to arm completion.

```
1. Browser
   User clicks e2, then e4.
   JS sends: POST /api/move  { "src": "e2", "dst": "e4" }

2. Flask (src/ui/app.py : api_move)
   Calls: backend.submit_human_move("e2", "e4", None)

3. QueuedUIBackend (src/ui/queued_backend.py)
   Enqueues UIRequest onto the main-thread queue.
   Blocks waiting for response (synchronous from caller's perspective).

4. Main thread (src/cli/play.py : while loop)
   Calls: backend.process_one(env=env)
   Dequeues the request and calls: orchestrator.submit_human_move("e2", "e4")

5. GameOrchestrator (src/chess_game/game_orchestrator.py)
   a. ChessService.validate_square_move("e2", "e4") → chess.Move(e2e4)
   b. MovePlanner.plan(move) → PhysicalPlan(commands=[ArmMoveCommand("white_pawn_e", "e2", "e4")])
   c. Sets is_busy=True
   d. Calls physical_executor.execute(plan)

6. PhysicalPlanExecutor (src/physical/plan_executor.py)
   Iterates plan.commands:
   - ArmMoveCommand → movement_executor.move_piece_between_squares(
       "white_pawn_e", "e2", "e4")

7. MovementExecutor (src/physical/movement_executor.py)
   a. occupancy.assert_piece_at("white_pawn_e", "e2")  ← validates expected state
   b. occupancy.assert_square_empty("e4")
   c. src_xy = board_mapper.square_name_to_xy("e2")  → [0.76, 0.1241]
   d. dst_xy = board_mapper.square_name_to_xy("e4")  → [0.76, 0.2841]
   e. env.set_active_piece("white_pawn_e")
   f. controller.run_full_move(src_xy, dst_xy)

8. Controller (ScriptedController or ModelEmbeddedController)
   Full pick-and-place sequence:
     run_transit(src_xy)   → arm moves to e2 at SAFE_Z (0.530 m)
     run_descend(src_xy)   → arm descends to HOVER_Z (0.460 m) above e2
     run_grasp()           → 6-phase scripted grasp pipeline
     run_ascend(src_xy)    → arm ascends to SAFE_Z with piece held
     run_transit(dst_xy)   → arm moves to e4 at SAFE_Z
     run_descend(dst_xy)   → arm descends to HOVER_Z above e4
     run_place(dst_xy)     → 6-phase scripted place pipeline
     run_ascend(dst_xy)    → arm ascends to SAFE_Z, piece placed

9. ChessTaskEnv (src/chess_env/task.py / simulation.py)
   Each run_* method calls env._set_action() and env._mujoco_step() in a loop.
   Abort conditions are monitored at every step.

10. Post-move snap (MovementExecutor)
    piece_pos = env.get_active_piece_position()
    If XY error ≤ 20 mm and Z error ≤ 10 mm:
      PieceTeleporter.teleport_piece_to_xyz(piece_id, exact_square_centre_xyz)
    occupancy.set_piece_square("white_pawn_e", "e4")

11. Return to home (PhysicalPlanExecutor.return_to_home)
    controller.run_transit(home_xy=[0.88, 0.2641])
    env.reset_arm_to_home_posture()   ← normalises redundant wrist/roll joints

12. GameOrchestrator (commit)
    chess_service.push(move)
    piece_tracker.apply_committed_move(move, plan)
    last_move = "e2e4"
    is_busy = False

13. Auto computer reply (if auto_computer_reply=True)
    orchestrator.let_computer_play_current_turn()
    → Stockfish selects a move → same pipeline executes for Black

14. Browser
    Receives MoveExecutionResult with embedded GameSnapshot.
    Renders new board state, highlights last move, updates move history.
```

---

## Entry Points

| Command | Source | Description |
|---|---|---|
| `robo-chess-play` | `src/cli/play.py:main` | Start Flask + MuJoCo game loop |
| `robo-chess-generate` | `src/cli/generate_scene.py:main` | Regenerate MuJoCo XML assets |
| `robo-chess-eval-stage` | `src/cli/eval_stage.py:main` | Per-stage waypoint accuracy (success %, error, crashes); supports `--visualize` |
| `robo-chess-eval-physics` | `src/cli/eval_physics.py:main` | Physics integrity: geometry, stability, reachability; supports `--visualize` |
| `robo-chess-eval-flow` | `src/cli/eval_flow.py:main` | Full piece-move flow: simple / complex / full board sweep; supports `--visualize` |
| `robo-chess-train train` | `training/cli/main.py:main` | Train one SAC specialist model |
| `robo-chess-train eval` | `training/cli/main.py:main` | Evaluate a trained model (delegates to eval-stage) |

---

## Key Design Decisions

### Dual-Controller Design
The arm can be driven by either the `ScriptedController` (deterministic proportional control) or the `ModelEmbeddedController` (SAC specialist models). Both expose the same interface (`run_transit`, `run_descend`, `run_ascend`, `run_grasp`, `run_place`, `run_full_move`). The scripted controller requires no trained models and is always available as a fallback.

### Specialist Model Architecture
Instead of training one monolithic model for the full pick-and-place task, three specialist SAC models handle separate movement phases: **transit** (horizontal sweep), **descend** (vertical approach), and **ascend** (vertical lift). Grasp and place remain scripted. This division allows each model to train on a narrow, well-defined task with appropriate reward shaping.

### Transfer Learning via the "Holding Object" Trick
All three specialist models are fine-tuned from a pretrained `FetchPickAndPlace-v4` SAC checkpoint. The pretrained model expects a 25-dimensional observation including an object position. During RoboChess training and inference, the object position in this observation is set equal to the gripper position ("as if the gripper is already holding the object"). This tricks the policy into applying its object-transport knowledge to pure gripper movement.

### Thread Safety
The MuJoCo simulation runs on the main thread. Flask runs on a daemon thread. All move requests are serialised through `QueuedUIBackend`, which enqueues requests and blocks until the main thread processes them. This avoids any concurrent MuJoCo calls.

### Scene Auto-Generation
The MuJoCo XML scene file contains clearly marked sections for board squares, chess pieces, and graveyard/reserve zones. These sections are regenerated automatically at startup from the YAML config. This ensures the visual scene always matches the config without requiring manual XML editing.
