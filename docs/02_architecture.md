# Architecture

## Layer Boundaries

RoboChess deliberately separates logical chess state, expected physical state,
and actual MuJoCo state.

| State | Owner | Meaning |
|---|---|---|
| Legal chess board | `ChessService` | The authoritative `python-chess.Board` |
| Logical piece IDs | `LogicalPieceTracker` | Which physical piece ID represents each logical square |
| Expected physical occupancy | `PhysicalOccupancy` | Where the physical layer expects active pieces to be |
| Actual simulated state | MuJoCo `model`/`data` | Freejoint poses, robot joints, contacts, velocities |

The chess board only advances after physical execution succeeds. This prevents
the engine and UI from seeing a move as committed when the simulated arm failed.

## Runtime Construction

`main.py` wires the full system:

```text
gym.make("ChessFetchTask-v0", show_chess_pieces=True, hide_object=True)
  -> ModelEmbeddedController
  -> BoardMapper
  -> PieceRegistry
  -> PhysicalOccupancy
  -> MovementExecutor
  -> PieceTeleporter
  -> PhysicalPlanExecutor
  -> ChessService
  -> LogicalPieceTracker
  -> GameOrchestrator
  -> QueuedUIBackend
  -> Flask app
```

The deployed model paths are resolved by `src.utils.io.resolve_model_paths()`.
Explicit CLI overrides win over `configs/deployed_models.yaml`.

## Move Lifecycle

For a normal move such as `e2e4`:

1. Browser sends `POST /api/move` with `{"src": "e2", "dst": "e4"}`.
2. Flask route calls `QueuedUIBackend.submit_human_move()`.
3. The request is queued and blocks until the main thread processes it.
4. `GameOrchestrator.submit_human_move()` verifies turn ownership.
5. `ChessService.validate_square_move()` verifies legality against
   `python-chess`.
6. `MovePlanner.plan()` creates a `PhysicalPlan`.
7. `PhysicalPlanExecutor.execute()` dispatches commands.
8. `MovementExecutor.move_piece_between_squares()` validates expected occupancy.
9. `ModelEmbeddedController.run_full_move()` performs pick and place.
10. `MovementExecutor` reconciles the final piece pose and snaps it exactly to
    the destination square if within tolerance.
11. `PhysicalPlanExecutor.return_to_home()` moves the arm home and restores the
    reset-time home posture.
12. `ChessService.push()` commits the move.
13. `LogicalPieceTracker.apply_committed_move()` updates piece ID mappings.
14. The result snapshot is returned to Flask and then to the browser.

## Physical Plan Commands

`MovePlanner` emits three command types:

| Command | Use |
|---|---|
| `ArmMoveCommand(piece_id, src_square, dst_square)` | Move a piece with the arm |
| `RemoveFromBoardCommand(piece_id, graveyard_slot)` | Capture removal before moving the capturing piece |
| `TeleportCommand(piece_id, destination_kind, destination_id)` | Promotion reserve and other instant repositioning |

Special chess cases:

- Captures remove the captured piece to a graveyard slot before the arm move.
- Castling emits two arm moves: king then rook.
- En passant removes the pawn from the passed-over square.
- Promotion moves the pawn to the destination, teleports that pawn to a reserve
  slot, then teleports an off-board reserve piece of the promoted type onto the
  promotion square.

## Threading Model

MuJoCo is treated as main-thread-only. The Flask app runs in a daemon thread, but
the backend calls that touch game execution are queued:

```text
Flask thread
  -> queue.Queue[UIRequest]
Main thread
  -> backend.process_one(env=env)
  -> orchestrator / physical executor / MuJoCo
```

`snapshot()` is the only backend call that does not enqueue. It returns a cached
snapshot under a lock. Mutating calls block until the main loop processes them.

## Model Control Strategy

The arm is controlled by a hybrid system:

- SAC specialist model for `transit`.
- SAC specialist model for `descend`.
- SAC specialist model for `ascend`.
- Scripted logic for grasp and place.
- Scripted transition logic between stages.

The three specialists use a 25-dimensional transfer observation compatible with
the pretrained Fetch PickAndPlace policy. Scripted/status paths use a smaller
native observation.

