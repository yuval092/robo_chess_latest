# Movement Pipeline

A legal chess move becomes robot work through a fixed sequence.

1. `ChessService` validates the UCI or source/destination move.
2. `MovePlanner` reads the current board and `LogicalPieceTracker`.
3. The planner emits a `PhysicalPlan`.
4. `PhysicalPlanExecutor` executes each command.
5. `MovementExecutor` moves board-to-board pieces with the existing waypoint controller.
6. `PieceTeleporter` handles non-board interactions: captured pieces, promoted pawns, and reserve pieces.
7. `GameOrchestrator` returns the arm home, pushes the chess move, updates the tracker, and publishes a new snapshot.

## Physical Commands

- `ArmMoveCommand`: pick a piece from one board square and place it on another board square.
- `RemoveFromBoardCommand`: teleport a captured piece to its graveyard slot.
- `TeleportCommand`: move pieces between promotion reserves, graveyards, and board squares.

## Failure Behavior

Move failures include the FEN before the move, attempted UCI, physical plan commands, failed stage, active piece position, and grip position in `eval_chess_game_flow.py`. The orchestrator leaves logical chess state unchanged on failed physical execution.
