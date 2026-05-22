# Chess Game Architecture

The chess refactor separates logical chess state from physical robot execution.

## Layers

- `ChessService` owns the `python-chess` board, legal move validation, SAN history, FEN, engine move selection, and game status.
- `LogicalPieceTracker` maps logical squares to stable physical piece ids such as `white_pawn_e`.
- `MovePlanner` converts a legal `chess.Move` into physical commands: arm moves, captures to graveyard, castling rook moves, en-passant removal, and promotion reserve swaps.
- `PhysicalPlanExecutor` executes those commands with `MovementExecutor` and `PieceTeleporter`.
- `GameOrchestrator` coordinates validation, planning, physical execution, home return, and logical commit. The chess board is only pushed after physical execution succeeds.
- `src/ui` exposes the orchestrator through Flask endpoints for board state, new game, human moves, and computer moves.

## Commit Rule

Physical movement is authoritative until a move succeeds. If the robot move fails, the chess board and logical tracker remain unchanged and the snapshot reports the error.

## Compatibility

Legacy MuJoCo names such as `object0`, `cube`, and `ChessFetchTask-v0` still exist where the environment and older tests depend on them. Public chess-game code uses piece, square, board, game, and physical-move terminology.
