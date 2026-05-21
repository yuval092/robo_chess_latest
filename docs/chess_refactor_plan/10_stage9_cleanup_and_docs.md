# Stage 9: Cleanup And Documentation

## Goal

Remove obsolete single-cube assumptions, update documentation, and leave the project in a coherent chess-game state.

## Cleanup Tasks

### Naming

Current names like `object0`, `cube`, and `ChessFetchTask-v0` can remain internally where compatibility requires, but public chess-game APIs should use:

- `piece`
- `square`
- `board`
- `game`
- `physical_move`

Do not rename everything at once if it risks breaking MuJoCo XML assumptions. Prioritize public interfaces and docs.

### Legacy Single-Cube Paths

After all chess scripts pass:

- Remove or clearly mark single-cube-only scripts.
- Keep legacy movement tests if they still protect the waypoint algorithm.
- Ensure `object0` is not the only supported active body.

### Documentation

Add:

```text
docs/chess_game/
  architecture.md
  coordinate_system.md
  movement_pipeline.md
  special_moves.md
  ui.md
  evaluation.md
```

Minimum content:

- How chess squares map to world coordinates.
- How a legal chess move becomes physical commands.
- Why graveyards and promotion reserves use teleportation.
- How to run UI.
- How to run evaluation suite.
- Known reachability limitations, if any.

### Config Documentation

Document:

- `configs/env.yaml` for physical robot/table settings.
- `configs/chess.yaml` for chess geometry, reserves, and game settings.
- Which file is source-of-truth for each constant.

### Logs And Debugging

Add structured debug output for failed moves:

- FEN before move.
- UCI move attempted.
- Physical plan commands.
- Failed command.
- Stage result and crash reason.
- Active piece position.
- Grip position.

This output should be easy to paste into a bug report.

## Final Validation

Run the full validation workflow:

```bash
pytest tests/ -v
python scripts/verify_physics.py
python scripts/eval_chess_reachability.py --all-squares --n-episodes 1
python scripts/eval_chess_piece_move.py --piece white_pawn_e --src e2 --dst e4
python scripts/eval_chess_piece_move.py --piece white_pawn_d --src d2 --dst d4 --starting-position crowded
python scripts/eval_chess_game_flow.py --moves e2e4,e7e5,g1f3,b8c6
python scripts/eval_special_moves.py --all
```

Manual UI validation:

```bash
python scripts/run_chess_ui.py --host 127.0.0.1 --port 8000 --visualize
```

Verify:

- New game starts correctly.
- Human move works.
- Computer reply works.
- Illegal move displays an error.
- Let-computer-play works on current side.
- Special move cases can be demonstrated through scripts.

## Definition Of Done

The chess-game refactor is complete when:

- The simulator loads a colored chess table with all pieces.
- The table surface itself is the chess board.
- Chess square centers use exact 8cm cell geometry, not the movement sampling grid.
- Crowded starting-position moves do not displace neighboring pieces.
- Every legal move type has a tested physical plan.
- The arm moves board-to-board pieces using the existing waypoint algorithm.
- Graveyard and promotion reserve interactions use teleportation only.
- UI controls and monitors the game.
- Evaluation scripts cover chess-specific behavior.
- Documentation explains the architecture and operational workflow.
