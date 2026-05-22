# Evaluation

Run from the repository root with the project virtualenv active, or prefix commands with `.venv/bin/`.

## Core Validation

```bash
pytest tests/ -v
python scripts/verify_physics.py
python scripts/eval_chess_reachability.py --all-squares --n-episodes 1
python scripts/eval_chess_piece_move.py --piece white_pawn_e --src e2 --dst e4
python scripts/eval_chess_piece_move.py --piece white_pawn_d --src d2 --dst d4 --starting-position crowded
python scripts/eval_chess_game_flow.py --verify-agreement
python scripts/eval_draw_conditions.py
python scripts/eval_special_moves.py --all
```

## Notes

The mechanically safer default game-flow smoke is:

```bash
python scripts/eval_chess_game_flow.py --moves e2e3,e7e6,g1f3,b8c6 --verify-agreement
```

The adjacent-pawn sequence `e2e4,e7e5,g1f3,b8c6` is legal and passes mocked planning, but real physics can fail while placing `e7e5` next to the pawn on `e4` with `ROTATION_FAILED (kinematic limit)`. Keep it as a regression scenario for future controller work rather than the default health check.

## Script Coverage

- `eval_chess_reachability.py`: exact square centers and arm reachability.
- `eval_chess_piece_move.py`: one physical board move, final placement, and non-moving piece displacement.
- `eval_chess_game_flow.py`: deterministic multi-move execution and board/tracker/occupancy/position agreement.
- `eval_draw_conditions.py`: stalemate, insufficient material, fifty-move claim, and threefold claim status.
- `eval_special_moves.py`: castling, en passant, promotion, and capture-promotion physical plans.
