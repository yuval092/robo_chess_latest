# Stage 8: Evaluation And Validation

## Goal

Update the evaluation suite so chess behavior is measurable, repeatable, and safe to evolve.

Evaluation scripts remain first-class project assets.

## Existing Scripts To Keep

Keep and update:

- `scripts/verify_physics.py`
- `scripts/eval_stages.py`
- `scripts/eval_sequence.py`
- `scripts/eval_stress.py`
- `scripts/visualize.py`
- `scripts/test_grasp_physics.py`

They may keep legacy single-cube modes temporarily, but must gain chess-aware modes where applicable.

## New Scripts

### eval_chess_reachability.py

Purpose:

- Validate every chess square center can be reached for transit, descend, and ascend.
- Use exact 8cm chess geometry from `configs/chess.yaml`, not the existing `env.yaml.edge_margin` sampling grid.

Command:

```bash
python scripts/eval_chess_reachability.py --all-squares --n-episodes 1
```

Output must include:

- square
- world xy
- transit success/error
- descend success/error
- ascend success/error
- crash reason if any

Exit non-zero on failure.

Required guardrail:

- Print the computed near/far rank centers.
- Fail if rank 1 center X is not `0.600` and rank 8 center X is not `1.160` for the default `center_xy: [0.88, 0.2641]`.
- This prevents accidentally validating the old 7.75cm sampling cells from `eval_stress.py`.

### eval_chess_piece_move.py

Purpose:

- Move a named piece from one square to another.
- Validate optional crowded-board safety cases.

Command:

```bash
python scripts/eval_chess_piece_move.py --piece white_pawn_e --src e2 --dst e4
python scripts/eval_chess_piece_move.py --piece white_pawn_d --src d2 --dst d4 --starting-position crowded
```

Required checks:

- Piece starts at source.
- Destination is empty.
- Physical move succeeds.
- Final piece center is within placement tolerance.
- In crowded mode, every non-moving active piece remains within 2mm of its original square center.

### eval_chess_game_flow.py

Purpose:

- Execute a deterministic list of legal chess moves through the full orchestrator.

Command:

```bash
python scripts/eval_chess_game_flow.py --moves e2e4,e7e5,g1f3,b8c6
```

Required checks:

- Each move is legal at the time it is requested.
- Physical execution succeeds.
- FEN matches expected state after each move.
- Piece tracker and chess board agree.
- Non-moving pieces are not displaced by crowded-board movement.

### eval_special_moves.py

Purpose:

- Exercise castling, en passant, capture, and promotion from constructed FEN states.

Command:

```bash
python scripts/eval_special_moves.py --case castling
python scripts/eval_special_moves.py --case en-passant
python scripts/eval_special_moves.py --case promotion
python scripts/eval_special_moves.py --case capture-promotion
```

Use mocked physical executor by default for fast validation. Add `--real-physics` for full MuJoCo runs.

## verify_physics.py Updates

Add checks:

- Red `target0` site is absent from loaded chess scene.
- 64 board square visuals exist and are non-colliding.
- 32 starting chess piece bodies exist.
- All chess piece collision cubes have correct dimensions, mass, and contact parameters.
- All chess visual STL geoms are non-colliding.
- Graveyard and reserve slots exist or their configured coordinates validate.
- All square centers are inside table extents.
- Kinematic reachability uses exact chess square centers from `BoardMapper`.
- The old `edge_margin` reachability grid may remain as a movement stress test, but it is not accepted as chess board validation.
- Crowded-board clearance test exists and is part of the required chess validation workflow.

Keep existing checks:

- XML integrity.
- Table geometry.
- Grasp physics.
- Static stability.
- Kinematic reachability.
- Teleport verification.

## Test Matrix

Pure logic:

```bash
pytest tests/chess_game -v
```

Physical utilities:

```bash
pytest tests/physical -v
```

Existing movement:

```bash
pytest tests/chess_env -v
```

Full suite:

```bash
pytest tests/ -v
```

Physics validation:

```bash
python scripts/verify_physics.py
```

Chess reachability:

```bash
python scripts/eval_chess_reachability.py --all-squares --n-episodes 1
```

Chess game smoke:

```bash
python scripts/eval_chess_game_flow.py --moves e2e4,e7e5,g1f3,b8c6
```

Special moves:

```bash
python scripts/eval_special_moves.py --all
```

## Acceptance Criteria

Before calling the refactor complete:

- All 64 square centers are physically reachable.
- All 64 square centers use exact 8cm chess-cell geometry.
- All 32 starting pieces are correctly placed.
- Crowded-board movement does not disturb neighboring pieces.
- Normal legal moves work end-to-end.
- Captures place captured pieces in correct graveyard by teleportation.
- Castling moves both king and rook physically.
- En passant removes the correct passed pawn by teleportation.
- Promotion replaces the pawn with the selected reserve piece.
- Illegal moves are rejected before physical execution.
- UI can drive at least one full human move and one computer reply.
- FEN, UI board, logical piece tracker, and physical occupancy agree after every successful move.
- Evaluation scripts exit non-zero on failures.

## Performance Metrics

Track:

- Average physical move duration.
- Per-square reachability success rate.
- Per-piece move success rate.
- Crowded-board neighbor displacement p50/p95/max.
- Capture success rate.
- Special move success rate.
- Placement error p50/p95/max.
- Grasp failure count by piece type and square.

Write metrics to stdout and optionally JSON:

```bash
python scripts/eval_chess_game_flow.py --moves ... --json-out logs/chess_eval.json
```
