# Evaluation Scripts

Evaluation scripts are standalone Python programs in `scripts/`. They run actual MuJoCo episodes and print pass/fail results. They are not pytest tests — they are manual verification tools and regression checks for the physical arm controller and game logic.

---

## 1. Overview

| Script | What it tests | Needs MuJoCo |
|---|---|---|
| `eval_stages.py` | Single-stage arm accuracy (transit, descend, ascend) | Yes |
| `eval_stress.py` | Corner/grid position stress test | Yes |
| `eval_chess_reachability.py` | All 64 board squares, arm can reach and return | Yes |
| `eval_chess_piece_move.py` | Single full board-to-board move | Yes |
| `eval_chess_game_flow.py` | Multi-turn game through `GameOrchestrator` | Yes |
| `eval_special_moves.py` | Castling, en passant, promotion command generation | No (mock executor) |
| `eval_draw_conditions.py` | Chess draw condition detection | No (mock executor) |
| `eval_sequence.py` | Sequential arm stages (transit → descend → ascend) | Yes |

---

## 2. eval_stages.py

**Purpose**: Measure per-stage accuracy of the `ScriptedController` in isolation.

**Usage**:
```bash
python scripts/eval_stages.py [--stages transit,descend,ascend] [--n-episodes 50] [--drift-limit 0.010] [--visualize] [--delay 0.02] [--debug]
```

**What it does**:
- Creates a `ChessFetchTask-v0` environment with `force_scenario` set to the target stage.
- Runs N episodes of each stage.
- For each episode: resets env, runs `ctrl.run_transit(target_xy)` / `run_descend()` / `run_ascend()`.
- Records: success/crash/timeout count, XY error (mm) at landing, step count.

**Reported metrics**:
- Success rate (%)
- Mean and max XY error (mm)
- Mean step count
- Crash rate (%)
- Crash reasons (if any)

**Expected output** (passing):
```
Stage: transit  | Success: 50/50 (100.0%) | Mean error: 2.3mm | Mean steps: 18
Stage: descend  | Success: 50/50 (100.0%) | Mean error: 1.1mm | Mean steps: 6
Stage: ascend   | Success: 50/50 (100.0%) | Mean error: 0.8mm | Mean steps: 6
```

---

## 3. eval_stress.py

**Purpose**: Stress-test the arm at board corners and arbitrary grid positions.

**Usage**:
```bash
python scripts/eval_stress.py [--chain full_move|pick|vertical] [--n-episodes 10] [--drift-limit 0.010] [--grid] [--grid-size 3] [--visualize]
```

**Modes**:
- Default (no `--grid`): Tests 5 positions — the 4 board corner square centres (a1, a8, h1, h8) and board centre.
- `--grid`: Tests an N×N grid of positions across the board (default 3×3 = 9 positions).

**Board boundary computation** (key detail fixed in refactor):
The script uses **chess square centre bounds**, not raw table edges:
```python
board_min_x = board_cx - half + cell / 2   # centre of a-file squares
board_max_x = board_cx + half - cell / 2   # centre of h-file squares
```
This ensures test positions are exactly on real chess square centres, eliminating false failures from positions outside the playable area.

**Chain modes**:
- `full_move`: full pick-and-place sequence (grasp → transit → place)
- `pick`: transit + descend + grasp only (no place)
- `vertical`: transit + descend + ascend only (no grasp)

**Expected output** (passing, corner mode):
```
a1: 10/10 success
a8: 10/10 success
h1: 10/10 success
h8: 10/10 success
center: 10/10 success
Overall: 50/50 (100.0%)
```

---

## 4. eval_chess_reachability.py

**Purpose**: Validate that the arm can reach all 64 board square centres via transit, descend, and ascend.

**Usage**:
```bash
python scripts/eval_chess_reachability.py [--n-episodes 1] [--drift-limit 0.010] [--visualize]
```

**What it does**:
1. Validates board geometry constants (rank1_y, rank8_y, fileA_x, fileH_x) against hardcoded expected values with nanometre tolerance.
2. For each of 64 squares: runs transit → descend → ascend. Reports per-square success/failure.
3. Final summary: N/64 squares reachable.

**Geometry validation** (values in metres):
```python
REQUIRED_RANK1_Y = -0.0159
REQUIRED_RANK8_Y = 0.5441
REQUIRED_FILE_A_X = 0.600
REQUIRED_FILE_H_X = 1.160
```
These are derived from `board.center_xy = [0.88, 0.2641]` and `cell_size_m = 0.08`. If the board config changes, these constants must be updated.

**Expected output** (all passing):
```
Board geometry: rank1_y=-0.0159m rank8_y=0.5441m fileA_x=0.600m fileH_x=1.160m
a1: transit OK, descend OK, ascend OK
...
h8: transit OK, descend OK, ascend OK
Reachability: 64/64 squares (100.0%)
```

---

## 5. eval_chess_piece_move.py

**Purpose**: Test a single full board-to-board arm move with a real chess piece in the scene.

**Usage**:
```bash
python scripts/eval_chess_piece_move.py [--piece white_pawn_e] [--src e2] [--dst e4] [--n-episodes 3] [--visualize]
```

**What it does**:
1. Creates `ChessFetchTask-v0` with chess pieces enabled.
2. Runs `MovementExecutor.move_piece_between_squares(piece_id, src, dst)`.
3. Verifies: final piece position within 2mm of target square; no other pieces moved.

**Expected output** (passing):
```
Move white_pawn_e e2→e4: 3/3 success, mean error 1.2mm
```

---

## 6. eval_chess_game_flow.py

**Purpose**: Run a multi-turn game through the full `GameOrchestrator` + physical executor stack.

**Usage**:
```bash
python scripts/eval_chess_game_flow.py [--moves e2e4,e7e5,g1f3] [--n-games 1] [--visualize]
```

**What it does**:
1. Constructs a full `GameOrchestrator` with live MuJoCo env and controller.
2. Plays the specified sequence of moves.
3. After each move: checks `snapshot.is_busy == False`, `snapshot.error == None`, FEN advances correctly, tracker state is consistent.

**Expected output** (passing):
```
Move 1: e2e4 → OK (turn: black, legal_moves: 20)
Move 2: e7e5 → OK (turn: white, legal_moves: 29)
Move 3: g1f3 → OK (turn: black, legal_moves: 29)
All moves executed successfully.
```

---

## 7. eval_special_moves.py

**Purpose**: Verify that all special chess moves generate the correct physical command sequence. Uses a **mock physical executor** — no MuJoCo needed.

**Usage**:
```bash
python scripts/eval_special_moves.py [--all | --castling | --en-passant | --promotion]
```

**What it tests**:

**Castling**:
- Kingside: `e1g1` → `ArmMove(king, e1, g1)` + `ArmMove(rook_h, h1, f1)`
- Queenside: `e1c1` → `ArmMove(king, e1, c1)` + `ArmMove(rook_a, a1, d1)`
- Black kingside/queenside variants

**En passant**:
- `e5d6` with black pawn on d5, en-passant square d6 → `Remove(black_pawn_d, slot_XX)` + `ArmMove(white_pawn_e, e5, d6)`
- Verifies the removed pawn is at d5 (not d6)

**Promotion**:
- `a7a8q` (queen) → `ArmMove(pawn_a, a7, a8)` + `Teleport(pawn_a, reserve)` + `Teleport(reserve_queen_1, square, a8)`
- Tests all promotion types: q, r, b, n
- Tests capture-promotion: `b7a8q` (pawn captures on promotion rank)

After running moves, verifies `LogicalPieceTracker` state:
- Promoted pawn is off-board (in promotion reserve zone, not on any square)
- Reserve queen is now at `a8`
- `piece_id_at("a8")` returns `white_reserve_queen_1`

**Expected output** (all passing):
```
Castling kingside: OK
Castling queenside: OK
En passant: OK
Promotion to queen: OK
Promotion to rook: OK
Promotion capture: OK
All special move tests passed.
```

---

## 8. eval_draw_conditions.py

**Purpose**: Verify that all chess draw conditions are correctly detected by `ChessService`. No MuJoCo needed.

**What it tests**:
- Stalemate (K vs K+Q, black to move into stalemate)
- Insufficient material (K vs K)
- 50-move rule (`can_claim_fifty_moves`)
- 75-move rule (`is_seventyfive_moves` — automatic)
- Threefold repetition (`can_claim_threefold_repetition`)
- Fivefold repetition (`is_fivefold_repetition` — automatic)

**Expected output** (all passing):
```
Stalemate: OK
Insufficient material: OK
50-move claim: OK
75-move auto: OK
Threefold claim: OK
Fivefold auto: OK
All draw conditions correctly detected.
```

---

## 9. eval_sequence.py

**Purpose**: Test the full three-stage arm sequence (transit → descend → ascend) at a specified position.

**Usage**:
```bash
python scripts/eval_sequence.py [--square e4] [--n-episodes 5] [--visualize] [--delay 0.02]
```

**What it does**:
1. Looks up the XY for the given square.
2. For each episode: transit to square, descend to hover, ascend back to safe Z.
3. Reports per-stage success rates and position errors.

---

## 10. Other Scripts

### `scripts/debug_one_move.py`

Interactive single-move debugger with detailed logging. Uses argparse:
```bash
python scripts/debug_one_move.py --piece white_pawn_e --src e2 --dst e4
```

Runs the full grasp/transit/place pipeline for one move and prints step-by-step stage results, finger joint values, and position errors. Used to diagnose specific failures.

### `scripts/analyze_debug_log.py`

Reads JSON debug logs from `logs/env_debug/` and prints statistics: stage success rates, mean step counts, crash reasons histogram.

### `scripts/verify_physics.py`

Loads the MuJoCo model, checks that all 32 piece freejoints are present, and verifies physics parameters (damping, geom sizes) against config values.

### `scripts/eval_grasp_physics.py`

Low-level test of finger physics during grasp. Drives fingers closed step by step and records joint values. Used to calibrate `grasp_verify_finger_threshold`.

### `scripts/visualize.py`

Open the MuJoCo viewer with the chess scene loaded. `visualize.py --setup` teleports all pieces to starting squares and shows the board.

---

## 11. Common Arguments

Most eval scripts accept these arguments via `src/utils/args.py:add_common_args()`:

| Argument | Default | Description |
|---|---|---|
| `--n-episodes` | varies | Number of episodes per test case |
| `--drift-limit` | `0.010` | Tube radius limit during descend/ascend (10mm) |
| `--visualize` | off | Open MuJoCo viewer during eval |
| `--delay` | `0.0` | Seconds between frames when visualizing |
| `--debug` | off | Enable verbose logging |

## Cleanup Refactor Evaluation Notes

`generate_scene.py` replaces the four older generator scripts. `visualize.py --setup` replaces the old setup-only viewer. `eval_grasp_physics.py` replaces the old pytest-collected grasp physics script name.

`eval_game_logic.py` exercises normal moves, captures, castling, en passant, and promotion through `GameOrchestrator` with `NoOpPhysicalExecutor`.

`eval_stages.py` defaults to integrated RL models. Use `--use-scripted-only` to opt into the scripted controller path.

`validate_deployed_models` checks that all checkpoint paths in `configs/deployed_models.yaml` exist before RL evaluation.
