# Stage 10: Board Reorientation — Arm Centered Between White and Black

## Goal

Reorient the chess board so the robot arm sits between the white and black sides. Currently the arm
at `x=0.56` sits at the near edge of white rank 1 (`x=0.600`), effectively placing the arm on
"white's side." After this stage the rank axis runs along **Y** and the file axis runs along **X**,
putting white pieces at low Y, black pieces at high Y, and the arm at `y=0.2641` — the lateral
center of the board — between both sides.

The arm base does **not** move. All changes are to the coordinate mapping and the XML positions of
the board squares and pieces.

## Geometry After Change

```
board_min_x = 0.88 - 0.32 = 0.56   (file a center: x=0.600)
board_max_x = 0.88 + 0.32 = 1.20   (file h center: x=1.160)
board_min_y = 0.2641 - 0.32 = -0.0559  (rank 1 center: y=-0.0159)
board_max_y = 0.2641 + 0.32 = 0.5841   (rank 8 center: y=0.5441)

Arm base: x=0.56, y=0.2641
Rank 4 center y: 0.2241    ← arm is between rank 4 and rank 5
Rank 5 center y: 0.3041

White side: ranks 1–2, y ∈ [-0.016, 0.064]
Black side: ranks 7–8, y ∈ [0.464, 0.544]
Arm y=0.2641 is centred between them.
```

Corner positions under the new mapping (file=X, rank=Y):

| Square | Old (rank=X, file=Y) | New (file=X, rank=Y) |
|--------|----------------------|----------------------|
| a1     | (0.600, −0.016)      | (0.600, −0.016)  ← same |
| h1     | (0.600, 0.544)       | (1.160, −0.016) |
| a8     | (1.160, −0.016)      | (0.600, 0.544) |
| h8     | (1.160, 0.544)       | (1.160, 0.544)   ← same |
| e1 (white king) | (0.600, 0.304) | (0.920, −0.016) |
| e8 (black king) | (1.160, 0.304) | (0.920, 0.544) |

## Files to Change

### 10.1 `configs/chess.yaml` — Orientation and Zone Positions

Change the orientation block from:

```yaml
orientation:
  white_side: near_arm
  file_axis: y
  rank_axis: x
  use_env_edge_margin_for_board: false
```

to:

```yaml
orientation:
  white_side: low_y
  file_axis: x
  rank_axis: y
  use_env_edge_margin_for_board: false
```

Also update the graveyard and promotion_reserve origin coordinates to match the new
white=low-Y / black=high-Y layout.  Replace the existing `graveyards:` and
`promotion_reserve:` blocks with:

```yaml
graveyards:
  white:
    origin_xyz: [0.640, -0.300, 0.015]   # floor, below white side (low Y)
    rows: 4
    cols: 4
  black:
    origin_xyz: [0.640, 0.680, 0.015]    # floor, above black side (high Y)
    rows: 4
    cols: 4

promotion_reserve:
  white:
    origin_xyz: [0.820, -0.300, 0.015]   # floor, same Y band as white graveyard
    rows: 8
    cols: 4
  black:
    origin_xyz: [0.820, 0.680, 0.015]    # floor, same Y band as black graveyard
    rows: 8
    cols: 4
```

Slot layout reminder: row increments along X (dimension 0), col increments along Y
(dimension 1). The `_slot_xyz` formula in `piece_teleport.py` is:
`origin + [row * spacing, col * spacing, 0]`.

White graveyard occupies x=[0.640, 0.775], y=[-0.300, -0.165].  
Black graveyard occupies x=[0.640, 0.775], y=[0.680, 0.815].  
White promotion reserve occupies x=[0.820, 1.135], y=[-0.300, -0.165].  
Black promotion reserve occupies x=[0.820, 1.135], y=[0.680, 0.815].

All four zones are outside the table footprint (table y: −0.086 → 0.614) and on the
floor (z=0.015 = piece cube half-height, so piece centres sit at z=0.030).

### 10.2 `src/chess_game/board_mapper.py` — `square_to_xy`

Replace the body of `square_to_xy`:

```python
# Before
def square_to_xy(self, square: chess.Square) -> np.ndarray:
    rank_index = chess.square_rank(square)
    file_index = chess.square_file(square)
    min_xy = self.board_min_xy
    return np.array(
        [
            min_xy[0] + (rank_index + 0.5) * self.geometry.cell_size_m,
            min_xy[1] + (file_index + 0.5) * self.geometry.cell_size_m,
        ],
        dtype=float,
    )

# After
def square_to_xy(self, square: chess.Square) -> np.ndarray:
    rank_index = chess.square_rank(square)
    file_index = chess.square_file(square)
    min_xy = self.board_min_xy
    return np.array(
        [
            min_xy[0] + (file_index + 0.5) * self.geometry.cell_size_m,   # X = file
            min_xy[1] + (rank_index + 0.5) * self.geometry.cell_size_m,   # Y = rank
        ],
        dtype=float,
    )
```

No other changes to `board_mapper.py` are needed. `board_min_xy`, `board_max_xy`,
`assert_on_board`, `nearest_square`, and all validation logic remain correct because
the board is still square and centred at the same point.

`nearest_square` currently decomposes the XY offset as:
```python
offset = (xy[:2] - min_xy) / self.geometry.cell_size_m
rank_index = int(math.floor(offset[0]))
file_index = int(math.floor(offset[1]))
```

This must be updated to match the new axis mapping:

```python
# After
offset = (xy[:2] - min_xy) / self.geometry.cell_size_m
file_index = min(max(int(math.floor(offset[0])), 0), self.geometry.board_size - 1)
rank_index = min(max(int(math.floor(offset[1])), 0), self.geometry.board_size - 1)
return chess.square(file_index, rank_index)
```

### 10.3 `tests/chess_game/test_board_mapper.py` — Update Hardcoded Expectations

`test_cell_spacing_is_8cm` currently asserts:
```python
assert np.isclose(mapper.square_name_to_xy("a2")[0] - mapper.square_name_to_xy("a1")[0], 0.08)  # rank changes X
assert np.isclose(mapper.square_name_to_xy("b1")[1] - mapper.square_name_to_xy("a1")[1], 0.08)  # file changes Y
```

With the new mapping these are both 0. Replace with:
```python
# rank increases along Y
assert np.isclose(mapper.square_name_to_xy("a2")[1] - mapper.square_name_to_xy("a1")[1], 0.08)
# file increases along X
assert np.isclose(mapper.square_name_to_xy("b1")[0] - mapper.square_name_to_xy("a1")[0], 0.08)
```

`test_a1_h1_a8_h8_positions_match_orientation` currently asserts:
```python
assert np.allclose(mapper.square_name_to_xy("a1"), [0.600, -0.0159])   # unchanged
assert np.allclose(mapper.square_name_to_xy("h1"), [0.600,  0.5441])   # changes
assert np.allclose(mapper.square_name_to_xy("a8"), [1.160, -0.0159])   # changes
assert np.allclose(mapper.square_name_to_xy("h8"), [1.160,  0.5441])   # unchanged
```

Replace with:
```python
assert np.allclose(mapper.square_name_to_xy("a1"), [0.600, -0.0159])
assert np.allclose(mapper.square_name_to_xy("h1"), [1.160, -0.0159])
assert np.allclose(mapper.square_name_to_xy("a8"), [0.600,  0.5441])
assert np.allclose(mapper.square_name_to_xy("h8"), [1.160,  0.5441])
```

`test_near_and_far_rank_centers_match_exact_8cm_geometry` currently checks X for d1/d8
(rank positions along X). With ranks along Y:

```python
# Before
assert np.isclose(mapper.square_name_to_xy("d1")[0], 0.600)
assert np.isclose(mapper.square_name_to_xy("d8")[0], 1.160)

# After
assert np.isclose(mapper.square_name_to_xy("d1")[1], -0.0159)   # rank 1 Y
assert np.isclose(mapper.square_name_to_xy("d8")[1],  0.5441)   # rank 8 Y
```

### 10.4 `scripts/eval_chess_reachability.py` — Update Geometry Validation

The `validate_rank_centers` function checks that rank-1 and rank-8 have the correct **X**
coordinate. After reorientation, ranks are identified by their **Y** coordinate. Rename
and rewrite the function:

```python
REQUIRED_RANK1_Y = -0.0159
REQUIRED_RANK8_Y =  0.5441
REQUIRED_FILE_A_X = 0.600
REQUIRED_FILE_H_X = 1.160
GEOMETRY_TOLERANCE_M = 1e-9

def validate_board_geometry(mapper: BoardMapper) -> None:
    rank1_y = mapper.square_name_to_xy("a1")[1]
    rank8_y = mapper.square_name_to_xy("a8")[1]
    file_a_x = mapper.square_name_to_xy("a1")[0]
    file_h_x = mapper.square_name_to_xy("h1")[0]
    print(f"Board geometry: rank1_y={rank1_y:.4f} rank8_y={rank8_y:.4f} "
          f"fileA_x={file_a_x:.3f} fileH_x={file_h_x:.3f}")
    if not math.isclose(rank1_y, REQUIRED_RANK1_Y, abs_tol=GEOMETRY_TOLERANCE_M):
        raise SystemExit(f"rank1 center Y must be {REQUIRED_RANK1_Y:.4f}m, got {rank1_y:.6f}m")
    if not math.isclose(rank8_y, REQUIRED_RANK8_Y, abs_tol=GEOMETRY_TOLERANCE_M):
        raise SystemExit(f"rank8 center Y must be {REQUIRED_RANK8_Y:.4f}m, got {rank8_y:.6f}m")
    if not math.isclose(file_a_x, REQUIRED_FILE_A_X, abs_tol=GEOMETRY_TOLERANCE_M):
        raise SystemExit(f"file-a center X must be {REQUIRED_FILE_A_X:.3f}m, got {file_a_x:.6f}m")
    if not math.isclose(file_h_x, REQUIRED_FILE_H_X, abs_tol=GEOMETRY_TOLERANCE_M):
        raise SystemExit(f"file-h center X must be {REQUIRED_FILE_H_X:.3f}m, got {file_h_x:.6f}m")
```

Update the call site from `validate_rank_centers(mapper)` to `validate_board_geometry(mapper)`.

### 10.5 Regenerate Board Squares

After changing `board_mapper.py`, re-run the board square generator. This rewrites all 64
square visual geoms in `pick_and_place.xml` with the new positions:

```bash
python scripts/generate_board_xml.py --write
```

Verify the first few lines of the generated fragment: rank 1 squares should all share
`y=-0.0159` (in table-local coordinates: `pos="-0.2800 ... 0.4006"` for file a through h
at rank 1).

Local coordinate reminder: table body is at world `(0.88, 0.2641, 0)`. The script
computes `local_x = world_x - 0.88`, `local_y = world_y - 0.2641`. For rank 1 / file a:
```
world_x = board_min_x + (file_a + 0.5)*0.08 = 0.56 + 0.04 = 0.60 → local_x = -0.28
world_y = board_min_y + (rank1  + 0.5)*0.08 = -0.0559 + 0.04 = -0.0159 → local_y = -0.28
```

All rank-1 squares will have `local_y = -0.28` (constant), varying `local_x` from
`-0.28` (file a) to `+0.28` (file h). This is the expected output after the script runs.

### 10.6 Regenerate Chess Piece Starting Positions

After changing `board_mapper.py`, re-run the piece XML generator. This writes all 32
active piece bodies and 64 reserve bodies to `pick_and_place.xml` at the correct new
coordinates:

```bash
python scripts/generate_pieces_xml.py --write
```

Verify spot-checks:
- `piece_white_king` body: should now be at pos `"0.9200 -0.0159 0.4150"` (e1, file=e=4, rank=1)
- `piece_black_king` body: should now be at pos `"0.9200 0.5441 0.4150"` (e8, file=e=4, rank=8)
- `piece_white_pawn_a` body: should be at pos `"0.6000 0.0641 0.4150"` (a2, file=a=0, rank=2)
- `piece_black_pawn_h` body: should be at pos `"1.1600 0.4641 0.4150"` (h7, file=h=7, rank=7)

White back rank (rank 1, y=-0.0159) should span from x=0.600 (rook a) to x=1.160 (rook h).
Black back rank (rank 8, y=0.5441) should span from x=0.600 (rook a) to x=1.160 (rook h).

### 10.7 `scripts/eval_chess_piece_move.py` — Check for Hardcoded Coordinates

Audit this script for any hardcoded X/Y positions that assume the old orientation. If any
exist, replace them with `mapper.square_name_to_xy(square_name)` calls.

### 10.8 `scripts/eval_chess_game_flow.py` — Update Smoke Test Comment

The comment in `DEFAULT_MOVES` explaining why e2e3/e7e6 were chosen instead of e2e4/e7e5
referenced crowded center squares. After reorientation, check whether the ROTATION_FAILED
risk still applies to the same sequence. The underlying kinematic limitation (arm cannot
verticalize at certain crowded positions) is independent of board orientation; only the
physical locations of the squares change. Update the comment to reference coordinates
rather than the old layout description.

## Validation

Run all checks in order. Do not proceed to the next check until the current one passes.

```bash
# 1. Unit tests — board_mapper, move_planner, orchestrator, UI
pytest tests/ -v --ignore=tests/integration -q

# 2. Verify correct geometry (a1, h1, a8, h8 corner positions)
python -c "
import sys; sys.path.insert(0,'.')
from src.chess_game.board_mapper import BoardMapper
import numpy as np
m = BoardMapper.from_configs()
assert np.allclose(m.square_name_to_xy('a1'), [0.600, -0.0159]), m.square_name_to_xy('a1')
assert np.allclose(m.square_name_to_xy('h1'), [1.160, -0.0159]), m.square_name_to_xy('h1')
assert np.allclose(m.square_name_to_xy('a8'), [0.600,  0.5441]), m.square_name_to_xy('a8')
assert np.allclose(m.square_name_to_xy('h8'), [1.160,  0.5441]), m.square_name_to_xy('h8')
print('Board geometry OK')
"

# 3. Verify XML was regenerated with correct positions
python -c "
text = open('chess_env/assets/pick_and_place.xml').read()
assert '0.9200 -0.0159' in text, 'white_king e1 position missing'
assert '0.9200 0.5441' in text, 'black_king e8 position missing'
print('XML positions OK')
"

# 4. Mock game flow
python scripts/eval_chess_game_flow.py --verify-agreement

# 5. Mock special moves
python scripts/eval_special_moves.py --all

# 6. Draw conditions
python scripts/eval_draw_conditions.py

# 7. Physics verification (no arm movement changes — should stay healthy)
python scripts/verify_physics.py

# 8. Exact chess reachability — CRITICAL
#    Validates all 64 new square centers under real arm physics.
#    Run this before calling Stage 10 complete.
python scripts/eval_chess_reachability.py --all-squares --n-episodes 1
```

## Stop Conditions

Do not proceed to Stage 11 if:

- Any of the 64 new square centers fails the reachability sweep. If squares fail, see the
  remediation checklist below.
- `piece_white_king` and `piece_black_king` are not at their respective new positions after
  running `generate_pieces_xml.py --write`.
- Any unit test fails after the board_mapper change.
- `test_nearest_square_roundtrip` fails (this verifies the inverse mapping is also correct).

## Reachability Remediation Checklist

If the reachability sweep fails after reorientation:

1. Check which squares fail. If failures are clustered at rank-1 or rank-8 Y extremes
   (`y=-0.016` or `y=0.544`), the arm's lateral reach is the constraint.
2. Try shifting `board.center_xy` in `chess.yaml` slightly toward `y=0.2641` on the failing
   side — this moves all squares but keeps the board centred on the arm.
3. Do NOT reduce `cell_size_m` below 0.08.
4. Do NOT move the arm base.
5. If failures occur at file-a (`x=0.600`) or file-h (`x=1.160`) columns, the issue is the
   same near-arm X limit that existed before; check `torso_height` is still 0.3661.

## Note on `arm_home_xy`

The current value `arm_home_xy: [0.680, 0.2641]` remains valid. In the new orientation,
x=0.680 is near file b (center x=0.680 = board_min_x + 1.5*0.08), and y=0.2641 is
exactly the arm's Y centre — between rank 4 and rank 5. The arm parks between the two
sides of the board while not obstructing either player's pieces.
