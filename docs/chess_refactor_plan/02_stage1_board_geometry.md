# Stage 1: Board Geometry And Visual Table

## Goal

Turn the table surface itself into a chess board with 8cm x 8cm cells, while preserving the table as the physical support surface.

No separate board body may be placed on the table.

## Coordinate Model

Use a board-centered model:

- Board cell size: `0.08m`
- Board width: `0.64m`
- Table half extent: `0.35m`
- Board margin on table: `(0.70 - 0.64) / 2 = 0.03m`
- This board geometry is independent of `env.yaml.edge_margin`. The edge margin remains a sampling parameter for existing movement/evaluation scripts.

Recommended orientation:

- Rank axis is world X.
- File axis is world Y.
- White side is near the arm.
- White pieces start on lower X ranks.
- Black pieces start on higher X ranks.

With `center_xy = [0.88, 0.2641]`:

```text
board_min_x = 0.88 - 0.32 = 0.56
board_max_x = 0.88 + 0.32 = 1.20
board_min_y = 0.2641 - 0.32 = -0.0559
board_max_y = 0.2641 + 0.32 = 0.5841
```

Square center formula:

```python
rank_index = chess.square_rank(square)  # a1 rank index = 0
file_index = chess.square_file(square)  # a-file index = 0

x = board_min_x + (rank_index + 0.5) * cell_size
y = board_min_y + (file_index + 0.5) * cell_size
z = table_surface_z + cube_height / 2
```

This produces near/far rank centers:

```text
rank 1 center X = 0.600
rank 8 center X = 1.160
file a center Y = -0.0159
file h center Y = 0.5441
```

These differ from the current status-doc `eval_stress.py` grid centers around `0.609..1.151`, because that script divides the 62cm edge-margin sampling range into 8 cells. The chess board must use the exact 64cm geometry above.

This maps:

- `a1` to near-arm, right side of board.
- `h1` to near-arm, left side of board.
- `a8` to far-arm, right side.
- `h8` to far-arm, left side.

If UI orientation prefers `a1` bottom-left from White's view, handle that in UI rendering. Do not contort physical coordinates for UI convenience.

## Implementation Tasks

### 1.1 Add BoardMapper

Create `src/chess_game/board_mapper.py`.

Required API:

```python
from dataclasses import dataclass
import chess
import numpy as np

@dataclass(frozen=True)
class BoardGeometry:
    center_xy: tuple[float, float]
    cell_size_m: float
    board_size: int
    table_surface_z: float
    cube_height_m: float
    table_half_x: float
    table_half_y: float

class BoardMapper:
    def __init__(self, geometry: BoardGeometry): ...
    def square_to_xy(self, square: chess.Square) -> np.ndarray: ...
    def square_name_to_xy(self, square_name: str) -> np.ndarray: ...
    def square_to_piece_xyz(self, square: chess.Square) -> np.ndarray: ...
    def all_square_centers(self) -> dict[str, np.ndarray]: ...
    def nearest_square(self, xy: np.ndarray) -> chess.Square: ...
    def assert_on_board(self, xy: np.ndarray) -> None: ...
```

Requirements:

- Use `python-chess` square indices internally.
- Return world-frame coordinates in meters.
- Validate cell size exactly `0.08`.
- Validate generated centers are inside table half extents with margin.
- Validate the board width is exactly `board_size * cell_size_m == 0.64`.
- Validate the table margin is at least `0.03m` on all sides.
- Keep all math deterministic and unit-testable without MuJoCo.

### 1.2 Add Board Materials

Update `chess_env/assets/shared.xml` with materials:

```xml
<material name="wood_table_mat" rgba="0.50 0.30 0.14 1"/>
<material name="chess_light_square_mat" rgba="0.86 0.78 0.63 1"/>
<material name="chess_dark_square_mat" rgba="0.35 0.20 0.11 1"/>
```

Use colors that distinguish cells clearly and still leave the table looking like wood.

### 1.2b Generate Board XML With A Script

Do not write the 64 square visual geoms by hand. Create `scripts/generate_board_xml.py`.

The script must:

1. Load `configs/chess.yaml` and instantiate `BoardMapper`.
2. Compute all 64 square center world XY coordinates.
3. Convert each to the local frame of `table0` body:
   - `table0` is at world `pos="0.88 0.2641 0"`.
   - Local XY = world XY − [0.88, 0.2641].
   - Local Z = world Z (since table0.z = 0), so just above table top = `0.4005`.
4. Assign `chess_light_square_mat` or `chess_dark_square_mat` based on `(rank + file) % 2`.
5. Write a `<!-- generated board squares -->` XML fragment.

This ensures square centers are exactly consistent with `BoardMapper.square_to_xy`, not hand-calculated.

**Important coordinate note**: The `table0` body is at world `(0.88, 0.2641, 0)`. Geoms declared inside `<body name="table0">` use coordinates relative to this body origin. The plan's example `pos="-0.280 -0.280 0.4006"` is correct: local a1 center is `(0.600 − 0.88, −0.0159 − 0.2641) = (−0.28, −0.28)`, and local Z 0.4005 sits just above the table surface top at local Z 0.400.

### 1.3 Color Squares On The Table Surface

Preferred implementation:

- Keep the existing `table0_surface` geom as the physical collision surface.
- Add 64 ultra-thin visual-only geoms slightly above the table top.
- Each square geom must have `contype="0"` and `conaffinity="0"` so it does not create collision or behave like an embedded board.
- Thickness must be visually negligible, e.g. `size="0.040 0.040 0.0005"` and `pos="square_local_x square_local_y 0.4006"` inside `table0`.

Example:

```xml
<geom name="board_a1_visual" type="box" size="0.040 0.040 0.0005"
      pos="-0.280 -0.280 0.4006" material="chess_light_square_mat"
      contype="0" conaffinity="0" mass="0"/>
```

Because the square overlays are visual-only and part of `table0`, the table remains the functional board surface.

Alternative implementation:

- Use a texture applied to the table material.
- Only choose this if texture support is simpler in the existing MuJoCo asset setup.

### 1.4 Add Board Labels For Debug Only

Do not add visible labels in the production scene. If labels are needed for debugging, make them optional in a separate debug XML or script overlay.

## Piece Stability Prerequisite

The square visual geoms have `contype="0" conaffinity="0"` and are collision-free, so they cannot interact with piece physics. However, from Stage 2 onward, 32 pieces will sit as freejoints on the table simultaneously. Freejoint `damping="0.5"` (from `chess.yaml`) must be used for piece joints. Verify that pieces do not drift more than 1mm over 5 simulated seconds with no arm interaction. Add this as a `verify_physics.py` check in Stage 2, but note the damping requirement here so Stage 1 does not create pieces with insufficient damping.

## Tests

Add `tests/chess_game/test_board_mapper.py`:

- `test_square_centers_are_64_unique_points`
- `test_all_square_centers_inside_table`
- `test_cell_spacing_is_8cm`
- `test_board_does_not_use_env_edge_margin`
- `test_a1_h1_a8_h8_positions_match_orientation`
- `test_nearest_square_roundtrip`
- `test_near_and_far_rank_centers_match_exact_8cm_geometry`

Add XML checks to `scripts/verify_physics.py`:

- 64 visual square geoms exist.
- All square visual geoms have `contype=0` and `conaffinity=0`.
- Table collision geom remains a single physical support surface.
- Square visual geom centers match `BoardMapper` exact-8cm centers, not `eval_stress.py` sampling-grid centers.

## Validation

Run:

```bash
pytest tests/chess_game/test_board_mapper.py -v
python scripts/verify_physics.py
python scripts/eval_chess_reachability.py --all-squares --n-episodes 1
```

Pass criteria:

- Board mapper tests pass.
- The table displays alternating square colors.
- No piece or cube collides with square overlay geoms.
- Reachability is unchanged from Stage 0.
