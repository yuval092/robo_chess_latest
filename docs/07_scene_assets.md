# Scene & Assets

## Board Geometry

The chess board is an 8×8 grid of 8 cm squares centred at world XY `[0.88, 0.2641]`. The total playable area is 64 cm × 64 cm. The table surface is at Z = 0.400 m.

### BoardMapper (`chess_game/board_mapper.py`)

Converts between python-chess square indices and world-frame XY coordinates.

```python
square_to_xy(square)          # chess.Square → np.ndarray [x, y]
square_name_to_xy(name)       # "e4" → np.ndarray [x, y]
square_to_piece_xyz(square)   # → [x, y, table_surface_z + piece_height/2]
all_square_centers()          # → dict[str, np.ndarray]  (used in tests)
```

Origin is the lower-left corner of a1 (`board_min_xy = center_xy - board_width/2`). Each square centre is at `min_xy + (file + 0.5) * cell_size`, `min_xy + (rank + 0.5) * cell_size`.

Configuration:

| Key | Value | Description |
|---|---|---|
| `board.cell_size_m` | 0.08 | Width of one square in metres |
| `board.board_size` | 8 | Squares per side |
| `board.center_xy` | [0.88, 0.2641] | World XY of board centre |
| `table_surface_z` | 0.400 | Z of table surface |

---

## Chess Piece Bodies

Each active piece has a dedicated MuJoCo body with:
- A **free joint** (`piece_{piece_id}:joint`) — allows unconstrained 6-DOF motion.
- A **collision cube** (`piece_{piece_id}_cube`) — used for grasp contact.
- A **visual geom** (`piece_{piece_id}_visual`) — rendered appearance.

Piece height is 30 mm. Piece centre sits at `TABLE_SURFACE_Z + 0.015` when flat on the table.

### `PIECE_COM_GRIP_OFFSET`

When the gripper holds a piece, the piece centre of mass sits ~15 mm below the grip site (`PIECE_COM_GRIP_OFFSET = 0.015`). This constant is used in:
- `_check_piece_held` — expected piece Z = `grip_z - 0.015`
- `_move_z` verify_held check — same offset

---

## Reserve Areas

Off-board storage areas for captured and promotion pieces, defined in `configs/chess.yaml`:

| Area | Color | Origin XYZ | Grid |
|---|---|---|---|
| Graveyard | White | [0.640, -0.300, 0.015] | 4×4 |
| Graveyard | Black | [0.640, 0.680, 0.015] | 4×4 |
| Promotion reserve | White | [0.820, -0.300, 0.015] | 8×4 |
| Promotion reserve | Black | [0.820, 0.680, 0.015] | 8×4 |

Slot spacing is 45 mm centre-to-centre. Slot IDs are `"slot_00"` through `"slot_NN"`. Row-major indexing: `row = slot // cols`, `col = slot % cols`.

---

## MuJoCo XML

The chess environment XML is at `chess_env/assets/pick_and_place.xml`. It extends the `gymnasium-robotics` Fetch Pick-and-Place model with:

- An 8×8 chess board texture and plane geometry.
- 32 active piece bodies (back rank + pawns for both colours).
- 64 reserve piece bodies (promotion + graveyard stock).
- One dummy `object0` freejoint body (required by the base Fetch environment, hidden during chess play).

The XML path is injected at runtime in `ChessSimulationEnv._load_chess_model` using a threading lock to handle parallel training environments safely.

---

## Coordinate Frame

All positions are in the MuJoCo world frame (metres):

- X: robot-forward axis (increasing toward the far edge of the board)
- Y: robot-lateral axis (increasing left when facing the board)
- Z: vertical (increasing upward)

The robot base is fixed. The Fetch torso is set to `torso_height = 0.370 m` for full-board arm reachability.
