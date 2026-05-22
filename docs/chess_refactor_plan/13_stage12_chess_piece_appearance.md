# Stage 12: Chess Piece Visual Appearance

## Goal

Verify and fix the visual appearance of chess pieces in the MuJoCo scene so that each piece
type is visually distinct and knights face their opponents. The cube collision bodies and
freejoint physics are unchanged; this stage touches only visual geoms, mesh declarations,
materials, and generation script logic.

## Current State

Discovered facts from reading the STL files:

| Piece  | Triangles | Height   | XY radius (approx) |
|--------|-----------|----------|---------------------|
| pawn   | 64        | 18.0 mm  | ±6.0 mm             |
| rook   | 64        | 20.0 mm  | ±7.0 mm             |
| knight | 20        | 21.0 mm  | asymmetric: x [−4.9, +6.0], y [±5.7] |
| bishop | 72        | 24.0 mm  | ±5.0 mm             |
| queen  | 23.0 mm   | 23.0 mm  | ±6.5 mm             |
| king   | 40        | 26.0 mm  | ±6.0 mm             |

All STL files are already in **meters**. The `visual_stl_scale: 0.020` entry in
`chess.yaml` was never applied to any geom or mesh declaration and is inconsistent with
the actual file sizes; it must be corrected.

Visual geoms are placed at `pos="0 0 0.017"` (17 mm above cube centre = just above the
cube top at ±15 mm). The tallest piece (king, 26 mm) reaches 0.415 + 0.017 + 0.026 =
0.458 m, which is 2 mm below `HOVER_Z = 0.460 m`. The STLs fit between the cube top and
the safe hover height.

Current generation code in `scripts/generate_pieces_xml.py` assigns:
- `white_piece_visual_mat` to all white pieces
- `black_piece_visual_mat` to all black pieces

There are no per-type materials. The pieces are distinguished by shape (STL mesh) only.

## Issues to Fix

### Issue 1: `visual_stl_scale` in `chess.yaml` is wrong and unused

The config says `visual_stl_scale: 0.020` but the STLs are already in meters and are
correctly sized without any scale factor. This key must be removed to avoid confusing
future implementers into adding a scale that would distort the pieces.

**Fix:** Remove the `visual_stl_scale` key from `configs/chess.yaml`.

### Issue 2: Knight STL has only 20 triangles — shape is very rough

20 triangles produce an angular, low-quality shape. All other pieces have at least 40
triangles. The knight STL must be replaced with a higher-quality mesh.

**Fix:** Replace `chess_env/stls/chess/knight.stl` with a mesh having at least 80
triangles. The new mesh must:
- Remain in meters
- Have z ranging from 0.0 (base) to ~0.021 m (height matches current 21 mm)
- Have XY extents within ±8 mm of centre so it clears the cube edges (cube half = 15 mm)
- Include a clear directional front face — the knight must "look" in the +X direction in
  its local frame

See the generation script section below for how the rotation is applied per color.

### Issue 3: Knight orientation not applied per color

The knight STL is asymmetric in X, meaning it has a natural "facing" direction.
White knights (on rank 1, at low Y) should face toward black (+Y direction).
Black knights (on rank 7–8, at high Y) should face toward white (−Y direction).
Currently no `euler` attribute is applied to knight visual geoms; all knights render in
the STL's default XY orientation.

**Fix:** Update `scripts/generate_pieces_xml.py` to add `euler="0 0 90"` for white
knights and `euler="0 0 270"` for black knights (rotate the natural +X-facing STL to
point in +Y for white and −Y for black).

### Issue 4: Pieces of the same color are visually identical (no per-type material)

All white pieces use `white_piece_visual_mat` (cream). This is intentional and acceptable
for the cube body, but the STL visual could benefit from a subtle colour variation that
helps distinguish piece types in the rendered view.

**Decision for this stage:** Keep per-color materials for the STL. Adding 12 per-type
materials (6 types × 2 colors) would clutter `shared.xml` with no gameplay benefit; the
STL shapes themselves provide sufficient distinction. This issue is documented here as a
deliberate omission and can be revisited only if UI user testing shows confusion.

## Implementation Tasks

### 12.1 Remove `visual_stl_scale` from `configs/chess.yaml`

Delete the line:
```yaml
  visual_stl_scale: 0.020
```

No other code references this key. Verify after deletion:

```bash
grep -r "visual_stl_scale" src/ scripts/ tests/
# Expected: no output
```

### 12.2 Replace Knight STL

Generate or source a replacement `knight.stl` with:
- At least 80 triangles
- All vertices in meters
- Z from 0.0 to ~0.021 m
- Natural facing direction: +X (the horse head points toward +X in local frame)

A parametric approach for a stylised chess knight (the shape need not be anatomically
realistic — a clear profile with a stepped base and a protruding head is sufficient):

The knight can be approximated as a stacked-cylinder body with an offset head:
- Base cylinder: radius 6 mm, height 12 mm (z=0 to 0.012)
- Neck: radius 3 mm, height 5 mm (z=0.012 to 0.017), centred at (0,0)
- Head: a box 8 mm × 6 mm × 4 mm (z=0.017 to 0.021), offset +2 mm in X from centre

Provide a Python generation script `scripts/generate_knight_stl.py` or create the STL
manually with Blender/FreeCAD. The exact shape is left to the implementer provided the
constraints above are met.

After replacing the file, run:
```bash
python -c "
import struct
with open('chess_env/stls/chess/knight.stl','rb') as f:
    f.read(80)
    n = struct.unpack('<I', f.read(4))[0]
print(f'Triangle count: {n}')
assert n >= 80, f'Need at least 80 triangles, got {n}'
print('OK')
"
```

### 12.3 Update `scripts/generate_pieces_xml.py` — Knight Rotation

Modify `piece_body_xml` to accept an optional `euler` argument for the visual geom.
Then in the calling code (inside `build_fragment`), pass the appropriate euler for knights:

```python
def piece_body_xml(
    piece_id: str,
    color: str,
    piece_type: str,
    xyz,
    *,
    initial_square: str | None = None,
    visual_euler: str | None = None,
) -> list[str]:
    body_name = f"piece_{piece_id}"
    material_prefix = "white" if color == "white" else "black"
    damping = load_config("chess")["pieces"]["freejoint_damping"]
    comment = f" <!-- {initial_square} -->" if initial_square else ""
    euler_attr = f' euler="{visual_euler}"' if visual_euler else ""
    return [
        f'\t\t<body name="{body_name}" pos="{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}">{comment}',
        f'\t\t\t<joint name="{body_name}:joint" type="free" damping="{damping}"/>',
        f'\t\t\t<geom name="{body_name}_cube" type="box" size="0.015 0.015 0.015"',
        f'\t\t\t      material="{material_prefix}_piece_cube_mat" mass="0.05" condim="6"',
        f'\t\t\t      friction="2.0 0.005 0.0001" solref="0.002 1" solimp="0.99 0.999 0.001"/>',
        f'\t\t\t<geom name="{body_name}_visual" type="mesh" mesh="chess_{piece_type}_mesh"',
        f'\t\t\t      pos="0 0 0.017" material="{material_prefix}_piece_visual_mat"',
        f'\t\t\t      contype="0" conaffinity="0" mass="0"{euler_attr}/>',
        f'\t\t\t<site name="{body_name}_site" pos="0 0 0" size="0.005"/>',
        "\t\t</body>",
    ]
```

In `build_fragment`, apply rotation for knights:
```python
for piece in registry.all_pieces():
    xyz = mapper.square_to_piece_xyz(chess.parse_square(piece.initial_square))
    visual_euler = None
    if piece.piece_type == "knight":
        # White knight faces +Y (toward black side); black knight faces -Y (toward white side)
        # Knight STL natural facing is +X; rotating 90° around Z gives +Y facing
        visual_euler = "0 0 90" if piece.color == "white" else "0 0 270"
    lines.extend(piece_body_xml(
        piece.piece_id, piece.color, piece.piece_type, xyz,
        initial_square=piece.initial_square,
        visual_euler=visual_euler,
    ))
```

Apply the same logic for reserve knights (passed through `reserve_piece_ids` loop).

After updating the script, re-run:
```bash
python scripts/generate_pieces_xml.py --write
```

Verify the XML contains the euler attribute for knights:
```bash
grep "white_knight.*euler\|euler.*knight" chess_env/assets/pick_and_place.xml | head -4
```

### 12.4 Visual Verification

Launch the scene and visually confirm:

1. White pieces are cream-coloured; black pieces are very dark.
2. Each piece type has a distinct silhouette (pawn smallest, king tallest with cross).
3. White knights face toward the black side; black knights face toward the white side.
4. All pieces sit cleanly on the cube top with no clipping through the table.
5. The STL visuals do not visually collide with pieces on adjacent squares.

```bash
python -c "
import gymnasium as gym, src.chess_env
env = gym.make('ChessFetchTask-v0', render_mode='human', show_chess_pieces=True)
env.reset()
input('Inspect piece visuals. Press Enter to continue.')
env.close()
"
```

Acceptance checklist:
- [ ] Pawn (rank 2) is the shortest piece — approximately 18 mm above cube top
- [ ] King (back rank) is the tallest piece — approximately 26 mm above cube top
- [ ] Knight's face/head points toward the opposing player
- [ ] No two piece types look identical in silhouette
- [ ] White and black pieces are clearly different colours
- [ ] Reserve pieces (graveyard area) also show the correct STL for each type

### 12.5 Update `tests/physical/test_piece_xml.py`

The test `test_piece_joint_cube_and_visual_properties` currently checks cube/visual geom
properties but not that each piece type uses the correct mesh. Add a check:

```python
def test_each_piece_type_uses_correct_mesh():
    env = gym.make("ChessFetchTask-v0", render_mode=None, show_chess_pieces=True)
    env.reset()
    uw = env.unwrapped
    registry = PieceRegistry()

    type_to_mesh = {
        "pawn":   "chess_pawn_mesh",
        "rook":   "chess_rook_mesh",
        "knight": "chess_knight_mesh",
        "bishop": "chess_bishop_mesh",
        "queen":  "chess_queen_mesh",
        "king":   "chess_king_mesh",
    }

    for piece in registry.all_pieces():
        expected_mesh = type_to_mesh[piece.piece_type]
        visual_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_GEOM, piece.visual_geom_name)
        mesh_id = uw.model.geom_dataid[visual_id]
        actual_mesh_name = mujoco.mj_id2name(uw.model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
        assert actual_mesh_name == expected_mesh, (
            f"{piece.piece_id}: expected mesh {expected_mesh}, got {actual_mesh_name}"
        )
    env.close()
```

### 12.6 Update `configs/chess.yaml` Documentation Comment

After removing `visual_stl_scale`, add a comment explaining the STL convention so future
contributors don't re-add a spurious scale:

```yaml
pieces:
  cube_half_extent_m: 0.015    # Physics collision body half-size
  cube_height_m: 0.030         # Full cube height; piece centre is at table_z + 0.015
  # STL files in chess_env/stls/chess/ are already in metres.
  # Visual geoms use pos="0 0 0.017" to sit on top of the cube (just above cube top at +0.015).
  # Piece heights: pawn 18mm, rook 20mm, knight 21mm, bishop 24mm, queen 23mm, king 26mm.
  # No mesh scale attribute is needed.
  freejoint_damping: 0.5
```

## Validation

```bash
# 1. All unit tests pass (includes mesh-type check added in 12.5)
pytest tests/ -v --ignore=tests/integration -q

# 2. visual_stl_scale is gone
grep -r "visual_stl_scale" . --include="*.py" --include="*.yaml" --include="*.xml"
# Expected: no output

# 3. Knight STL has >= 80 triangles
python -c "
import struct
with open('chess_env/stls/chess/knight.stl','rb') as f:
    f.read(80); n = struct.unpack('<I',f.read(4))[0]
assert n >= 80, n
print(f'Knight triangles: {n} OK')
"

# 4. Knights in XML have euler attribute
grep -c 'euler="0 0 90"\|euler="0 0 270"' chess_env/assets/pick_and_place.xml
# Expected: 4  (2 white knights + 2 black knights)

# 5. Eval scripts still pass after XML regeneration
python scripts/eval_special_moves.py --all
python scripts/eval_chess_game_flow.py --verify-agreement
```

## Stop Conditions

Do not call Stage 12 complete if:

- Any unit test fails.
- `visual_stl_scale` still appears anywhere in the codebase.
- Knight STL has fewer than 80 triangles.
- White and black knights do not face opposite directions in the rendered scene.
- Any piece's visual geom visually intersects the table surface (check that all z
  positions are at or above `table_surface_z + cube_height_m / 2 + 0.017 = 0.432 m`).
