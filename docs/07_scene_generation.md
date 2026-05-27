# Scene Generation

## Overview

The MuJoCo scene XML is not hand-edited for board layout, piece placement, or zone geometry. These sections are procedurally generated from the YAML configs and injected into the XML file at startup. The STL mesh files for chess pieces are also procedurally generated from parametric geometry functions.

---

## `environment_generation.py` (`src/chess_env/environment_generation.py`)

### Auto-Generation at Startup

`ensure_environment_generated()` is called from `ChessSimulationEnv.__init__`:

```python
def ensure_environment_generated(scene_path=DEFAULT_SCENE_PATH):
    before = scene_path.read_text()
    after = _replace_marked_fragment(before, BOARD_START_MARKER, BOARD_END_MARKER, build_board_fragment())
    after = _replace_marked_fragment(after, ZONES_START_MARKER, ZONES_END_MARKER, build_zones_fragment())
    after = _replace_marked_fragment(after, PIECES_START_MARKER, PIECES_END_MARKER, build_pieces_fragment())
    if after != before:
        scene_path.write_text(after)
```

If the generated content matches what's already in the file (config unchanged), the file is not rewritten. This means startup is fast in normal operation.

### Marker Pairs

The scene XML contains paired comment markers delimiting each auto-generated section:

| Section | Start marker | End marker |
|---|---|---|
| Board squares | `<!-- generated board squares start -->` | `<!-- generated board squares end -->` |
| Chess pieces | `<!-- generated chess pieces start -->` | `<!-- generated chess pieces end -->` |
| Zone markers | `<!-- generated zone markers start -->` | `<!-- generated zone markers end -->` |

`_replace_marked_fragment(text, start, end, fragment)` finds the markers and replaces everything between (and including) them with the new fragment.

---

## XML Fragment Builders

### `build_board_fragment()` — 64 Board Square Geoms

```python
for square in chess.SQUARES:
    name = chess.square_name(square)          # "a1", "b2", ...
    xy = mapper.square_to_xy(square)          # world XY centre
    local_x = xy[0] - table_cx               # relative to table centre
    local_y = xy[1] - table_cy
    material = "chess_dark_square_mat" if (rank + file) % 2 == 0 else "chess_light_square_mat"
    lines.append(
        f'<geom name="board_{name}_visual" type="box" '
        f'size="{square_half:.3f} {square_half:.3f} 0.0005" '
        f'pos="{local_x:.4f} {local_y:.4f} {z:.4f}" '
        f'material="{material}" contype="0" conaffinity="0" mass="0"/>'
    )
```

All board geoms use `contype="0" conaffinity="0"` — they are purely visual and do not participate in collision detection.

### `build_pieces_fragment()` — 32+64 Piece Bodies

For each active piece in `PieceRegistry().all_pieces()` (32 active pieces):

```xml
<body name="piece_white_pawn_e" pos="X Y Z">  <!-- e2 -->
    <joint name="piece_white_pawn_e:joint" type="free" damping="8.0"/>
    <geom name="piece_white_pawn_e_cube" type="box" size="0.015 0.015 0.015"
          material="white_piece_cube_mat" rgba="0 0 0 0" mass="0.35" condim="6"
          friction="2.0 0.005 0.0001" solref="0.002 1" solimp="0.99 0.999 0.001"/>
    <geom name="piece_white_pawn_e_visual" type="mesh" mesh="chess_pawn_mesh"
          pos="0 0 -0.016" material="white_piece_visual_mat"
          contype="0" conaffinity="0" mass="0"/>
    <site name="piece_white_pawn_e_site" pos="0 0 0" size="0.005"/>
</body>
```

Key design decisions:
- The **cube geom** is the physics body: 30mm cube, mass 0.35 kg, high friction (`2.0`), tight soft contacts (`solimp="0.99 0.999 0.001"`). This is what the gripper actually grasps.
- The **visual mesh geom** sits `pos="0 0 -0.016"` below the cube centre. It has `contype="0" conaffinity="0"` (no collision).
- The **free joint** `damping="8.0"` prevents pieces from sliding or tumbling under small perturbations.
- `rgba="0 0 0 0"` makes the cube geom fully transparent — only the mesh is visible.

For the 64 reserve pieces from `reserve_piece_ids()` (promotion), the same structure is used but placed at off-board reserve positions.

Knights are given an `euler` rotation attribute to face the correct direction:
```python
def knight_visual_euler(color):
    return "0 0 1.5707963268" if color == "white" else "0 0 4.7123889804"
```
White knights face forward (+X); black knights face backward (−X, rotated 270°).

### `build_zones_fragment()` — Graveyard and Reserve Visuals

Four coloured rectangular box geoms mark the off-board zones:

```python
ZONE_SPECS = [
    ("white_graveyard", "graveyards", "white", "white_graveyard_mat"),
    ("black_graveyard", "graveyards", "black", "black_graveyard_mat"),
    ("white_reserve",   "promotion_reserve", "white", "white_reserve_mat"),
    ("black_reserve",   "promotion_reserve", "black", "black_reserve_mat"),
]
```

Each zone geom spans the full grid area with a 5mm margin.

---

## STL Mesh Generation

### `regenerate_stls(out_dir)` — Entry Point

```python
for name, builder in PIECE_BUILDERS.items():
    write_binary_stl(out_dir / f"{name}.stl", name, builder())
```

`PIECE_BUILDERS = {"pawn": pawn_triangles, "rook": rook_triangles, ...}`

Meshes are written to `chess_env/stls/chess/`. They are pre-committed to the repository; regeneration is only needed if geometry constants change.

### Geometry Primitives

All pieces share a common **base** and are differentiated by their upper section.

#### `piece_base()` — Shared Base Geometry

A tapered cylindrical base built from frustum sections:

```
z=0.000–0.003: flare up (19mm→22mm radius)
z=0.003–0.006: flat rim (22mm)
z=0.006–0.009: taper (22→18mm)
z=0.009–0.024: body taper (18→13mm)
z=0.024–0.026: bulge (13→15mm)
z=0.026–0.029: taper (15→12mm)
z=0.029–0.032: neck taper (12→10mm)
```

The base is identical for all six piece types.

#### Geometry Primitive Functions

```python
frustum_triangles(r_bottom, r_top, z_bottom, z_top, segments) → list[triangle]
cone_triangles(r_base, z_base, z_tip, segments) → list[triangle]
box_triangles(center, size) → list[triangle]   # axis-aligned box
sphere_triangles(center, radius, rings, segments, z_scale) → list[triangle]
offset_tris(tris, dx, dy, dz) → list[triangle]  # translate a triangle list
normal(a, b, c) → (nx, ny, nz)                  # face normal
```

All geometry is defined in metres relative to the piece origin at base centre (z=0).

#### Per-Piece Upper Sections

| Piece | Upper geometry |
|---|---|
| Pawn | Narrow neck → collar → sphere head (radius 11.5mm, z-scale 0.9) |
| Rook | Shaft → collar → parapet cylinder → 4 cardinal + 4 corner battlement boxes |
| Knight | Shaft → body block → neck/snout block → ears → nostril boxes → mane strip |
| Bishop | Shaft → body bell → shoulder → sphere head → cross notch → finial cone |
| Queen | Shaft → body bell → shoulder → neck → 5 crown spikes (cones) → central orb |
| King | Shaft → body bell → shoulder → spire → vertical cross bar → horizontal bar + top |

All dimensions are defined as named constants with a piece prefix (e.g., `_PAWN_HEAD_R`, `_ROOK_BATTLEMENT_Z`). Changing any constant requires re-running `robo-chess-generate stls`.

#### `write_binary_stl(path, name, triangles)`

Writes a binary STL file with:
- 80-byte ASCII header: `"RoboChess {name} visual mesh"` padded to 80 bytes
- 4-byte triangle count
- Per-triangle: 3×3 floats (normal) + 3×3×3 floats (vertices) + 2-byte attribute (`0`)

---

## Manual Regeneration

The `robo-chess-generate` CLI (`src/cli/generate_scene.py`) provides sub-commands:

```bash
# Print generated XML fragments
robo-chess-generate board
robo-chess-generate pieces
robo-chess-generate zones

# Write generated XML fragments into chess_env/assets/pick_and_place.xml
robo-chess-generate board --write
robo-chess-generate pieces --write
robo-chess-generate zones --write

# Regenerate STL meshes only
robo-chess-generate stls

# Regenerate XML fragments only
robo-chess-generate all
```

When to regenerate:
- **Scene**: after changing `configs/chess.yaml` (board geometry, graveyard layout, piece config)
- **STLs**: after changing STL geometry constants in `environment_generation.py`
- Automatic at startup: `regenerate_environment()` validates/regenerates scene fragments only; STLs are not regenerated unless `regenerate_environment(include_stls=True)` or `robo-chess-generate stls` is used.

---

## Scene File Structure (`chess_env/assets/pick_and_place.xml`)

The scene XML inherits from the Fetch robot definition and adds:

1. **Materials**: `chess_light_square_mat`, `chess_dark_square_mat`, `white_piece_cube_mat`, `black_piece_cube_mat`, `white_piece_visual_mat`, `black_piece_visual_mat`, graveyard/reserve zone materials
2. **Meshes**: 6 chess piece STL meshes loaded from `chess_env/stls/chess/`
3. **Table**: A flat 70cm × 70cm box at z=0.400m
4. **Auto-generated sections**: Board squares, chess piece bodies, zone markers
5. **object0**: A separate 30mm cube used during RL training (hidden in chess game mode)
6. **goal0**: The FetchPickAndPlace goal sphere marker (suppressed by `_render_callback`)

The XML is relative to the project root (resolved by the XML path injection mechanism in `ChessSimulationEnv`).
