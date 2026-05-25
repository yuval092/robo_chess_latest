# Scene Generation, XML, and STL Meshes

## Overview

The MuJoCo scene (`chess_env/assets/pick_and_place.xml`) is not hand-authored in its chess-specific sections. Instead, `src/chess_env/environment_generation.py` generates three XML fragments at startup and injects them into the file using marker-delimited text replacement. The STL mesh files for chess pieces are also procedurally generated.

---

## Startup Generation Flow

```python
# scripts/run_chess_ui.py:114
regenerate_environment()
```

`regenerate_environment()` calls `regenerate_scene()`, which:
1. Reads the existing XML file.
2. Replaces the **board squares** fragment (64 coloured box geoms).
3. Replaces the **zones** fragment (graveyard + reserve zone marker geoms).
4. Replaces the **chess pieces** fragment (96 chess piece bodies).
5. Writes the modified XML back to disk.

STL regeneration (`include_stls=True`) is only done explicitly (e.g., `scripts/generate_scene.py stls`).

---

## XML Fragment Injection

`_replace_marked_fragment(text, start_marker, end_marker, fragment)` locates a pair of XML comment markers and replaces everything between them:

```python
# environment_generation.py:32
start = text.index(start_marker)
end = text.index(end_marker) + len(end_marker)
return text[:start] + fragment + text[end:]
```

This raises `ValueError` if either marker is missing. The three marker pairs are:

| Section | Start Marker | End Marker |
|---------|-------------|------------|
| Board squares | `<!-- generated board squares start -->` | `<!-- generated board squares end -->` |
| Chess pieces | `<!-- generated chess pieces start -->` | `<!-- generated chess pieces end -->` |
| Zone markers | `<!-- generated zone markers start -->` | `<!-- generated zone markers end -->` |

---

## Board Fragment

`build_board_fragment()` generates 64 thin box geoms (0.5 mm thick), one per chess square:
- Alternating `chess_dark_square_mat` / `chess_light_square_mat` materials based on `(rank + file) % 2`.
- `contype="0" conaffinity="0"` — purely visual, no collision.
- XY positions computed from `BoardMapper.square_to_xy()`.

---

## Chess Piece Bodies (`build_pieces_fragment`)

For each of the 96 piece bodies (32 active + 64 reserve), `piece_body_xml()` generates:

```xml
<body name="piece_white_pawn_e" pos="0.8400 0.1841 0.4150">  <!-- e2 -->
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
- The **cube geom** is the physics body (30 mm × 30 mm × 30 mm). It handles all contact with fingers and table. `rgba="0 0 0 0"` makes it invisible (only the mesh is shown).
- `condim=6` enables full friction model (torsional + rolling friction).
- The **mesh geom** is visual only (`contype="0" conaffinity="0"`, `mass="0"`). It is offset by `pos="0 0 -0.016"` to sit at the correct visual height relative to the cube centre.
- `freejoint_damping=8.0` — high angular damping prevents pieces from spinning or toppling.
- **Knights** get an `euler` attribute to orient the horse correctly: white `"0 0 1.5707963268"`, black `"0 0 4.7123889804"` (90° and 270° respectively).

---

## Zone Fragment (`build_zones_fragment`)

Generates 4 zone marker geoms:
- `white_graveyard`: 4×4 grid at `[0.640, -0.300, 0.015]` (bottom of board)
- `black_graveyard`: 4×4 grid at `[0.640, 0.680, 0.015]` (top of board)
- `white_reserve`: 8×4 grid at `[0.820, -0.300, 0.015]` (promotion reserve, white)
- `black_reserve`: 8×4 grid at `[0.820, 0.680, 0.015]` (promotion reserve, black)

These are coloured but non-colliding box geoms used purely for visual reference in the simulator window.

---

## STL Mesh Generation

Chess piece STL files live in `chess_env/stls/chess/`: `pawn.stl`, `rook.stl`, `knight.stl`, `bishop.stl`, `queen.stl`, `king.stl`.

They are generated procedurally using primitive shapes assembled from frustum (truncated cone) and box triangles:

### Geometry Builders (in `environment_generation.py`)

Each piece is built from a list of triangle functions:

| Function | Shape | Used for |
|----------|-------|---------|
| `frustum_triangles(r_bottom, r_top, z_bot, z_top, segments)` | Truncated cone side + end caps | Bodies, necks, heads |
| `box_triangles(centre, size)` | Axis-aligned box (12 triangles) | Flat bases and decorative elements |

**Pawn** (`build_pawn`): base frustum → waist taper → ball head.  
**Rook** (`build_rook`): base → column → battlements (4 small boxes on top).  
**Knight** (`build_knight`): base → column → angled horse-head approximation.  
**Bishop** (`build_bishop`): base → waist → mitre top + cross decoration.  
**Queen** (`build_queen`): base → waist → crown (frustum with spikes).  
**King** (`build_king`): base → waist → crown + cross on top.

### Binary STL Format

`write_binary_stl(path, name, triangles)` writes the [binary STL](https://en.wikipedia.org/wiki/STL_(file_format)) format:
- 80-byte header.
- `uint32` triangle count.
- For each triangle: 3× `float32` normal + 3× 3× `float32` vertex + `uint16` attribute byte count.

All STL coordinates are in **metres** so no `scale` attribute is needed in the MuJoCo `<mesh>` declarations.

### MuJoCo Mesh Asset Declaration

In the XML `<asset>` section:
```xml
<mesh name="chess_pawn_mesh" file="stls/chess/pawn.stl"/>
<mesh name="chess_rook_mesh" file="stls/chess/rook.stl"/>
<!-- etc. -->
```

The relative path is resolved from the XML file's directory (`chess_env/assets/`).

---

## Material Definitions

Materials are declared in the XML `<asset>` section (not generated — hand-authored):

| Material name | Used for |
|---------------|---------|
| `white_piece_visual_mat` | White piece visual mesh |
| `black_piece_visual_mat` | Black piece visual mesh |
| `white_piece_cube_mat` | White cube collision body (invisible) |
| `black_piece_cube_mat` | Black cube collision body (invisible) |
| `chess_light_square_mat` | Light-coloured board squares |
| `chess_dark_square_mat` | Dark-coloured board squares |
| `white_graveyard_mat`, `black_graveyard_mat` | Graveyard zone markers |
| `white_reserve_mat`, `black_reserve_mat` | Promotion reserve zone markers |

---

## Graveyard and Reserve Slot Coordinates

Pieces removed from the board (captures) and promoted-out pawns are placed off-board.

**Graveyard slots** (`piece_teleport.py:teleport_piece_to_graveyard`):
- Slot ID format: `"slot_00"`, `"slot_01"`, … `"slot_15"` (4 rows × 4 cols).
- `_slot_xyz(cfg, spacing, slot_id)` converts index to world XY using origin + row/col × spacing.
- Slot index validated with `re.fullmatch(r"slot_(\d+)", slot_id)`.

**Promotion reserve slots** (`teleport_piece_to_promotion_reserve`):
- 8 × 4 grid (8 pieces per type: Q, R, B, N × 4 cols).
- `piece_type_offset = {"queen": 0, "rook": 8, "bishop": 16, "knight": 24}`.
- Slot computed from `piece_type_offset + reserve_index`.

## Cleanup Refactor Notes

Scene XML generation is lazy and idempotent via `ensure_environment_generated()` during `ChessSimulationEnv` construction. Importing `src.chess_env` alone does not write generated assets.

Manual regeneration uses `python scripts/generate_scene.py all` or the subcommands `board`, `pieces`, `zones`, and `stls`.

The upstream demo scenes `push.xml`, `reach.xml`, and `slide.xml` were removed. `robot.xml` and `shared.xml` remain required includes for `pick_and_place.xml`.
