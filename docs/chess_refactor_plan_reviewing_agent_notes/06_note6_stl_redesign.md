# Note 6: Chess Piece STL Redesign

## Problem
Current STLs are small cylinders (radius 5–7mm) placed at `pos="0 0 0.017"` ON TOP of the cube.
The cube half-extent is 15mm, so the cube body is much larger than the STL visual.
Pieces look like 30mm cubes with tiny shape markings, not like chess pieces.

## Design Requirements
- STL base must have ≥30×30mm footprint to fully cover the cube
- STL z=0 aligns with cube bottom: geom pos must be `pos="0 0 -0.015"`
- STL spans z=0.000 to z=0.030 covering the full cube (30mm)
- Distinctive shape extends above z=0.030
- All vertices in metres
- contype=0, conaffinity=0, mass=0 — zero physics effect (unchanged)
- HOVER_Z constraint: top of tallest piece must be < 0.460m
  - King at 55mm: body_center(0.415) - 0.015 + 0.055 = 0.455m ✓

## Piece Dimensions

| Piece  | STL total height | Shape above cube |
|--------|-----------------|-----------------|
| pawn   | 45 mm | Tapered neck + round head |
| rook   | 46 mm | Cylinder + 4 battlements |
| knight | 48 mm | Cylinder neck + offset head (+X facing) |
| bishop | 50 mm | Tapered body + mitre point |
| queen  | 50 mm | Cylinder + 5 crown spikes |
| king   | 55 mm | Cylinder + cross (vertical + horizontal bars) |

## Box Base (shared)
All pieces have a 30×30×30mm box base (z=0 to z=0.030):
```python
box_triangles(center=(0, 0, 0.015), size=(0.030, 0.030, 0.030))
```
This perfectly covers the cube body.

## Changes

### 6.1 Rewrite `scripts/generate_chess_stls.py`
- Add `frustum_triangles(r_bottom, r_top, z_bottom, z_top, segments)`
- Add `cone_triangles(r_base, z_base, z_tip, segments)`
- Replace PIECE_SPECS cylinder approach with per-piece builder functions
- Each builder: box_base() + distinctive shape above z=0.030

### 6.2 Update `scripts/generate_pieces_xml.py`
Change visual geom pos from `pos="0 0 0.017"` to `pos="0 0 -0.015"` (STL z=0 at cube bottom).

### 6.3 Regenerate STLs and XML
```bash
python scripts/generate_chess_stls.py
python scripts/generate_pieces_xml.py --write
```

## Validation
- Each STL has ≥80 triangles (much more with box base)
- `python scripts/visualize_chess_setup.py` — cube not visible through STL
- King top < HOVER_Z (0.460m)
