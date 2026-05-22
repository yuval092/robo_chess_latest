"""
Generate chess piece STL meshes that wrap the 30mm cube collision body.

Design: no flat box slab — each piece is one continuous shape.
- STL geom placed at pos="0 0 -0.016" (1mm below cube centre)
- Wide frustum base: r=22mm at z=0 → r=13mm at z=0.032
  r=22mm covers cube corners (diagonal 21.2mm) with all 24 segments (inscribed r=21.8mm)
  z=0.032 is 1mm above cube top (avoids z-fighting on top face)
- Piece-specific body continues above z=0.032
- King top at STL z=0.060 → body z=0.044 → world z=0.459m < HOVER_Z 0.460m ✓
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct

BASE_R    = 0.022    # wide base radius — covers cube corners at 21.2mm
NECK_R    = 0.013    # piece body radius at top of base section
BASE_BOT  = 0.000    # STL z of cube bottom (1mm below actual cube bottom after -0.016 offset)
BASE_TOP  = 0.032    # STL z of cube top + 1mm margin
SEGS      = 24       # smooth circular cross-sections


def normal(a, b, c):
    ux, uy, uz = (b[i] - a[i] for i in range(3))
    vx, vy, vz = (c[i] - a[i] for i in range(3))
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / length, ny / length, nz / length


def box_triangles(center, size):
    cx, cy, cz = center
    sx, sy, sz = (dim / 2.0 for dim in size)
    v = {
        "000": (cx - sx, cy - sy, cz - sz),
        "001": (cx - sx, cy - sy, cz + sz),
        "010": (cx - sx, cy + sy, cz - sz),
        "011": (cx - sx, cy + sy, cz + sz),
        "100": (cx + sx, cy - sy, cz - sz),
        "101": (cx + sx, cy - sy, cz + sz),
        "110": (cx + sx, cy + sy, cz - sz),
        "111": (cx + sx, cy + sy, cz + sz),
    }
    return [
        (v["000"], v["100"], v["110"]), (v["000"], v["110"], v["010"]),
        (v["001"], v["011"], v["111"]), (v["001"], v["111"], v["101"]),
        (v["000"], v["001"], v["101"]), (v["000"], v["101"], v["100"]),
        (v["010"], v["110"], v["111"]), (v["010"], v["111"], v["011"]),
        (v["000"], v["010"], v["011"]), (v["000"], v["011"], v["001"]),
        (v["100"], v["101"], v["111"]), (v["100"], v["111"], v["110"]),
    ]


def frustum_triangles(r_bottom, r_top, z_bottom, z_top, segments):
    """Frustum (truncated cone). r_bottom==r_top → cylinder."""
    angles = [2 * math.pi * i / segments for i in range(segments)]
    bot = [(r_bottom * math.cos(a), r_bottom * math.sin(a), z_bottom) for a in angles]
    top = [(r_top   * math.cos(a), r_top   * math.sin(a), z_top)    for a in angles]
    bc = (0.0, 0.0, z_bottom)
    tc = (0.0, 0.0, z_top)
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        tris.append((bc, bot[j], bot[i]))
        tris.append((tc, top[i], top[j]))
        tris.append((bot[i], bot[j], top[j]))
        tris.append((bot[i], top[j], top[i]))
    return tris


def cone_triangles(r_base, z_base, z_tip, segments):
    """Solid cone pointing upward."""
    tip = (0.0, 0.0, z_tip)
    angles = [2 * math.pi * i / segments for i in range(segments)]
    base = [(r_base * math.cos(a), r_base * math.sin(a), z_base) for a in angles]
    bc = (0.0, 0.0, z_base)
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        tris.append((bc, base[j], base[i]))
        tris.append((base[i], base[j], tip))
    return tris


def offset_tris(tris, dx, dy, dz=0.0):
    """Translate a triangle list."""
    return [tuple((x + dx, y + dy, z + dz) for x, y, z in tri) for tri in tris]


def piece_base():
    """Wide-to-neck frustum covering the cube body. Shared by all pieces."""
    return frustum_triangles(BASE_R, NECK_R, BASE_BOT, BASE_TOP, SEGS)


def pawn_triangles():
    """Pawn: tapered base + waist + round head. Total 49mm."""
    tris = piece_base()
    tris += frustum_triangles(NECK_R, 0.008, BASE_TOP, 0.040, 16)   # waist narrows
    tris += frustum_triangles(0.008, 0.012, 0.040, 0.045, 12)       # head widens
    tris += cone_triangles(0.012, 0.045, 0.049, 12)                  # head cap
    return tris


def rook_triangles():
    """Rook: tapered base + cylinder + platform + 4 battlements. Total 50mm."""
    tris = piece_base()
    tris += frustum_triangles(NECK_R, 0.011, BASE_TOP, 0.042, 16)   # body
    tris += frustum_triangles(0.011, 0.014, 0.042, 0.044, 16)       # platform flare
    for dx, dy in [(0.011, 0.0), (-0.011, 0.0), (0.0, 0.011), (0.0, -0.011)]:
        tris += offset_tris(
            box_triangles((0.0, 0.0, 0.047), (0.008, 0.008, 0.006)), dx, dy
        )
    return tris


def knight_triangles():
    """Knight: tapered base + neck + offset head facing +X. Total 52mm."""
    tris = piece_base()
    tris += frustum_triangles(NECK_R, 0.008, BASE_TOP, 0.040, 16)   # neck taper
    tris += box_triangles((0.007, 0.0, 0.046), (0.020, 0.012, 0.012))  # head (+X offset)
    tris += box_triangles((0.014, 0.0, 0.041), (0.008, 0.007, 0.004))  # snout
    return tris


def bishop_triangles():
    """Bishop: tapered base + long body taper + mitre tip. Total 54mm."""
    tris = piece_base()
    tris += frustum_triangles(NECK_R, 0.004, BASE_TOP, 0.050, 20)   # long taper
    tris += cone_triangles(0.004, 0.050, 0.054, 12)                  # mitre tip
    return tris


def queen_triangles():
    """Queen: tapered base + body + crown with 5 spikes. Total 54mm."""
    tris = piece_base()
    tris += frustum_triangles(NECK_R, 0.010, BASE_TOP, 0.046, 16)   # body
    tris += frustum_triangles(0.010, 0.013, 0.046, 0.048, 16)       # crown base flare
    # 5 crown spikes at r=0.011 from centre, evenly spaced
    for i in range(5):
        angle = 2 * math.pi * i / 5
        cx = 0.011 * math.cos(angle)
        cy = 0.011 * math.sin(angle)
        tris += offset_tris(cone_triangles(0.006, 0.048, 0.054, 8), cx, cy)
    return tris


def king_triangles():
    """King: tapered base + body + prominent cross. Total 60mm."""
    tris = piece_base()
    tris += frustum_triangles(NECK_R, 0.010, BASE_TOP, 0.050, 16)   # body
    tris += box_triangles((0.0, 0.0, 0.055), (0.005, 0.005, 0.010))  # cross vertical
    tris += box_triangles((0.0, 0.0, 0.053), (0.018, 0.005, 0.005))  # cross horizontal
    return tris


PIECE_BUILDERS = {
    "pawn":   pawn_triangles,
    "rook":   rook_triangles,
    "knight": knight_triangles,
    "bishop": bishop_triangles,
    "queen":  queen_triangles,
    "king":   king_triangles,
}


def write_binary_stl(path: Path, name: str, triangles) -> None:
    header = f"RoboChess {name} visual mesh".encode("ascii")[:80].ljust(80, b"\0")
    with path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            nx, ny, nz = normal(*tri)
            values = [nx, ny, nz]
            for vertex in tri:
                values.extend(vertex)
            f.write(struct.pack("<12fH", *values, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chess piece STL meshes.")
    parser.add_argument("--out-dir", default="chess_env/stls/chess")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in PIECE_BUILDERS.items():
        triangles = builder()
        write_binary_stl(out_dir / f"{name}.stl", name, triangles)
        print(f"  {name}: {len(triangles)} triangles")
    print(f"Generated {len(PIECE_BUILDERS)} STL meshes in {out_dir}")


if __name__ == "__main__":
    main()
