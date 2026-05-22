"""
Generate Staunton-inspired chess piece STL meshes wrapping the 30mm cube.

Profile (shared base):
  z=0.000→0.006  Base disc:  r=22mm flat ring (clearly visible pedestal)
  z=0.006→0.030  Taper:      r=22→11mm steep shoulder
  z=0.030→0.032  Waist:      r=11→10mm tight neck (cube top +1mm margin)
  piece body starts at z=0.032, r=10mm

STL geom pos="0 0 -0.016":  1mm below cube centre → no co-planar z-fight.
King world top: 0.415 - 0.016 + 0.058 = 0.457m  < HOVER_Z 0.460m ✓
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct

BASE_R    = 0.022   # covers cube corners (diagonal 21.2mm); 24-seg min=21.8mm
DISC_TOP  = 0.006   # disc height — visually clear flat pedestal
TAPER_TOP = 0.030   # taper end (z just below cube top at 0.031)
WAIST_R   = 0.010   # waist width at taper end
BASE_TOP  = 0.032   # transition to piece body (cube top +1mm margin)
SEGS      = 24      # segments for base; fewer for smaller features


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
    return [tuple((x + dx, y + dy, z + dz) for x, y, z in tri) for tri in tris]


def piece_base():
    """Staunton-style pedestal: flat disc + steep shoulder taper + waist."""
    tris = frustum_triangles(BASE_R, BASE_R,   0.000, DISC_TOP,  SEGS)   # disc
    tris += frustum_triangles(BASE_R, WAIST_R,  DISC_TOP, TAPER_TOP, SEGS) # shoulder
    tris += frustum_triangles(WAIST_R, WAIST_R, TAPER_TOP, BASE_TOP, SEGS) # waist stub
    return tris


def pawn_triangles():
    """Pawn: pedestal + narrow waist + round ball head. Total 50mm."""
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, 0.007, BASE_TOP, 0.038, 16)  # neck narrows
    tris += frustum_triangles(0.007, 0.015, 0.038, 0.044, 14)       # ball widens
    tris += frustum_triangles(0.015, 0.015, 0.044, 0.047, 14)       # ball equator
    tris += cone_triangles(0.015, 0.047, 0.050, 14)                  # ball cap
    return tris


def rook_triangles():
    """Rook: pedestal + cylinder + wide battlemented platform. Total 52mm."""
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, 0.012, BASE_TOP, 0.042, 16)  # body
    tris += frustum_triangles(0.012, 0.018, 0.042, 0.044, 16)       # platform flare
    tris += frustum_triangles(0.018, 0.018, 0.044, 0.046, 16)       # platform rim
    # 4 battlements (merlons) at N/S/E/W; 4 gaps (crenels) between them
    for dx, dy in [(0.013, 0.0), (-0.013, 0.0), (0.0, 0.013), (0.0, -0.013)]:
        tris += offset_tris(box_triangles((0, 0, 0.049), (0.010, 0.010, 0.006)), dx, dy)
    return tris


def knight_triangles():
    """Knight: pedestal + neck + large horse-head offset in +X. Total 54mm."""
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, 0.007, BASE_TOP, 0.040, 16)   # neck taper
    # Main head block offset strongly in +X (horse face direction)
    tris += box_triangles((0.009, 0.0, 0.047), (0.026, 0.016, 0.014)) # head
    tris += box_triangles((0.018, 0.0, 0.042), (0.010, 0.009, 0.006)) # snout
    # Ear nub at top-front
    tris += box_triangles((0.006, 0.0, 0.054), (0.006, 0.006, 0.004)) # ear
    return tris


def bishop_triangles():
    """Bishop: pedestal + slim body + ring collar + tall mitre. Total 56mm."""
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, 0.008, BASE_TOP, 0.040, 16)   # lower body
    tris += frustum_triangles(0.008, 0.011, 0.040, 0.042, 14)        # collar flare
    tris += frustum_triangles(0.011, 0.008, 0.042, 0.044, 14)        # collar taper back
    tris += frustum_triangles(0.008, 0.004, 0.044, 0.052, 16)        # slim upper body
    tris += cone_triangles(0.004, 0.052, 0.056, 12)                   # mitre tip
    return tris


def queen_triangles():
    """Queen: pedestal + curved body + wide crown with 5 tall spikes. Total 54mm."""
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, 0.008, BASE_TOP, 0.038, 16)   # waist narrows
    tris += frustum_triangles(0.008, 0.014, 0.038, 0.046, 16)        # body swells
    tris += frustum_triangles(0.014, 0.017, 0.046, 0.048, 16)        # crown base flare
    # 5 crown spikes: tall cones at r=0.014 from centre
    for i in range(5):
        angle = 2 * math.pi * i / 5
        cx = 0.014 * math.cos(angle)
        cy = 0.014 * math.sin(angle)
        tris += offset_tris(cone_triangles(0.007, 0.048, 0.054, 10), cx, cy)
    # Central orb
    tris += frustum_triangles(0.004, 0.006, 0.048, 0.051, 10)
    tris += cone_triangles(0.006, 0.051, 0.054, 10)
    return tris


def king_triangles():
    """King: pedestal + curved body + large cross. Total ~57mm."""
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, 0.008, BASE_TOP, 0.038, 16)   # waist
    tris += frustum_triangles(0.008, 0.013, 0.038, 0.048, 16)        # body swells
    tris += frustum_triangles(0.013, 0.008, 0.048, 0.050, 12)        # shoulder step
    # Cross: vertical bar + horizontal bar
    tris += box_triangles((0.0, 0.0, 0.054), (0.006, 0.006, 0.008))  # vertical
    tris += box_triangles((0.0, 0.0, 0.052), (0.022, 0.006, 0.006))  # horizontal
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
