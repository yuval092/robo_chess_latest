"""
Generate simple low-poly visual STL meshes for chess piece identities.

These meshes are visual-only. The physical collision remains the 30mm cube.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import struct


PIECE_SPECS = {
    "pawn": (0.006, 0.018, 16),
    "rook": (0.007, 0.020, 16),
    "knight": (0.006, 0.021, 5),
    "bishop": (0.005, 0.024, 18),
    "queen": (0.0065, 0.023, 12),
    "king": (0.006, 0.026, 10),
}


def normal(a, b, c):
    ux, uy, uz = (b[i] - a[i] for i in range(3))
    vx, vy, vz = (c[i] - a[i] for i in range(3))
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / length, ny / length, nz / length


def cylinder_triangles(radius: float, height: float, segments: int):
    bottom = (0.0, 0.0, 0.0)
    top = (0.0, 0.0, height)
    vertices = []
    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        vertices.append((radius * math.cos(angle), radius * math.sin(angle), 0.0))
    top_vertices = [(x, y, height) for x, y, _ in vertices]

    triangles = []
    for i in range(segments):
        j = (i + 1) % segments
        triangles.append((bottom, vertices[j], vertices[i]))
        triangles.append((top, top_vertices[i], top_vertices[j]))
        triangles.append((vertices[i], vertices[j], top_vertices[j]))
        triangles.append((vertices[i], top_vertices[j], top_vertices[i]))
    return triangles


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
    parser = argparse.ArgumentParser(description="Generate simple chess visual STL meshes.")
    parser.add_argument("--out-dir", default="chess_env/stls/chess")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (radius, height, segments) in PIECE_SPECS.items():
        write_binary_stl(out_dir / f"{name}.stl", name, cylinder_triangles(radius, height, segments))
    print(f"Generated {len(PIECE_SPECS)} STL meshes in {out_dir}")


if __name__ == "__main__":
    main()
