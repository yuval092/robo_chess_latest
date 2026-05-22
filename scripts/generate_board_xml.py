"""
Generate visual-only chess board square geoms for the table0 body.

The generated positions come from BoardMapper, so XML visuals stay aligned
with the exact 8cm chess geometry used by game logic and validation.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import chess

sys.path.append(os.getcwd())

from src.chess_game.board_mapper import BoardMapper
from src.utils.config import load_config


START_MARKER = "\t\t\t<!-- generated board squares start -->"
END_MARKER = "\t\t\t<!-- generated board squares end -->"


def build_fragment() -> str:
    mapper = BoardMapper.from_configs()
    env_cfg = load_config("env")
    table_cx, table_cy = env_cfg["table_center_xy"]
    square_half = mapper.geometry.cell_size_m / 2.0
    z = mapper.geometry.table_surface_z + 0.0006

    lines = [START_MARKER]
    for square in chess.SQUARES:
        name = chess.square_name(square)
        xy = mapper.square_to_xy(square)
        local_x = xy[0] - table_cx
        local_y = xy[1] - table_cy
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        material = "chess_dark_square_mat" if (rank + file) % 2 == 0 else "chess_light_square_mat"
        lines.append(
            "\t\t\t"
            f'<geom name="board_{name}_visual" type="box" '
            f'size="{square_half:.3f} {square_half:.3f} 0.0005" '
            f'pos="{local_x:.4f} {local_y:.4f} {z:.4f}" '
            f'material="{material}" contype="0" conaffinity="0" mass="0"/>'
        )
    lines.append(END_MARKER)
    return "\n".join(lines)


def update_scene(scene_path: Path) -> None:
    text = scene_path.read_text()
    start = text.index(START_MARKER)
    end = text.index(END_MARKER) + len(END_MARKER)
    updated = text[:start] + build_fragment() + text[end:]
    scene_path.write_text(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chess board visual geoms.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Update chess_env/assets/pick_and_place.xml in place.",
    )
    args = parser.parse_args()

    if args.write:
        scene_path = Path("chess_env/assets/pick_and_place.xml")
        update_scene(scene_path)
        print(f"Updated {scene_path}")
    else:
        print(build_fragment())


if __name__ == "__main__":
    main()
