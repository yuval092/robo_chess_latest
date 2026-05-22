"""
Generate MuJoCo XML bodies for active chess pieces and promotion reserves.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.append(os.getcwd())

import chess

from src.chess_game.board_mapper import BoardMapper
from src.physical.piece_registry import PieceRegistry, reserve_piece_ids
from src.utils.config import load_config


START_MARKER = "\t\t<!-- generated chess pieces start -->"
END_MARKER = "\t\t<!-- generated chess pieces end -->"


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
        f'\t\t\t      material="{material_prefix}_piece_cube_mat" rgba="0 0 0 0" mass="0.35" condim="6"',
        f'\t\t\t      friction="2.0 0.005 0.0001" solref="0.002 1" solimp="0.99 0.999 0.001"/>',
        f'\t\t\t<geom name="{body_name}_visual" type="mesh" mesh="chess_{piece_type}_mesh"',
        f'\t\t\t      pos="0 0 -0.016" material="{material_prefix}_piece_visual_mat"',
        f'\t\t\t      contype="0" conaffinity="0" mass="0"{euler_attr}/>',
        f'\t\t\t<site name="{body_name}_site" pos="0 0 0" size="0.005"/>',
        "\t\t</body>",
    ]


def reserve_position(index: int, color: str, piece_type: str) -> tuple[float, float, float]:
    cfg = load_config("chess")
    reserve_cfg = cfg["promotion_reserve"][color]
    spacing = cfg["reserves"]["promotion_slot_spacing_m"]
    origin = reserve_cfg["origin_xyz"]
    piece_type_offset = {"queen": 0, "rook": 8, "bishop": 16, "knight": 24}[piece_type]
    slot = piece_type_offset + index
    row = slot // reserve_cfg["cols"]
    col = slot % reserve_cfg["cols"]
    return origin[0] + row * spacing, origin[1] + col * spacing, origin[2]


def build_fragment() -> str:
    mapper = BoardMapper.from_configs()
    registry = PieceRegistry()
    lines = [START_MARKER]

    for piece in registry.all_pieces():
        xyz = mapper.square_to_piece_xyz(chess.parse_square(piece.initial_square))
        visual_euler = knight_visual_euler(piece.color) if piece.piece_type == "knight" else None
        lines.extend(
            piece_body_xml(
                piece.piece_id,
                piece.color,
                piece.piece_type,
                xyz,
                initial_square=piece.initial_square,
                visual_euler=visual_euler,
            )
        )

    for piece_id, color, piece_type in reserve_piece_ids():
        reserve_index = int(piece_id.rsplit("_", 1)[1]) - 1
        xyz = reserve_position(reserve_index, color, piece_type)
        visual_euler = knight_visual_euler(color) if piece_type == "knight" else None
        lines.extend(piece_body_xml(piece_id, color, piece_type, xyz, visual_euler=visual_euler))

    lines.append(END_MARKER)
    return "\n".join(lines)


def knight_visual_euler(color: str) -> str:
    return "0 0 1.5707963268" if color == "white" else "0 0 4.7123889804"


def update_scene(scene_path: Path) -> None:
    text = scene_path.read_text()
    start = text.index(START_MARKER)
    end = text.index(END_MARKER) + len(END_MARKER)
    scene_path.write_text(text[:start] + build_fragment() + text[end:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chess piece XML bodies.")
    parser.add_argument("--write", action="store_true", help="Update pick_and_place.xml in place.")
    args = parser.parse_args()

    if args.write:
        path = Path("chess_env/assets/pick_and_place.xml")
        update_scene(path)
        print(f"Updated {path}")
    else:
        print(build_fragment())


if __name__ == "__main__":
    main()
