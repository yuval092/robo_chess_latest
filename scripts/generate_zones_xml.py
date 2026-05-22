"""Generate visual floor zone geoms for graveyard and promotion reserve areas."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.append(os.getcwd())

from src.utils.config import load_config


START_MARKER = "\t\t<!-- generated zone markers start -->"
END_MARKER = "\t\t<!-- generated zone markers end -->"

ZONE_SPECS = [
    ("white_graveyard", "graveyards", "white", "white_graveyard_mat"),
    ("black_graveyard", "graveyards", "black", "black_graveyard_mat"),
    ("white_reserve", "promotion_reserve", "white", "white_reserve_mat"),
    ("black_reserve", "promotion_reserve", "black", "black_reserve_mat"),
]


def _zone_geom(name: str, cfg: dict, spacing: float, material: str) -> str:
    origin = cfg["origin_xyz"]
    rows = cfg["rows"]
    cols = cfg["cols"]
    margin = 0.005
    half_x = (rows * spacing) / 2.0 + margin
    half_y = (cols * spacing) / 2.0 + margin
    center_x = origin[0] + (rows - 1) * spacing / 2.0
    center_y = origin[1] + (cols - 1) * spacing / 2.0
    center_z = 0.002
    return (
        f'\t\t<geom name="zone_{name}" type="box" '
        f'size="{half_x:.4f} {half_y:.4f} 0.002" '
        f'pos="{center_x:.4f} {center_y:.4f} {center_z:.4f}" '
        f'material="{material}" contype="0" conaffinity="0" mass="0"/>'
    )


def build_fragment() -> str:
    cfg = load_config("chess")
    graveyard_spacing = cfg["reserves"]["graveyard_slot_spacing_m"]
    promotion_spacing = cfg["reserves"]["promotion_slot_spacing_m"]
    lines = [START_MARKER]
    for short_name, section_key, color, material in ZONE_SPECS:
        zone_cfg = cfg[section_key][color]
        spacing = promotion_spacing if "reserve" in short_name else graveyard_spacing
        lines.append(_zone_geom(short_name, zone_cfg, spacing, material))
    lines.append(END_MARKER)
    return "\n".join(lines)


def update_scene(scene_path: Path) -> None:
    text = scene_path.read_text()
    start = text.index(START_MARKER)
    end = text.index(END_MARKER) + len(END_MARKER)
    scene_path.write_text(text[:start] + build_fragment() + text[end:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate zone visual floor geoms.")
    parser.add_argument("--write", action="store_true", help="Update chess_env/assets/pick_and_place.xml in place.")
    args = parser.parse_args()
    if args.write:
        scene_path = Path("chess_env/assets/pick_and_place.xml")
        update_scene(scene_path)
        print(f"Updated {scene_path}")
    else:
        print(build_fragment())


if __name__ == "__main__":
    main()
