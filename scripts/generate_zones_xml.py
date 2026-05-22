"""CLI wrapper for chess zone XML generation."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.getcwd())

from src.chess_env.environment_generation import build_zones_fragment, update_zones_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate zone visual floor geoms.")
    parser.add_argument("--write", action="store_true", help="Update chess_env/assets/pick_and_place.xml in place.")
    args = parser.parse_args()

    if args.write:
        scene_path = Path("chess_env/assets/pick_and_place.xml")
        update_zones_scene(scene_path)
        print(f"Updated {scene_path}")
    else:
        print(build_zones_fragment())


if __name__ == "__main__":
    main()
