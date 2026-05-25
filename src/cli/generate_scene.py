"""CLI tool for regenerating chess scene assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.chess_env.environment_generation import (
    PIECE_BUILDERS,
    build_board_fragment,
    build_pieces_fragment,
    build_zones_fragment,
    regenerate_environment,
    regenerate_stls,
    update_board_scene,
    update_pieces_scene,
    update_zones_scene,
)

SCENE_PATH = Path("chess_env/assets/pick_and_place.xml")


def cmd_board(args: argparse.Namespace) -> None:
    """Print or write the generated board XML fragment."""
    if args.write:
        update_board_scene(SCENE_PATH)
        print(f"Updated board fragment in {SCENE_PATH}")
    else:
        print(build_board_fragment())


def cmd_pieces(args: argparse.Namespace) -> None:
    """Print or write the generated pieces XML fragment."""
    if args.write:
        update_pieces_scene(SCENE_PATH)
        print(f"Updated pieces fragment in {SCENE_PATH}")
    else:
        print(build_pieces_fragment())


def cmd_zones(args: argparse.Namespace) -> None:
    """Print or write the generated zone XML fragment."""
    if args.write:
        update_zones_scene(SCENE_PATH)
        print(f"Updated zone fragment in {SCENE_PATH}")
    else:
        print(build_zones_fragment())


def cmd_stls(args: argparse.Namespace) -> None:
    """Regenerate STL meshes into the requested directory."""
    regenerate_stls(Path(args.out_dir))
    print(f"Generated {len(PIECE_BUILDERS)} STL meshes in {args.out_dir}")


def cmd_all(args: argparse.Namespace) -> None:
    """Regenerate XML fragments and STL meshes."""
    regenerate_environment()
    print("Full environment regeneration complete.")


def main() -> None:
    """Parse arguments and run the requested generation command."""
    parser = argparse.ArgumentParser(description="Regenerate RoboChess scene assets.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_board = sub.add_parser("board")
    p_board.add_argument("--write", action="store_true")
    p_board.set_defaults(func=cmd_board)

    p_pieces = sub.add_parser("pieces")
    p_pieces.add_argument("--write", action="store_true")
    p_pieces.set_defaults(func=cmd_pieces)

    p_zones = sub.add_parser("zones")
    p_zones.add_argument("--write", action="store_true")
    p_zones.set_defaults(func=cmd_zones)

    p_stls = sub.add_parser("stls")
    p_stls.add_argument("--out-dir", default="chess_env/stls/chess")
    p_stls.set_defaults(func=cmd_stls)

    p_all = sub.add_parser("all")
    p_all.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
