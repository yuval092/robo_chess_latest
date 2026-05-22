"""CLI wrapper for generated chess STL visual meshes."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, os.getcwd())

from src.chess_env.environment_generation import PIECE_BUILDERS, regenerate_stls


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chess piece STL meshes.")
    parser.add_argument("--out-dir", default="chess_env/stls/chess")
    args = parser.parse_args()

    out_dir = regenerate_stls(Path(args.out_dir))
    for name in PIECE_BUILDERS:
        print(f"  {name}: generated")
    print(f"Generated {len(PIECE_BUILDERS)} STL meshes in {out_dir}")


if __name__ == "__main__":
    main()
