"""Validate that all deployed model paths in configs/deployed_models.yaml exist."""

from __future__ import annotations

import sys
from pathlib import Path

from src.utils.io import load_config


def missing_deployed_models() -> list[tuple[str, str]]:
    """Return deployed model entries whose checkpoint files are missing."""
    cfg = load_config("deployed_models")
    return [(stage, path) for stage, path in cfg.items() if not Path(path).exists()]


def validate_or_exit() -> None:
    """Print a clear error and exit if any deployed model file is missing."""
    missing = missing_deployed_models()
    if not missing:
        return
    print("ERROR: Missing deployed model files:")
    for stage, path in missing:
        print(f"  {stage}: {path}")
    print()
    print("To run with scripted-only mode instead:")
    print("  python scripts/eval_stages.py --use-scripted-only ...")
    sys.exit(1)


def main() -> None:
    """Validate deployed model files and print the result."""
    validate_or_exit()
    print(f"OK: all {len(load_config('deployed_models'))} deployed model files present")


if __name__ == "__main__":
    main()
