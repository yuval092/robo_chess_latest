"""Shared argparse utilities for RoboChess CLI commands."""

from __future__ import annotations

import argparse


def add_model_path_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add --transit-model, --descend-model, --ascend-model override flags."""
    parser.add_argument(
        "--transit-model",
        metavar="PATH",
        help="Path to transit model ZIP. Overrides configs/deployed_models.yaml.",
    )
    parser.add_argument(
        "--descend-model",
        metavar="PATH",
        help="Path to descend model ZIP. Overrides configs/deployed_models.yaml.",
    )
    parser.add_argument(
        "--ascend-model",
        metavar="PATH",
        help="Path to ascend model ZIP. Overrides configs/deployed_models.yaml.",
    )
    return parser


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add common evaluation flags shared by stage and flow evaluators."""
    add_model_path_args(parser)
    parser.add_argument(
        "--drift-limit",
        type=float,
        default=None,
        metavar="METRES",
        help="Override the tube constraint radius for descend/ascend stages, in metres.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open the MuJoCo viewer window while evaluation runs.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        metavar="SECS",
        help="Per-step sleep in seconds when --visualize is active (default: 0.0).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose per-step environment logs.",
    )
    return parser


def model_overrides_from_args(args: argparse.Namespace) -> dict[str, str]:
    """Build a stage→path overrides dict from parsed --*-model args."""
    overrides: dict[str, str] = {}
    if args.transit_model:
        overrides["transit"] = args.transit_model
    if args.descend_model:
        overrides["descend"] = args.descend_model
    if args.ascend_model:
        overrides["ascend"] = args.ascend_model
    return overrides


def resolve_model_paths(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Resolve deployed model paths with explicit CLI overrides applied."""
    from src.utils.io import load_config

    deployed = load_config("deployed_models")
    overrides = overrides or {}
    paths = {
        stage: overrides.get(stage) or deployed.get(stage)
        for stage in ("transit", "descend", "ascend")
    }
    missing = [stage for stage, path in paths.items() if not path]
    if missing:
        flags = ", ".join(f"--{stage}-model" for stage in missing)
        raise ValueError(
            f"Missing model path(s) for {', '.join(missing)}. "
            f"Set configs/deployed_models.yaml or pass {flags}."
        )
    return paths
