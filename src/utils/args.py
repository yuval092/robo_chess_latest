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
