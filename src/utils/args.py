"""
Shared argparse utilities for all RoboChess evaluation scripts.
"""
import argparse


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add visualization and common flags to any ArgumentParser."""
    parser.add_argument(
        "--visualize", action="store_true",
        help="Open a MuJoCo viewer window (requires display)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Sleep seconds between steps when visualizing (default: 0)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable environment debug logging"
    )
    parser.add_argument(
        "--n-episodes", type=int, default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--drift-limit", type=float, default=0.010,
        help="Tube constraint radius in meters for descend/ascend (default: 0.010)"
    )
    return parser


def make_env(args, force_scenario=None, hide_object=True):
    """
    Create the gymnasium environment using parsed args.
    Handles render_mode selection based on --visualize flag.
    """
    import gymnasium as gym
    import src.chess_env  # trigger registration

    render_mode = "human" if args.visualize else None
    env = gym.make(
        "ChessFetchTask-v0",
        render_mode=render_mode,
        force_scenario=force_scenario,
        hide_object=hide_object,
        debug=args.debug,
    )
    return env
