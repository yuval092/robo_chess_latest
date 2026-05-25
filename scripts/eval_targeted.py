"""
Targeted evaluation of specific squares known to be problematic.

Tests transit TO a set of destination squares from multiple source squares,
reporting per-destination success rates.

Usage:
    PYTHONPATH=. python scripts/eval_targeted.py --transit-model <path> [--n-reps 5]
    PYTHONPATH=. python scripts/eval_targeted.py  # uses deployed model from configs/deployed_models.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import mujoco
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym

from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_game.board_mapper import BoardMapper
from src.utils.io import load_config

# Previously problematic destination squares (highest priority)
PROBLEM_DESTINATIONS = ["a5", "c1"]

# Structural coverage: corners, near-corners, edge midpoints, center
CORNER_SQUARES = ["a1", "h1", "a8", "h8"]
NEAR_CORNER = ["b1", "g1", "b8", "g8"]
EDGE_MID = ["a4", "h4", "a5", "h5", "d1", "d8"]
CENTER = ["d4", "e5"]

ALL_DST = list(
    dict.fromkeys(
        PROBLEM_DESTINATIONS + CORNER_SQUARES + NEAR_CORNER + EDGE_MID + CENTER
    )
)

# Source squares: 10 spread across the board (center, corners, edges, mid-inner)
SOURCE_SQUARES = [
    "d4",
    "e5",  # center
    "a1",
    "h8",  # far corners
    "b2",
    "g7",  # near-corner
    "h4",
    "a5",  # edge
    "c3",
    "f6",  # mid-inner ring
]


def sq_xy(name: str, bm: BoardMapper) -> np.ndarray:
    """Return sq xy."""
    return bm.square_name_to_xy(name)


def reposition_arm(inner, xy: np.ndarray, safe_z: float):
    """Return reposition arm."""
    target_pos = np.array([xy[0], xy[1], safe_z])
    inner._move_mocap_to(
        target_pos, inner.VERTICAL_QUAT, max_steps=250, tolerance=0.003
    )
    mujoco.mj_forward(inner.model, inner.data)


def run_eval(ctrl, inner, bm, env_cfg, n_reps: int):
    """Run eval."""
    safe_z = env_cfg["safe_z"]

    # dst → {attempts, successes, error_mm_list}
    dst_stats: dict[str, dict] = {
        d: {"attempts": 0, "successes": 0, "errors": []} for d in ALL_DST
    }
    # (src, dst) → list of bools
    pair_results: dict[tuple, list] = defaultdict(list)

    total = len(ALL_DST) * len(SOURCE_SQUARES) * n_reps
    done = 0

    for dst_name in ALL_DST:
        dst_xy = sq_xy(dst_name, bm)
        for src_name in SOURCE_SQUARES:
            if src_name == dst_name:
                continue
            src_xy = sq_xy(src_name, bm)

            for _ in range(n_reps):
                inner.grasp_mode = False
                inner.finger_target_joint = inner.FINGER_CLOSED_JOINT
                reposition_arm(inner, src_xy, safe_z)

                result = ctrl.run_transit(dst_xy)
                dst_stats[dst_name]["attempts"] += 1
                if result.success:
                    dst_stats[dst_name]["successes"] += 1
                dst_stats[dst_name]["errors"].append(result.error_mm)
                pair_results[(src_name, dst_name)].append(result.success)

                done += 1
                if done % 20 == 0:
                    print(f"  progress: {done}/{total}", flush=True)

    return dst_stats, pair_results


def print_results(dst_stats: dict, pair_results: dict):
    """Print results output."""
    print("\n" + "=" * 65)
    print(
        f"{'DESTINATION':<12} {'SUCCESS%':>9} {'ATTEMPTS':>9} {'AVG_ERR_MM':>12} {'STATUS':>10}"
    )
    print("-" * 65)

    warned = []
    for dst in ALL_DST:
        s = dst_stats[dst]
        if s["attempts"] == 0:
            continue
        pct = s["successes"] / s["attempts"] * 100
        avg_err = np.mean(s["errors"]) if s["errors"] else 0.0
        label = ""
        if dst in PROBLEM_DESTINATIONS:
            label = "<-- WAS BAD"
        status = "OK" if pct >= 95 else ("WARN" if pct >= 80 else "FAIL")
        print(
            f"{dst:<12} {pct:>8.1f}% {s['attempts']:>9}  {avg_err:>10.1f}   {status}  {label}"
        )
        if status != "OK":
            warned.append((dst, pct, s))

    print("=" * 65)

    if warned:
        print("\nFailed source→destination breakdown:")
        for dst, pct, s in warned:
            fails = [
                (src, results)
                for (src, d), results in pair_results.items()
                if d == dst and not all(results)
            ]
            fails.sort(key=lambda x: sum(x[1]) / len(x[1]))
            for src, results in fails[:10]:
                sr = sum(results) / len(results) * 100
                print(f"  {src} → {dst}: {sr:.0f}% ({sum(results)}/{len(results)})")
    else:
        print("\nAll destinations PASSED (≥95%).")


def main():
    """Parse command-line arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Targeted problematic-square transit evaluation."
    )
    parser.add_argument(
        "--transit-model",
        type=str,
        default=None,
        help="Path to transit model zip. Defaults to configs/deployed_models.yaml transit",
    )
    parser.add_argument(
        "--n-reps",
        type=int,
        default=3,
        help="Repetitions per (src, dst) pair (default: 3)",
    )
    args = parser.parse_args()

    cfg = load_config("deployed_models")
    env_cfg = load_config("env")
    transit_path = args.transit_model or cfg.get("transit")
    if not transit_path:
        print("ERROR: No transit model path found.")
        sys.exit(1)
    print(f"Transit model: {transit_path}")
    print(f"n_reps: {args.n_reps}")
    print(f"Destinations: {ALL_DST}")
    print(f"Sources: {SOURCE_SQUARES}")
    total = len(ALL_DST) * len(SOURCE_SQUARES) * args.n_reps
    print(f"Total episodes: {total}\n")

    env = gym.make("ChessFetchTask-v0", force_scenario="transit")
    inner = env.unwrapped
    bm = BoardMapper.from_configs()

    ctrl = ModelEmbeddedController(env)
    ctrl.load_model("transit", transit_path)

    env.reset()

    dst_stats, pair_results = run_eval(ctrl, inner, bm, env_cfg, args.n_reps)
    print_results(dst_stats, pair_results)

    env.close()


if __name__ == "__main__":
    main()
