"""
eval_all_cells_rl.py — All-cells-to-all-cells evaluation using ModelEmbeddedController.

Tests representative source/destination pairs across all 64 squares.
Records success/failure, which stage failed, and position patterns.

Usage:
    PYTHONPATH=. python scripts/eval_all_cells_rl.py [--n-reps 3] [--mode pawn|key|full]
    PYTHONPATH=. python scripts/eval_all_cells_rl.py --mode pawn --test-mode transit_only --n-reps 3
"""

import argparse
import os
import sys
import time
import traceback
from collections import defaultdict

import chess
import mujoco
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym

from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_game.board_mapper import BoardMapper
from src.utils.io import load_config

# ── Model loading ───────────────────────────────────────────────────────────────


def load_controller(env, deployed_cfg):
    """Load controller."""
    ctrl = ModelEmbeddedController(env)
    dp = deployed_cfg
    ctrl.load_model("transit", dp["transit"])
    ctrl.load_model("descend", dp["descend"])
    ctrl.load_model("ascend", dp["ascend"])
    return ctrl


# ── Per-episode helpers ─────────────────────────────────────────────────────────


def reposition_arm(inner, target_xy, safe_z):
    """Move arm to target_xy at safe_z using mocap."""
    target_pos = np.array([target_xy[0], target_xy[1], safe_z])
    inner._move_mocap_to(
        target_pos, inner.VERTICAL_QUAT, max_steps=250, tolerance=0.003
    )
    mujoco.mj_forward(inner.model, inner.data)


def run_transit_episode(ctrl, inner, src_xy, dst_xy, safe_z, env_cfg):
    """
    Reposition arm to src_xy, then run transit to dst_xy.
    Does NOT call env.reset() - uses _move_mocap_to for positioning.
    """
    inner.grasp_mode = False
    inner.finger_target_joint = inner.FINGER_CLOSED_JOINT

    # Move to src
    reposition_arm(inner, src_xy, safe_z)

    # Run transit
    result = ctrl.run_transit(dst_xy)
    return {
        "success": result.success,
        "failed_at": None if result.success else "transit",
        "transit": result,
        "descend": None,
        "ascend": None,
    }


def run_three_stage_episode(ctrl, inner, src_xy, dst_xy, safe_z, hover_z, env_cfg):
    """
    Full transit->descend->ascend test. Arm starts at src_xy, reaches dst_xy.
    """
    inner.grasp_mode = False
    inner.finger_target_joint = inner.FINGER_CLOSED_JOINT

    # Move to src
    reposition_arm(inner, src_xy, safe_z)

    # --- Transit ---
    transit_result = ctrl.run_transit(dst_xy)
    if not transit_result.success:
        return {
            "success": False,
            "failed_at": "transit",
            "transit": transit_result,
            "descend": None,
            "ascend": None,
        }

    # --- Descend ---
    nom_exit = np.array([dst_xy[0], dst_xy[1], safe_z])
    goal_descend = np.array([dst_xy[0], dst_xy[1], hover_z])
    ctrl.transition("descend", goal_descend, nom_exit, dst_xy)

    descend_result = ctrl.run_descend(dst_xy)
    if not descend_result.success:
        return {
            "success": False,
            "failed_at": "descend",
            "transit": transit_result,
            "descend": descend_result,
            "ascend": None,
        }

    # --- Ascend ---
    nom_exit_hover = np.array([dst_xy[0], dst_xy[1], hover_z])
    goal_ascend = np.array([dst_xy[0], dst_xy[1], safe_z])
    ctrl.transition("ascend", goal_ascend, nom_exit_hover, dst_xy)

    ascend_result = ctrl.run_ascend(dst_xy)
    return {
        "success": ascend_result.success,
        "failed_at": None if ascend_result.success else "ascend",
        "transit": transit_result,
        "descend": descend_result,
        "ascend": ascend_result,
    }


# ── Test pair selection ─────────────────────────────────────────────────────────


def get_test_pairs(mode: str, bm: BoardMapper):
    """Returns list of (src_name, dst_name, src_xy, dst_xy) tuples."""

    def sq_xy(name):
        """Return sq xy."""
        return bm.square_name_to_xy(name)

    pairs = []
    seen = set()

    def add_pair(src, dst):
        """Return add pair."""
        if src == dst:
            return
        k = (src, dst)
        if k not in seen:
            seen.add(k)
            pairs.append((src, dst, sq_xy(src), sq_xy(dst)))

    if mode in ("pawn", "full"):
        # White pawns (rank 2) → 3 destinations each
        for file_char in "abcdefgh":
            src = f"{file_char}2"
            for dst in [f"{file_char}4", f"{file_char}3", f"{file_char}5"]:
                add_pair(src, dst)

        # Black pawns (rank 7) → 3 destinations each
        for file_char in "abcdefgh":
            src = f"{file_char}7"
            for dst in [f"{file_char}5", f"{file_char}6", f"{file_char}4"]:
                add_pair(src, dst)

    if mode in ("key", "full"):
        key_squares = [
            # Corners
            "a1",
            "h1",
            "a8",
            "h8",
            # Near-corner pieces
            "b1",
            "g1",
            "b8",
            "g8",
            # Center
            "d4",
            "e4",
            "d5",
            "e5",
            # Mid-board edges
            "a4",
            "h4",
            "a5",
            "h5",
        ]
        for src in key_squares:
            for dst in key_squares:
                dist = np.linalg.norm(sq_xy(src) - sq_xy(dst))
                if dist >= 0.15:  # at least ~2 squares
                    add_pair(src, dst)

    if mode == "full":
        # Comprehensive: every file vs every rank, filtered for min distance
        for sf in "abcdefgh":
            for sr in "12345678":
                src = f"{sf}{sr}"
                for df in "abcdefgh":
                    for dr in "12345678":
                        dst = f"{df}{dr}"
                        dist = np.linalg.norm(sq_xy(src) - sq_xy(dst))
                        if dist >= 0.24:  # ~3 squares
                            add_pair(src, dst)

    return pairs


# ── Main evaluation ─────────────────────────────────────────────────────────────


def run_evaluation(pairs, ctrl, inner, env, env_cfg, n_reps=1, test_mode="three_stage"):
    """Run all (src, dst) pairs, n_reps times each."""
    safe_z = env_cfg["safe_z"]
    hover_z = env_cfg["hover_z"]

    results = []

    print(
        f"\n{'Pair':>12} | {'Rep':>4} | {'Result':>10} | {'Failed@':>8} | "
        f"{'Transit':>9} | {'Descend':>9} | {'Ascend':>9}"
    )
    print("-" * 82)

    for pair_idx, (src_name, dst_name, src_xy, dst_xy) in enumerate(pairs):
        for rep in range(n_reps):
            # Use env.reset() to start each episode cleanly
            obs, info = env.reset()
            inner.grasp_mode = False
            inner.finger_target_joint = inner.FINGER_CLOSED_JOINT

            try:
                if test_mode == "transit_only":
                    result = run_transit_episode(
                        ctrl, inner, src_xy, dst_xy, safe_z, env_cfg
                    )
                else:
                    result = run_three_stage_episode(
                        ctrl, inner, src_xy, dst_xy, safe_z, hover_z, env_cfg
                    )
            except Exception as e:
                result = {
                    "success": False,
                    "failed_at": "EXCEPTION",
                    "error": str(e),
                    "tb": traceback.format_exc(),
                    "transit": None,
                    "descend": None,
                    "ascend": None,
                }

            success = result["success"]
            failed_at = result.get("failed_at", "") or ""

            t_err = (
                f"{result['transit'].error_mm:.1f}mm"
                if result.get("transit")
                else "---"
            )
            d_err = (
                f"{result['descend'].error_mm:.1f}mm"
                if result.get("descend")
                else "---"
            )
            a_err = (
                f"{result['ascend'].error_mm:.1f}mm" if result.get("ascend") else "---"
            )

            status = "SUCCESS" if success else f"FAIL@{failed_at}"
            print(
                f"{src_name}->{dst_name:>4} | {rep + 1:>4} | {status:>10} | "
                f"{failed_at:>8} | {t_err:>9} | {d_err:>9} | {a_err:>9}"
            )
            sys.stdout.flush()

            if not success and "tb" in result:
                print(f"  EXCEPTION: {result['error']}")
                print(result["tb"][:300])

            results.append(
                {
                    "src": src_name,
                    "dst": dst_name,
                    "src_xy": src_xy.copy(),
                    "dst_xy": dst_xy.copy(),
                    "rep": rep,
                    "success": success,
                    "failed_at": failed_at,
                    "result": result,
                }
            )

    return results


def analyze_results(results):
    """Analyze patterns in results and print summary."""
    pair_stats = defaultdict(lambda: {"successes": 0, "total": 0, "failures": []})
    src_stats = defaultdict(lambda: {"successes": 0, "total": 0})
    dst_stats = defaultdict(lambda: {"successes": 0, "total": 0})
    stage_failures = defaultdict(int)

    for r in results:
        k = (r["src"], r["dst"])
        pair_stats[k]["total"] += 1
        src_stats[r["src"]]["total"] += 1
        dst_stats[r["dst"]]["total"] += 1

        if r["success"]:
            pair_stats[k]["successes"] += 1
            src_stats[r["src"]]["successes"] += 1
            dst_stats[r["dst"]]["successes"] += 1
        else:
            pair_stats[k]["failures"].append(r["failed_at"])
            stage_failures[r["failed_at"] or "unknown"] += 1

    total = len(results)
    successes = sum(1 for r in results if r["success"])

    print(f"\n{'=' * 82}")
    print(f"OVERALL: {successes}/{total} ({successes / total:.1%}) success")

    print("\nStage failure breakdown:")
    for stage, count in sorted(stage_failures.items(), key=lambda x: -x[1]):
        print(f"  {stage}: {count} failures ({count / total:.1%} of total)")

    print("\nFailing pairs (success < 100%):")
    failing_pairs = [
        (k, v) for k, v in pair_stats.items() if v["successes"] < v["total"]
    ]
    failing_pairs.sort(key=lambda x: x[1]["successes"] / x[1]["total"])
    for (src, dst), stats in failing_pairs[:40]:
        rate = stats["successes"] / stats["total"]
        print(
            f"  {src}->{dst}: {stats['successes']}/{stats['total']} ({rate:.0%}) "
            f"failures={stats['failures']}"
        )

    print("\nWorst source squares:")
    src_fail = [(sq, v) for sq, v in src_stats.items() if v["successes"] < v["total"]]
    src_fail.sort(key=lambda x: x[1]["successes"] / x[1]["total"])
    for sq, stats in src_fail[:20]:
        rate = stats["successes"] / stats["total"]
        print(f"  {sq}: {stats['successes']}/{stats['total']} ({rate:.0%})")

    print("\nWorst destination squares:")
    dst_fail = [(sq, v) for sq, v in dst_stats.items() if v["successes"] < v["total"]]
    dst_fail.sort(key=lambda x: x[1]["successes"] / x[1]["total"])
    for sq, dst_s in dst_fail[:20]:
        rate = dst_s["successes"] / dst_s["total"]
        print(f"  {sq}: {dst_s['successes']}/{dst_s['total']} ({rate:.0%})")

    # 8x8 grid for destination success rates
    print("\n8x8 board map (destination, % success rate):")
    print("       a     b     c     d     e     f     g     h")
    for rank in range(7, -1, -1):
        row = f"  {rank + 1}  "
        for file_idx in range(8):
            sq_name = chess.square_name(chess.square(file_idx, rank))
            s = dst_stats.get(sq_name, {"successes": 0, "total": 0})
            if s["total"] == 0:
                row += "  -- "
            else:
                rate = s["successes"] / s["total"]
                if rate == 1.0:
                    row += " 100  "
                elif rate == 0.0:
                    row += "   0  "
                else:
                    row += f" {rate * 100:3.0f}  "
        print(row)

    # 8x8 grid for source success rates
    print("\n8x8 board map (source square, % success rate):")
    print("       a     b     c     d     e     f     g     h")
    for rank in range(7, -1, -1):
        row = f"  {rank + 1}  "
        for file_idx in range(8):
            sq_name = chess.square_name(chess.square(file_idx, rank))
            s = src_stats.get(sq_name, {"successes": 0, "total": 0})
            if s["total"] == 0:
                row += "  -- "
            else:
                rate = s["successes"] / s["total"]
                if rate == 1.0:
                    row += " 100  "
                elif rate == 0.0:
                    row += "   0  "
                else:
                    row += f" {rate * 100:3.0f}  "
        print(row)

    return pair_stats, src_stats, dst_stats


def main():
    """Parse command-line arguments and run the script."""
    parser = argparse.ArgumentParser(description="All-cells RL model evaluation.")
    parser.add_argument("--n-reps", type=int, default=3, help="Repetitions per pair")
    parser.add_argument("--mode", choices=["pawn", "key", "full"], default="pawn")
    parser.add_argument(
        "--test-mode", choices=["transit_only", "three_stage"], default="three_stage"
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    args = parser.parse_args()

    env_cfg = load_config("env")
    deployed_cfg = load_config("deployed_models")
    bm = BoardMapper.from_configs()

    pairs = get_test_pairs(args.mode, bm)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]

    print(f"Mode: {args.mode}, test_mode: {args.test_mode}")
    print(
        f"Total pairs: {len(pairs)}, reps/pair: {args.n_reps}, total episodes: {len(pairs) * args.n_reps}"
    )

    env = gym.make(
        "ChessFetchTask-v0",
        force_scenario="transit",
        show_chess_pieces=True,
        hide_object=True,
        render_mode=None,
    )

    ctrl = load_controller(env, deployed_cfg)
    inner = env.unwrapped

    t_start = time.time()
    results = run_evaluation(
        pairs, ctrl, inner, env, env_cfg, n_reps=args.n_reps, test_mode=args.test_mode
    )
    elapsed = time.time() - t_start

    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / len(results):.2f}s/episode)")
    analyze_results(results)

    env.close()
    failures = [result for result in results if not result["success"]]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
