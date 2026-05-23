"""
eval_sequence.py — Full scenario chain evaluation using ScriptedController.

Usage:
    python scripts/eval_sequence.py
        [--chain full_move|pick|vertical]
        [--src-xy "0.88 0.2641"] [--dst-xy "1.00 0.40"]
        [--n-episodes 20] [--drift-limit 0.010]
        [--visualize] [--delay 0.02] [--debug]
"""
import sys, os, argparse, time
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

import src.chess_env
from src.chess_env.controller import ScriptedController, SequenceResult
from src.utils.args import add_common_args, make_env

CHAIN_CHOICES = ["full_move", "pick", "vertical"]


def run_sequence_episodes(args, src_xy, dst_xy) -> list:
    """Run N episodes of the specified chain. Returns list of SequenceResult."""
    hide_object = args.chain not in ("pick", "full_move")
    env = make_env(args, force_scenario="transit", hide_object=hide_object)
    render_fn = env.render if args.visualize else None
    if args.use_rl_models:
        from src.chess_env.model_controller import ModelEmbeddedController
        from src.utils.config import load_config

        train_cfg = load_config("training")
        model_paths = train_cfg.get("deployed_models", {})
        ctrl = ModelEmbeddedController(
            env=env,
            render_fn=render_fn,
            render_delay=args.delay,
        )
        ctrl.load_available(
            transit_path=args.transit_model or model_paths.get("transit"),
            descend_path=args.descend_model or model_paths.get("descend"),
            ascend_path=args.ascend_model or model_paths.get("ascend"),
        )
    else:
        ctrl = ScriptedController(env, drift_limit=args.drift_limit,
                                  render_fn=render_fn, render_delay=args.delay)
    inner = env.unwrapped
    results = []

    for ep in range(args.n_episodes):
        # Force start near home_pos
        inner.force_start_pos = inner.HOME_POS.copy()
        if args.chain in ("full_move", "pick"):
            inner.force_cube_pos = np.array([src_xy[0], src_xy[1],
                                              inner.TABLE_SURFACE_Z + inner.CUBE_HEIGHT / 2])
        obs, _ = env.reset()

        if args.chain == "full_move":
            result = ctrl.run_full_move(src_xy, dst_xy)
        elif args.chain == "pick":
            result = ctrl.run_pick_sequence(src_xy)
        elif args.chain == "vertical":
            from src.chess_env.waypoints import SAFE_Z, HOVER_Z
            nom_exit = np.array([src_xy[0], src_xy[1], SAFE_Z])
            goal_descend = np.array([src_xy[0], src_xy[1], HOVER_Z])
            ctrl.transition("descend", goal_descend, nom_exit, src_xy)
            d = ctrl.run_descend(src_xy)
            a = ctrl.run_ascend(src_xy) if d.success else None
            
            result = SequenceResult(
                success=d.success and (a is not None and a.success),
                stage_results=[("descend", d)] + ([("ascend", a)] if a else []),
                failed_at=None if (d.success and a and a.success) else ("descend" if not d.success else "ascend"),
                grasp_quality=None
            )
        results.append(result)

    env.close()
    return results


def print_summary(results: list, chain: str):
    n = len(results)
    if n == 0:
        print("No results to summarize.")
        return
    successes = sum(r.success for r in results)
    print(f"\n{'='*65}")
    print(f"Chain: {chain}  |  Episodes: {n}  |  Full success: {successes}/{n} ({successes/n:.1%})")

    # Per-stage breakdown
    stage_stats = {}
    for r in results:
        for name, sr in r.stage_results:
            if name not in stage_stats:
                stage_stats[name] = {"ok": 0, "fail": 0, "reasons": {}}
            if sr.success:
                stage_stats[name]["ok"] += 1
            else:
                stage_stats[name]["fail"] += 1
                reason = sr.crash_reason or "UNKNOWN"
                stage_stats[name]["reasons"][reason] = stage_stats[name]["reasons"].get(reason, 0) + 1

    print(f"\n{'Stage':<12} {'Success':>9} {'Fail':>7}")
    print("-" * 35)
    for name, s in stage_stats.items():
        total = s["ok"] + s["fail"]
        print(f"{name:<12} {s['ok']:>5}/{total}    ", end="")
        if s["reasons"]:
            print(", ".join(f"{r}×{c}" for r, c in s["reasons"].items()))
        else:
            print()
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Full scenario chain evaluation.")
    parser.add_argument("--chain", type=str, default="full_move", choices=CHAIN_CHOICES)
    parser.add_argument("--src-xy", type=str, default="0.88 0.2641",
                        help="Source XY: two space-separated floats")
    parser.add_argument("--dst-xy", type=str, default="1.00 0.40",
                        help="Destination XY: two space-separated floats")
    parser.add_argument(
        "--use-rl-models",
        action="store_true",
        help="Use RL models instead of the scripted controller",
    )
    parser.add_argument("--transit-model", type=str, default=None)
    parser.add_argument("--descend-model", type=str, default=None)
    parser.add_argument("--ascend-model", type=str, default=None)
    parser = add_common_args(parser)
    args = parser.parse_args()

    src_xy = np.array([float(x) for x in args.src_xy.split()])
    dst_xy = np.array([float(x) for x in args.dst_xy.split()])

    print(f"Chain: {args.chain}, src={src_xy}, dst={dst_xy}, n={args.n_episodes}")
    results = run_sequence_episodes(args, src_xy, dst_xy)
    print_summary(results, args.chain)


if __name__ == "__main__":
    main()
