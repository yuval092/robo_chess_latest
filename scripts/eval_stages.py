"""
eval_stages.py — Per-stage accuracy evaluation using ScriptedController.

Usage:
    python scripts/eval_stages.py [--stages transit,descend,ascend]
        [--n-episodes 50] [--drift-limit 0.010] [--visualize] [--delay 0.02] [--debug]
"""
import sys, os, argparse, time
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

import src.chess_env
from src.chess_env.controller import ScriptedController
from src.utils.args import add_common_args, make_env

STAGE_CHOICES = ["transit", "descend", "ascend"]


def evaluate_stage(stage: str, args) -> dict:
    """Run N episodes of a single stage and collect accuracy statistics."""
    env = make_env(args, force_scenario=stage)
    ctrl = ScriptedController(env, drift_limit=args.drift_limit)
    inner = env.unwrapped

    successes = 0
    crashes = 0
    timeouts = 0
    errors_mm = []
    step_counts = []
    crash_reasons = {}

    for ep in range(args.n_episodes):
        obs, info = env.reset()
        target_xy = inner.goal_pos[:2].copy()

        if stage == "transit":
            result = ctrl.run_transit(target_xy)
        elif stage == "descend":
            result = ctrl.run_descend(target_xy)
        elif stage == "ascend":
            result = ctrl.run_ascend(target_xy)

        if result.success:
            successes += 1
            errors_mm.append(result.error_mm)
        elif result.crash_reason == "TIMEOUT":
            timeouts += 1
        else:
            crashes += 1
            reason = result.crash_reason or "UNKNOWN"
            crash_reasons[reason] = crash_reasons.get(reason, 0) + 1
        step_counts.append(result.steps)

    env.close()
    return {
        "stage": stage,
        "n_episodes": args.n_episodes,
        "success_rate": successes / args.n_episodes if args.n_episodes > 0 else 0,
        "crash_rate": crashes / args.n_episodes if args.n_episodes > 0 else 0,
        "timeout_rate": timeouts / args.n_episodes if args.n_episodes > 0 else 0,
        "avg_error_mm": np.mean(errors_mm) if errors_mm else 0.0,
        "p95_error_mm": float(np.percentile(errors_mm, 95)) if errors_mm else 0.0,
        "avg_steps": np.mean(step_counts) if step_counts else 0,
        "crash_reasons": crash_reasons,
    }


def print_summary(results: list):
    print("\n" + "=" * 70)
    print(f"{'Stage':<12} {'Success':>8} {'Crash':>8} {'Timeout':>9} {'AvgErr':>9} {'P95Err':>9}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['stage']:<12} "
            f"{r['success_rate']:>7.1%} "
            f"{r['crash_rate']:>7.1%} "
            f"{r['timeout_rate']:>8.1%} "
            f"{r['avg_error_mm']:>8.1f}mm "
            f"{r['p95_error_mm']:>8.1f}mm"
        )
        if r["crash_reasons"]:
            for reason, count in r["crash_reasons"].items():
                print(f"  Crash breakdown: {reason}: {count}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Per-stage accuracy evaluation.")
    parser.add_argument(
        "--stages", type=str, default="transit,descend,ascend",
        help="Comma-separated list of stages to evaluate"
    )
    parser = add_common_args(parser)
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip() in STAGE_CHOICES]
    if not stages:
        print(f"No valid stages specified. Choose from: {STAGE_CHOICES}")
        sys.exit(1)

    results = []
    for stage in stages:
        print(f"\nEvaluating stage: {stage} ({args.n_episodes} episodes)...")
        result = evaluate_stage(stage, args)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
s = []
    for stage in stages:
        print(f"\nEvaluating stage: {stage} ({args.n_episodes} episodes)...")
        result = evaluate_stage(stage, args)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
