"""robo-chess-eval-stage — per-stage waypoint accuracy evaluation."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from tqdm import tqdm

import src.chess_env  # noqa: F401 — registers ChessFetchTask-v0
from src.chess_env.model_controller import ModelEmbeddedController
from src.utils.args import add_eval_args, model_overrides_from_args, resolve_model_paths


# ── Chain definitions ──────────────────────────────────────────────────────

_CHAINS: dict[str, list[str]] = {
    "full_move": ["transit", "descend", "ascend"],
    "pick": ["transit", "descend"],
    "vertical": ["descend", "ascend"],
}

_STAGE_RUN = {
    "transit": lambda ctrl, xy: ctrl.run_transit(xy),
    "descend": lambda ctrl, xy: ctrl.run_descend(xy),
    "ascend": lambda ctrl, xy: ctrl.run_ascend(xy),
}


# ── Core evaluation loop ───────────────────────────────────────────────────

def _eval_one_stage(
    stage: str,
    episodes: int,
    model_overrides: dict[str, str],
    drift_limit: float | None,
    debug: bool,
    visualize: bool = False,
    delay: float = 0.0,
) -> dict:
    """
    Run `episodes` episodes for a single stage and return a metrics dict.

    Returns
    -------
    dict with keys:
        stage, n_episodes, success_rate, crash_rate, timeout_rate,
        avg_error_mm, p95_error_mm, avg_steps, crash_reasons
    """
    env = gym.make(
        "ChessFetchTask-v0",
        render_mode="human" if visualize else None,
        force_scenario=stage,
        hide_object=True,
        debug=debug,
    )
    if drift_limit is not None:
        env.unwrapped.env_cfg["eval_drift_limit"] = drift_limit

    render_fn = env.render if visualize else None
    model_paths = resolve_model_paths(model_overrides)
    ctrl = ModelEmbeddedController(env=env, render_fn=render_fn, render_delay=delay)
    ctrl.load_model(stage, model_paths[stage])

    inner = env.unwrapped
    run_stage = _STAGE_RUN[stage]

    successes = 0
    crashes = 0
    timeouts = 0
    errors_mm: list[float] = []
    step_counts: list[int] = []
    crash_reasons: dict[str, int] = {}

    print(
        f"[eval-stage] stage={stage}  controller=model-hybrid  episodes={episodes}"
    )

    try:
        with tqdm(total=episodes, desc=f"{stage:8s}", unit="ep") as pbar:
            for _ in range(episodes):
                env.reset()
                target_xy = inner.goal_pos[:2].copy()
                result = run_stage(ctrl, target_xy)

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

                pbar.update(1)
                pbar.set_postfix({"ok%": f"{100 * successes / pbar.n:.0f}"})
    finally:
        env.close()

    n = episodes or 1
    return {
        "stage": stage,
        "n_episodes": episodes,
        "success_rate": successes / n,
        "crash_rate": crashes / n,
        "timeout_rate": timeouts / n,
        "avg_error_mm": float(np.mean(errors_mm)) if errors_mm else 0.0,
        "p95_error_mm": float(np.percentile(errors_mm, 95)) if errors_mm else 0.0,
        "avg_steps": float(np.mean(step_counts)) if step_counts else 0.0,
        "crash_reasons": crash_reasons,
    }


# ── Summary printer ────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    header = (
        f"\n{'Stage':<12} {'Success':>8} {'Crash':>8} {'Timeout':>9}"
        f" {'AvgErr':>9} {'P95Err':>9} {'Steps':>7}"
    )
    sep = "=" * 68
    print(sep)
    print(header)
    print("-" * 68)
    for r in results:
        print(
            f"{r['stage']:<12} "
            f"{r['success_rate']:>7.1%} "
            f"{r['crash_rate']:>7.1%} "
            f"{r['timeout_rate']:>8.1%} "
            f"{r['avg_error_mm']:>8.2f}mm "
            f"{r['p95_error_mm']:>8.2f}mm "
            f"{r['avg_steps']:>6.1f}"
        )
    print(sep)

    for r in results:
        if r["crash_reasons"]:
            print(f"\n  {r['stage']} crash breakdown:")
            for reason, count in sorted(
                r["crash_reasons"].items(), key=lambda x: -x[1]
            ):
                pct = 100.0 * count / r["n_episodes"]
                print(f"    {reason:<50s}  {count:4d}  ({pct:.1f}%)")
    print()


# ── Public callable API ────────────────────────────────────────────────────

def run_eval_stage(
    stages: list[str],
    episodes: int,
    model_overrides: dict[str, str] | None = None,
    drift_limit: float | None = None,
    debug: bool = False,
    visualize: bool = False,
    delay: float = 0.0,
) -> list[dict]:
    """
    Evaluate one or more waypoint stages and print a summary table.

    Parameters
    ----------
    stages : list of "transit" | "descend" | "ascend"
    episodes : episodes to run per stage
    model_overrides : per-stage model path overrides, e.g. {"transit": "/path/to/model.zip"}
    drift_limit : tube constraint radius override (metres); None = use config default
    debug : enable verbose per-step environment logs
    visualize : open the MuJoCo viewer window
    delay : per-step render delay in seconds (only used when visualize=True)

    Returns
    -------
    list of per-stage metrics dicts (same order as `stages`)
    """
    model_overrides = model_overrides or {}
    results = []
    for stage in stages:
        r = _eval_one_stage(
            stage=stage,
            episodes=episodes,
            model_overrides=model_overrides,
            drift_limit=drift_limit,
            debug=debug,
            visualize=visualize,
            delay=delay,
        )
        results.append(r)
    _print_summary(results)
    return results


# ── Entry point ────────────────────────────────────────────────────────────

_STAGE_HELP = (
    "Which stage(s) to evaluate. "
    "Single stages: transit, descend, ascend. "
    "Shortcuts: "
    "  all        = transit + descend + ascend (default); "
    "  full_move  = transit + descend + ascend (same as all, full pipeline name); "
    "  pick       = transit + descend; "
    "  vertical   = descend + ascend."
)

# Extended stage choices include the chain shortcuts so --stage is the single flag.
_STAGE_CHOICES = ["transit", "descend", "ascend", "all"] + list(_CHAINS.keys())


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robo-chess-eval-stage",
        description=(
            "Evaluate waypoint-stage accuracy for RoboChess.\n\n"
            "Runs N episodes per stage and reports:\n"
            "  - Success rate, crash rate, timeout rate\n"
            "  - Average and P95 positional error\n"
            "  - Average step count\n"
            "  - Per-crash-reason breakdown\n\n"
            "Examples:\n"
            "  robo-chess-eval-stage --stage transit --episodes 100\n"
            "  robo-chess-eval-stage --stage transit --transit-model models/transit.zip\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--stage",
        choices=_STAGE_CHOICES,
        default="all",
        metavar="STAGE",
        help=_STAGE_HELP + f"  Choices: {', '.join(_STAGE_CHOICES)}. (default: all)",
    )

    p.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes to run per stage. (default: 50)",
    )
    add_eval_args(p)
    return p


def main() -> None:
    """Entry point for robo-chess-eval-stage."""
    args = _build_parser().parse_args()

    # Resolve stage argument (includes chain shortcuts)
    if args.stage in _CHAINS:
        stages = _CHAINS[args.stage]
    elif args.stage == "all":
        stages = ["transit", "descend", "ascend"]
    else:
        stages = [args.stage]

    run_eval_stage(
        stages=stages,
        episodes=args.episodes,
        model_overrides=model_overrides_from_args(args),
        drift_limit=args.drift_limit,
        debug=args.debug,
        visualize=args.visualize,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
