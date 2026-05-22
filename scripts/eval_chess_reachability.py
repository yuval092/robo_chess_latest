"""
Exact chess-board reachability sweep.

Validates all 64 configured 8cm chess square centers through transit,
descend, and ascend using the scripted waypoint controller.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.append(os.getcwd())

import src.chess_env
from src.chess_env.controller import ScriptedController
from src.chess_game.board_mapper import BoardMapper
from src.utils.args import add_common_args, make_env


REQUIRED_RANK1_X = 0.600
REQUIRED_RANK8_X = 1.160
GEOMETRY_TOLERANCE_M = 1e-9


def validate_rank_centers(mapper: BoardMapper) -> None:
    rank1_x = mapper.square_name_to_xy("a1")[0]
    rank8_x = mapper.square_name_to_xy("a8")[0]
    print(f"Rank center x: rank1={rank1_x:.3f}m rank8={rank8_x:.3f}m")
    if not math.isclose(rank1_x, REQUIRED_RANK1_X, abs_tol=GEOMETRY_TOLERANCE_M):
        raise SystemExit(f"rank1 center x must be {REQUIRED_RANK1_X:.3f}m, got {rank1_x:.6f}m")
    if not math.isclose(rank8_x, REQUIRED_RANK8_X, abs_tol=GEOMETRY_TOLERANCE_M):
        raise SystemExit(f"rank8 center x must be {REQUIRED_RANK8_X:.3f}m, got {rank8_x:.6f}m")


def run_square(square_name: str, square_xy: np.ndarray, args, env, ctrl) -> list[dict]:
    results = []
    inner = env.unwrapped

    for episode in range(args.n_episodes):
        inner.force_start_pos = inner.HOME_POS.copy()
        env.reset()

        stage_results = []
        transit = ctrl.run_transit(square_xy)
        stage_results.append(("transit", transit))

        if transit.success:
            descend_goal = np.array([square_xy[0], square_xy[1], inner.HOVER_Z])
            transit_exit = np.array([square_xy[0], square_xy[1], inner.SAFE_Z])
            try:
                ctrl.transition("descend", descend_goal, transit_exit, square_xy)
                descend = ctrl.run_descend(square_xy)
            except Exception as exc:
                descend = None
                stage_results.append(("descend", exc))
            else:
                stage_results.append(("descend", descend))

            if descend is not None and descend.success:
                ascend_goal = np.array([square_xy[0], square_xy[1], inner.SAFE_Z])
                descend_exit = np.array([square_xy[0], square_xy[1], inner.HOVER_Z])
                try:
                    ctrl.transition("ascend", ascend_goal, descend_exit, square_xy)
                    ascend = ctrl.run_ascend(square_xy)
                except Exception as exc:
                    stage_results.append(("ascend", exc))
                else:
                    stage_results.append(("ascend", ascend))

        success = all(
            not isinstance(result, Exception) and result.success
            for _, result in stage_results
        )
        results.append(
            {
                "square": square_name,
                "episode": episode,
                "success": success,
                "stages": stage_results,
            }
        )

    return results


def format_stage(result) -> str:
    if isinstance(result, Exception):
        return f"EXCEPTION {result}"
    if result.success:
        return f"ok {result.error_mm:.1f}mm"
    return f"FAIL {result.crash_reason} {result.error_mm:.1f}mm"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exact 8cm chess-square reachability.")
    parser.add_argument("--all-squares", action="store_true", help="Validate all 64 squares.")
    parser.add_argument("--squares", nargs="*", help="Optional square names, e.g. a1 e4 h8.")
    parser = add_common_args(parser)
    args = parser.parse_args()

    mapper = BoardMapper.from_configs()
    validate_rank_centers(mapper)
    centers = mapper.all_square_centers()
    if args.squares:
        selected = {name: centers[name] for name in args.squares}
    elif args.all_squares:
        selected = centers
    else:
        selected = {name: centers[name] for name in ("a1", "h1", "a8", "h8", "d4", "e5")}

    print(f"Chess reachability: {len(selected)} squares × {args.n_episodes} episode(s)")
    failures = []
    max_error_mm = 0.0

    env = make_env(args, force_scenario="transit", hide_object=True)
    render_fn = env.render if args.visualize else None
    ctrl = ScriptedController(
        env,
        drift_limit=args.drift_limit,
        render_fn=render_fn,
        render_delay=args.delay,
    )

    try:
        for square_name, xy in selected.items():
            square_results = run_square(square_name, xy, args, env, ctrl)
            for result in square_results:
                stage_text = []
                for stage, stage_result in result["stages"]:
                    if not isinstance(stage_result, Exception):
                        max_error_mm = max(max_error_mm, stage_result.error_mm)
                    stage_text.append(f"{stage}={format_stage(stage_result)}")
                status = "OK" if result["success"] else "FAIL"
                print(f"{square_name:>2} ({xy[0]:.3f}, {xy[1]:.3f}) ep={result['episode']} {status}  " + "  ".join(stage_text))
                if not result["success"]:
                    failures.append(result)
    finally:
        env.close()

    print(f"\nMax final stage error: {max_error_mm:.1f}mm")
    if failures:
        print(f"FAILED: {len(failures)} square episode(s) were unreachable.")
        sys.exit(1)
    print("All selected exact-8cm chess square centers are reachable.")


if __name__ == "__main__":
    main()
