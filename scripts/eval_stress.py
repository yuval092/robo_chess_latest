"""
eval_stress.py — Corner/position stress test using ScriptedController.

Usage:
    python scripts/eval_stress.py
        [--chain full_move|pick|vertical]
        [--n-episodes 10] [--drift-limit 0.010]
        [--grid] [--grid-size 3]
        [--visualize] [--delay 0.0] [--debug]

--grid: Test a grid of NxN positions across the board (instead of just corners+center)
"""

import argparse

import numpy as np

from src.chess_env.controller import ScriptedController, SequenceResult
from src.chess_env.waypoints import HOVER_Z, SAFE_Z
from src.utils.args import add_common_args, make_env
from src.utils.io import load_config


def get_test_positions(args) -> list:
    """Return list of (name, src_xy, dst_xy) test cases."""
    chess_cfg = load_config("chess")
    board_cx, board_cy = chess_cfg["board"]["center_xy"]
    cell = chess_cfg["board"]["cell_size_m"]
    n_cells = chess_cfg["board"]["board_size"]
    half = n_cells * cell / 2

    # Chess square center boundaries: first/last square centres are half-cell inside the board edge
    board_min_x = board_cx - half + cell / 2
    board_max_x = board_cx + half - cell / 2
    board_min_y = board_cy - half + cell / 2
    board_max_y = board_cy + half - cell / 2

    center = np.array([board_cx, board_cy])

    if args.grid:
        n = args.grid_size
        xs = np.linspace(board_min_x, board_max_x, n)
        ys = np.linspace(board_min_y, board_max_y, n)
        positions = []
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                positions.append((f"grid_{xi}_{yi}", np.array([x, y])))
        return [(name, src, center) for name, src in positions]
    else:
        # 4 corner chess square centres + board centre
        corners = [
            ("a1", np.array([board_min_x, board_min_y])),
            ("a8", np.array([board_min_x, board_max_y])),
            ("h1", np.array([board_max_x, board_min_y])),
            ("h8", np.array([board_max_x, board_max_y])),
            ("center", center),
        ]
        return [(name, src, center) for name, src in corners]


def run_position(name, src_xy, dst_xy, args) -> dict:
    """Run N episodes at a specific source position."""
    hide_object = args.chain not in ("pick", "full_move")
    env = make_env(args, force_scenario="transit", hide_object=hide_object)
    render_fn = env.render if args.visualize else None
    ctrl = ScriptedController(
        env, drift_limit=args.drift_limit, render_fn=render_fn, render_delay=args.delay
    )
    inner = env.unwrapped

    successes = 0
    failures = []

    for _ in range(args.n_episodes):
        inner.force_start_pos = inner.HOME_POS.copy()
        if args.chain in ("full_move", "pick"):
            inner.force_cube_pos = np.array(
                [src_xy[0], src_xy[1], inner.TABLE_SURFACE_Z + inner.CUBE_HEIGHT / 2]
            )
        obs, _ = env.reset()

        if args.chain == "full_move":
            result = ctrl.run_full_move(src_xy, dst_xy)
        elif args.chain == "pick":
            result = ctrl.run_pick_sequence(src_xy)
        elif args.chain == "vertical":
            # Transition manually to descend
            nom_exit = np.array([src_xy[0], src_xy[1], SAFE_Z])
            goal_descend = np.array([src_xy[0], src_xy[1], HOVER_Z])
            ctrl.transition("descend", goal_descend, nom_exit, src_xy)
            d = ctrl.run_descend(src_xy)
            a = ctrl.run_ascend(src_xy) if d.success else None

            result = SequenceResult(
                success=d.success and (a is not None and a.success),
                stage_results=[("descend", d)] + ([("ascend", a)] if a else []),
                failed_at=None
                if (d.success and a and a.success)
                else ("descend" if not d.success else "ascend"),
                grasp_quality=None,
            )

        if result.success:
            successes += 1
        else:
            failures.append(result.failed_at or "unknown")

    env.close()
    return {
        "name": name,
        "src_xy": src_xy,
        "successes": successes,
        "n": args.n_episodes,
        "rate": successes / args.n_episodes if args.n_episodes > 0 else 0,
        "failure_stages": failures,
    }


def main():
    """Parse command-line arguments and run the script."""
    parser = argparse.ArgumentParser(description="Corner/grid stress test.")
    parser.add_argument(
        "--chain", type=str, default="pick", choices=["full_move", "pick", "vertical"]
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Test a grid of positions instead of corners+center",
    )
    parser.add_argument(
        "--grid-size", type=int, default=3, help="Grid dimension NxN (default: 3)"
    )
    parser = add_common_args(parser)
    args = parser.parse_args()

    test_positions = get_test_positions(args)
    print(
        f"Stress test: {len(test_positions)} positions × {args.n_episodes} episodes = "
        f"{len(test_positions) * args.n_episodes} total"
    )

    all_results = []
    for name, src_xy, dst_xy in test_positions:
        print(f"  Testing {name} {src_xy}...")
        r = run_position(name, src_xy, dst_xy, args)
        all_results.append(r)
        print(f"    {r['successes']}/{r['n']} ({r['rate']:.0%})")

    # Summary
    print(f"\n{'=' * 55}")
    print(f"{'Position':<18} {'XY':>20} {'Rate':>8}")
    print("-" * 55)
    for r in all_results:
        xy_str = f"({r['src_xy'][0]:.3f}, {r['src_xy'][1]:.3f})"
        status = "OK" if r["rate"] >= 0.8 else "LOW"
        print(f"{r['name']:<18} {xy_str:>20} {r['rate']:>7.0%}  {status}")

    total_episodes = sum(r["n"] for r in all_results)
    if total_episodes > 0:
        overall = sum(r["successes"] for r in all_results) / total_episodes
        print(f"\nOverall success rate: {overall:.1%}")
    print("=" * 55)


if __name__ == "__main__":
    main()
