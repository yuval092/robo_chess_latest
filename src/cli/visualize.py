"""Human-render loop using ScriptedController."""

from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

from src.chess_env.controller import ScriptedController
from src.physical.piece_registry import PieceRegistry
from src.physical.piece_teleport import PieceTeleporter
from src.utils.args import add_common_args


def _teleport_starting_pieces(env) -> None:
    """Teleport all active chess pieces to their starting squares."""
    teleporter = PieceTeleporter(env)
    for piece in PieceRegistry().all_pieces():
        teleporter.teleport_piece_to_square(piece.piece_id, piece.initial_square)


def main() -> None:
    """Run the visualization loop."""
    parser = argparse.ArgumentParser(description="Human-render visualization.")
    parser.add_argument(
        "--scenario",
        type=str,
        default="transit",
        choices=["transit", "descend", "ascend", "pick", "full_move"],
    )
    parser.add_argument("--src-xy", type=str, default="0.88 0.2641")
    parser.add_argument("--dst-xy", type=str, default="1.00 0.40")
    parser.add_argument("--episodes", type=int, default=0, help="0 = infinite")
    parser.add_argument(
        "--wait", action="store_true", help="Wait for Enter between episodes"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Teleport all pieces to starting squares before rendering.",
    )
    parser = add_common_args(parser)
    args = parser.parse_args()
    args.visualize = True

    src_xy = np.array([float(x) for x in args.src_xy.split()])
    dst_xy = np.array([float(x) for x in args.dst_xy.split()])

    if args.scenario in ("transit", "descend", "ascend"):
        force_scenario = args.scenario
    else:
        force_scenario = "transit"

    env = gym.make(
        "ChessFetchTask-v0",
        render_mode="human",
        force_scenario=force_scenario,
        debug=args.debug,
        show_chess_pieces=args.setup,
        hide_object=args.setup,
    )
    ctrl = ScriptedController(
        env,
        drift_limit=args.drift_limit,
        render_fn=env.render,
        render_delay=args.delay,
    )
    inner = env.unwrapped

    ep = 0
    try:
        while args.episodes == 0 or ep < args.episodes:
            ep += 1
            inner.force_start_pos = inner.HOME_POS.copy()
            if args.scenario in ("pick", "full_move"):
                inner.force_cube_pos = np.array(
                    [
                        src_xy[0],
                        src_xy[1],
                        inner.TABLE_SURFACE_Z + inner.CUBE_HEIGHT / 2,
                    ]
                )

            env.reset()
            if args.setup:
                _teleport_starting_pieces(env)
            env.render()

            result = None
            if args.scenario == "transit":
                target_xy = inner.goal_pos[:2].copy()
                result = ctrl.run_transit(target_xy)
            elif args.scenario == "descend":
                target_xy = inner.goal_pos[:2].copy()
                result = ctrl.run_descend(target_xy)
            elif args.scenario == "ascend":
                target_xy = inner.goal_pos[:2].copy()
                result = ctrl.run_ascend(target_xy)
            elif args.scenario == "pick":
                result = ctrl.run_pick_sequence(src_xy)
            elif args.scenario == "full_move":
                result = ctrl.run_full_move(src_xy, dst_xy)

            if result is None:
                raise RuntimeError(f"Unsupported scenario: {args.scenario}")
            res_str = "SUCCESS" if result.success else f"FAIL ({result.crash_reason})"
            print(f"Ep {ep}: {res_str}")

            if args.wait:
                input("Press Enter for next episode...")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
