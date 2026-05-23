"""
Exhaustive physical board-square reachability check.

Runs a real scripted pick/place cycle for one physical piece from each source
square to each destination square. Other pieces are moved out of the way so the
test measures arm reachability/tangling, not chess legality or crowded-board
collisions.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import chess
import gymnasium as gym
import numpy as np

sys.path.append(os.getcwd())

import src.chess_env
from src.chess_env.controller import ScriptedController
from src.chess_env.task import HOME_POSTURE_JOINTS
from src.chess_game.board_mapper import BoardMapper
from src.physical.movement_executor import MovementExecutor
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry
from src.physical.piece_teleport import PieceTeleporter
from src.physical.plan_executor import PhysicalPlanExecutor


@dataclass
class MoveCheckFailure:
    src: str
    dst: str
    kind: str
    error: str | None
    stages: str


@dataclass
class MoveCheckSummary:
    total: int
    passed: int
    failures: list[MoveCheckFailure]


def square_names() -> list[str]:
    return [chess.square_name(square) for square in chess.SQUARES]


def robot_home_qpos(env) -> np.ndarray:
    return np.array([
        env.data.qpos[env.model.joint(name).qposadr[0]]
        for name in HOME_POSTURE_JOINTS
    ])


def hide_other_pieces(env, teleporter: PieceTeleporter, registry: PieceRegistry, moving_piece_id: str) -> None:
    z = env.TABLE_Z + env.CUBE_HEIGHT / 2.0
    index = 0
    for piece_id in registry.starting_square_map():
        if piece_id == moving_piece_id:
            continue
        x = 2.0 + 0.05 * (index // 8)
        y = 2.0 + 0.05 * (index % 8)
        teleporter.teleport_piece_to_xyz(piece_id, np.array([x, y, z]))
        index += 1


def format_stages(stage_results: list) -> str:
    return ", ".join(
        f"{name}:{stage.success}:{stage.error_mm:.1f}mm:{stage.crash_reason}"
        for name, stage in stage_results
    )


def selected_pairs(from_square: str | None, to_square: str | None, include_same: bool) -> list[tuple[str, str]]:
    sources = [from_square] if from_square else square_names()
    destinations = [to_square] if to_square else square_names()
    pairs = []
    for src in sources:
        for dst in destinations:
            if src == dst and not include_same:
                continue
            pairs.append((src, dst))
    return pairs


def run_all_square_moves(
    *,
    piece_id: str,
    from_square: str | None = None,
    to_square: str | None = None,
    include_same: bool = False,
    max_cases: int | None = None,
    stop_on_failure: bool = False,
    check_home: bool = True,
    home_tolerance: float = 1e-9,
    drift_limit: float = 0.010,
    visualize: bool = False,
    delay: float = 0.0,
) -> MoveCheckSummary:
    pairs = selected_pairs(from_square, to_square, include_same)
    if max_cases is not None:
        pairs = pairs[:max_cases]

    render_mode = "human" if visualize else None
    env = gym.make(
        "ChessFetchTask-v0",
        render_mode=render_mode,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
    )

    passed = 0
    failures: list[MoveCheckFailure] = []

    try:
        env.reset()
        inner = env.unwrapped
        mapper = BoardMapper.from_configs()
        registry = PieceRegistry()
        teleporter = PieceTeleporter(env, mapper)
        hide_other_pieces(inner, teleporter, registry, piece_id)
        render_fn = env.render if visualize else None
        controller = ScriptedController(env, drift_limit=drift_limit, render_fn=render_fn, render_delay=delay)
        occupancy = PhysicalOccupancy({piece_id: None})
        movement = MovementExecutor(env, controller, mapper, occupancy)
        physical = PhysicalPlanExecutor(movement, teleporter, occupancy, controller=controller, env=env)
        home_qpos = robot_home_qpos(inner).copy()

        for index, (src, dst) in enumerate(pairs, start=1):
            inner.clear_active_piece()
            teleporter.teleport_piece_to_square(piece_id, src)
            occupancy.reset({piece_id: src})

            result = movement.move_piece_between_squares(piece_id, src, dst)
            if not result.success:
                failure = MoveCheckFailure(src, dst, "move", result.error, format_stages(result.stage_results))
                failures.append(failure)
                print(f"{index}/{len(pairs)} {src}->{dst}: FAIL {failure.error} [{failure.stages}]")
                env.reset()
                hide_other_pieces(inner, teleporter, registry, piece_id)
                home_qpos = robot_home_qpos(inner).copy()
                if stop_on_failure:
                    break
                continue

            if check_home:
                home_result = physical.return_to_home()
                if not home_result.success:
                    failure = MoveCheckFailure(src, dst, "return_home", home_result.error, "")
                    failures.append(failure)
                    print(f"{index}/{len(pairs)} {src}->{dst}: FAIL return_home {home_result.error}")
                    env.reset()
                    hide_other_pieces(inner, teleporter, registry, piece_id)
                    home_qpos = robot_home_qpos(inner).copy()
                    if stop_on_failure:
                        break
                    continue

                home_delta = float(np.max(np.abs(robot_home_qpos(inner) - home_qpos)))
                if home_delta > home_tolerance:
                    failure = MoveCheckFailure(
                        src,
                        dst,
                        "home_qpos",
                        f"delta={home_delta:.3e} > tolerance={home_tolerance:.3e}",
                        format_stages(result.stage_results),
                    )
                    failures.append(failure)
                    print(f"{index}/{len(pairs)} {src}->{dst}: FAIL {failure.error}")
                    if stop_on_failure:
                        break
                    continue

            passed += 1
            max_error = max(stage.error_mm for _, stage in result.stage_results)
            print(f"{index}/{len(pairs)} {src}->{dst}: ok max_stage_err={max_error:.1f}mm")
    finally:
        env.close()

    return MoveCheckSummary(total=len(pairs), passed=passed, failures=failures)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exhaustively test physical moves between board squares.")
    parser.add_argument("--piece", default="black_rook_a", help="Physical piece id to reuse for all checks.")
    parser.add_argument("--from-square", help="Optional source square filter, e.g. a8.")
    parser.add_argument("--to-square", help="Optional destination square filter, e.g. h1.")
    parser.add_argument("--include-same", action="store_true", help="Include same-square no-op checks.")
    parser.add_argument("--max-cases", type=int, help="Run only the first N generated cases.")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--skip-home-check", action="store_true")
    parser.add_argument("--home-tolerance", type=float, default=1e-9)
    parser.add_argument("--drift-limit", type=float, default=0.010)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    summary = run_all_square_moves(
        piece_id=args.piece,
        from_square=args.from_square,
        to_square=args.to_square,
        include_same=args.include_same,
        max_cases=args.max_cases,
        stop_on_failure=args.stop_on_failure,
        check_home=not args.skip_home_check,
        home_tolerance=args.home_tolerance,
        drift_limit=args.drift_limit,
        visualize=args.visualize,
        delay=args.delay,
    )

    print(f"SUMMARY passed={summary.passed}/{summary.total} failures={len(summary.failures)}")
    for failure in summary.failures:
        print(f"FAILURE {failure.src}->{failure.dst} kind={failure.kind} error={failure.error}")

    if summary.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
