"""Root runtime entry point for RoboChess play and all-square verification."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field

import chess
import gymnasium as gym
import numpy as np
from tqdm import tqdm

import src.chess_env  # noqa: F401 - registers ChessFetchTask-v0
from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_game.board_mapper import BoardMapper
from src.chess_game.chess_service import ChessService
from src.chess_game.game_orchestrator import GameOrchestrator
from src.chess_game.move_planner import LogicalPieceTracker
from src.physical.movement_executor import MovementExecutor
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry
from src.physical.piece_teleport import PieceTeleporter
from src.physical.plan_executor import PhysicalPlanExecutor
from src.ui.app import create_app
from src.ui.queued_backend import QueuedUIBackend
from src.utils.io import load_config

DEFAULT_ALL_SQUARE_PIECE = "black_rook_a"


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


@dataclass
class FlowResult:
    src: str
    dst: str
    success: bool
    stage_results: list = field(default_factory=list)
    failure_reason: str = ""


def resolve_model_paths(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Resolve deployed model paths with explicit CLI overrides applied."""
    deployed = load_config("deployed_models")
    overrides = overrides or {}
    paths = {
        stage: overrides.get(stage) or deployed.get(stage)
        for stage in ("transit", "descend", "ascend")
    }
    missing = [stage for stage, path in paths.items() if not path]
    if missing:
        flags = ", ".join(f"--{stage}-model" for stage in missing)
        raise ValueError(
            f"Missing model path(s) for {', '.join(missing)}. "
            f"Set configs/deployed_models.yaml or pass {flags}."
        )
    return paths


def model_overrides_from_args(args: argparse.Namespace) -> dict[str, str]:
    return {
        stage: path
        for stage, path in {
            "transit": args.transit_model,
            "descend": args.descend_model,
            "ascend": args.ascend_model,
        }.items()
        if path
    }


def add_model_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--transit-model", metavar="PATH", help="Transit model ZIP override.")
    parser.add_argument("--descend-model", metavar="PATH", help="Descend model ZIP override.")
    parser.add_argument("--ascend-model", metavar="PATH", help="Ascend model ZIP override.")


def build_controller(
    env,
    visualize: bool,
    delay: float,
    transit_model: str | None,
    descend_model: str | None,
    ascend_model: str | None,
) -> ModelEmbeddedController:
    model_paths = resolve_model_paths(
        {
            stage: path
            for stage, path in {
                "transit": transit_model,
                "descend": descend_model,
                "ascend": ascend_model,
            }.items()
            if path
        }
    )
    controller = ModelEmbeddedController(
        env=env,
        render_fn=env.render if visualize else None,
        render_delay=delay,
    )
    controller.load_all(
        transit_path=model_paths["transit"],
        descend_path=model_paths["descend"],
        ascend_path=model_paths["ascend"],
    )
    return controller


def build_orchestrator(
    env,
    visualize: bool,
    delay: float,
    transit_model: str | None = None,
    descend_model: str | None = None,
    ascend_model: str | None = None,
) -> GameOrchestrator:
    registry = PieceRegistry()
    occupancy = PhysicalOccupancy(registry.starting_square_map())
    board_mapper = BoardMapper.from_configs()
    controller = build_controller(
        env, visualize, delay, transit_model, descend_model, ascend_model
    )
    movement_executor = MovementExecutor(env, controller, board_mapper, occupancy)
    physical_executor = PhysicalPlanExecutor(
        movement_executor,
        PieceTeleporter(env, board_mapper),
        occupancy,
        controller=controller,
        env=env,
    )
    engine_cfg = load_config("chess").get("engine")
    return GameOrchestrator(
        ChessService(engine_cfg=engine_cfg),
        physical_executor,
        LogicalPieceTracker(),
        engine_cfg=engine_cfg,
    )


def run_play(args: argparse.Namespace) -> int:
    env = gym.make(
        "ChessFetchTask-v0",
        render_mode="human" if args.visualize else None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
        debug=args.debug,
    )
    env.reset()
    overrides = model_overrides_from_args(args)
    orchestrator = build_orchestrator(
        env,
        args.visualize,
        args.delay,
        transit_model=overrides.get("transit"),
        descend_model=overrides.get("descend"),
        ascend_model=overrides.get("ascend"),
    )
    backend = QueuedUIBackend(orchestrator)
    app = create_app(backend)
    server = threading.Thread(
        target=lambda: app.run(
            host=args.host,
            port=args.port,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
    )
    server.start()
    print(f"RoboChess UI running at http://{args.host}:{args.port}")

    try:
        while True:
            backend.process_one(env=env, timeout=0.05)
            if args.visualize:
                env.render()
            time.sleep(0.01)
    except KeyboardInterrupt:
        return 0
    finally:
        env.close()


def _make_flow_env(debug: bool, visualize: bool) -> gym.Env:
    return gym.make(
        "ChessFetchTask-v0",
        render_mode="human" if visualize else None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
        debug=debug,
    )


def _hide_other_pieces(inner, teleporter: PieceTeleporter, piece_id: str) -> None:
    registry = PieceRegistry()
    z = inner.TABLE_Z + inner.CUBE_HEIGHT / 2.0
    for idx, other_id in enumerate(registry.starting_square_map()):
        if other_id == piece_id:
            continue
        teleporter.teleport_piece_to_xyz(
            other_id,
            np.array([2.0 + 0.05 * (idx // 8), 2.0 + 0.05 * (idx % 8), z]),
        )


def _format_stages(stage_results) -> str:
    return ", ".join(
        f"{name}:{sr.success}:{sr.error_mm:.1f}mm:{sr.crash_reason}"
        for name, sr in stage_results
    )


def _setup_all_square_env(
    *,
    debug: bool,
    drift_limit: float | None,
    model_overrides: dict[str, str],
    piece_id: str,
    visualize: bool,
    delay: float,
) -> tuple:
    env = _make_flow_env(debug, visualize)
    env.reset()
    if drift_limit is not None:
        env.unwrapped.env_cfg["eval_drift_limit"] = drift_limit
    inner = env.unwrapped
    mapper = BoardMapper.from_configs()
    teleporter = PieceTeleporter(env, mapper)
    _hide_other_pieces(inner, teleporter, piece_id)
    model_paths = resolve_model_paths(model_overrides)
    controller = ModelEmbeddedController(
        env=env,
        render_fn=env.render if visualize else None,
        render_delay=delay,
    )
    controller.load_all(
        transit_path=model_paths["transit"],
        descend_path=model_paths["descend"],
        ascend_path=model_paths["ascend"],
    )
    occupancy = PhysicalOccupancy({piece_id: None})
    movement = MovementExecutor(env, controller, mapper, occupancy)
    physical = PhysicalPlanExecutor(
        movement, teleporter, occupancy, controller=controller, env=env
    )
    return env, inner, teleporter, occupancy, movement, physical


def _run_pairs(
    *,
    pairs: list[tuple[str, str]],
    piece_id: str,
    model_overrides: dict[str, str],
    visualize: bool,
    delay: float,
    debug: bool,
    drift_limit: float | None,
    check_home: bool,
    stop_on_failure: bool,
) -> list[FlowResult]:
    env, inner, teleporter, occupancy, movement, physical = _setup_all_square_env(
        debug=debug,
        drift_limit=drift_limit,
        model_overrides=model_overrides,
        piece_id=piece_id,
        visualize=visualize,
        delay=delay,
    )
    results: list[FlowResult] = []
    try:
        with tqdm(total=len(pairs), desc="all-square", unit="move") as pbar:
            for src, dst in pairs:
                inner.clear_active_piece()
                teleporter.teleport_piece_to_square(piece_id, src)
                occupancy.reset({piece_id: src})
                move = movement.move_piece_between_squares(piece_id, src, dst)
                result = FlowResult(
                    src=src,
                    dst=dst,
                    success=move.success,
                    stage_results=move.stage_results,
                    failure_reason=move.error or "",
                )
                results.append(result)

                if result.success and check_home:
                    home_result = physical.return_to_home()
                    if not home_result.success:
                        result.success = False
                        result.failure_reason = f"return_home: {home_result.error}"

                if not result.success:
                    env.reset()
                    _hide_other_pieces(inner, teleporter, piece_id)
                    if stop_on_failure:
                        pbar.update(1)
                        return results

                ok = sum(item.success for item in results)
                pbar.update(1)
                pbar.set_postfix({"ok%": f"{100 * ok / len(results):.0f}"})
    finally:
        env.close()
    return results


def run_all_square_moves(
    *,
    piece_id: str = DEFAULT_ALL_SQUARE_PIECE,
    from_square: str | None = None,
    to_square: str | None = None,
    include_same: bool = False,
    max_cases: int | None = None,
    stop_on_failure: bool = False,
    check_home: bool = True,
    drift_limit: float | None = None,
    visualize: bool = False,
    delay: float = 0.0,
    debug: bool = False,
    model_overrides: dict[str, str] | None = None,
) -> MoveCheckSummary:
    sources = [from_square] if from_square else [chess.square_name(sq) for sq in chess.SQUARES]
    destinations = [to_square] if to_square else [chess.square_name(sq) for sq in chess.SQUARES]
    pairs = [(src, dst) for src in sources for dst in destinations if src != dst or include_same]
    if max_cases is not None:
        pairs = pairs[:max_cases]

    results = _run_pairs(
        pairs=pairs,
        piece_id=piece_id,
        model_overrides=model_overrides or {},
        visualize=visualize,
        delay=delay,
        debug=debug,
        drift_limit=drift_limit,
        check_home=check_home,
        stop_on_failure=stop_on_failure,
    )
    failures = [
        MoveCheckFailure(
            result.src,
            result.dst,
            "return_home" if result.failure_reason.startswith("return_home:") else "move",
            result.failure_reason,
            _format_stages(result.stage_results),
        )
        for result in results
        if not result.success
    ]
    return MoveCheckSummary(total=len(pairs), passed=len(results) - len(failures), failures=failures)


def print_all_square_summary(summary: MoveCheckSummary) -> None:
    print()
    print(f"=== all-square physical move sweep: {summary.passed}/{summary.total} passed ===")
    if summary.failures:
        print("Failures:")
        for failure in summary.failures:
            print(
                f"  {failure.src}->{failure.dst} [{failure.kind}] "
                f"{failure.error or ''} stages={failure.stages}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Start RoboChess or run the 64x63 physical move sweep.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Flask bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Flask bind port.")
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        help="Run play mode without the MuJoCo viewer.",
    )
    parser.set_defaults(visualize=True)
    parser.add_argument("--delay", type=float, default=0.0, help="Per-step render delay.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose environment logs.")
    parser.add_argument(
        "--all-square-test",
        action="store_true",
        help="Run all 4,032 source/destination moves with black_rook_a.",
    )
    parser.add_argument(
        "--visualize-test",
        action="store_true",
        help="Open the MuJoCo viewer during --all-square-test.",
    )
    parser.add_argument(
        "--drift-limit",
        type=float,
        default=None,
        metavar="METRES",
        help="Override the evaluation drift tolerance.",
    )
    add_model_path_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.all_square_test:
        summary = run_all_square_moves(
            piece_id=DEFAULT_ALL_SQUARE_PIECE,
            visualize=args.visualize_test,
            delay=args.delay,
            debug=args.debug,
            drift_limit=args.drift_limit,
            model_overrides=model_overrides_from_args(args),
        )
        print_all_square_summary(summary)
        sys.exit(0 if not summary.failures and summary.passed == summary.total else 1)

    sys.exit(run_play(args))


if __name__ == "__main__":
    main()
