"""Root runtime entry point for RoboChess play."""

from __future__ import annotations

# Must import before gym.make to avoid a CUDA init conflict between MuJoCo's
# OpenGL context and triton (imported lazily by torch._dynamo on first optimizer use).
# noqa: F401 to avoid "imported but unused" lint error since we don't directly reference this module.
import torch._dynamo  # noqa: F401

import argparse
import sys
import threading
import time

import gymnasium as gym

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
from src.utils.io import load_config, resolve_model_paths


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
    model_paths = resolve_model_paths({
        "transit": transit_model,
        "descend": descend_model,
        "ascend": ascend_model,
    })
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
    orchestrator = build_orchestrator(
        env,
        args.visualize,
        args.delay,
        transit_model=args.transit_model,
        descend_model=args.descend_model,
        ascend_model=args.ascend_model,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Start RoboChess play mode.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Flask bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Flask bind port.")
    parser.add_argument(
        "--no-visualize",
        dest="visualize",
        action="store_false",
        default=True,
        help="Run play mode without the MuJoCo viewer.",
    )
    parser.add_argument("--delay", type=float, default=0.0, help="Per-step render delay.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose environment logs.")
    add_model_path_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(run_play(args))


if __name__ == "__main__":
    main()
