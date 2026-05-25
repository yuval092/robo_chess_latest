"""Packaged command-line entry point for the RoboChess web UI."""

import argparse
import threading
import time

import gymnasium as gym

from src.chess_env.controller import ScriptedController
from src.chess_env.environment_generation import regenerate_environment
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


def build_controller(
    env,
    visualize: bool,
    delay: float,
    use_rl_models: bool,
    transit_model: str | None,
    descend_model: str | None,
    ascend_model: str | None,
):
    """Run build controller logic."""
    render_fn = env.render if visualize else None
    env_cfg = load_config("env")
    if not use_rl_models:
        return ScriptedController(
            env,
            drift_limit=env_cfg["drift_limit_end"],
            render_fn=render_fn,
            render_delay=delay,
        )

    model_paths = load_config("deployed_models")
    controller = ModelEmbeddedController(
        env=env, render_fn=render_fn, render_delay=delay
    )
    controller.load_available(
        transit_path=transit_model or model_paths.get("transit"),
        descend_path=descend_model or model_paths.get("descend"),
        ascend_path=ascend_model or model_paths.get("ascend"),
    )
    return controller


def build_orchestrator(
    env,
    visualize: bool,
    delay: float,
    use_rl_models: bool = True,
    transit_model: str | None = None,
    descend_model: str | None = None,
    ascend_model: str | None = None,
):
    """Run build orchestrator logic."""
    registry = PieceRegistry()
    occupancy = PhysicalOccupancy(registry.starting_square_map())
    board_mapper = BoardMapper.from_configs()
    controller = build_controller(
        env,
        visualize,
        delay,
        use_rl_models,
        transit_model,
        descend_model,
        ascend_model,
    )
    movement_executor = MovementExecutor(env, controller, board_mapper, occupancy)
    physical_executor = PhysicalPlanExecutor(
        movement_executor,
        PieceTeleporter(env, board_mapper),
        occupancy,
        controller=controller,
        env=env,
    )
    return GameOrchestrator(
        ChessService(),
        physical_executor,
        LogicalPieceTracker(),
    )


def main():
    """Run main logic."""
    parser = argparse.ArgumentParser(description="Run the local RoboChess web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--use-scripted-controller",
        action="store_true",
        help="Use the scripted controller instead of the configured RL movement models.",
    )
    parser.add_argument(
        "--transit-model",
        default=None,
        help="Override the configured transit model path.",
    )
    parser.add_argument(
        "--descend-model",
        default=None,
        help="Override the configured descend model path.",
    )
    parser.add_argument(
        "--ascend-model",
        default=None,
        help="Override the configured ascend model path.",
    )
    args = parser.parse_args()

    regenerate_environment()

    env = gym.make(
        "ChessFetchTask-v0",
        render_mode="human" if args.visualize else None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
    )
    env.reset()
    orchestrator = build_orchestrator(
        env,
        args.visualize,
        args.delay,
        use_rl_models=not args.use_scripted_controller,
        transit_model=args.transit_model,
        descend_model=args.descend_model,
        ascend_model=args.ascend_model,
    )
    backend = QueuedUIBackend(orchestrator)
    app = create_app(backend)

    server = threading.Thread(
        target=lambda: app.run(
            host=args.host, port=args.port, debug=False, use_reloader=False
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
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
