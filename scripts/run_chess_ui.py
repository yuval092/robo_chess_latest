import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass

import gymnasium as gym

sys.path.append(os.getcwd())

import src.chess_env
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
from src.utils.config import load_config


@dataclass
class UIRequest:
    name: str
    args: tuple
    kwargs: dict
    response: queue.Queue


class QueuedUIBackend:
    def __init__(self, orchestrator: GameOrchestrator):
        self._orchestrator = orchestrator
        self._requests: queue.Queue[UIRequest] = queue.Queue()
        self._snapshot = orchestrator.snapshot()
        self._lock = threading.Lock()

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def new_game(self):
        return self._call("new_game")

    def submit_human_move(self, src: str, dst: str, promotion: str | None = None):
        return self._call("submit_human_move", src, dst, promotion)

    def let_computer_play_current_turn(self):
        return self._call("let_computer_play_current_turn")

    def process_one(self, env=None, timeout=0.05):
        try:
            request = self._requests.get(timeout=timeout)
        except queue.Empty:
            return
        try:
            if request.name == "new_game" and env is not None:
                env.reset()
                physical_executor = getattr(self._orchestrator, "physical_executor", None)
                if hasattr(physical_executor, "reset_board_state"):
                    physical_executor.reset_board_state()
                elif hasattr(physical_executor, "reset_occupancy"):
                    physical_executor.reset_occupancy()
            result = getattr(self._orchestrator, request.name)(*request.args, **request.kwargs)
            with self._lock:
                self._snapshot = result.snapshot if hasattr(result, "snapshot") else result
            request.response.put((True, result))
        except Exception as exc:
            request.response.put((False, exc))

    def _call(self, name: str, *args, **kwargs):
        response: queue.Queue = queue.Queue(maxsize=1)
        self._requests.put(UIRequest(name, args, kwargs, response))
        ok, result = response.get()
        if not ok:
            raise result
        return result


def build_controller(
    env,
    visualize: bool,
    delay: float,
    use_rl_models: bool,
    transit_model: str | None,
    descend_model: str | None,
    ascend_model: str | None,
):
    render_fn = env.render if visualize else None
    if not use_rl_models:
        return ScriptedController(env, drift_limit=0.010, render_fn=render_fn, render_delay=delay)

    train_cfg = load_config("training")
    model_paths = train_cfg.get("deployed_models", {})
    controller = ModelEmbeddedController(env=env, render_fn=render_fn, render_delay=delay)
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
        target=lambda: app.run(host=args.host, port=args.port, debug=False, use_reloader=False),
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
