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


def build_orchestrator(env, visualize: bool, delay: float):
    registry = PieceRegistry()
    occupancy = PhysicalOccupancy(registry.starting_square_map())
    board_mapper = BoardMapper.from_configs()
    render_fn = env.render if visualize else None
    controller = ScriptedController(env, drift_limit=0.010, render_fn=render_fn, render_delay=delay)
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
    args = parser.parse_args()

    env = gym.make(
        "ChessFetchTask-v0",
        render_mode="human" if args.visualize else None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
    )
    env.reset()
    orchestrator = build_orchestrator(env, args.visualize, args.delay)
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
