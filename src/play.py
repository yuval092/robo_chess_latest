"""Installed play-mode entry point for RoboChess."""

from __future__ import annotations

# Must import before gym.make to avoid a CUDA init conflict between MuJoCo's
# OpenGL context and triton (imported lazily by torch._dynamo on first optimizer use).
# noqa: F401 to avoid "imported but unused" lint error since we don't directly reference this module.
import torch._dynamo  # noqa: F401

import sys
import threading
import time

import gymnasium as gym

import src.chess_env  # noqa: F401 - registers chess Train/Play envs
from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_game.chess_service import ChessService
from src.chess_game.game_orchestrator import GameOrchestrator
from src.chess_game.move_planner import LogicalPieceTracker
from src.physical.plan_executor import PhysicalPlanExecutor
from src.ui.app import create_app
from src.ui.queued_backend import QueuedUIBackend
from src.utils.config_validation import validate_config
from src.utils.io import load_config, resolve_model_paths

HOST = "127.0.0.1"
PORT = 9999


def build_controller(env) -> ModelEmbeddedController:
    model_paths = resolve_model_paths()
    controller = ModelEmbeddedController(env=env)
    controller.load_all(
        transit_path=model_paths["transit"],
        descend_path=model_paths["descend"],
        ascend_path=model_paths["ascend"],
    )
    return controller


def build_orchestrator(env) -> GameOrchestrator:
    controller = build_controller(env)
    physical_executor = PhysicalPlanExecutor(env, controller)
    engine_cfg = load_config("chess").get("engine")
    return GameOrchestrator(
        ChessService(engine_cfg=engine_cfg),
        physical_executor,
        LogicalPieceTracker(),
        engine_cfg=engine_cfg,
    )


def run_game() -> int:
    env = None
    orchestrator = None
    try:
        validate_config()
        env = gym.make(
            "ChessFetchTask-Play-v0",
            render_mode="human",
            show_chess_pieces=True,
            debug=False,
        )
        env.reset()
        orchestrator = build_orchestrator(env)
    except Exception as exc:
        if env is not None:
            env.close()
        print(f"RoboChess startup failed: {exc}", file=sys.stderr)
        return 1
    backend = QueuedUIBackend(orchestrator, env)
    app = create_app(backend)
    server = threading.Thread(
        target=lambda: app.run(
            host=HOST,
            port=PORT,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
    )
    server.start()
    print(f"RoboChess UI running at http://{HOST}:{PORT}")

    try:
        while True:
            backend.process_request(timeout=0.05)
            env.render()
            time.sleep(0.01)
    except KeyboardInterrupt:
        return 0
    finally:
        if orchestrator is not None:
            orchestrator.chess_service.close()
        if env is not None:
            env.close()


def main() -> None:
    if len(sys.argv) > 1:
        print("RoboChess play mode does not accept CLI flags.", file=sys.stderr)
        sys.exit(2)
    sys.exit(run_game())
