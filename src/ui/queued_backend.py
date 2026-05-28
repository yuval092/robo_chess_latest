"""Thread-safe UIBackend wrapper that serialises calls through a queue."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass

from src.chess_game.game_orchestrator import GameOrchestrator


@dataclass
class UIRequest:
    """Queued UI backend request and response channel."""

    name: str
    args: tuple
    kwargs: dict
    response: queue.Queue


class QueuedUIBackend:
    """Serialize UI backend calls onto the main simulation thread."""

    def __init__(self, orchestrator: GameOrchestrator):
        self._orchestrator = orchestrator
        self._requests: queue.Queue[UIRequest] = queue.Queue()
        self._snapshot = orchestrator.snapshot()
        self._lock = threading.Lock()

    def snapshot(self):
        """Return the latest cached snapshot."""
        with self._lock:
            return self._snapshot

    def new_game(self):
        """Queue a new-game request."""
        return self._call("new_game")

    def submit_human_move(self, src: str, dst: str, promotion: str | None = None):
        """Queue a human move request."""
        return self._call("submit_human_move", src, dst, promotion)

    def let_computer_play_current_turn(self):
        """Queue a computer-turn request."""
        return self._call("let_computer_play_current_turn")

    def process_one(self, env=None, timeout=0.05):
        """Process one queued request on the caller's thread."""
        try:
            request = self._requests.get(timeout=timeout)
        except queue.Empty:
            return
        try:
            if request.name == "new_game" and env is not None:
                env.reset()
                physical_executor = getattr(
                    self._orchestrator, "physical_executor", None
                )
                if hasattr(physical_executor, "reset_board_state"):
                    physical_executor.reset_board_state()
                elif hasattr(physical_executor, "reset_occupancy"):
                    physical_executor.reset_occupancy()
            result = getattr(self._orchestrator, request.name)(
                *request.args, **request.kwargs
            )
            with self._lock:
                self._snapshot = (
                    result.snapshot if hasattr(result, "snapshot") else result
                )
            request.response.put((True, result))
        except Exception as exc:
            request.response.put((False, exc))

    def _call(self, name: str, *args, **kwargs):
        """Queue a request and wait for its result."""
        response: queue.Queue = queue.Queue(maxsize=1)
        self._requests.put(UIRequest(name, args, kwargs, response))
        ok, result = response.get()
        if not ok:
            raise result
        return result
