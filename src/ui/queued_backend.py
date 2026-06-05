"""Thread-safe UIBackend wrapper that serialises calls through a queue."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

from src.chess_game.game_orchestrator import GameOrchestrator, MoveExecutionResult


@dataclass
class UIRequest:
    """Queued UI backend request and response channel."""

    fn: Callable[[], object]
    response: queue.Queue


class QueuedUIBackend:
    """Serialize UI backend calls onto the main simulation thread."""

    def __init__(self, orchestrator: GameOrchestrator, env=None):
        self._orchestrator = orchestrator
        self._env = env
        self._requests: queue.Queue[UIRequest] = queue.Queue()
        self._snapshot = orchestrator.snapshot()
        self._lock = threading.Lock()

    def snapshot(self):
        """Return the latest cached snapshot."""
        with self._lock:
            return self._snapshot

    def new_game(self):
        """Queue a new-game request."""
        def _fn():
            try:
                if self._env is not None:
                    self._env.reset()
            except Exception as exc:
                return self._orchestrator.fault(f"NEW_GAME_RESET_FAILED: {exc}")
            return self._orchestrator.new_game()
        return self._enqueue(_fn)

    def submit_human_move(self, src: str, dst: str, promotion: str | None = None):
        """Queue a human move request."""
        return self._enqueue(
            lambda: self._orchestrator.submit_human_move(src, dst, promotion)
        )

    def let_computer_play_current_turn(self):
        """Queue a computer-turn request."""
        return self._enqueue(self._orchestrator.let_computer_play_current_turn)

    def process_request(self, timeout=0.05):
        """Process one queued request on the caller's thread."""
        try:
            request = self._requests.get(timeout=timeout)
        except queue.Empty:
            return
        try:
            result = request.fn()
            with self._lock:
                self._snapshot = result.snapshot if isinstance(result, MoveExecutionResult) else result
            request.response.put((True, result))
        except Exception as exc:
            request.response.put((False, exc))

    def _enqueue(self, fn: Callable[[], object]):
        """Queue a callable and block until the main thread executes it."""
        response: queue.Queue = queue.Queue(maxsize=1)
        self._requests.put(UIRequest(fn, response))
        ok, result = response.get()
        if not ok:
            raise result
        return result
