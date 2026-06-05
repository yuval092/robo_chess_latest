import queue
import threading

from src.ui.queued_backend import QueuedUIBackend


class FakeOrchestrator:
    def snapshot(self):
        return "initial"

    def new_game(self):
        return "new"

    def fault(self, error):
        return f"faulted: {error}"


class FakeEnv:
    def __init__(self, reset_error=None):
        self.reset_count = 0
        self._reset_error = reset_error

    def reset(self):
        self.reset_count += 1
        if self._reset_error is not None:
            raise self._reset_error


def test_new_game_request_resets_env():
    orchestrator = FakeOrchestrator()
    env = FakeEnv()
    backend = QueuedUIBackend(orchestrator, env)
    result_queue = queue.Queue(maxsize=1)

    thread = threading.Thread(target=lambda: result_queue.put(backend.new_game()))
    thread.start()
    backend.process_request(timeout=1.0)
    thread.join(timeout=1.0)

    assert result_queue.get_nowait() == "new"
    assert env.reset_count == 1
    assert backend.snapshot() == "new"


def test_new_game_reset_failure_faults_orchestrator():
    orchestrator = FakeOrchestrator()
    env = FakeEnv(reset_error=RuntimeError("RESET_ARM_START_MOVE_FAILED"))
    backend = QueuedUIBackend(orchestrator, env)
    result_queue = queue.Queue(maxsize=1)

    thread = threading.Thread(target=lambda: result_queue.put(backend.new_game()))
    thread.start()
    backend.process_request(timeout=1.0)
    thread.join(timeout=1.0)

    result = result_queue.get_nowait()
    assert result == "faulted: NEW_GAME_RESET_FAILED: RESET_ARM_START_MOVE_FAILED"
    # The failed reset must not silently leave the previous game playable.
    assert backend.snapshot() == result
