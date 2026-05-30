import queue
import threading

from src.ui.queued_backend import QueuedUIBackend


class FakeOrchestrator:
    def snapshot(self):
        return "initial"

    def new_game(self):
        return "new"


class FakeEnv:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


def test_new_game_request_resets_env():
    orchestrator = FakeOrchestrator()
    backend = QueuedUIBackend(orchestrator)
    env = FakeEnv()
    result_queue = queue.Queue(maxsize=1)

    thread = threading.Thread(target=lambda: result_queue.put(backend.new_game()))
    thread.start()
    backend.process_request(env=env, timeout=1.0)
    thread.join(timeout=1.0)

    assert result_queue.get_nowait() == "new"
    assert env.reset_count == 1
    assert backend.snapshot() == "new"
