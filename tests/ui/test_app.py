from dataclasses import dataclass

import pytest

from src.chess_game.chess_service import ChessService
from src.chess_game.game_orchestrator import GameOrchestrator
from src.chess_game.move_planner import LogicalPieceTracker
from src.ui.app import create_app


@dataclass
class FakePhysicalResult:
    success: bool
    error: str | None = None


class FakePhysicalExecutor:
    def __init__(self, success=True):
        self.success = success
        self.plans = []

    def execute(self, plan):
        self.plans.append(plan)
        return FakePhysicalResult(self.success, None if self.success else "FAILED")

    def return_to_home(self):
        return FakePhysicalResult(True)

    def reset_board_state(self):
        pass


_ENGINE_CFG = {"stockfish_path": "stockfish", "skill_level": 1}


@pytest.fixture
def client():
    orchestrator = GameOrchestrator(
        ChessService(engine_cfg=_ENGINE_CFG),
        FakePhysicalExecutor(),
        LogicalPieceTracker(),
        auto_computer_reply=False,
        engine_cfg=_ENGINE_CFG,
    )
    app = create_app(orchestrator)
    app.config.update(TESTING=True)
    yield app.test_client()
    orchestrator.chess_service.close()


def test_snapshot_returns_initial_board(client):
    response = client.get("/api/snapshot")

    assert response.status_code == 200
    data = response.get_json()
    assert data["board"]["e2"] == "P"
    assert data["turn"] == "white"
    assert len(data["legal_moves"]) == 20


def test_index_renders_interactive_controls(client):
    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    for element_id in [
        "board",
        "new-game",
        "computer",
        "refresh",
        "flip-board",
        "history",
    ]:
        assert f'id="{element_id}"' in html
    assert "busy-overlay" not in html


def test_new_game_endpoint_resets_snapshot(client):
    client.post("/api/move", json={"src": "e2", "dst": "e4"})

    response = client.post("/api/new-game", json={})

    assert response.status_code == 200
    data = response.get_json()
    assert data["board"]["e2"] == "P"
    assert data["board"]["e4"] is None
    assert data["move_history_san"] == []


def test_illegal_move_returns_400(client):
    response = client.post("/api/move", json={"src": "e2", "dst": "e5"})

    assert response.status_code == 400
    assert response.get_json()["accepted"] is False


def test_legal_move_returns_result_snapshot(client):
    response = client.post("/api/move", json={"src": "e2", "dst": "e4"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["accepted"] is True
    assert data["physical_success"] is True
    assert data["snapshot"]["board"]["e4"] == "P"


def test_let_computer_play_returns_legal_move(client):
    response = client.post("/api/let-computer-play", json={})

    assert response.status_code == 200
    data = response.get_json()
    assert data["accepted"] is True
    assert data["move_uci"] is not None



