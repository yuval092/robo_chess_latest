from dataclasses import dataclass

import chess

from src.chess_game.chess_service import ChessService
from src.chess_game.game_orchestrator import GameOrchestrator
from src.chess_game.move_planner import LogicalPieceTracker


@dataclass
class FakePhysicalResult:
    success: bool
    error: str | None = None


class FakePhysicalExecutor:
    def __init__(self, success=True):
        self.success = success
        self.plans = []
        self.home_calls = 0

    def execute(self, plan):
        self.plans.append(plan)
        return FakePhysicalResult(self.success, None if self.success else "ROBOT_FAILED")

    def return_to_home(self):
        self.home_calls += 1
        return FakePhysicalResult(True)


def make_orchestrator(executor=None, *, auto=False, human_color="white"):
    return GameOrchestrator(
        ChessService(),
        executor or FakePhysicalExecutor(),
        LogicalPieceTracker(),
        auto_computer_reply=auto,
        human_color=human_color,
    )


def test_illegal_move_rejects_without_calling_physical_executor():
    executor = FakePhysicalExecutor()
    orchestrator = make_orchestrator(executor)

    result = orchestrator.submit_human_move("e2", "e5")

    assert not result.accepted
    assert executor.plans == []


def test_legal_move_calls_executor_and_commits_after_success():
    executor = FakePhysicalExecutor(success=True)
    orchestrator = make_orchestrator(executor)

    result = orchestrator.submit_human_move("e2", "e4")

    assert result.accepted
    assert result.physical_success
    assert not result.snapshot.is_busy
    assert executor.plans[0].chess_move_uci == "e2e4"
    assert executor.home_calls == 1
    assert orchestrator.chess_service.board.piece_at(chess.E4).symbol() == "P"
    assert orchestrator.piece_tracker.piece_id_at("e4") == "white_pawn_e"


def test_physical_failure_leaves_fen_unchanged():
    executor = FakePhysicalExecutor(success=False)
    orchestrator = make_orchestrator(executor)
    fen_before = orchestrator.chess_service.fen()

    result = orchestrator.submit_human_move("e2", "e4")

    assert result.accepted
    assert not result.physical_success
    assert not result.snapshot.is_busy
    assert orchestrator.chess_service.fen() == fen_before
    assert orchestrator.piece_tracker.piece_id_at("e2") == "white_pawn_e"


def test_computer_move_uses_same_execution_path():
    executor = FakePhysicalExecutor(success=True)
    orchestrator = make_orchestrator(executor, human_color="black")

    result = orchestrator.let_computer_play_current_turn()

    assert result.accepted
    assert result.physical_success
    assert executor.plans
    assert result.move_uci in result.snapshot.move_history_san[-1] or result.move_uci is not None


def test_busy_state_rejects_overlapping_request():
    orchestrator = make_orchestrator()
    orchestrator.is_busy = True

    result = orchestrator.submit_human_move("e2", "e4")

    assert not result.accepted
    assert "busy" in result.error


def test_auto_computer_reply_runs_after_human_success():
    executor = FakePhysicalExecutor(success=True)
    orchestrator = make_orchestrator(executor, auto=True)

    result = orchestrator.submit_human_move("e2", "e4")

    assert result.accepted
    assert result.physical_success
    assert len(executor.plans) == 2
    assert len(orchestrator.chess_service.san_history()) == 2
