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
    def __init__(self, success=True, home_success=True):
        self.success = success
        self.home_success = home_success
        self.plans = []
        self.home_calls = 0
        self.reset_calls = 0

    def execute(self, plan):
        self.plans.append(plan)
        return FakePhysicalResult(
            self.success, None if self.success else "ROBOT_FAILED"
        )

    def return_to_home(self):
        self.home_calls += 1
        return FakePhysicalResult(
            self.home_success, None if self.home_success else "HOME_FAILED"
        )

    def reset_board_state(self):
        self.reset_calls += 1


def make_orchestrator(executor=None, *, auto=False, human_color="white", service=None):
    return GameOrchestrator(
        service or ChessService(),
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
    assert orchestrator.chess_service.fen() == fen_before
    assert orchestrator.piece_tracker.piece_id_at("e2") == "white_pawn_e"
    assert result.snapshot.state == "FAULTED"


def test_faulted_state_blocks_moves_until_new_game():
    executor = FakePhysicalExecutor(success=False)
    orchestrator = make_orchestrator(executor)

    first = orchestrator.submit_human_move("e2", "e4")
    second = orchestrator.submit_human_move("d2", "d4")
    snapshot = orchestrator.new_game()

    assert first.snapshot.state == "FAULTED"
    assert not second.accepted
    assert second.error == "System is faulted. Start a new game to recover."
    assert snapshot.state == "READY"
    assert executor.reset_calls == 1


def test_engine_failure_enters_fault_state_without_traceback():
    orchestrator = make_orchestrator()

    result = orchestrator.let_computer_play_current_turn()

    assert not result.accepted
    assert result.snapshot.state == "FAULTED"
    assert result.error.startswith("ENGINE_ERROR:")


def test_home_return_failure_commits_move_then_faults():
    executor = FakePhysicalExecutor(success=True, home_success=False)
    orchestrator = make_orchestrator(executor)

    result = orchestrator.submit_human_move("e2", "e4")

    assert result.accepted
    assert result.physical_success
    assert result.error == "HOME_FAILED"
    assert result.snapshot.state == "FAULTED"
    assert orchestrator.chess_service.board.piece_at(chess.E4).symbol() == "P"


def test_computer_move_uses_same_execution_path():
    executor = FakePhysicalExecutor(success=True)
    service = ChessService(engine_cfg={"stockfish_path": "stockfish", "skill_level": 1})
    orchestrator = make_orchestrator(executor, human_color="black", service=service)

    result = orchestrator.let_computer_play_current_turn()
    service.close()

    assert result.accepted
    assert result.physical_success
    assert executor.plans
    assert (
        result.move_uci in result.snapshot.move_history_san[-1]
        or result.move_uci is not None
    )


def test_auto_computer_reply_runs_after_human_success():
    executor = FakePhysicalExecutor(success=True)
    service = ChessService(engine_cfg={"stockfish_path": "stockfish", "skill_level": 1})
    orchestrator = make_orchestrator(executor, auto=True, service=service)

    result = orchestrator.submit_human_move("e2", "e4")
    service.close()

    assert result.accepted
    assert result.physical_success
    assert len(executor.plans) == 2
    assert len(orchestrator.chess_service.san_history()) == 2


def test_computer_turn_rejects_finished_game_without_physical_move():
    service = ChessService("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    executor = FakePhysicalExecutor(success=True)
    orchestrator = make_orchestrator(executor, service=service)

    result = orchestrator.let_computer_play_current_turn()

    assert not result.accepted
    assert not result.physical_success
    assert "White wins" in result.error
    assert result.snapshot.status.is_checkmate
    assert executor.plans == []
