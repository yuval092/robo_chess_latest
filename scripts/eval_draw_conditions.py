import os
import sys

sys.path.append(os.getcwd())

from src.chess_game.chess_service import ChessService
from src.chess_game.game_orchestrator import GameOrchestrator
from src.chess_game.move_planner import LogicalPieceTracker
from src.physical.plan_executor import PhysicalExecutionResult


class MockPhysicalExecutor:
    def execute(self, plan):
        return PhysicalExecutionResult(True, [(command, True) for command in plan.commands])

    def return_to_home(self):
        return PhysicalExecutionResult(True, [])


def orchestrator_for(service: ChessService) -> GameOrchestrator:
    return GameOrchestrator(
        service,
        MockPhysicalExecutor(),
        LogicalPieceTracker.empty(),
        human_color="white",
        auto_computer_reply=False,
    )


def assert_status(name: str, service: ChessService, **expected) -> None:
    snapshot = orchestrator_for(service).snapshot()
    failures = []
    for field, wanted in expected.items():
        actual = getattr(snapshot.status, field)
        if actual != wanted:
            failures.append(f"{field}: expected {wanted!r}, got {actual!r}")
    if failures:
        raise SystemExit(f"{name} failed: " + "; ".join(failures))
    print(
        f"{name}: ok outcome={snapshot.status.outcome} "
        f"game_over={snapshot.status.is_game_over} legal_moves={len(snapshot.legal_moves)}"
    )


def threefold_service() -> ChessService:
    service = ChessService()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"):
        service.push(service.validate_uci(uci))
    return service


def main() -> None:
    assert_status(
        "stalemate",
        ChessService("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"),
        is_stalemate=True,
        is_game_over=True,
        outcome="1/2-1/2",
    )
    assert_status(
        "insufficient_material",
        ChessService("8/8/8/8/8/8/8/K1k5 w - - 0 1"),
        is_insufficient_material=True,
        is_game_over=True,
        outcome="1/2-1/2",
    )
    assert_status(
        "fifty_move_claim",
        ChessService("8/8/8/8/8/8/8/K1k5 w - - 100 1"),
        can_claim_fifty_moves=True,
        is_game_over=True,
        outcome="1/2-1/2",
    )
    assert_status(
        "threefold_claim",
        threefold_service(),
        can_claim_threefold_repetition=True,
        is_game_over=True,
        outcome="1/2-1/2",
    )
    print("Draw-condition evaluation passed.")


if __name__ == "__main__":
    main()
