"""Logical-only evaluation of GameOrchestrator through representative sequences."""

from __future__ import annotations

import argparse

from src.chess_game.chess_service import ChessService
from src.chess_game.game_orchestrator import GameOrchestrator
from src.chess_game.move_planner import LogicalPieceTracker
from src.physical.noop_executor import NoOpPhysicalExecutor

SCHOLAR_MATE_MOVES = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "a7a6", "h5f7"]
CASTLING_SEQUENCE = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1"]
EN_PASSANT_SEQ = ["e2e4", "d7d5", "e4e5", "f7f5", "e5f6"]
PROMOTION_FEN = "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1"


def _orchestrator_for(name: str) -> GameOrchestrator:
    """Build an orchestrator for a named logical sequence."""
    if name != "promotion":
        return GameOrchestrator.create_headless(
            human_color="both", auto_computer_reply=False
        )

    tracker = LogicalPieceTracker.empty()
    tracker.set_piece_at("a7", "white_pawn_a")
    tracker.set_piece_at("b8", "black_rook_b")
    return GameOrchestrator(
        ChessService(PROMOTION_FEN),
        NoOpPhysicalExecutor(),
        tracker,
        human_color="both",
        auto_computer_reply=False,
    )


def check_sequence(name: str, moves: list[str], verbose: bool) -> bool:
    """Play a move sequence and verify no errors occur."""
    orchestrator = _orchestrator_for(name)
    for uci in moves:
        src, dst = uci[:2], uci[2:4]
        promotion = uci[4] if len(uci) == 5 else None
        result = orchestrator.submit_human_move(src, dst, promotion)
        if not result.accepted:
            if verbose:
                print(f"  {name}: FAIL at {uci}: {result.error}")
            return False
    if verbose:
        print(f"  {name}: OK ({len(moves)} moves)")
    return True


def main() -> None:
    """Run all logical game sequences and report failures."""
    parser = argparse.ArgumentParser(description="Logical game-flow evaluation.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    sequences = [
        ("scholar_mate", SCHOLAR_MATE_MOVES),
        ("castling", CASTLING_SEQUENCE),
        ("en_passant", EN_PASSANT_SEQ),
        ("promotion", ["a7b8q"]),
    ]
    failures = sum(
        0 if check_sequence(name, moves, args.verbose) else 1
        for name, moves in sequences
    )
    if failures:
        print(f"\n{failures} sequence(s) FAILED.")
        raise SystemExit(1)
    print("All game logic sequences passed.")


if __name__ == "__main__":
    main()
