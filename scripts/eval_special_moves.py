import argparse
import os
import sys

import chess
import gymnasium as gym

sys.path.append(os.getcwd())

import src.chess_env
from src.chess_env.controller import ScriptedController
from src.chess_game.board_mapper import BoardMapper
from src.chess_game.chess_service import ChessService
from src.chess_game.game_orchestrator import GameOrchestrator
from src.chess_game.move_planner import (
    ArmMoveCommand,
    LogicalPieceTracker,
    RemoveFromBoardCommand,
    TeleportCommand,
)
from src.physical.movement_executor import MovementExecutor
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry
from src.physical.piece_teleport import PieceTeleporter
from src.physical.plan_executor import PhysicalExecutionResult, PhysicalPlanExecutor


class MockPhysicalExecutor:
    def __init__(self):
        self.plans = []

    def execute(self, plan):
        self.plans.append(plan)
        return PhysicalExecutionResult(True, [(command, True) for command in plan.commands])

    def return_to_home(self):
        return PhysicalExecutionResult(True, [])


def command_signature(command) -> tuple:
    if isinstance(command, ArmMoveCommand):
        return ("arm", command.piece_id, command.src_square, command.dst_square)
    if isinstance(command, RemoveFromBoardCommand):
        return ("remove", command.piece_id, command.graveyard_slot)
    if isinstance(command, TeleportCommand):
        return ("teleport", command.piece_id, command.destination_kind, command.destination_id)
    raise TypeError(f"Unsupported command {command}")


CASES = {
    "castling": {
        "fen": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "uci": "e1g1",
        "pieces": {"e1": "white_king", "h1": "white_rook_h"},
        "expected": [
            ("arm", "white_king", "e1", "g1"),
            ("arm", "white_rook_h", "h1", "f1"),
        ],
    },
    "en_passant": {
        "fen": "4k3/8/8/3pPp2/8/8/8/4K3 w - d6 0 1",
        "uci": "e5d6",
        "pieces": {"e5": "white_pawn_e", "d5": "black_pawn_d"},
        "expected": [
            ("remove", "black_pawn_d", "slot_00"),
            ("arm", "white_pawn_e", "e5", "d6"),
        ],
    },
    "promotion": {
        "fen": "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "uci": "a7a8q",
        "pieces": {"a7": "white_pawn_a"},
        "expected": [
            ("arm", "white_pawn_a", "a7", "a8"),
            ("teleport", "white_pawn_a", "promotion_reserve", "slot_00"),
            ("teleport", "white_reserve_queen_1", "square", "a8"),
        ],
    },
    "capture_promotion": {
        "fen": "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "uci": "a7b8q",
        "pieces": {"a7": "white_pawn_a", "b8": "black_rook_b"},
        "expected": [
            ("remove", "black_rook_b", "slot_00"),
            ("arm", "white_pawn_a", "a7", "b8"),
            ("teleport", "white_pawn_a", "promotion_reserve", "slot_00"),
            ("teleport", "white_reserve_queen_1", "square", "b8"),
        ],
    },
}


def tracker_for(case: dict) -> LogicalPieceTracker:
    tracker = LogicalPieceTracker.empty()
    for square, piece_id in case["pieces"].items():
        tracker.set_piece_at(square, piece_id)
    return tracker


def run_mock_case(name: str, case: dict) -> None:
    executor = MockPhysicalExecutor()
    orchestrator = GameOrchestrator(
        ChessService(case["fen"]),
        executor,
        tracker_for(case),
        human_color="white",
        auto_computer_reply=False,
    )
    move = chess.Move.from_uci(case["uci"])
    result = orchestrator.submit_human_move(
        chess.square_name(move.from_square),
        chess.square_name(move.to_square),
        chess.piece_symbol(move.promotion) if move.promotion else None,
    )
    if not result.accepted or not result.physical_success:
        raise SystemExit(f"{name} failed: {result.error}")
    actual = [command_signature(command) for command in executor.plans[-1].commands]
    if actual != case["expected"]:
        raise SystemExit(f"{name} command mismatch: expected {case['expected']}, got {actual}")
    print(f"{name}: ok move={result.move_uci} fen={result.snapshot.fen}")


def run_real_case(name: str, case: dict, args) -> None:
    env = gym.make(
        "ChessFetchTask-v0",
        render_mode="human" if args.visualize else None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
    )
    try:
        env.reset()
        mapper = BoardMapper.from_configs()
        tracker = tracker_for(case)
        teleporter = PieceTeleporter(env, mapper)
        offboard_slots = {"white": 0, "black": 0}
        for piece_id in PieceRegistry().starting_square_map():
            color = "white" if piece_id.startswith("white_") else "black"
            teleporter.teleport_piece_to_graveyard(piece_id, f"slot_{offboard_slots[color]:02d}")
            offboard_slots[color] += 1
        for square, piece_id in case["pieces"].items():
            teleporter.teleport_piece_to_square(piece_id, square)
        occupancy = PhysicalOccupancy({piece_id: square for square, piece_id in case["pieces"].items()})
        controller = ScriptedController(
            env,
            drift_limit=args.drift_limit,
            render_fn=env.render if args.visualize else None,
            render_delay=args.delay,
        )
        physical_executor = PhysicalPlanExecutor(
            MovementExecutor(env, controller, mapper, occupancy),
            teleporter,
            occupancy,
            controller=controller,
            env=env,
        )
        orchestrator = GameOrchestrator(
            ChessService(case["fen"]),
            physical_executor,
            tracker,
            human_color="white",
            auto_computer_reply=False,
        )
        move = chess.Move.from_uci(case["uci"])
        result = orchestrator.submit_human_move(
            chess.square_name(move.from_square),
            chess.square_name(move.to_square),
            chess.piece_symbol(move.promotion) if move.promotion else None,
        )
        if not result.accepted or not result.physical_success:
            raise SystemExit(f"{name} real-physics failed: {result.error}")
        print(f"{name}: real-physics ok move={result.move_uci} fen={result.snapshot.fen}")
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate special chess moves through planner/orchestrator.")
    parser.add_argument("--all", action="store_true", help="Run all mocked special-move cases. This is the default.")
    parser.add_argument("--case", choices=sorted(CASES), help="Run one case instead of all mocked cases.")
    parser.add_argument("--real-physics", action="store_true", help="Run the selected case with MuJoCo physics.")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--drift-limit", type=float, default=0.010)
    args = parser.parse_args()

    selected = {args.case: CASES[args.case]} if args.case else CASES
    if args.real_physics and len(selected) != 1:
        raise SystemExit("--real-physics requires --case to avoid a long destructive scenario sequence")

    for name, case in selected.items():
        if args.real_physics:
            run_real_case(name, case, args)
        else:
            run_mock_case(name, case)
    print("Special-move evaluation passed.")


if __name__ == "__main__":
    main()
