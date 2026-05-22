import chess
import gymnasium as gym

import src.chess_env
from src.chess_env.controller import ScriptedController
from src.chess_game.chess_service import ChessService
from src.chess_game.game_orchestrator import GameOrchestrator
from src.chess_game.move_planner import LogicalPieceTracker
from src.chess_game.board_mapper import BoardMapper
from src.physical.movement_executor import MovementExecutor
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry
from src.physical.piece_teleport import PieceTeleporter
from src.physical.plan_executor import PhysicalPlanExecutor


def test_real_physical_single_turn_commits_after_success():
    env = gym.make("ChessFetchTask-v0", render_mode=None, show_chess_pieces=True, hide_object=True, force_scenario="transit")
    env.reset()
    registry = PieceRegistry()
    occupancy = PhysicalOccupancy(registry.starting_square_map())
    board_mapper = BoardMapper.from_configs()
    controller = ScriptedController(env, drift_limit=0.010)
    movement_executor = MovementExecutor(env, controller, board_mapper, occupancy)
    physical_executor = PhysicalPlanExecutor(
        movement_executor,
        PieceTeleporter(env, board_mapper),
        occupancy,
        controller=controller,
        env=env,
    )
    orchestrator = GameOrchestrator(
        ChessService(),
        physical_executor,
        LogicalPieceTracker(),
        auto_computer_reply=False,
    )

    result = orchestrator.submit_human_move("e2", "e4")

    assert result.accepted
    assert result.physical_success
    assert orchestrator.chess_service.board.piece_at(chess.E4).symbol() == "P"
    assert orchestrator.piece_tracker.piece_id_at("e4") == "white_pawn_e"
    assert occupancy.square_of_piece("white_pawn_e") == "e4"
    assert result.snapshot.fen == orchestrator.chess_service.fen()
    env.close()
