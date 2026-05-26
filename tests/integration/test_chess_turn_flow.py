import chess
import gymnasium as gym
import numpy as np

from tests.integration.flow_helpers import run_flow
from src.chess_env.controller import ScriptedController
from src.chess_game.board_mapper import BoardMapper
from src.chess_game.chess_service import ChessService
from src.chess_game.game_orchestrator import GameOrchestrator
from src.chess_game.move_planner import LogicalPieceTracker
from src.physical.movement_executor import MovementExecutor
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry
from src.physical.piece_teleport import PieceTeleporter
from src.physical.plan_executor import PhysicalPlanExecutor

ARM_HOME_JOINTS = (
    "robot0:torso_lift_joint",
    "robot0:shoulder_pan_joint",
    "robot0:shoulder_lift_joint",
    "robot0:upperarm_roll_joint",
    "robot0:elbow_flex_joint",
    "robot0:forearm_roll_joint",
    "robot0:wrist_flex_joint",
    "robot0:wrist_roll_joint",
    "robot0:l_gripper_finger_joint",
    "robot0:r_gripper_finger_joint",
)


def robot_home_qpos(uw):
    return np.array(
        [uw.data.qpos[uw.model.joint(name).qposadr[0]] for name in ARM_HOME_JOINTS]
    )


def test_real_physical_single_turn_commits_after_success():
    env = gym.make(
        "ChessFetchTask-v0",
        render_mode=None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
    )
    env.reset()
    home_qpos = robot_home_qpos(env.unwrapped)
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
    assert np.allclose(robot_home_qpos(env.unwrapped), home_qpos)
    env.close()


def test_edge_pawn_move_after_prior_home_returns_does_not_timeout():
    env = gym.make(
        "ChessFetchTask-v0",
        render_mode=None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
    )
    try:
        env.reset()
        mapper = BoardMapper.from_configs()
        occupancy = PhysicalOccupancy(PieceRegistry().starting_square_map())
        controller = ScriptedController(env, drift_limit=0.010)
        physical_executor = PhysicalPlanExecutor(
            MovementExecutor(env, controller, mapper, occupancy),
            PieceTeleporter(env, mapper),
            occupancy,
            controller=controller,
            env=env,
        )

        run_flow(
            ["g2g4", "h7h6", "a2a4"],
            physical_executor,
            LogicalPieceTracker(),
            env,
            5.0,
            mapper=mapper,
            occupancy=occupancy,
            verify_agreement=True,
        )
    finally:
        env.close()
