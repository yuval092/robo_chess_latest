import chess
import gymnasium as gym
import numpy as np

from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_game.chess_service import ChessService
from src.chess_game.game_orchestrator import GameOrchestrator
from src.chess_game.move_planner import LogicalPieceTracker
from src.physical.plan_executor import PhysicalPlanExecutor
from src.utils.io import resolve_model_paths
from tests.integration.flow_helpers import run_flow

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


def make_controller(env):
    model_paths = resolve_model_paths()
    controller = ModelEmbeddedController(env)
    controller.load_all(
        model_paths["transit"],
        model_paths["descend"],
        model_paths["ascend"],
    )
    return controller


def test_real_physical_single_turn_commits_after_success():
    env = gym.make(
        "ChessFetchTask-Play-v0",
        render_mode=None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
    )
    env.reset()
    home_qpos = robot_home_qpos(env.unwrapped)
    controller = make_controller(env)
    physical_executor = PhysicalPlanExecutor(env, controller)
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
    assert physical_executor.occupancy.square_of_piece("white_pawn_e") == "e4"
    assert result.snapshot.fen == orchestrator.chess_service.fen()
    assert np.allclose(robot_home_qpos(env.unwrapped), home_qpos)
    env.close()


def test_edge_pawn_move_after_prior_home_returns_does_not_timeout():
    env = gym.make(
        "ChessFetchTask-Play-v0",
        render_mode=None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
    )
    try:
        env.reset()
        controller = make_controller(env)
        physical_executor = PhysicalPlanExecutor(env, controller)

        run_flow(
            ["g2g4", "h7h6", "a2a4"],
            physical_executor,
            LogicalPieceTracker(),
            env,
            5.0,
            mapper=physical_executor.movement_executor.board_mapper,
            occupancy=physical_executor.occupancy,
            verify_agreement=True,
        )
    finally:
        env.close()
