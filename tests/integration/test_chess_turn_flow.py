import chess
import gymnasium as gym
import numpy as np

from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_game.chess_service import ChessService
from src.chess_game.game_orchestrator import GameOrchestrator
from src.chess_game.move_planner import (
    ArmMoveCommand,
    LogicalPieceTracker,
    MovePlanner,
    RemoveFromBoardCommand,
    TeleportCommand,
)
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_teleport import PieceTeleporter
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


def test_promotion_then_move_promoted_piece():
    """Regression: promoted pawn must go to graveyard (not stack on reserve piece),
    and the reserve piece that appears on the board must be arm-moveable.

    Covers:
    - _build_promotion_commands emits RemoveFromBoardCommand (not TeleportCommand to promotion_reserve)
    - set_active_piece works for reserve piece IDs (e.g. white_reserve_queen_1)
    """
    # Minimal legal position: white pawn on a7 ready to promote, both kings present.
    FEN = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"

    env = gym.make(
        "ChessFetchTask-Play-v0",
        render_mode=None,
        show_chess_pieces=False,  # all pieces hidden; we place only what the test needs
        hide_object=True,
        force_scenario="transit",
    )
    try:
        env.reset()

        # Place the pawn at a7; everything else stays hidden so a8 is physically clear.
        PieceTeleporter(env).teleport_piece_to_square("white_pawn_a", "a7")

        service = ChessService(starting_fen=FEN)
        tracker = LogicalPieceTracker({"white_pawn_a": "a7"})
        occupancy = PhysicalOccupancy({"white_pawn_a": "a7"})
        controller = make_controller(env)
        physical_executor = PhysicalPlanExecutor(env, controller, occupancy)

        # ── Move 1: promote pawn a7→a8=Q ──────────────────────────────────────
        move1 = service.parse_uci("a7a8q")
        plan1 = MovePlanner(service.board, tracker).plan(move1)

        # Verify plan shape: arm picks up pawn, pawn removed to graveyard, reserve queen appears.
        assert sum(isinstance(c, ArmMoveCommand) for c in plan1) == 1
        assert sum(isinstance(c, RemoveFromBoardCommand) for c in plan1) == 1
        assert sum(isinstance(c, TeleportCommand) for c in plan1) == 1

        result1 = physical_executor.execute(plan1)
        assert result1.success, f"Promotion failed: {result1.error}"

        physical_executor.return_to_home()
        service.push(move1)
        tracker.apply_plan(plan1)

        promoted_id = tracker.piece_id_at("a8")
        assert promoted_id is not None and "reserve_queen" in promoted_id
        assert tracker.piece_id_at("a7") is None

        # ── Move 2: physically move the promoted queen a8→b7 ─────────────────
        # Drive the arm directly (no chess-service turn validation needed here;
        # the regression is that set_active_piece must accept reserve piece IDs).
        plan2 = [ArmMoveCommand(promoted_id, "a8", "b7")]
        result2 = physical_executor.execute(plan2)
        assert result2.success, f"Promoted queen move failed: {result2.error}"

        tracker.apply_plan(plan2)

        assert tracker.piece_id_at("b7") == promoted_id
        assert tracker.piece_id_at("a8") is None

    finally:
        env.close()
