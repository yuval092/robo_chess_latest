"""Physical plan executor for arm, teleport, and remove commands."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_game.board_mapper import BoardMapper
from src.chess_game.move_planner import (
    ArmMoveCommand,
    RemoveFromBoardCommand,
    TeleportCommand,
)
from src.physical.movement_executor import MovementExecutor
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry
from src.physical.piece_teleport import PieceTeleporter
from src.utils.io import load_config


@dataclass(frozen=True)
class PhysicalExecutionResult:
    success: bool
    command_results: list
    error: str | None = None


class PhysicalPlanExecutor:
    def __init__(self, env, controller: ModelEmbeddedController, occupancy: PhysicalOccupancy | None = None):
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env
        self.controller = controller
        board_mapper = BoardMapper.from_configs()
        self.occupancy = occupancy or PhysicalOccupancy(PieceRegistry().starting_square_map())
        self.piece_teleporter = PieceTeleporter(env, board_mapper)
        self.movement_executor = MovementExecutor(
            env, controller, board_mapper, self.occupancy, self.piece_teleporter
        )
        self._handlers: dict[type, Callable] = {
            RemoveFromBoardCommand: self._handle_remove,
            ArmMoveCommand: self._handle_arm_move,
            TeleportCommand: self._handle_teleport,
        }

    def execute(self, plan: list) -> PhysicalExecutionResult:
        """Execute all commands in a plan and stop on first arm-move failure."""
        results = []
        try:
            for command in plan:
                handler = self._handlers.get(type(command))
                if handler is None:
                    return PhysicalExecutionResult(
                        False, results, f"Unsupported command {command}"
                    )
                result = handler(command)
                results.append((command, result))
                if result is not None and not result.success:
                    return PhysicalExecutionResult(False, results, result.error)
        except Exception as exc:
            return PhysicalExecutionResult(False, results, str(exc))
        return PhysicalExecutionResult(True, results)

    def _handle_remove(self, command: RemoveFromBoardCommand) -> None:
        """Teleport a captured piece to graveyard and clear occupancy."""
        self.piece_teleporter.teleport_piece_to_graveyard(
            command.piece_id, command.graveyard_slot
        )
        self.occupancy.set_piece_square(command.piece_id, None)

    def _handle_arm_move(self, command: ArmMoveCommand):
        """Execute a board-to-board arm move."""
        return self.movement_executor.move_piece_between_squares(
            command.piece_id, command.src_square, command.dst_square
        )

    def _handle_teleport(self, command: TeleportCommand) -> None:
        """Teleport a piece to the requested destination kind."""
        new_square = None
        if command.destination_kind == "promotion_reserve":
            self.piece_teleporter.teleport_piece_to_promotion_reserve(
                command.piece_id, command.destination_id
            )
        elif command.destination_kind == "graveyard":
            self.piece_teleporter.teleport_piece_to_graveyard(
                command.piece_id, command.destination_id
            )
        elif command.destination_kind == "square":
            self.piece_teleporter.teleport_piece_to_square(
                command.piece_id, command.destination_id
            )
            new_square = command.destination_id
        else:
            raise ValueError(f"Unknown teleport destination {command.destination_kind}")
        self.occupancy.set_piece_square(command.piece_id, new_square)

    def return_to_home(self) -> PhysicalExecutionResult:
        home_xy = np.array(load_config("env")["home_position_xy"])
        result = self.controller.run_transit(home_xy)
        command_results = [("return_to_home", result)]
        if not result.success:
            return PhysicalExecutionResult(False, command_results, result.crash_reason)

        posture_result = self.env.reset_arm_to_home_posture()
        if not posture_result.get("success", False):
            return PhysicalExecutionResult(
                False, command_results, posture_result.get("reason")
            )

        return PhysicalExecutionResult(True, command_results, None)

    def reset_board_state(self) -> None:
        """Reset physical chess pieces and expected occupancy to the standard start."""
        starting_square_map = PieceRegistry().starting_square_map()
        self.env._reset_chess_piece_bodies()
        self.env.clear_active_piece()
        self.occupancy.reset(starting_square_map)
