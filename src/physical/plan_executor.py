from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.chess_game.move_planner import ArmMoveCommand, PhysicalPlan, RemoveFromBoardCommand, TeleportCommand
from src.physical.movement_executor import MovementExecutor
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry
from src.physical.piece_teleport import PieceTeleporter
from src.utils.config import load_config


@dataclass(frozen=True)
class PhysicalExecutionResult:
    success: bool
    command_results: list
    error: str | None = None


class PhysicalPlanExecutor:
    def __init__(
        self,
        movement_executor: MovementExecutor,
        piece_teleporter: PieceTeleporter,
        occupancy: PhysicalOccupancy,
        controller=None,
        env=None,
    ):
        self.movement_executor = movement_executor
        self.piece_teleporter = piece_teleporter
        self.occupancy = occupancy
        self.controller = controller
        self.env = env.unwrapped if hasattr(env, "unwrapped") else env

    def execute(self, plan: PhysicalPlan) -> PhysicalExecutionResult:
        results = []
        try:
            for command in plan.commands:
                if isinstance(command, RemoveFromBoardCommand):
                    self.piece_teleporter.teleport_piece_to_graveyard(command.piece_id, command.graveyard_slot)
                    self.occupancy.set_piece_square(command.piece_id, None)
                    results.append((command, True))
                elif isinstance(command, ArmMoveCommand):
                    result = self.movement_executor.move_piece_between_squares(
                        command.piece_id, command.src_square, command.dst_square
                    )
                    results.append((command, result))
                    if not result.success:
                        return PhysicalExecutionResult(False, results, result.error)
                elif isinstance(command, TeleportCommand):
                    if command.destination_kind == "promotion_reserve":
                        self.piece_teleporter.teleport_piece_to_promotion_reserve(command.piece_id, command.destination_id)
                        self.occupancy.set_piece_square(command.piece_id, None)
                    elif command.destination_kind == "graveyard":
                        self.piece_teleporter.teleport_piece_to_graveyard(command.piece_id, command.destination_id)
                        self.occupancy.set_piece_square(command.piece_id, None)
                    elif command.destination_kind == "square":
                        self.piece_teleporter.teleport_piece_to_square(command.piece_id, command.destination_id)
                        self.occupancy.set_piece_square(command.piece_id, command.destination_id)
                    else:
                        return PhysicalExecutionResult(False, results, f"Unknown teleport destination {command.destination_kind}")
                    results.append((command, True))
                else:
                    return PhysicalExecutionResult(False, results, f"Unsupported command {command}")
        except Exception as exc:
            return PhysicalExecutionResult(False, results, str(exc))
        return PhysicalExecutionResult(True, results)

    def return_to_home(self) -> PhysicalExecutionResult:
        if self.controller is None or self.env is None:
            return PhysicalExecutionResult(True, [], None)
        home_xy = np.array(load_config("chess")["game"]["arm_home_xy"])
        result = self.controller.run_transit(home_xy)
        return PhysicalExecutionResult(result.success, [("return_to_home", result)], result.crash_reason)

    def reset_board_state(self) -> None:
        """Reset physical chess pieces and expected occupancy to the standard start."""
        starting_square_map = PieceRegistry().starting_square_map()
        if self.env is not None and hasattr(self.env, "_reset_chess_piece_bodies"):
            self.env._reset_chess_piece_bodies()
            if hasattr(self.env, "clear_active_piece"):
                self.env.clear_active_piece()
        else:
            for piece_id, square in starting_square_map.items():
                self.piece_teleporter.teleport_piece_to_square(piece_id, square)
        self.occupancy.reset(starting_square_map)

    def reset_occupancy(self) -> None:
        self.reset_board_state()
