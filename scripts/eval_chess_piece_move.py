"""Evaluate physical movement of selected chess pieces."""
import argparse

import gymnasium as gym
import numpy as np

from src.chess_env.controller import ScriptedController
from src.chess_game.board_mapper import BoardMapper
from src.physical.movement_executor import MovementExecutor
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry


def piece_position(env, piece_id: str) -> np.ndarray:
    """Return piece position."""
    inner = env.unwrapped if hasattr(env, "unwrapped") else env
    previous = inner.active_piece_id
    inner.set_active_piece(piece_id)
    position = inner.get_active_piece_position()
    if previous is None:
        inner.clear_active_piece()
    else:
        inner.set_active_piece(previous)
    return position


def assert_piece_at_square(
    env, mapper: BoardMapper, piece_id: str, square: str, tolerance_mm: float
) -> float:
    """Assert piece at square."""
    expected = mapper.square_to_piece_xyz(__import__("chess").parse_square(square))
    actual = piece_position(env, piece_id)
    error_mm = float(np.linalg.norm(actual - expected) * 1000.0)
    if error_mm > tolerance_mm:
        raise SystemExit(
            f"{piece_id} is not at {square}: error={error_mm:.1f}mm tolerance={tolerance_mm:.1f}mm"
        )
    return error_mm


def main() -> None:
    """Parse command-line arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Execute one physical chess-piece board move."
    )
    parser.add_argument("--piece", required=True, help="Piece id, e.g. white_pawn_e")
    parser.add_argument("--src", required=True, help="Source square, e.g. e2")
    parser.add_argument("--dst", required=True, help="Destination square, e.g. e4")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--drift-limit", type=float, default=0.010)
    parser.add_argument(
        "--starting-position",
        choices=("crowded",),
        default="crowded",
        help="Initial board layout. The current chess eval uses the crowded standard start.",
    )
    parser.add_argument("--final-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--nonmoving-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--skip-nonmoving-check", action="store_true")
    args = parser.parse_args()

    env = gym.make(
        "ChessFetchTask-v0",
        render_mode="human" if args.visualize else None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
    )
    try:
        env.reset()
        render_fn = env.render if args.visualize else None
        controller = ScriptedController(
            env,
            drift_limit=args.drift_limit,
            render_fn=render_fn,
            render_delay=args.delay,
        )
        registry = PieceRegistry()
        mapper = BoardMapper.from_configs()
        starting_square_map = registry.starting_square_map()
        occupancy = PhysicalOccupancy(starting_square_map)
        occupancy.assert_piece_at(args.piece, args.src)
        occupancy.assert_square_empty(args.dst)
        src_error_mm = assert_piece_at_square(
            env, mapper, args.piece, args.src, args.final_tolerance_mm
        )
        nonmoving_start = {
            piece_id: piece_position(env, piece_id)
            for piece_id in starting_square_map
            if piece_id != args.piece
        }
        executor = MovementExecutor(
            env,
            controller,
            mapper,
            occupancy,
        )
        result = executor.move_piece_between_squares(args.piece, args.src, args.dst)
        print(
            f"success={result.success} piece={result.piece_id} {result.src_square}->{result.dst_square} error={result.error}"
        )
        print(f"start_check {args.piece}@{args.src} error={src_error_mm:.1f}mm")
        for stage_name, stage in result.stage_results:
            print(
                f"{stage_name:<8} success={stage.success} err={stage.error_mm:.1f}mm reason={stage.crash_reason}"
            )
        if not result.success:
            raise SystemExit(1)
        final_error_mm = assert_piece_at_square(
            env, mapper, args.piece, args.dst, args.final_tolerance_mm
        )
        print(f"final_check {args.piece}@{args.dst} error={final_error_mm:.1f}mm")
        if not args.skip_nonmoving_check:
            max_displacement_mm = 0.0
            worst_piece = None
            for piece_id, start_pos in nonmoving_start.items():
                displacement_mm = float(
                    np.linalg.norm(piece_position(env, piece_id) - start_pos) * 1000.0
                )
                if displacement_mm > max_displacement_mm:
                    max_displacement_mm = displacement_mm
                    worst_piece = piece_id
            print(
                f"nonmoving_max_displacement piece={worst_piece} displacement={max_displacement_mm:.1f}mm"
            )
            if max_displacement_mm > args.nonmoving_tolerance_mm:
                raise SystemExit(
                    f"Non-moving piece displacement exceeded tolerance: "
                    f"{worst_piece} {max_displacement_mm:.1f}mm > {args.nonmoving_tolerance_mm:.1f}mm"
                )
    finally:
        env.close()


if __name__ == "__main__":
    main()
