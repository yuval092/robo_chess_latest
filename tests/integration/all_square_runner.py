"""Opt-in exhaustive physical move sweep helpers for integration tests."""

from __future__ import annotations

from dataclasses import dataclass, field

import chess
import gymnasium as gym
import numpy as np
from tqdm import tqdm

import src.chess_env  # noqa: F401 - registers chess Train/Play envs
from src.chess_env.model_controller import ModelEmbeddedController
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry
from src.physical.piece_teleport import PieceTeleporter
from src.physical.plan_executor import PhysicalPlanExecutor
from src.utils.io import resolve_model_paths

DEFAULT_ALL_SQUARE_PIECE = "black_rook_a"


@dataclass
class MoveCheckFailure:
    src: str
    dst: str
    kind: str
    error: str | None
    stages: str


@dataclass
class MoveCheckSummary:
    total: int
    passed: int
    failures: list[MoveCheckFailure]


@dataclass
class FlowResult:
    src: str
    dst: str
    success: bool
    stage_results: list = field(default_factory=list)
    failure_reason: str = ""


def _make_flow_env(debug: bool, visualize: bool) -> gym.Env:
    return gym.make(
        "ChessFetchTask-Play-v0",
        render_mode="human" if visualize else None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
        debug=debug,
    )


def _hide_other_pieces(inner, teleporter: PieceTeleporter, piece_id: str) -> None:
    registry = PieceRegistry()
    z = inner.TABLE_SURFACE_Z + inner.PIECE_HEIGHT / 2.0
    for idx, other_id in enumerate(registry.starting_square_map()):
        if other_id == piece_id:
            continue
        teleporter.teleport_piece_to_xyz(
            other_id,
            np.array([2.0 + 0.05 * (idx // 8), 2.0 + 0.05 * (idx % 8), z]),
        )


def _format_stages(stage_results) -> str:
    return ", ".join(
        f"{name}:{sr.success}:{sr.error_mm:.1f}mm:{sr.crash_reason}"
        for name, sr in stage_results
    )


def _setup_all_square_env(
    *,
    debug: bool,
    drift_limit: float | None,
    model_overrides: dict[str, str],
    piece_id: str,
    visualize: bool,
    delay: float,
) -> tuple:
    env = _make_flow_env(debug, visualize)
    env.reset()
    if drift_limit is not None:
        env.unwrapped.env_cfg["eval_drift_limit"] = drift_limit
    inner = env.unwrapped
    model_paths = resolve_model_paths(model_overrides)
    controller = ModelEmbeddedController(
        env=env,
        render_fn=env.render if visualize else None,
        render_delay=delay,
    )
    controller.load_all(
        transit_path=model_paths["transit"],
        descend_path=model_paths["descend"],
        ascend_path=model_paths["ascend"],
    )
    occupancy = PhysicalOccupancy({piece_id: None})
    physical = PhysicalPlanExecutor(env, controller, occupancy)
    _hide_other_pieces(inner, physical.piece_teleporter, piece_id)
    return env, inner, physical.piece_teleporter, physical.occupancy, physical.movement_executor, physical


def _run_pairs(
    *,
    pairs: list[tuple[str, str]],
    piece_id: str,
    model_overrides: dict[str, str],
    visualize: bool,
    delay: float,
    debug: bool,
    drift_limit: float | None,
    check_home: bool,
    stop_on_failure: bool,
) -> list[FlowResult]:
    env, inner, teleporter, occupancy, movement, physical = _setup_all_square_env(
        debug=debug,
        drift_limit=drift_limit,
        model_overrides=model_overrides,
        piece_id=piece_id,
        visualize=visualize,
        delay=delay,
    )
    results: list[FlowResult] = []
    try:
        with tqdm(total=len(pairs), desc="all-square", unit="move") as pbar:
            for src, dst in pairs:
                inner.clear_active_piece()
                teleporter.teleport_piece_to_square(piece_id, src)
                occupancy.reset({piece_id: src})
                move = movement.move_piece_between_squares(piece_id, src, dst)
                result = FlowResult(
                    src=src,
                    dst=dst,
                    success=move.success,
                    stage_results=move.stage_results,
                    failure_reason=move.error or "",
                )
                results.append(result)

                if result.success and check_home:
                    home_result = physical.return_to_home()
                    if not home_result.success:
                        result.success = False
                        result.failure_reason = f"return_home: {home_result.error}"

                if not result.success:
                    env.reset()
                    _hide_other_pieces(inner, teleporter, piece_id)
                    if stop_on_failure:
                        pbar.update(1)
                        return results

                ok = sum(item.success for item in results)
                pbar.update(1)
                pbar.set_postfix({"ok%": f"{100 * ok / len(results):.0f}"})
    finally:
        env.close()
    return results


def run_all_square_moves(
    *,
    piece_id: str = DEFAULT_ALL_SQUARE_PIECE,
    from_square: str | None = None,
    to_square: str | None = None,
    include_same: bool = False,
    max_cases: int | None = None,
    stop_on_failure: bool = False,
    check_home: bool = True,
    drift_limit: float | None = None,
    visualize: bool = False,
    delay: float = 0.0,
    debug: bool = False,
    model_overrides: dict[str, str] | None = None,
) -> MoveCheckSummary:
    sources = (
        [from_square] if from_square else [chess.square_name(sq) for sq in chess.SQUARES]
    )
    destinations = (
        [to_square] if to_square else [chess.square_name(sq) for sq in chess.SQUARES]
    )
    pairs = [
        (src, dst)
        for src in sources
        for dst in destinations
        if src != dst or include_same
    ]
    if max_cases is not None:
        pairs = pairs[:max_cases]

    results = _run_pairs(
        pairs=pairs,
        piece_id=piece_id,
        model_overrides=model_overrides or {},
        visualize=visualize,
        delay=delay,
        debug=debug,
        drift_limit=drift_limit,
        check_home=check_home,
        stop_on_failure=stop_on_failure,
    )
    failures = [
        MoveCheckFailure(
            result.src,
            result.dst,
            "return_home" if result.failure_reason.startswith("return_home:") else "move",
            result.failure_reason,
            _format_stages(result.stage_results),
        )
        for result in results
        if not result.success
    ]
    return MoveCheckSummary(
        total=len(pairs),
        passed=len(results) - len(failures),
        failures=failures,
    )
