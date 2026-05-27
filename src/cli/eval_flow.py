"""robo-chess-eval-flow — full arm piece-move evaluation using production code."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field

import chess
import gymnasium as gym
import numpy as np
from tqdm import tqdm

import src.chess_env  # noqa: F401 — registers ChessFetchTask-v0
from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_game.board_mapper import BoardMapper
from src.physical.movement_executor import MovementExecutor
from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry
from src.physical.piece_teleport import PieceTeleporter
from src.physical.plan_executor import PhysicalPlanExecutor
from src.utils.args import add_eval_args, model_overrides_from_args, resolve_model_paths


# ── Board cell constants ───────────────────────────────────────────────────

ALL_CELLS: list[str] = [chess.square_name(sq) for sq in chess.SQUARES]

CORNER_CELLS = ["a1", "h1", "a8", "h8"]
NEAR_CORNERS = [
    "b1", "g1", "b8", "g8",
    "a2", "h2", "a7", "h7",
]
EDGE_MIDPOINTS = ["a4", "h4", "d1", "d8", "e1", "e8"]
HARD_DESTINATIONS = ["a5", "c1"]
CENTER_CELLS = ["d4", "e5"]

COMPLEX_DESTINATIONS: list[str] = list(
    dict.fromkeys(
        HARD_DESTINATIONS + CORNER_CELLS + NEAR_CORNERS + EDGE_MIDPOINTS + CENTER_CELLS
    )
)
COMPLEX_SOURCES: list[str] = [
    "d4", "e5",   # center
    "a1", "h8",   # far corners
    "b2", "g7",   # near-corner
    "h4", "a5",   # edge
    "c3", "f6",   # mid-inner ring
]

# Default piece ID for flow evaluation (matches eval_all_square_moves.py default)
_DEFAULT_PIECE = "black_rook_a"


# ── Result dataclasses ────────────────────────────────────────────────────

@dataclass
class MoveCheckFailure:
    """One failed move in a run_all_square_moves sweep."""

    src: str
    dst: str
    kind: str
    error: str | None
    stages: str


@dataclass
class MoveCheckSummary:
    """Aggregate result from run_all_square_moves."""

    total: int
    passed: int
    failures: list[MoveCheckFailure]


@dataclass
class FlowResult:
    src: str
    dst: str
    success: bool
    stage_results: list = field(default_factory=list)   # list[(name, StageResult)]
    failure_reason: str = ""


# ── Environment helpers ────────────────────────────────────────────────────

def _make_flow_env(debug: bool, visualize: bool = False) -> gym.Env:
    return gym.make(
        "ChessFetchTask-v0",
        render_mode="human" if visualize else None,
        show_chess_pieces=True,
        hide_object=True,
        force_scenario="transit",
        debug=debug,
    )


def _hide_other_pieces(inner, teleporter: PieceTeleporter, piece_id: str) -> None:
    """Teleport every piece except `piece_id` far off the board."""
    registry = PieceRegistry()
    z = inner.TABLE_Z + inner.CUBE_HEIGHT / 2.0
    for idx, other_id in enumerate(registry.starting_square_map()):
        if other_id == piece_id:
            continue
        x = 2.0 + 0.05 * (idx // 8)
        y = 2.0 + 0.05 * (idx % 8)
        teleporter.teleport_piece_to_xyz(other_id, np.array([x, y, z]))


def _make_controller(
    env: gym.Env,
    model_overrides: dict[str, str],
    render_fn=None,
    delay: float = 0.0,
) -> ModelEmbeddedController:
    model_paths = resolve_model_paths(model_overrides)
    ctrl = ModelEmbeddedController(env=env, render_fn=render_fn, render_delay=delay)
    ctrl.load_all(
        transit_path=model_paths["transit"],
        descend_path=model_paths["descend"],
        ascend_path=model_paths["ascend"],
    )
    return ctrl


def _setup_eval_env(
    debug: bool,
    drift_limit: float | None,
    model_overrides: dict[str, str],
    piece_id: str,
    visualize: bool = False,
    delay: float = 0.0,
) -> tuple:
    """Create and reset a flow env with controller, teleporter, occupancy, and movement executor."""
    env = _make_flow_env(debug, visualize=visualize)
    env.reset()
    if drift_limit is not None:
        env.unwrapped.env_cfg["eval_drift_limit"] = drift_limit
    inner = env.unwrapped
    mapper = BoardMapper.from_configs()
    teleporter = PieceTeleporter(env, mapper)
    _hide_other_pieces(inner, teleporter, piece_id)
    render_fn = env.render if visualize else None
    ctrl = _make_controller(env, model_overrides, render_fn=render_fn, delay=delay)
    occupancy = PhysicalOccupancy({piece_id: None})
    movement = MovementExecutor(env, ctrl, mapper, occupancy)
    return env, inner, teleporter, ctrl, occupancy, movement


# ── Single-move executor ───────────────────────────────────────────────────

def _run_one_move(
    movement: MovementExecutor,
    piece_id: str,
    src: str,
    dst: str,
) -> FlowResult:
    result = movement.move_piece_between_squares(piece_id, src, dst)
    return FlowResult(
        src=src,
        dst=dst,
        success=result.success,
        stage_results=result.stage_results,
        failure_reason=result.error or "",
    )


# ── Mode implementation ────────────────────────────────────────────────────

def _format_stages(stage_results) -> str:
    return ", ".join(
        f"{name}:{sr.success}:{sr.error_mm:.1f}mm:{sr.crash_reason}"
        for name, sr in stage_results
    )


def _pairs_for_mode(args: argparse.Namespace) -> tuple[list[tuple[str, str]], int]:
    if args.mode == "simple":
        src = args.src or random.choice(ALL_CELLS)
        dst = args.dst or random.choice([cell for cell in ALL_CELLS if cell != src])
        return [(src, dst)], args.episodes
    if args.mode == "complex":
        pairs = [
            (src, dst)
            for dst in COMPLEX_DESTINATIONS
            for src in COMPLEX_SOURCES
            if src != dst
        ]
        reps = args.episodes if args.episodes != 1 else 3
        return pairs, reps
    return [(src, dst) for src in ALL_CELLS for dst in ALL_CELLS if src != dst], args.episodes


def _run_pairs(
    *,
    pairs: list[tuple[str, str]],
    reps: int,
    piece_id: str,
    model_overrides: dict[str, str],
    visualize: bool,
    delay: float,
    debug: bool,
    drift_limit: float | None,
    check_home: bool,
    stop_on_failure: bool = False,
    progress_desc: str = "moves",
) -> list[FlowResult]:
    total = len(pairs) * reps
    env, inner, teleporter, ctrl, occupancy, movement = _setup_eval_env(
        debug,
        drift_limit,
        model_overrides,
        piece_id,
        visualize=visualize,
        delay=delay,
    )
    physical = PhysicalPlanExecutor(movement, teleporter, occupancy, controller=ctrl, env=env)
    results: list[FlowResult] = []
    try:
        with tqdm(total=total, desc=progress_desc, unit="move") as pbar:
            for src, dst in pairs:
                for _ in range(reps):
                    inner.clear_active_piece()
                    teleporter.teleport_piece_to_square(piece_id, src)
                    occupancy.reset({piece_id: src})
                    result = _run_one_move(movement, piece_id, src, dst)
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


def _run_mode(args: argparse.Namespace, model_overrides: dict[str, str]) -> list[FlowResult]:
    pairs, reps = _pairs_for_mode(args)
    print(
        f"[eval-flow] {args.mode}  controller=model-hybrid"
        f"  pairs={len(pairs)}  reps={reps}  total={len(pairs) * reps}"
    )
    return _run_pairs(
        pairs=pairs,
        reps=reps,
        piece_id=args.piece,
        model_overrides=model_overrides,
        visualize=args.visualize,
        delay=args.delay,
        debug=args.debug,
        drift_limit=args.drift_limit,
        check_home=args.mode == "full",
        progress_desc=args.mode,
    )


# ── Summary printer ────────────────────────────────────────────────────────

def _print_summary(results: list[FlowResult], mode: str) -> None:
    total = len(results)
    ok = sum(r.success for r in results)
    print()
    print(f"=== robo-chess-eval-flow  mode={mode}  moves={total} ===")
    print(f"  Success : {ok:5d} / {total}  ({100 * ok / max(total, 1):.1f}%)")
    print(f"  Failures: {total - ok:5d} / {total}  ({100 * (total - ok) / max(total, 1):.1f}%)")

    # Per-stage breakdown
    stage_ok: dict[str, int] = {}
    stage_total: dict[str, int] = {}
    for r in results:
        for stage_name, stage_res in r.stage_results:
            stage_total[stage_name] = stage_total.get(stage_name, 0) + 1
            if stage_res.success:
                stage_ok[stage_name] = stage_ok.get(stage_name, 0) + 1
    if stage_total:
        print()
        print("  Per-stage breakdown:")
        for stage_name, n in stage_total.items():
            ok_s = stage_ok.get(stage_name, 0)
            print(f"    {stage_name:10s}  {ok_s:5d}/{n}  ({100 * ok_s / n:.1f}%)")

    # Failure reasons
    reasons: dict[str, int] = {}
    for r in results:
        if not r.success and r.failure_reason:
            reasons[r.failure_reason] = reasons.get(r.failure_reason, 0) + 1
    if reasons:
        print()
        print("  Failure reasons:")
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason:<55s}  {count:4d}  ({100 * count / max(total, 1):.1f}%)")

    # Worst destination cells (when multiple destinations are tested)
    dst_set = {r.dst for r in results}
    if len(dst_set) > 1:
        dst_ok_map: dict[str, list[bool]] = {}
        for r in results:
            dst_ok_map.setdefault(r.dst, []).append(r.success)
        dst_rates = {d: sum(v) / len(v) for d, v in dst_ok_map.items()}
        worst_dsts = sorted(dst_rates.items(), key=lambda x: x[1])[:5]
        print()
        print("  Worst destination cells:")
        for cell, rate in worst_dsts:
            print(f"    {cell}  {100 * rate:.0f}% ok  ({sum(dst_ok_map[cell])}/{len(dst_ok_map[cell])})")

    print()


# ── Public library API ────────────────────────────────────────────────────

def run_all_square_moves(
    *,
    piece_id: str = _DEFAULT_PIECE,
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
) -> MoveCheckSummary:
    """
    Run physical board-move checks for one piece across all (src, dst) pairs.

    Equivalent to ``robo-chess-eval-flow --mode full`` but returns a structured
    ``MoveCheckSummary`` suitable for programmatic use (e.g. test suites).

    Parameters
    ----------
    piece_id : physical piece ID to move for all checks
    from_square : restrict source squares to this single square (None = all)
    to_square : restrict destination squares to this single square (None = all)
    include_same : include same-square no-op moves
    max_cases : run only the first N generated (src, dst) pairs
    stop_on_failure : stop after the first failed move
    check_home : after each successful move, call return_to_home()
    drift_limit : tube constraint radius override (metres)
    visualize : open MuJoCo viewer
    delay : render delay in seconds per frame
    debug : enable verbose environment logging
    """
    sources = [from_square] if from_square else [chess.square_name(sq) for sq in chess.SQUARES]
    destinations = [to_square] if to_square else [chess.square_name(sq) for sq in chess.SQUARES]
    pairs = [
        (s, d)
        for s in sources
        for d in destinations
        if s != d or include_same
    ]
    if max_cases is not None:
        pairs = pairs[:max_cases]

    results = _run_pairs(
        pairs=pairs,
        reps=1,
        piece_id=piece_id,
        model_overrides={},
        visualize=visualize,
        delay=delay,
        debug=debug,
        drift_limit=drift_limit,
        check_home=check_home,
        stop_on_failure=stop_on_failure,
        progress_desc="all-square",
    )
    failures = []
    for result in results:
        if not result.success:
            kind = "return_home" if result.failure_reason.startswith("return_home:") else "move"
            failures.append(
                MoveCheckFailure(
                    result.src,
                    result.dst,
                    kind,
                    result.failure_reason,
                    _format_stages(result.stage_results),
                )
            )
    passed = len(results) - len(failures)
    return MoveCheckSummary(total=len(pairs), passed=passed, failures=failures)


# ── Entry point ────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="robo-chess-eval-flow",
        description=(
            "Evaluate the full arm piece-move flow using production code. "
            "Three modes: simple (one move), complex (known-hard cells), "
            "full (all 4032 board pairs)."
        ),
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["simple", "complex", "full"],
        help=(
            "Evaluation mode. "
            "simple: one move (optional --src/--dst, random if unspecified). "
            "complex: targeted corners/edges/hard cells. "
            "full: brute-force all 64×63 cell pairs."
        ),
    )
    p.add_argument(
        "--src",
        metavar="CELL",
        help="Source cell in algebraic notation (simple mode only, e.g. e2). Default: random.",
    )
    p.add_argument(
        "--dst",
        metavar="CELL",
        help="Destination cell (simple mode only, e.g. e4). Default: random.",
    )
    p.add_argument(
        "--piece",
        default=_DEFAULT_PIECE,
        help=f"Physical piece ID to move (default: {_DEFAULT_PIECE}).",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=1,
        help=(
            "Repetitions per (src, dst) pair. "
            "Default: 1 for simple and full modes, 3 for complex mode."
        ),
    )
    add_eval_args(p)
    return p


def main() -> None:
    """Entry point for robo-chess-eval-flow."""
    args = _build_parser().parse_args()

    model_overrides = model_overrides_from_args(args)

    results = _run_mode(args, model_overrides)

    _print_summary(results, args.mode)
    sys.exit(0 if all(r.success for r in results) else 1)


if __name__ == "__main__":
    main()
