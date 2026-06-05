"""Validate required RoboChess configuration keys and model paths."""

from __future__ import annotations

import math
from pathlib import Path

from src.utils.io import load_config

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _PROJECT_ROOT / candidate


def _require_keys(config: dict, keys: list[str], label: str) -> None:
    for key in keys:
        assert key in config, f"Missing {label} key: {key}"


def _number(config: dict, key: str, label: str, *, positive: bool = False) -> float:
    value = config.get(key)
    assert isinstance(value, (int, float)), f"{label}.{key} must be numeric"
    assert math.isfinite(float(value)), f"{label}.{key} must be finite"
    if positive:
        assert float(value) > 0.0, f"{label}.{key} must be > 0"
    return float(value)


def _positive_numbers(config: dict, label: str, keys: tuple[str, ...]) -> None:
    for key in keys:
        _number(config, key, label, positive=True)


def _vector(config: dict, key: str, label: str, length: int) -> None:
    value = config.get(key)
    assert isinstance(value, list) and len(value) == length, (
        f"{label}.{key} must be a list of {length} numbers"
    )
    for item in value:
        assert isinstance(item, (int, float)) and math.isfinite(float(item)), (
            f"{label}.{key} must contain only finite numbers"
        )


def validate_config() -> None:
    """
    Raise AssertionError if any required config schema check fails.

    Checks:
    - Required keys in configs/env.yaml
    - Board geometry consistency in configs/chess.yaml
    - Deployed model paths present in configs/deployed_models.yaml
    """
    env = load_config("env")
    chess = load_config("chess")
    physics = load_config("physics")
    deployed = load_config("deployed_models")

    _require_keys(env, [
        "grasp_z",
        "hover_z",
        "safe_z",
        "floor_limit",
        "table_surface_z",
        "home_position_xy",
        "reconcile_xy_tolerance_m",
        "reconcile_z_tolerance_m",
        "rl_max_steps_per_stage",
    ], "env")
    _positive_numbers(env, "env", (
        "grasp_z",
        "hover_z",
        "safe_z",
        "floor_limit",
        "table_surface_z",
        "reconcile_xy_tolerance_m",
        "reconcile_z_tolerance_m",
    ))
    safe_z, hover_z, floor_limit = env["safe_z"], env["hover_z"], env["floor_limit"]
    assert safe_z > hover_z, "env.safe_z must be above env.hover_z"
    assert hover_z > floor_limit, "env.hover_z must be above env.floor_limit"
    _number(env, "rl_max_steps_per_stage", "env", positive=True)
    _vector(env, "home_position_xy", "env", 2)

    board = chess.get("board")
    pieces = chess.get("pieces")
    game = chess.get("game")
    engine = chess.get("engine")
    assert isinstance(board, dict), "chess.board must be a mapping"
    assert isinstance(pieces, dict), "chess.pieces must be a mapping"
    assert isinstance(game, dict), "chess.game must be a mapping"
    assert isinstance(engine, dict), "chess.engine must be a mapping"
    _number(board, "cell_size_m", "chess.board", positive=True)
    assert board.get("board_size") == 8, "chess.board.board_size must be 8"
    _vector(board, "center_xy", "chess.board", 2)
    _number(pieces, "piece_height_m", "chess.pieces", positive=True)
    assert game.get("human_color") in {"white", "black", "both"}, (
        "chess.game.human_color must be white, black, or both"
    )
    assert isinstance(game.get("auto_computer_reply"), bool), (
        "chess.game.auto_computer_reply must be boolean"
    )
    stockfish_path = engine.get("stockfish_path")
    assert stockfish_path is None or isinstance(stockfish_path, str), (
        "chess.engine.stockfish_path must be a string or null"
    )
    skill = engine.get("skill_level")
    assert isinstance(skill, int) and 0 <= skill <= 20, (
        "chess.engine.skill_level must be an integer from 0 to 20"
    )
    think_time = _number(engine, "think_time_s", "chess.engine", positive=True)
    assert think_time <= 30.0, "chess.engine.think_time_s must be <= 30.0"

    _require_keys(physics, [
        "vertical_quat",
        "initial_qpos",
        "env_setup_steps",
        "pos_ctrl_scale",
        "settle_tolerance",
    ], "physics")
    _vector(physics, "vertical_quat", "physics", 4)
    assert sum(float(v) * float(v) for v in physics["vertical_quat"]) > 0.0, (
        "physics.vertical_quat must be non-zero"
    )
    _vector(physics, "initial_qpos", "physics", 2)
    _number(physics, "pos_ctrl_scale", "physics", positive=True)
    _number(physics, "settle_tolerance", "physics", positive=True)
    assert isinstance(physics.get("env_setup_steps"), int) and physics["env_setup_steps"] >= 0, (
        "physics.env_setup_steps must be a non-negative integer"
    )

    for stage in ("transit", "descend", "ascend"):
        assert stage in deployed, f"Missing deployed model path: {stage}"
        assert isinstance(deployed[stage], str), f"Model path must be string: {stage}"
        assert _path(deployed[stage]).exists(), f"Model path missing: {deployed[stage]}"
