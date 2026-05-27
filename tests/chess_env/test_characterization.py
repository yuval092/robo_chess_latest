"""Characterization tests for cleanup refactor behavior locks."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

import src.chess_env.model_controller as model_controller
from src.chess_env.environment_generation import DEFAULT_SCENE_PATH
from src.chess_game.board_mapper import BoardMapper


def test_pick_and_place_xml_loads_in_mujoco() -> None:
    """Construct the ChessFetchTask-v0 MuJoCo env without XML load errors."""
    import src.chess_env  # noqa: F401

    env = gym.make("ChessFetchTask-v0", render_mode=None)
    env.close()


def test_model_load_restores_observation_space_after_success_and_failure(
    monkeypatch,
) -> None:
    """Ensure model loading restores wrapped and unwrapped observation state."""
    import src.chess_env  # noqa: F401

    def fake_load(path, env):
        if path == "bad.zip":
            raise RuntimeError("load failed")
        return object()

    monkeypatch.setattr(SAC, "load", fake_load)

    env = gym.make("ChessFetchTask-v0", render_mode=None)
    unwrapped = env.unwrapped
    wrapped_space = env.observation_space
    unwrapped_space = unwrapped.observation_space
    flag_name = (
        "_use_transfer_obs"
        if hasattr(unwrapped, "_use_transfer_obs")
        else "_use_phase9_obs"
    )
    transfer_flag = getattr(unwrapped, flag_name)

    controller = model_controller.ModelEmbeddedController(env)
    controller.load_model("transit", "ok.zip")

    assert env.observation_space == wrapped_space
    assert unwrapped.observation_space == unwrapped_space
    assert getattr(unwrapped, flag_name) == transfer_flag

    try:
        controller.load_model("transit", "bad.zip")
    except RuntimeError:
        pass

    assert env.observation_space == wrapped_space
    assert unwrapped.observation_space == unwrapped_space
    assert getattr(unwrapped, flag_name) == transfer_flag
    env.close()


def test_board_mapper_all_64_square_coordinates() -> None:
    """Verify BoardMapper.from_configs maps every square to expected 8 cm centres."""
    mapper = BoardMapper.from_configs()
    for file_idx, file_name in enumerate("abcdefgh"):
        for rank_idx in range(8):
            square = f"{file_name}{rank_idx + 1}"
            expected = np.array([0.600 + file_idx * 0.08, -0.0159 + rank_idx * 0.08])
            assert np.allclose(mapper.square_name_to_xy(square), expected)


def test_ensure_environment_generated_is_idempotent_and_absolute() -> None:
    """Call environment generation twice and verify identical XML via absolute path."""
    try:
        from src.chess_env.environment_generation import ensure_environment_generated
    except ImportError:
        from src.chess_env.environment_generation import (
            regenerate_scene as ensure_environment_generated,
        )

    scene_path = Path(DEFAULT_SCENE_PATH).resolve()
    ensure_environment_generated(scene_path)
    first = scene_path.read_text()
    ensure_environment_generated(scene_path)
    second = scene_path.read_text()

    assert scene_path.is_absolute()
    assert first == second


def test_import_src_chess_env_writes_no_files() -> None:
    """Importing src.chess_env must not mutate generated asset files."""
    tracked = [
        Path("chess_env/assets/pick_and_place.xml"),
        Path("chess_env/stls/chess/pawn.stl"),
    ]
    before = {path: path.stat().st_mtime_ns for path in tracked if path.exists()}

    for key in list(sys.modules):
        if key == "src.chess_env" or key.startswith("src.chess_env."):
            del sys.modules[key]
    importlib.import_module("src.chess_env")

    after = {path: path.stat().st_mtime_ns for path in tracked if path.exists()}
    assert before == after
