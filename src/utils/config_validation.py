"""Validate required RoboChess configuration keys and model paths."""

from __future__ import annotations

from pathlib import Path

from src.utils.io import load_config


def validate_config() -> None:
    """
    Raise AssertionError if any required config schema check fails.

    Checks:
    - Required keys in configs/env.yaml
    - Board geometry consistency in configs/chess.yaml
    - Deployed model paths present in configs/deployed_models.yaml
    - Base training model file exists on disk
    """
    env = load_config("env")
    chess = load_config("chess")
    deployed = load_config("deployed_models")

    for key in [
        "transit_tolerance_m",
        "vertical_tolerance_m",
        "step_gain",
        "min_step_size_m",
        "max_step_size_m",
        "transit_max_steps",
        "vertical_max_steps",
        "grasp_verify_drift_mm",
        "reconcile_xy_tolerance_m",
        "reconcile_z_tolerance_m",
        "rl_max_steps_per_stage",
    ]:
        assert key in env, f"Missing env key: {key}"

    board = chess["board"]
    validation = board["validation"]
    assert validation["required_cell_size_m"] == board["cell_size_m"]
    assert (
        validation["required_board_width_m"]
        == board["board_size"] * board["cell_size_m"]
    )
    assert "reachability_expected" in chess

    for stage in ("transit", "descend", "ascend"):
        assert stage in deployed, f"Missing deployed model path: {stage}"
        assert isinstance(deployed[stage], str), f"Model path must be string: {stage}"

    base_model = load_config("training")["base_model"]
    assert Path(base_model).exists(), f"Base model missing: {base_model}"
