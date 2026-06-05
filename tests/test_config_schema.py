"""Configuration schema tests for RoboChess YAML files."""

from __future__ import annotations

import copy

import pytest

import src.utils.config_validation as config_validation
import src.utils.io as io_utils
from src.utils.config_validation import validate_config
from src.utils.io import load_config


def test_config_schema_is_valid() -> None:
    """Validate required config keys and internal geometry consistency."""
    validate_config()


def test_config_validation_rejects_invalid_engine_skill(monkeypatch) -> None:
    configs = {
        name: copy.deepcopy(load_config(name))
        for name in ("env", "chess", "physics", "deployed_models")
    }
    configs["chess"]["engine"]["skill_level"] = 99

    monkeypatch.setattr(config_validation, "load_config", lambda name: configs[name])

    with pytest.raises(AssertionError, match="skill_level"):
        validate_config()


def test_resolve_model_paths_uses_project_root_for_relative_paths(monkeypatch, tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for stage in ("transit", "descend", "ascend"):
        (model_dir / f"{stage}.zip").write_text("model")

    monkeypatch.setattr(io_utils, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        io_utils,
        "load_config",
        lambda name: {
            "transit": "models/transit.zip",
            "descend": "models/descend.zip",
            "ascend": "models/ascend.zip",
        },
    )

    paths = io_utils.resolve_model_paths()

    assert paths == {
        stage: str(model_dir / f"{stage}.zip")
        for stage in ("transit", "descend", "ascend")
    }
