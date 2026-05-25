"""Configuration schema tests for RoboChess YAML files."""

from __future__ import annotations

from scripts.validate_config import validate_config


def test_config_schema_is_valid() -> None:
    """Validate required config keys and internal geometry consistency."""
    validate_config()
