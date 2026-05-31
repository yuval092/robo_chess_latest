"""Shared I/O utilities: YAML config loading and logging setup."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def load_config(config_name: str) -> dict:
    """Load and return a YAML config by name from configs/."""
    path = _CONFIG_ROOT / f"{config_name}.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_model_paths(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Resolve deployed model paths with explicit overrides applied."""
    deployed = load_config("deployed_models")
    overrides = overrides or {}
    paths = {
        stage: overrides.get(stage) or deployed.get(stage)
        for stage in ("transit", "descend", "ascend")
    }
    missing = [stage for stage, path in paths.items() if not path]
    if missing:
        flags = ", ".join(f"--{stage}-model" for stage in missing)
        raise ValueError(
            f"Missing model path(s) for {', '.join(missing)}. "
            f"Set configs/deployed_models.yaml or pass {flags}."
        )
    not_found = [stage for stage, path in paths.items() if not os.path.exists(path)]
    if not_found:
        raise FileNotFoundError(
            f"Model file(s) not found for: {', '.join(not_found)}. "
            f"Check paths in configs/deployed_models.yaml."
        )
    return paths


def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Create a file-backed logger at INFO level, idempotent on repeated calls."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
        )
        logger.addHandler(handler)
    return logger
