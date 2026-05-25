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
