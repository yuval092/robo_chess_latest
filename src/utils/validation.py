from __future__ import annotations

import numpy as np


def ensure_finite_array(name: str, value, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name}_INVALID_SHAPE expected {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"INVALID_SENSOR_DATA {name}")
    return array


def ensure_finite_scalar(name: str, value) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}_INVALID_VALUE") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"INVALID_SENSOR_DATA {name}")
    return scalar
