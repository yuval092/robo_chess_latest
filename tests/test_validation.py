import numpy as np
import pytest

from src.utils.validation import ensure_finite_array, ensure_finite_scalar


def test_ensure_finite_array_rejects_nan():
    with pytest.raises(ValueError, match="INVALID_SENSOR_DATA"):
        ensure_finite_array("sensor", np.array([0.0, np.nan]), (2,))


def test_ensure_finite_scalar_rejects_inf():
    with pytest.raises(ValueError, match="INVALID_SENSOR_DATA"):
        ensure_finite_scalar("sensor", float("inf"))
