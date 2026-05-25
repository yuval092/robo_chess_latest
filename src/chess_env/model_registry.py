"""Load and store specialist SAC models keyed by stage name."""

from __future__ import annotations

import contextlib
import io

from stable_baselines3 import SAC

from src.chess_env.transfer_obs import transfer_obs_enabled


class ModelRegistry:
    """Loads SAC specialist models and provides access by stage name."""

    KNOWN_STAGES = frozenset({"transit", "descend", "ascend"})

    def __init__(self, wrapped_env) -> None:
        """Initialise this object."""
        self._wrapped_env = wrapped_env
        self._models: dict[str, SAC | None] = {
            stage: None for stage in self.KNOWN_STAGES
        }

    def load(self, stage: str, path: str) -> None:
        """Load one stage's model from a checkpoint path."""
        if stage not in self.KNOWN_STAGES:
            raise ValueError(f"Unknown stage: {stage!r}")
        if not path:
            raise ValueError(f"No model path provided for {stage!r}")

        with transfer_obs_enabled(self._wrapped_env):
            with contextlib.redirect_stdout(io.StringIO()):
                self._models[stage] = SAC.load(path, env=self._wrapped_env)

    def get(self, stage: str) -> SAC | None:
        """Return the loaded model for a stage, or None if not loaded."""
        return self._models.get(stage)

    def available_stages(self) -> list[str]:
        """Return the list of stages with a loaded model."""
        return [stage for stage, model in self._models.items() if model is not None]
