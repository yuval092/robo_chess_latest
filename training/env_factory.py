"""Factory functions for creating monitored training and evaluation environments."""
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

import src.chess_env  # noqa: F401 - register chess Train/Play envs


def make_train_env(
    stage: str,
    drift_curriculum_steps: int,
    debug: bool = False,
    fixed_drift: bool = False,
):
    """Creates a monitored, curriculum-enabled training env for one stage."""
    def _init():
        return Monitor(gym.make(
            "ChessFetchTask-Train-v0",
            force_scenario=stage,
            drift_curriculum_steps=drift_curriculum_steps,
            debug=debug,
            fixed_drift=fixed_drift,
        ))

    return _init


def make_eval_env(stage: str, eval_drift_limit: float = 0.005, debug: bool = False):
    """Creates a monitored evaluation env locked to one stage."""
    def _init():
        return Monitor(gym.make(
            "ChessFetchTask-Train-v0",
            force_scenario=stage,
            force_drift_limit=eval_drift_limit,
            debug=debug,
        ))

    return _init
