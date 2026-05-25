"""Training environment factory functions and wrapper registry."""
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

import src.chess_env  # noqa: F401 - register ChessFetchTask-v0
from training.envs.ascend_env import AscendTrainEnv
from training.envs.descend_env import DescendTrainEnv
from training.envs.transit_env import TransitTrainEnv

WRAPPER_MAP = {
    "transit": TransitTrainEnv,
    "ascend": AscendTrainEnv,
    "descend": DescendTrainEnv,
}


def make_train_env(
    stage: str,
    drift_curriculum_steps: int,
    debug: bool = False,
    fixed_drift: bool = False,
):
    """Creates a monitored, curriculum-enabled training env for one stage."""
    def _init():
        """Create and return a monitored wrapped environment."""
        base = gym.make(
            "ChessFetchTask-v0",
            force_scenario=stage,
            drift_curriculum_steps=drift_curriculum_steps,
            debug=debug,
            fixed_drift=fixed_drift,
        )
        wrapped = WRAPPER_MAP[stage](base)
        return Monitor(wrapped)

    return _init


def make_eval_env(stage: str, eval_drift_limit: float = 0.005, debug: bool = False):
    """Creates a monitored evaluation env locked to one stage."""
    def _init():
        """Create and return a monitored wrapped environment."""
        base = gym.make(
            "ChessFetchTask-v0",
            force_scenario=stage,
            force_drift_limit=eval_drift_limit,
            debug=debug,
        )
        wrapped = WRAPPER_MAP[stage](base)
        return Monitor(wrapped)

    return _init
