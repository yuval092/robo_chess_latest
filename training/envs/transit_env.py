"""Training wrapper for the transit specialist policy."""
import gymnasium as gym

from src.chess_env.base_env import TRANSFER_OBS_SPACE


class TransitTrainEnv(gym.Wrapper):
    """
    Training wrapper for the transit specialist model.

    The base env is locked to the scenario by gym.make(force_scenario=...);
    this wrapper only enables transfer observations for SB3.
    """

    def __init__(self, env: gym.Env):
        """Return init."""
        super().__init__(env)
        uw = env.unwrapped
        uw._use_transfer_obs = True
        uw.observation_space = TRANSFER_OBS_SPACE
        self.observation_space = TRANSFER_OBS_SPACE
