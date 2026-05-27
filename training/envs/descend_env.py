"""Training wrapper for the descend specialist policy."""
import gymnasium as gym

from src.chess_env.transfer_obs import TRANSFER_OBS_SPACE


class DescendTrainEnv(gym.Wrapper):
    """Training wrapper for the descend specialist model."""

    def __init__(self, env: gym.Env):
        """Return init."""
        super().__init__(env)
        uw = env.unwrapped
        uw._use_transfer_obs = True
        uw.observation_space = TRANSFER_OBS_SPACE
        self.observation_space = TRANSFER_OBS_SPACE
