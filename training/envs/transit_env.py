import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TransitTrainEnv(gym.Wrapper):
    """
    Training wrapper for the transit specialist model.

    The base env is locked to the scenario by gym.make(force_scenario=...);
    this wrapper only enables Phase-9 observations for SB3.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        uw = env.unwrapped
        uw._use_phase9_obs = True
        phase9_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float64),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
        })
        uw.observation_space = phase9_space
        self.observation_space = phase9_space
