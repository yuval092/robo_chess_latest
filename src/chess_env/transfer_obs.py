"""Converts the chess environment's native observation into the 25-D format
expected by the pretrained FetchPickAndPlace-v4 SAC models."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
from gymnasium import spaces

TRANSFER_OBS_SPACE = spaces.Dict(
    {
        "observation": spaces.Box(-np.inf, np.inf, shape=(25,), dtype=np.float64),
        "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
        "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
    }
)


def enable_transfer_obs(env) -> None:
    """Enable transfer observations on both the wrapper and unwrapped env."""
    unwrapped = env.unwrapped
    unwrapped._use_transfer_obs = True
    unwrapped.observation_space = TRANSFER_OBS_SPACE
    env.observation_space = TRANSFER_OBS_SPACE


@contextmanager
def transfer_obs_enabled(env) -> Generator[None, None, None]:
    """Enable transfer observations and restore wrapper/unwrapped state on exit."""
    unwrapped = env.unwrapped
    saved_flag = unwrapped._use_transfer_obs
    saved_unwrapped_space = unwrapped.observation_space
    saved_wrapper_space = env.observation_space
    try:
        enable_transfer_obs(env)
        yield
    finally:
        unwrapped._use_transfer_obs = saved_flag
        unwrapped.observation_space = saved_unwrapped_space
        env.observation_space = saved_wrapper_space
