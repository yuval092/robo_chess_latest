import gymnasium as gym
import numpy as np

from src.utils.io import load_config

SAFE_Z = load_config("env")["safe_z"]


def test_soft_reset_flow():
    env = gym.make("ChessFetchTask-Play-v0", render_mode=None)
    env.reset()

    # Simulate a success state
    uw = env.unwrapped
    uw.current_scenario = "transit"
    nominal_xy = np.array([0.5, 0.5])
    exit_wp = np.array([0.5, 0.5, SAFE_Z])
    next_goal = np.array([0.5, 0.5, 0.460])  # descend (HOVER_Z)

    obs = uw.soft_reset(
        new_scenario="descend",
        new_goal_pos=next_goal,
        nominal_exit_pos=exit_wp,
        nominal_xy=nominal_xy,
    )

    assert isinstance(obs, dict)
    assert "observation" in obs
    assert uw.current_scenario == "descend"
    assert np.allclose(uw.goal, next_goal)
    assert uw.episode_steps == 0
    env.close()


def test_soft_reset_finger_validation():
    env = gym.make("ChessFetchTask-Play-v0", render_mode=None)
    env.reset()
    uw = env.unwrapped

    nominal_xy = np.array([0.5, 0.5])
    exit_wp = np.array([0.5, 0.5, SAFE_Z])
    next_goal = np.array([0.5, 0.5, 0.460])  # HOVER_Z

    # This should pass without error with the current physics
    obs = uw.soft_reset(
        new_scenario="descend",
        new_goal_pos=next_goal,
        nominal_exit_pos=exit_wp,
        nominal_xy=nominal_xy,
    )
    assert isinstance(obs, dict)
    env.close()
