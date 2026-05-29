"""Gymnasium environment registration for RoboChess training and play envs."""

from gymnasium.envs.registration import register

register(
    id="ChessFetchTask-Train-v0",
    entry_point="src.chess_env.training_env:ChessTrainingEnv",
    max_episode_steps=200,
)

register(
    id="ChessFetchTask-Play-v0",
    entry_point="src.chess_env.production_env:ChessProductionEnv",
)
