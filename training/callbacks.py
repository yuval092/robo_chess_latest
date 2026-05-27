"""Training callbacks for SAC specialist fine-tuning."""
import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from src.utils.io import load_config, setup_logger

progress_logger = setup_logger("training_progress", "logs/training_progress.log")


class DetailedLoggingCallback(BaseCallback):
    """
    Logs rolling reward, episode length, and success rate during training.
    """

    def __init__(self, verbose=0):
        """Return init."""
        super().__init__(verbose)
        cfg = load_config("training")
        self.log_freq = cfg["log_freq"]
        self.window = cfg["moving_avg_window"]
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.last_log_step = 0

    def _on_step(self) -> bool:
        """Return on step."""
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.successes.append(float(info.get("is_success", 0.0)))

        if (
            self.num_timesteps - self.last_log_step >= self.log_freq
            and self.episode_rewards
        ):
            self.last_log_step = self.num_timesteps
            mean_rew = np.mean(self.episode_rewards[-self.window:])
            mean_len = np.mean(self.episode_lengths[-self.window:])
            mean_suc = np.mean(self.successes[-self.window:]) * 100.0
            progress_logger.info(
                "Step %d | Reward: %.2f | Len: %.1f | Success: %.1f%%",
                self.num_timesteps,
                mean_rew,
                mean_len,
                mean_suc,
            )
        return True


class SuccessRateEvalCallback(EvalCallback):
    """
    Saves latest and best checkpoints based on evaluation success rate.
    """

    def __init__(self, *args, save_path: str, name: str, **kwargs):
        """Return init."""
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        self.name = name
        self._best_success = -1.0
        self.last_mean_success = -1.0

    def _on_step(self) -> bool:
        """Return on step."""
        result = super()._on_step()
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return result

        success_buf = getattr(self, "_is_success_buffer", None)
        if success_buf is None and getattr(self, "evaluations_successes", None):
            success_buf = self.evaluations_successes[-1]
        if success_buf is None:
            return result

        latest = float(np.mean(success_buf))
        self.last_mean_success = latest
        os.makedirs(self.save_path, exist_ok=True)
        self.model.save(os.path.join(self.save_path, f"latest_model_{self.name}"))

        if latest > self._best_success:
            self._best_success = latest
            best_path = os.path.join(self.save_path, f"best_model_{self.name}.zip")
            self.model.save(os.path.join(self.save_path, f"best_model_{self.name}"))
            progress_logger.info(
                "*** NEW BEST [%s] *** Success: %.1f%% -> saved to %s",
                self.name.upper(),
                latest * 100.0,
                best_path,
            )
        return result
