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
        self._collect_finished_episodes()
        if self._should_log():
            self._log_progress()
        return True

    def _collect_finished_episodes(self) -> None:
        """Accumulate stats from any episodes that completed this step."""
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.successes.append(float(info.get("is_success", 0.0)))

    def _should_log(self) -> bool:
        return (
            self.num_timesteps - self.last_log_step >= self.log_freq
            and bool(self.episode_rewards)
        )

    def _log_progress(self) -> None:
        self.last_log_step = self.num_timesteps
        progress_logger.info(
            "Step %d | Reward: %.2f | Len: %.1f | Success: %.1f%%",
            self.num_timesteps,
            np.mean(self.episode_rewards[-self.window:]),
            np.mean(self.episode_lengths[-self.window:]),
            np.mean(self.successes[-self.window:]) * 100.0,
        )


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
        result = super()._on_step()
        if not self._is_eval_step():
            return result
        success_rate = self._read_success_rate()
        self._save_latest_checkpoint(success_rate)
        if success_rate > self._best_success:
            self._update_best_checkpoint(success_rate)
        return result

    def _is_eval_step(self) -> bool:
        return self.eval_freq > 0 and self.n_calls % self.eval_freq == 0

    def _read_success_rate(self) -> float | None:
        """Return mean success rate from the just-completed eval run, or None if empty."""
        buf = self._is_success_buffer
        return float(np.mean(buf)) if buf else -1.0

    def _save_latest_checkpoint(self, success_rate: float) -> None:
        self.last_mean_success = success_rate
        os.makedirs(self.save_path, exist_ok=True)
        self.model.save(os.path.join(self.save_path, f"latest_model_{self.name}"))

    def _update_best_checkpoint(self, success_rate: float) -> None:
        """Save best checkpoint and log when success rate improves."""
        self._best_success = success_rate
        best_path = os.path.join(self.save_path, f"best_model_{self.name}.zip")
        self.model.save(os.path.join(self.save_path, f"best_model_{self.name}"))
        progress_logger.info(
            "*** NEW BEST [%s] *** Success: %.1f%% -> saved to %s",
            self.name.upper(),
            success_rate * 100.0,
            best_path,
        )
