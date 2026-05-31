"""SAC specialist trainer for RoboChess movement stages."""
import os
from datetime import datetime

import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import src.chess_env  # noqa: F401 - register chess Train/Play envs
from src.utils.io import load_config
from training.callbacks import DetailedLoggingCallback, SuccessRateEvalCallback
from training.env_factory import make_eval_env, make_train_env


class SACTrainer:
    """
    Orchestrates SAC fine-tuning for a single movement specialist.
    """

    def __init__(
        self,
        stage: str,
        num_envs: int = None,
        debug: bool = False,
        fixed_drift: bool = False,
    ):
        """Return init."""
        assert stage in {"transit", "descend", "ascend"}, f"Unknown stage: {stage}"
        self.stage = stage
        self.cfg = load_config("training")
        self.env_cfg = load_config("env")
        self.num_envs = num_envs or self.cfg["num_envs"]
        self.debug = debug
        self.fixed_drift = fixed_drift
        self.total_timesteps = self.cfg["total_timesteps"]

    def train(self, model_path: str = None, save_dir: str = None) -> str:
        """Run the full training loop for this specialist stage."""
        model_path = model_path or self.cfg["base_model"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = save_dir or f"checkpoints/{self.stage}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        print(
            f"[SACTrainer] Stage={self.stage} | Workers={self.num_envs} | "
            f"Timesteps={self.total_timesteps:,}"
        )
        print(f"[SACTrainer] Base model: {model_path}")
        print(f"[SACTrainer] Save dir:   {save_dir}")

        train_env, eval_env = self._build_envs()
        model = self._load_model(model_path, train_env, timestamp)

        print(f"[SACTrainer] Starting {self.total_timesteps:,} steps...")
        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=self._build_callbacks(eval_env, save_dir),
                reset_num_timesteps=True,
                progress_bar=True,
            )
            final_path = os.path.join(save_dir, f"final_{self.stage}.zip")
            model.save(final_path)
            print(f"[SACTrainer] Training complete! Final model: {final_path}")
            return final_path
        finally:
            train_env.close()
            eval_env.close()

    def _build_envs(self):
        """Create the vectorised training env and the single-process eval env."""
        drift_steps = self.env_cfg["drift_curriculum_steps"]
        env_fns = [
            make_train_env(
                self.stage,
                drift_curriculum_steps=drift_steps,
                debug=self.debug,
                fixed_drift=self.fixed_drift,
            )
            for _ in range(self.num_envs)
        ]
        train_env = (
            DummyVecEnv(env_fns)
            if self.num_envs == 1
            else SubprocVecEnv(env_fns, start_method="fork")
        )
        eval_env = DummyVecEnv(
            [make_eval_env(self.stage, self.env_cfg["eval_drift_limit"], debug=False)]
        )
        return train_env, eval_env

    def _build_callbacks(self, eval_env, save_dir: str) -> CallbackList:
        """Build the eval + logging callback list."""
        eval_freq = max(self.cfg["eval_freq"] // self.num_envs, 1)
        cb_eval = SuccessRateEvalCallback(
            eval_env,
            save_path=save_dir,
            name=self.stage,
            eval_freq=eval_freq,
            n_eval_episodes=self.cfg["n_eval_episodes"],
            verbose=1,
        )
        return CallbackList([DetailedLoggingCallback(), cb_eval])

    def _load_model(self, model_path: str, train_env, timestamp: str) -> SAC:
        """Load a SAC checkpoint and apply training hyperparameters from config."""
        model = SAC.load(
            model_path,
            env=train_env,
            custom_objects={
                "learning_rate": float(self.cfg["learning_rate"]),
                "batch_size": self.cfg["batch_size"],
                "target_entropy": float(self.cfg["target_entropy"]),
                "buffer_size": self.cfg["buffer_size"],
            },
            verbose=1,
        )
        model.tensorboard_log = f"logs/{self.stage}_{timestamp}/tensorboard/"
        model.learning_starts = self.cfg["learning_starts"]

        # Reset entropy coefficient and its optimizer to the configured starting values.
        # SAC.load preserves the checkpoint's ent_coef, which may be too low/high for
        # fine-tuning on a new stage.
        initial_ent_coef = float(self.cfg["initial_ent_coef"])
        ent_coef_lr = float(self.cfg["ent_coef_lr"])
        model.log_ent_coef = th.log(th.ones(1) * initial_ent_coef).to(model.device)
        model.log_ent_coef = th.nn.Parameter(model.log_ent_coef, requires_grad=True)
        model.ent_coef_optimizer = th.optim.Adam([model.log_ent_coef], lr=ent_coef_lr)

        return model
