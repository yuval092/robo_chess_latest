import os
from datetime import datetime

import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import src.chess_env  # noqa: F401 - register ChessFetchTask-v0
from src.utils.config import load_config
from training.callbacks import DetailedLoggingCallback, SuccessRateEvalCallback
from training.envs import make_eval_env, make_train_env


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
        assert stage in {"transit", "descend", "ascend"}, f"Unknown stage: {stage}"
        self.stage = stage
        self.cfg = load_config("training")
        self.env_cfg = load_config("env")
        self.num_envs = num_envs or self.cfg.get("num_envs", 8)
        self.debug = debug
        self.fixed_drift = fixed_drift
        self.total_timesteps = self.cfg.get("total_timesteps", 1_000_000)

    def train(self, model_path: str = None, save_dir: str = None):
        """
        Run the full training loop for this specialist stage.
        """
        model_path = model_path or self.cfg.get(
            "base_model", "models/sac-FetchPickAndPlace-v4.zip"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = save_dir or f"checkpoints/{self.stage}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        print(
            f"[SACTrainer] Stage={self.stage} | Workers={self.num_envs} | "
            f"Timesteps={self.total_timesteps:,}"
        )
        print(f"[SACTrainer] Base model: {model_path}")
        print(f"[SACTrainer] Save dir:   {save_dir}")

        drift_steps = self.env_cfg.get("drift_curriculum_steps", 62_500)
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

        eval_drift_limit = self.env_cfg.get("eval_drift_limit", 0.005)
        eval_env = DummyVecEnv([make_eval_env(self.stage, eval_drift_limit, debug=False)])

        eval_freq = max(self.cfg.get("eval_freq", 10_000) // self.num_envs, 1)
        cb_eval = SuccessRateEvalCallback(
            eval_env,
            save_path=save_dir,
            name=self.stage,
            eval_freq=eval_freq,
            n_eval_episodes=self.cfg.get("n_eval_episodes", 50),
            verbose=1,
        )
        callbacks = CallbackList([DetailedLoggingCallback(), cb_eval])

        model = SAC.load(
            model_path,
            env=train_env,
            custom_objects={
                "learning_rate": float(self.cfg.get("learning_rate", 5e-5)),
                "batch_size": self.cfg.get("batch_size", 512),
                "target_entropy": float(self.cfg.get("target_entropy", -4.0)),
                "buffer_size": self.cfg.get("buffer_size", 300_000),
            },
            verbose=1,
        )
        model.tensorboard_log = f"logs/{self.stage}_{timestamp}/tensorboard/"
        model.learning_starts = self.cfg.get("learning_starts", 10_000)

        initial_ent_coef = float(self.cfg.get("initial_ent_coef", 0.1))
        ent_coef_lr = float(self.cfg.get("ent_coef_lr", 1e-3))
        model.log_ent_coef = th.log(th.ones(1) * initial_ent_coef).to(model.device)
        model.log_ent_coef = th.nn.Parameter(model.log_ent_coef, requires_grad=True)
        model.ent_coef_optimizer = th.optim.Adam([model.log_ent_coef], lr=ent_coef_lr)

        print(f"[SACTrainer] Starting {self.total_timesteps:,} steps...")
        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
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
