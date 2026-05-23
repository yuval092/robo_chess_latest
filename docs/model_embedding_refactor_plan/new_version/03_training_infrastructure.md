# Training Infrastructure

## Overview

Training is orchestrated by a `SACTrainer` class that manages:
- Parallel environment vectorisation (`SubprocVecEnv` for speed)
- Isolated per-scenario evaluation environments
- SB3 callbacks for logging and model saving
- Loading from pretrained weights + entropy reset

We **keep checkpointing simple**: save `best_model_{stage}.zip` whenever success improves.
No auto-resume, no complex checkpoint scanning.

---

## 1. `configs/training.yaml` — All Hyperparameters

Create this file (does not exist yet in our project):

```yaml
# configs/training.yaml

# Base pretrained model (FetchPickAndPlace-v4 transfer weights)
base_model: "models/sac-FetchPickAndPlace-v4.zip"

# Parallelism
num_envs: 8              # SubprocVecEnv workers per training run

# Training length
total_timesteps: 1000000  # 1M per specialist stage

# SAC Hyperparameters (fine-tuning values, not scratch values)
learning_rate: 0.00005   # 5e-5: conservative LR for fine-tuning
batch_size: 512
target_entropy: -4.0
learning_starts: 10000   # Steps before first gradient update
buffer_size: 300000      # Replay buffer (reduced from default to avoid OOM)

# Entropy reset on model load
# Pretrained weights have near-zero entropy (~0.003).
# Resetting to 0.1 re-enables exploration on the new reward surface.
initial_ent_coef: 0.1
ent_coef_lr: 0.001

# Evaluation
eval_freq: 10000         # Run eval every N total steps
n_eval_episodes: 50      # Episodes per evaluation run

# Logging
log_freq: 2000           # How often DetailedLoggingCallback prints (total steps)
moving_avg_window: 100   # Episodes for rolling success/reward average
```

---

## 2. `training/callbacks.py`

Three SB3 callback classes, adapted from the reference project:

```python
# training/callbacks.py
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from src.utils.logger import setup_logger
from src.utils.config import load_config

progress_logger = setup_logger("training_progress", "logs/training_progress.log")


class DetailedLoggingCallback(BaseCallback):
    """
    Logs per-episode stats (reward, length, success rate) every log_freq steps.
    Also breaks down success by scenario name (from info["scenario"]).
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        cfg = load_config("training")
        self.log_freq = cfg.get("log_freq", 2000)
        self.window = cfg.get("moving_avg_window", 100)
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.last_log_step = 0

    def _on_step(self) -> bool:
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
                self.num_timesteps, mean_rew, mean_len, mean_suc
            )
        return True


class SuccessRateEvalCallback(EvalCallback):
    """
    Extends EvalCallback to:
    1. Save 'latest_model_{name}.zip' every eval
    2. Save 'best_model_{name}.zip' whenever success rate improves
    3. Track last_mean_success for CombinedSuccessCallback
    """

    def __init__(self, *args, save_path: str, name: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = save_path
        self.name = name
        self._best_success = -1.0
        self.last_mean_success = -1.0

    def _on_step(self) -> bool:
        result = super()._on_step()
        # NOTE: EvalCallback stores success flags in `self.evaluations_results` in modern SB3.
        # The older `self._is_success_buffer` was renamed. Use `self.last_mean_reward` as a
        # proxy or check `self.evaluations_successes`. The safest approach is to read from
        # `self.locals["infos"]` if needed, or use `self.evaluations_successes[-1]` after
        # EvalCallback has run. We use the parent's internal buffer name here:
        success_buf = getattr(self, "_is_success_buffer", None) or \
                      (self.evaluations_successes[-1] if getattr(self, "evaluations_successes", None) else None)
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if success_buf is not None:
                latest = float(np.mean(success_buf))
                self.last_mean_success = latest

                os.makedirs(self.save_path, exist_ok=True)
                # Always save latest
                self.model.save(os.path.join(self.save_path, f"latest_model_{self.name}"))

                # Save best if improved
                if latest > self._best_success:
                    self._best_success = latest
                    self.model.save(os.path.join(self.save_path, f"best_model_{self.name}"))
                    progress_logger.info(
                        "*** NEW BEST [%s] *** Success: %.1f%% -> saved to %s",
                        self.name.upper(), latest * 100.0,
                        os.path.join(self.save_path, f"best_model_{self.name}.zip")
                    )
        return result
```

---

## 3. `training/trainer.py` — `SACTrainer` Class

```python
# training/trainer.py
import os
from datetime import datetime
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.utils.config import load_config
from training.callbacks import DetailedLoggingCallback, SuccessRateEvalCallback
from training.envs import make_train_env, make_eval_env
import src.chess_env  # registers ChessFetchTask-v0


class SACTrainer:
    """
    Orchestrates SAC fine-tuning for a single specialist stage.
    
    Args:
        stage:      "transit" | "descend" | "ascend"
        num_envs:   Number of SubprocVecEnv workers (default from training.yaml)
        debug:      Enable env debug logging
        fixed_drift: Skip curriculum, use DRIFT_LIMIT_END from the start
    """

    def __init__(self, stage: str, num_envs: int = None, debug: bool = False, fixed_drift: bool = False):
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
        
        Args:
            model_path: Path to .zip file to fine-tune from.
                        Defaults to configs/training.yaml base_model.
            save_dir:   Directory for checkpoints and final model.
                        Defaults to checkpoints/{stage}_{timestamp}/
        """
        model_path = model_path or self.cfg.get("base_model", "models/sac-FetchPickAndPlace-v4.zip")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = save_dir or f"checkpoints/{self.stage}_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)

        print(f"[SACTrainer] Stage={self.stage} | Workers={self.num_envs} | Timesteps={self.total_timesteps:,}")
        print(f"[SACTrainer] Base model: {model_path}")
        print(f"[SACTrainer] Save dir:   {save_dir}")

        # ── Curriculum Steps (per worker) ─────────────────────────────────
        # env.yaml: drift_curriculum_steps is the PER-WORKER step count.
        # With 8 workers the total drift steps = 8 × drift_curriculum_steps.
        # The archive used `target_curriculum_total: 500000` and computed
        # per_worker = total // num_envs. We store per-worker directly to avoid
        # the circular divide-and-multiply pattern.
        drift_curriculum_steps_per_worker = self.env_cfg.get("drift_curriculum_steps", 62_500)

        # ── Training Environments ─────────────────────────────────────────
        train_env = SubprocVecEnv([
            make_train_env(
                self.stage,
                drift_curriculum_steps=drift_curriculum_steps_per_worker,
                debug=self.debug,
                fixed_drift=self.fixed_drift,
            )
            for _ in range(self.num_envs)
        ])

        # ── Evaluation Environments (one per scenario, locked) ────────────
        eval_drift_limit = self.env_cfg.get("eval_drift_limit", 0.005)
        eval_env = DummyVecEnv([make_eval_env(self.stage, eval_drift_limit, debug=False)])

        # ── Callbacks ─────────────────────────────────────────────────────
        eval_freq = max(self.cfg.get("eval_freq", 10_000) // self.num_envs, 1)
        n_eval_ep = self.cfg.get("n_eval_episodes", 50)

        cb_eval = SuccessRateEvalCallback(
            eval_env,
            save_path=save_dir,
            name=self.stage,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_ep,
            verbose=1,
        )
        callbacks = CallbackList([DetailedLoggingCallback(), cb_eval])

        # ── Load Model ────────────────────────────────────────────────────
        # NOTE: hyperparameter overrides must go through custom_objects, not
        # top-level kwargs. SAC.load() restores internal state (including the
        # replay buffer) from the zip, and only reads custom_objects as overrides.
        # Passing buffer_size as a plain kwarg has no effect on the loaded buffer.
        model = SAC.load(
            model_path,
            env=train_env,
            custom_objects={
                "learning_rate":  float(self.cfg.get("learning_rate", 5e-5)),
                "batch_size":     self.cfg.get("batch_size", 512),
                "target_entropy": float(self.cfg.get("target_entropy", -4.0)),
                "buffer_size":    self.cfg.get("buffer_size", 300_000),
            },
            verbose=1,
        )
        model.tensorboard_log = f"logs/{self.stage}_{timestamp}/tensorboard/"
        model.learning_starts = self.cfg.get("learning_starts", 10_000)

        # ── Entropy Reset ─────────────────────────────────────────────────
        # Pretrained weights have near-zero entropy (~0.003), which causes the
        # model to be over-committed to old behaviours. Reset to 0.1 to allow
        # fresh exploration on the new reward landscape.
        initial_ent_coef = float(self.cfg.get("initial_ent_coef", 0.1))
        ent_coef_lr = float(self.cfg.get("ent_coef_lr", 1e-3))
        model.log_ent_coef = th.log(th.ones(1) * initial_ent_coef).to(model.device)
        model.log_ent_coef = th.nn.Parameter(model.log_ent_coef, requires_grad=True)
        model.ent_coef_optimizer = th.optim.Adam([model.log_ent_coef], lr=ent_coef_lr)

        # ── Train ─────────────────────────────────────────────────────────
        print(f"[SACTrainer] Starting {self.total_timesteps:,} steps...")
        model.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=True,
            progress_bar=True,
        )

        # ── Save Final Model ──────────────────────────────────────────────
        final_path = os.path.join(save_dir, f"final_{self.stage}.zip")
        model.save(final_path)
        print(f"[SACTrainer] Training complete! Final model: {final_path}")

        train_env.close()
        eval_env.close()
        return final_path
```

---

## 4. `scripts/train_rl.py` — CLI Entry Point

```python
#!/usr/bin/env python3
"""
Train a specialist SAC model for one movement stage.

Usage:
    python scripts/train_rl.py --stage transit
    python scripts/train_rl.py --stage descend --envs 4 --timesteps 500000
    python scripts/train_rl.py --stage ascend --model checkpoints/transit_.../best_model_transit.zip
    python scripts/train_rl.py --stage transit --fixed-drift  # skip curriculum
"""
import sys
import os
import argparse

sys.path.append(os.getcwd())

from training.trainer import SACTrainer


def main():
    parser = argparse.ArgumentParser(description="Train a specialist SAC model for RoboChess.")
    parser.add_argument(
        "--stage", required=True, choices=["transit", "descend", "ascend"],
        help="Which movement stage to train."
    )
    parser.add_argument(
        "--envs", type=int, default=None,
        help="Number of parallel training environments (default: from training.yaml)."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to .zip model to fine-tune from (default: training.yaml base_model)."
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training timesteps (default: training.yaml total_timesteps)."
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to save checkpoints and final model."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable verbose environment logging."
    )
    parser.add_argument(
        "--fixed-drift", action="store_true",
        help="Skip drift curriculum; use final DRIFT_LIMIT_END immediately."
    )
    args = parser.parse_args()

    trainer = SACTrainer(
        stage=args.stage,
        num_envs=args.envs,
        debug=args.debug,
        fixed_drift=args.fixed_drift,
    )
    if args.timesteps:
        trainer.total_timesteps = args.timesteps

    trainer.train(model_path=args.model, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
```

---

## 5. Training a Specialist: Step-by-Step

```bash
# Step 1: Train transit specialist
python scripts/train_rl.py --stage transit --envs 8 --timesteps 1000000

# Step 2: Train descend specialist  
python scripts/train_rl.py --stage descend --envs 8 --timesteps 1000000

# Step 3: Train ascend specialist
python scripts/train_rl.py --stage ascend --envs 8 --timesteps 1000000

# Optional: Start from an existing checkpoint instead of the pretrained base
python scripts/train_rl.py --stage transit --model checkpoints/transit_.../best_model_transit.zip

# Optional: Skip drift curriculum for debugging/quick testing
python scripts/train_rl.py --stage transit --fixed-drift --timesteps 100000
```

Each run creates a `checkpoints/{stage}_{timestamp}/` directory with:
- `best_model_{stage}.zip` — best performing checkpoint during training
- `latest_model_{stage}.zip` — most recent checkpoint
- `final_{stage}.zip` — model at end of training

Tensorboard logs appear in `logs/{stage}_{timestamp}/tensorboard/`.

---

## 6. Why `SubprocVecEnv`, Not `DummyVecEnv`

`DummyVecEnv` runs all environments sequentially in a single process. With 8 workers this gives
no speedup (Python GIL blocks true parallelism for CPU-bound MuJoCo simulation).

`SubprocVecEnv` spawns each worker as a separate subprocess — true parallelism. With 8 workers,
data collection is ~8× faster. Training that would take 8 hours takes ~1 hour.

The downside is slightly more complex debugging. Use `--envs 1` (which degrades to single-process)
for debugging, and scale up to 8 for real training runs.

> [!NOTE]
> On Windows, `SubprocVecEnv` requires the training script to be run under `if __name__ == "__main__":`
> or via the CLI entry point (`scripts/train_rl.py`). This is already handled by the script.
