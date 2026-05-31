"""robo-chess-train — train a specialist SAC model."""

from __future__ import annotations
from training.trainer import SACTrainer
import argparse


def _build_parser() -> argparse.ArgumentParser:
    train_parser = argparse.ArgumentParser(
        prog="robo-chess-train",
        description="Train a RoboChess specialist RL model.",
    )

    train_parser.add_argument(
        "--stage",
        required=True,
        choices=["transit", "descend", "ascend"],
        help="Which movement stage to train.",
    )
    train_parser.add_argument(
        "--envs",
        type=int,
        default=None,
        help="Number of parallel training environments (default: from configs/training.yaml).",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Resume from this checkpoint or base model path (default: from configs/training.yaml).",
    )
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (default: from configs/training.yaml).",
    )
    train_parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to write the model ZIP (default: checkpoints/<stage>_<timestamp>/).",
    )
    train_parser.add_argument(
        "--fixed-drift",
        action="store_true",
        help="Skip drift curriculum; use the final drift limit from step 1. " \
        "use it when fine-tuning a model that was already trained with the curriculum.",
    )
    train_parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose training and environment logs.",
    )

    return train_parser


def _run_train(args: argparse.Namespace) -> None:
    print(f"[robo-chess-train] stage={args.stage}  envs={args.envs}  timesteps={args.timesteps}")
    trainer = SACTrainer(
        stage=args.stage,
        num_envs=args.envs,
        debug=args.debug,
        fixed_drift=args.fixed_drift,
    )
    if args.timesteps is not None:
        trainer.total_timesteps = args.timesteps

    trainer.train(model_path=args.model, save_dir=args.save_dir)
    print("[robo-chess-train] Training complete.")

def main() -> None:
    """Entry point for robo-chess-train."""
    parser = _build_parser()
    args = parser.parse_args()

    _run_train(args)


if __name__ == "__main__":
    main()
