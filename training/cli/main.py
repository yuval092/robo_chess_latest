"""robo-chess-train — train a specialist SAC model."""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robo-chess-train",
        description="Train a RoboChess specialist RL model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train a specialist SAC model.")
    train_p.add_argument(
        "--stage",
        required=True,
        choices=["transit", "descend", "ascend"],
        help="Which movement stage to train.",
    )
    train_p.add_argument(
        "--envs",
        type=int,
        default=None,
        help="Number of parallel training environments (default: from configs/training.yaml).",
    )
    train_p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Resume from this checkpoint or base model path (default: from configs/training.yaml).",
    )
    train_p.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (default: from configs/training.yaml).",
    )
    train_p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to write the model ZIP (default: checkpoints/<stage>_<timestamp>/).",
    )
    train_p.add_argument(
        "--fixed-drift",
        action="store_true",
        help="Skip drift curriculum; use the final drift limit from step 1.",
    )
    train_p.add_argument(
        "--debug",
        action="store_true",
        help="Verbose training and environment logs.",
    )

    return parser


def _run_train(args: argparse.Namespace) -> None:
    from training.trainer import SACTrainer

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
