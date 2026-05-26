"""robo-chess-train — train or evaluate a specialist SAC model."""

from __future__ import annotations

import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robo-chess-train",
        description="Train or evaluate a RoboChess specialist RL model.",
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

    # ── eval ───────────────────────────────────────────────────────────────
    eval_p = sub.add_parser(
        "eval",
        help=(
            "Evaluate a trained model on a movement stage. "
            "Delegates to robo-chess-eval-stage logic."
        ),
    )
    eval_p.add_argument(
        "--stage",
        required=True,
        choices=["transit", "descend", "ascend"],
        help="Stage to evaluate.",
    )
    eval_p.add_argument(
        "--model",
        required=True,
        help="Path to the model ZIP to evaluate.",
    )
    eval_p.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes to run (default: 50).",
    )
    eval_p.add_argument(
        "--drift-limit",
        type=float,
        default=None,
        help="Override drift tolerance in metres (default: from config).",
    )
    eval_p.add_argument(
        "--debug",
        action="store_true",
        help="Verbose debug logs forwarded to the evaluation runner.",
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


def _run_eval(args: argparse.Namespace) -> None:
    from src.cli.eval_stage import run_eval_stage

    run_eval_stage(
        stages=[args.stage],
        episodes=args.episodes,
        controller="model",
        model_overrides={args.stage: args.model},
        drift_limit=args.drift_limit,
        debug=args.debug,
    )


def main() -> None:
    """Entry point for robo-chess-train."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        _run_train(args)
    else:
        _run_eval(args)


if __name__ == "__main__":
    main()
