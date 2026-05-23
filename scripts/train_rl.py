#!/usr/bin/env python3
"""
Train a specialist SAC model for one movement stage.
"""
import argparse
import os
import sys

sys.path.append(os.getcwd())

from training.trainer import SACTrainer


def main():
    parser = argparse.ArgumentParser(description="Train a specialist SAC model for RoboChess.")
    parser.add_argument("--stage", required=True, choices=["transit", "descend", "ascend"])
    parser.add_argument("--envs", type=int, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--fixed-drift",
        action="store_true",
        help="Skip drift curriculum; use final drift limit immediately.",
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
