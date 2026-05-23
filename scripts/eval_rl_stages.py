#!/usr/bin/env python3
"""
Standalone per-stage RL model evaluation.
"""
import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

sys.path.append(os.getcwd())

import src.chess_env  # noqa: F401 - register ChessFetchTask-v0
from src.utils.config import load_config
from training.envs import WRAPPER_MAP


def evaluate(
    stage: str,
    model_path: str,
    n_episodes: int,
    visualize: bool = False,
    delay: float = 0.0,
):
    env_cfg = load_config("env")
    eval_drift = env_cfg.get("eval_drift_limit", 0.005)
    render_mode = "human" if visualize else None
    env = gym.make(
        "ChessFetchTask-v0",
        force_scenario=stage,
        force_drift_limit=eval_drift,
        render_mode=render_mode,
    )
    env = WRAPPER_MAP[stage](env)
    model = SAC.load(model_path, env=env)

    successes = 0
    crashes = 0
    timeouts = 0
    total_reward = 0.0

    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_success = False
            ep_crash = False
            crash_reason = None

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated

                if visualize:
                    env.render()
                    if delay > 0:
                        time.sleep(delay)

                if terminated:
                    if info.get("is_success"):
                        ep_success = True
                    elif info.get("crash_reason"):
                        ep_crash = True
                        crash_reason = info["crash_reason"]

            if ep_success:
                successes += 1
            elif ep_crash:
                crashes += 1
            else:
                timeouts += 1
            total_reward += ep_reward

            print(
                f"Episode {ep + 1:3d}/{n_episodes} | "
                f"{'SUCCESS' if ep_success else ('CRASH' if ep_crash else 'TIMEOUT'):8s} | "
                f"reward={ep_reward:8.1f}"
                + (f" | {crash_reason}" if crash_reason else "")
            )

        print(f"\n{'=' * 60}")
        print(f"Stage: {stage.upper()} | Model: {model_path}")
        print(f"  Success:  {successes:3d}/{n_episodes} ({successes / n_episodes:.1%})")
        print(f"  Crashes:  {crashes:3d}/{n_episodes} ({crashes / n_episodes:.1%})")
        print(f"  Timeouts: {timeouts:3d}/{n_episodes} ({timeouts / n_episodes:.1%})")
        print(f"  Avg Reward: {total_reward / n_episodes:.2f}")
        print(f"{'=' * 60}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["transit", "descend", "ascend"])
    parser.add_argument("--model", required=True, help="Path to .zip model file")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    evaluate(args.stage, args.model, args.n_episodes, args.visualize, args.delay)


if __name__ == "__main__":
    main()
