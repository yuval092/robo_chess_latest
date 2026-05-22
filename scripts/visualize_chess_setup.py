import argparse
import time

import gymnasium as gym

import src.chess_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the chess board with all active pieces visible.")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    env = gym.make("ChessFetchTask-v0", render_mode="human", show_chess_pieces=True, hide_object=True)
    env.reset()
    end_time = time.time() + args.seconds
    while time.time() < end_time:
        env.render()
        time.sleep(0.02)
    env.close()


if __name__ == "__main__":
    main()
