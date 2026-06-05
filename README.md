# RoboChess

Project: RoboChess.
Students: Alon Sternberg, Yuval Farkash.
Description: MuJoCo simulation of a robotic chess player. A simulated Fetch arm picks up and places chess pieces on an 8x8 board.

## Install

0. (Optional) Create a python virtual environment (venv): `python3 -m venv .venv && source .venv/bin/activate`

1. Install the game: `pip3 install -e .`

2. Install Stockfish, and make sure the executable is on `PATH` env var:
    1. Ubuntu: `sudo apt install stockfish`.
    2. Windows: https://stockfishchess.org/download/

## Play

In order to start playing the chess game, follow the instructions:

1. Start the game: `robo-chess-play`

2. Open the browser at `http://localhost:9999`. The UI should be presented to you.

## Training

In case you wish to train a new RL model, use the `robo-chess-train` command.
You can run `robo-chess-train --help` to see all of the different parameters.

## Tests

Feel free to run our unit & integration tests: `PYTHONPATH=. pytest`
