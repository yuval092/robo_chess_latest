# Coordinate System

Chess square centers are configured in `configs/chess.yaml` and mapped by `BoardMapper`.

## Board Geometry

- Cell size: exactly `0.08m`.
- Board size: `8 x 8`, total width `0.64m`.
- Near rank center x: rank 1 is `0.600m`.
- Far rank center x: rank 8 is `1.160m`.
- Files advance along world Y.
- Ranks advance along world X.

`a1` is the near-left square from White's perspective and is dark. Board square geoms are visual-only; `table0_surface` remains the physical support geom.

## Source Of Truth

- `configs/chess.yaml`: board center, cell size, piece dimensions, reserve/graveyard layout, game settings.
- `configs/env.yaml`: table dimensions, table surface height, robot/environment constants.
- `BoardMapper`: all square-to-world and world-to-square conversions.

Use `scripts/eval_chess_reachability.py` to verify the configured centers and arm reachability.
