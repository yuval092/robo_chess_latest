# Coordinate System

Chess square centers are configured in `configs/chess.yaml` and mapped by `BoardMapper`.

## Board Geometry

- Cell size: exactly `0.08m`.
- Board size: `8 x 8`, total width `0.64m`.
- File A center x: `0.600m`.
- File H center x: `1.160m`.
- Rank 1 center y: `-0.0159m`.
- Rank 8 center y: `0.5441m`.
- Files advance along world X.
- Ranks advance along world Y.

`a1` is the low-X, low-Y corner and is dark. White pieces start on low Y, black pieces start on high Y, and the robot arm is centered laterally between the two sides. Board square geoms are visual-only; `table0_surface` remains the physical support geom.

## Off-Board Zones

Captured pieces and promotion reserves use floor zones outside the table footprint:

- white graveyard: low Y, x `0.640..0.775`, y `-0.300..-0.165`
- black graveyard: high Y, x `0.640..0.775`, y `0.680..0.815`
- white promotion reserve: low Y, x `0.820..1.135`, y `-0.300..-0.165`
- black promotion reserve: high Y, x `0.820..1.135`, y `0.680..0.815`

## Source Of Truth

- `configs/chess.yaml`: board center, cell size, piece dimensions, reserve/graveyard layout, game settings.
- `configs/env.yaml`: table dimensions, table surface height, robot/environment constants.
- `BoardMapper`: all square-to-world and world-to-square conversions.

Use `scripts/eval_chess_reachability.py` to verify the configured centers and arm reachability.
