# Testing

## Test Layout

```text
tests/
  test_config_schema.py
  chess_env/
  chess_game/
  physical/
  ui/
  integration/
```

The suite mixes fast pure-Python tests with MuJoCo-backed tests. Stockfish tests
require the configured executable to be available.

## Core Commands

Default suite:

```bash
pytest
```

Focused logic-only checks:

```bash
pytest tests/chess_game tests/ui/test_app.py tests/test_config_schema.py
```

Environment and physical checks:

```bash
pytest tests/chess_env tests/physical
```

Opt-in exhaustive board sweep:

```bash
RUN_EXHAUSTIVE_PHYSICAL_MOVES=1 pytest tests/integration/test_all_square_moves.py
```

## `tests/chess_game`

| File | Coverage |
|---|---|
| `test_board_mapper.py` | Square-to-world geometry, board bounds, spacing, nearest-square roundtrip |
| `test_chess_service.py` | Legal moves, illegal moves, SAN, checkmate/stalemate, promotion, engine move, save/load |
| `test_game_orchestrator.py` | Commit ordering, physical failure behavior, busy state, new game, computer replies |
| `test_move_planner.py` | Normal moves, captures, castling, en passant, promotion commands, logical tracker |

These tests are mostly independent of MuJoCo.

## `tests/chess_env`

| File | Coverage |
|---|---|
| `test_characterization.py` | XML loads; transfer observation state restores after model loading success/failure |
| `test_static_assets.py` | Static XML sections, piece bodies, STL detail/clearance constraints |
| `test_task_chaining.py` | `soft_reset()`, finger transition diagnostics |
| `test_waypoints.py` | Scenario transition validation and goal derivation |

These protect environment and model-interface invariants.

## `tests/physical`

| File | Coverage |
|---|---|
| `test_piece_registry.py` | Active piece IDs, starting squares, reserve IDs, occupancy reset |
| `test_piece_teleport.py` | Active piece routing, square teleports, graveyard teleports |
| `test_scene_physics.py` | MuJoCo scene physics invariants |
| `test_zone_alignment.py` | Graveyard/reserve slots remain within configured zones and outside board range |

## `tests/ui`

| File | Coverage |
|---|---|
| `test_app.py` | Flask endpoints, snapshots, move submission, new game, computer route, busy state |
| `test_run_chess_ui_backend.py` | `QueuedUIBackend` request processing, env reset, physical reset, snapshot update |

## `tests/integration`

| File | Coverage |
|---|---|
| `test_chess_piece_move.py` | Real MuJoCo physical piece movement |
| `test_chess_turn_flow.py` | End-to-end chess turn with physical execution |
| `test_all_square_moves.py` | Opt-in 64x63 source/destination sweep |
| `flow_helpers.py` | Shared integration helpers |
| `all_square_runner.py` | Sweep support |

The all-square sweep is intentionally skipped unless
`RUN_EXHAUSTIVE_PHYSICAL_MOVES=1` is set.

## When to Add Tests

Add or update tests when changing:

- Board geometry or config schema.
- XML/STL assets or piece naming.
- Z levels, grasp thresholds, or collision geometry.
- Move planning for chess special cases.
- Physical command dispatch.
- UI route shapes or snapshot fields.
- Transfer observation shape or model loading behavior.

