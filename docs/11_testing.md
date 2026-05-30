# Testing

## Test Suite Layout

```
tests/
├── conftest.py                        # Shared fixtures (env factory, controller stubs)
├── test_config_schema.py              # Config key presence and type checks
│
├── chess_env/
│   ├── test_characterization.py       # Observation shape, reset, scenario sampling
│   ├── test_static_assets.py          # XML loading, board geometry, piece bodies
│   └── test_task_chaining.py          # Soft-reset and stage transition tests
│
├── chess_game/
│   ├── test_board_mapper.py           # Square→XY conversions, table bounds
│   ├── test_chess_service.py          # Move validation, UCI parsing, SAN history
│   ├── test_game_orchestrator.py      # Human/computer move flow, new_game, snapshot
│   └── test_move_planner.py           # Plan generation for normal/capture/castle/promotion
│
├── physical/
│   ├── test_piece_registry.py         # Piece ID structure, starting square map
│   ├── test_piece_teleport.py         # Teleport to square/graveyard, occupancy update
│   ├── test_scene_physics.py          # Full physics smoke tests (all squares reachable)
│   └── test_zone_alignment.py         # Reserve zone geometry checks
│
├── integration/
│   ├── flow_helpers.py                # Shared integration test utilities
│   ├── test_chess_piece_move.py       # Single-move arm pipeline tests
│   ├── test_chess_turn_flow.py        # Full turn flow (plan → execute → home)
│   ├── test_all_square_moves.py       # Exhaustive 64×63 sweep (opt-in)
│   └── all_square_runner.py           # Runner for the all-square sweep
│
└── ui/
    ├── test_app.py                    # Flask route unit tests with a stub backend
    └── test_run_chess_ui_backend.py   # QueuedUIBackend threading tests
```

---

## Key Test Categories

### Unit Tests

Fast, no MuJoCo. Cover chess logic, board mapping, piece registry, occupancy, move planning, and UI routes. Run with `pytest tests/ -m "not slow"` (or just `pytest tests/` — MuJoCo tests are marked slow or require the env).

### Integration Tests

Require a full MuJoCo environment and loaded SAC models. Marked `@pytest.mark.slow` or guarded by environment variables.

- **`test_chess_piece_move.py`** — verifies a single arm move from a specific square.
- **`test_chess_turn_flow.py`** — verifies a full game turn including home return and occupancy update.

### Exhaustive All-Square Sweep

```bash
RUN_EXHAUSTIVE_PHYSICAL_MOVES=1 pytest tests/integration/test_all_square_moves.py
```

Runs all 64×63 source→destination combinations using a single black rook (`black_rook_a` by default). After each move, optionally returns arm to home (`check_home=True`). Stops on first failure by default (`stop_on_failure=True`).

`run_all_square_moves()` in `all_square_runner.py` accepts:

| Param | Default | Description |
|---|---|---|
| `piece_id` | `"black_rook_a"` | Piece to test with |
| `from_square` | all | Restrict source square |
| `to_square` | all | Restrict destination square |
| `max_cases` | all | Cap number of pairs |
| `stop_on_failure` | `True` | Stop on first failure |
| `check_home` | `True` | Verify home return after each move |
| `drift_limit` | config value | Override eval drift limit |
| `visualize` | `False` | Open MuJoCo viewer |
| `model_overrides` | `{}` | Override model paths |

Returns `MoveCheckSummary(total, passed, failures)`. Each `MoveCheckFailure` includes `src`, `dst`, `kind`, `error`, and `stages` (formatted stage results with crash reasons).

---

## Config Schema Test

`tests/test_config_schema.py` validates that required keys exist in `env.yaml` and that deployed model paths resolve to existing files. Run as part of the standard test suite.
