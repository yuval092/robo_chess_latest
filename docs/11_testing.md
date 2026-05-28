# Testing

## Overview

The test suite covers all major layers of the system: chess logic, physical layer, static scene assets, MuJoCo environment, UI, and integration. Tests are written with `pytest` and organised into a clear directory structure matching the source tree.

```
tests/
├── conftest.py                     — Shared pytest configuration
├── test_config_schema.py           — Config file validation smoke test
├── chess_env/                      — MuJoCo environment and RL infrastructure
│   ├── test_characterization.py    — Behaviour locks: XML loading, obs space restore
│   ├── test_static_assets.py        — Static scene XML and STL asset correctness
│   ├── test_task_chaining.py       — soft_reset, transition_validate, stage chaining
│   └── test_waypoints.py           — Waypoint constants, chain validation, goal derivation
├── chess_game/                     — Chess logic (no MuJoCo required)
│   ├── test_board_mapper.py        — Square-to-XY mapping and geometry
│   ├── test_chess_service.py       — ChessService: moves, validation, engine, persistence
│   ├── test_game_orchestrator.py   — GameOrchestrator: move flow, physical/logical state
│   └── test_move_planner.py        — MovePlanner and LogicalPieceTracker
├── physical/                       — Physical layer (MuJoCo required for teleport tests)
│   ├── test_piece_registry.py      — PieceRegistry, PhysicalPiece, reserve IDs
│   ├── test_piece_teleport.py      — PieceTeleporter, PhysicalPlanExecutor
│   ├── test_piece_xml.py           — MuJoCo body/joint/geom existence for all pieces
│   └── test_zone_alignment.py      — Graveyard/reserve slot geometry
├── ui/                             — Flask API and QueuedUIBackend
│   ├── test_app.py                 — REST endpoint integration with fake executor
│   └── test_run_chess_ui_backend.py — QueuedUIBackend thread safety
└── integration/                    — End-to-end with real MuJoCo + scripted controller
    ├── test_chess_piece_move.py    — Physical piece moves on real board
    ├── test_chess_turn_flow.py     — Full game turn with real arm execution
    └── test_all_square_moves.py    — Board reachability across all src→dst pairs
```

---

## Test Infrastructure

### `tests/conftest.py`

Shared pytest configuration. The conftest currently contains only the shared pytest module docstring; fixtures are defined locally in each test module for clarity.

### `tests/test_config_schema.py`

```python
def test_config_schema_is_valid() -> None:
    validate_config()
```

Calls the `validate_config()` function from `src/utils/config_validation.py`. This ensures all YAML config files have required keys and satisfy internal geometry constraints (Z-level ordering, board dimensions, etc.). This test runs without MuJoCo and completes in under 100ms.

---

## `tests/chess_game/` — Chess Logic Tests

No MuJoCo dependency. These tests run fast and are appropriate for CI.

### `test_chess_service.py`

Tests `ChessService` in isolation.

| Test | What it verifies |
|---|---|
| `test_initial_board_fen_is_standard` | New game starts at the standard FEN |
| `test_legal_move_e2e4_accepted_and_turn_alternates` | Legal move is accepted; turn flips to Black; SAN history is updated |
| `test_illegal_move_rejected` | `validate_uci("e2e5")` raises `IllegalMoveError` |
| `test_square_move_validation` | `validate_square_move("e2", "e4")` returns the correct `chess.Move` |
| `test_check_and_checkmate_detection_fools_mate` | Fool's mate sequence produces `is_checkmate=True`, `outcome="0-1"` |
| `test_stalemate_detection` | Pre-loaded stalemate FEN is detected via `is_stalemate` |
| `test_castling_legal_move_appears_when_path_clear` | `"e1g1"` appears in legal moves when path is clear |
| `test_en_passant_legal_move_appears` | En passant move appears when en passant square is set in FEN |
| `test_promotion_requires_promotion_piece` | Promotion without piece raises error; with `"q"` it succeeds |
| `test_choose_engine_move_returns_legal_move` | Stockfish returns a move in the legal move set |
| `test_save_and_load_roundtrip` | JSON save/load preserves FEN and move history |

### `test_game_orchestrator.py`

Tests `GameOrchestrator` using `FakePhysicalExecutor` (no arm, configurable success).

**Fake infrastructure:**
```python
class FakePhysicalExecutor:
    def execute(plan): returns FakePhysicalResult(success)
    def return_to_home(): returns FakePhysicalResult(True)
```

| Test | What it verifies |
|---|---|
| `test_illegal_move_rejects_without_calling_physical_executor` | Illegal move returns `accepted=False` without touching the executor |
| `test_legal_move_calls_executor_and_commits_after_success` | Legal move triggers `execute()`, `return_to_home()`, `chess_service.push()`, and `piece_tracker` update |
| `test_physical_failure_leaves_fen_unchanged` | When executor returns `success=False`, board FEN stays at pre-move state |

Additional tests (full file) cover:
- `is_busy` flag clears after any outcome
- `new_game()` resets all state
- `auto_computer_reply` triggers engine after human move
- Computer move commits correctly
- `submit_human_move` on wrong turn is rejected

### `test_move_planner.py`

Tests `MovePlanner` command generation for all move types.

| Test | What it verifies |
|---|---|
| `test_normal_move_emits_one_arm_command` | e2e4 → `[ArmMoveCommand("white_pawn_e", "e2", "e4")]` |
| `test_capture_emits_remove_then_arm` | Capture → `[RemoveFromBoardCommand(...), ArmMoveCommand(...)]` |
| `test_castling_emits_king_then_rook_arm_moves` | Kingside castle → `[ArmMoveCommand(king), ArmMoveCommand(rook)]` |
| `test_en_passant_removes_passed_over_pawn` | En passant → removes pawn at the captured square (one rank behind the to-square) |
| `test_promotion_emits_arm_and_two_teleports` | Promotion → `[ArmMoveCommand(pawn), TeleportCommand(pawn→reserve), TeleportCommand(reserve_piece→board)]` |

Also tests `LogicalPieceTracker`: `piece_id_at`, `apply_committed_move`, `reserve_piece_for`, `next_graveyard_slot`, and that the internal bijection is maintained after each update.

### `test_board_mapper.py`

Tests `BoardMapper` coordinate geometry.

| Test | What it verifies |
|---|---|
| `test_square_centers_are_64_unique_points` | All 64 squares map to distinct XY coordinates |
| `test_all_square_centers_inside_table` | All coordinates are within table bounds |
| `test_cell_spacing_is_8cm` | Adjacent squares are exactly 80mm apart in both axes |
| `test_board_does_not_use_env_edge_margin` | Board geometry uses `chess.yaml`, not `env.yaml:edge_margin` |
| `test_a1_h1_a8_h8_positions_match_orientation` | Corner squares match expected world coordinates exactly |
| `test_nearest_square_roundtrip` | `nearest_square(square_to_xy(sq)) == sq` for all squares |

---

## `tests/chess_env/` — MuJoCo Environment Tests

These tests require a working MuJoCo installation and `ChessFetchTask-v0` registration.

### `test_characterization.py`

Behaviour-locking tests that protect against regressions from refactoring.

| Test | What it verifies |
|---|---|
| `test_pick_and_place_xml_loads_in_mujoco` | `gym.make("ChessFetchTask-v0")` constructs successfully without XML errors |
| `test_model_load_restores_observation_space_after_success_and_failure` | After `load_model()` (success or exception), both wrapped and unwrapped observation spaces and `_use_transfer_obs` flag are restored to their pre-load state |

The second test uses `monkeypatch` to inject a fake `SAC.load` that either succeeds or raises, verifying that `transfer_obs_enabled` context manager correctly restores state in both paths.

### `test_static_assets.py`

Tests checked-in scene assets.

| Test | What it verifies |
|---|---|
| `test_static_scene_contains_canonical_chess_sections` | XML contains exactly 64 board geoms, 4 zone geoms, and 96 piece bodies (32 active + 64 reserve) |
| `test_static_chess_stls_are_detailed_and_below_hover_clearance` | Each STL mesh has >=1000 triangles and all Z vertices are <= 60.5mm (below `HOVER_Z` clearance) |

The STL test reads binary STL files directly and extracts Z values from each triangle's vertices to verify no mesh exceeds the hover clearance height.

### `test_task_chaining.py`

Tests stage-to-stage transition infrastructure.

| Test | What it verifies |
|---|---|
| `test_transition_validate` | `transition_validate()` returns expected diagnostic dict with `grip_speed_mm_s`, `is_velocity_ok`, and correct error calculation |
| `test_soft_reset_flow` | `soft_reset(new_scenario, new_goal_pos, ...)` updates `current_scenario`, `goal`, resets `episode_steps`, returns valid obs dict and info with `halt_steps`/`align_steps` |
| `test_soft_reset_finger_validation` | `soft_reset` checks finger preconditions and returns appropriate finger state in info |

### `test_waypoints.py`

Tests the `src/chess_env/waypoints` module.

| Test | What it verifies |
|---|---|
| `test_validate_chain_valid` | `validate_chain(["transit", "descend", "ascend"])` does not raise |
| `test_validate_chain_invalid` | `validate_chain(["transit", "ascend"])` raises `ValueError` (invalid transition) |
| `test_derive_goal_pos` | `derive_goal_pos("descend", cell)` returns `[cell_x, cell_y, HOVER_Z]` |
| `test_derive_goal_pos_all_scenarios` | All three scenarios map to their expected Z levels |
| `test_validate_chain_invalid_transitions` | Various invalid transitions are all rejected |

---

## `tests/physical/` — Physical Layer Tests

### `test_piece_registry.py`

Tests `PieceRegistry` and related utilities (no MuJoCo required for most).

| Test | What it verifies |
|---|---|
| `test_active_registry_has_32_unique_pieces` | Exactly 32 active pieces, all with unique IDs and unique starting squares |
| `test_starting_square_map_contains_expected_pieces` | Key pieces (white_king→e1, white_queen→d1, etc.) are at correct starting squares |
| `test_ids_for_color` | Each colour has exactly 16 active pieces |
| `test_reserve_piece_ids_cover_64_reserves` | 64 reserve pieces with all unique IDs |
| `test_physical_occupancy_reset_restores_starting_map` | `PhysicalOccupancy.reset()` brings state back to starting positions |

### `test_piece_teleport.py`

Tests `PieceTeleporter` and `PhysicalPlanExecutor` with a real MuJoCo environment.

| Test | What it verifies |
|---|---|
| `test_set_active_piece_routes_cube_position_to_selected_piece` | After `set_active_piece("white_pawn_e")`, `get_cube_position()` returns the pawn's freejoint position |
| `test_teleport_piece_to_square_sets_pose_and_zeroes_velocity` | Teleport to e4 sets qpos to expected XYZ and zeroes all 6 DOF velocity components |
| `test_teleport_piece_to_graveyard_slot` | `teleport_piece_to_graveyard("white_pawn_e", "slot_03")` places piece at correct grid coordinate |

These tests directly read `data.qpos` and `data.qvel` from MuJoCo to verify the low-level freejoint manipulation.

### `test_piece_xml.py`

Tests that the static scene XML contains all expected MuJoCo objects.

| Test | What it verifies |
|---|---|
| `test_all_active_and_reserve_piece_bodies_exist` | Every active piece has body, joint, cube geom, and visual geom registered in MuJoCo model |
| (additional) | All 64 reserve piece bodies exist; all geom/site names follow naming convention |

Uses `mujoco.mj_name2id()` to verify that each named element is present in the compiled model.

### `test_zone_alignment.py`

Tests graveyard and promotion reserve geometry (no MuJoCo required).

| Test | What it verifies |
|---|---|
| `test_all_slots_inside_zone_extent` (parametrized × 4) | Every slot in both graveyards and both promotion reserves falls within the zone's bounding box with 5mm margin |
| `test_zones_do_not_overlap_board_y_range` | Graveyard and reserve Y coordinates do not overlap the board's Y extent |

Parametrized over `(white graveyards, black graveyards, white reserve, black reserve)`. Validates slot index arithmetic: `row = slot // cols`, `col = slot % cols`.

---

## `tests/ui/` — UI Tests

### `test_app.py`

Flask REST API integration tests using `app.test_client()`. Uses a real `GameOrchestrator` with `FakePhysicalExecutor` (no arm) and a real Stockfish instance.

```python
@pytest.fixture
def client():
    orchestrator = GameOrchestrator(ChessService(engine_cfg=...), FakePhysicalExecutor(), LogicalPieceTracker(), ...)
    app = create_app(orchestrator)
    app.config.update(TESTING=True)
    yield app.test_client()
```

| Test | What it verifies |
|---|---|
| `test_snapshot_returns_initial_board` | `GET /api/snapshot` returns 200, board `e2=="P"`, turn `"white"`, 20 legal moves |
| `test_index_renders_interactive_controls` | `GET /` returns HTML containing expected element IDs (`board`, `new-game`, `computer`, etc.) |
| `test_new_game_endpoint_resets_snapshot` | After a move, `POST /api/new-game` resets board to starting position |
| (additional) | `POST /api/move` with valid src/dst returns 200; invalid move returns 400; missing fields returns 400 with error |
| (additional) | `POST /api/let-computer-play` returns 200 and a valid computer move |
| `test_busy_state_is_represented` | Busy state appears in `GET /api/snapshot` |

### `test_run_chess_ui_backend.py`

Tests `QueuedUIBackend` threading behaviour.

| Test | What it verifies |
|---|---|
| `test_new_game_request_resets_env_and_physical_occupancy` | When `backend.new_game()` is called from a thread and `process_one(env=env)` is called from the main thread: `env.reset()` is called once, the physical executor reset hook is called once, result is returned to the calling thread, and `backend.snapshot()` reflects the new state |

Uses `threading.Thread` to simulate the Flask worker/main-thread split. Verifies that the queue bridge correctly serialises the call and delivers the response.

---

## `tests/integration/` — End-to-End Tests

These tests require a full MuJoCo environment with chess pieces. They are slower (each test executes a real arm move).

### `test_chess_piece_move.py`

Physical piece moves on a real board with `ModelEmbeddedController`.

```python
@pytest.mark.parametrize(
    ("piece_id", "src", "dst", "dst_square"),
    [
        ("white_pawn_e", "e2", "e4", chess.E4),
        ("white_knight_g", "g1", "f3", chess.F3),
        ("white_queen", "d1", "h5", chess.H5),
    ],
)
def test_move_piece_in_crowded_starting_position(...):
```

For each parametrized case:
1. Creates a full `MovementExecutor` with `ModelEmbeddedController` and real `BoardMapper`
2. Calls `move_piece_between_squares(piece_id, src, dst)`
3. Asserts `result.success`
4. Reads the piece's freejoint position from `data.qpos` and verifies it is within 2mm of the target square's XYZ
5. Verifies all other pieces are within 2mm of their starting positions (no collision side-effects)

### `test_chess_turn_flow.py`

Full game turn with real arm execution — the most comprehensive integration test.

Builds a complete stack: `PieceRegistry → PhysicalOccupancy → BoardMapper → ModelEmbeddedController → MovementExecutor → PhysicalPlanExecutor → GameOrchestrator`.

| Test | What it verifies |
|---|---|
| `test_real_physical_single_turn_commits_after_success` | A single human move (e2e4) executes through the full stack: plan generation → physical execution → post-snap teleport → return-to-home → `chess_service.push()` → `piece_tracker.apply_committed_move()`. Final board FEN reflects the move. Arm returns to home posture within tolerance. |

Also verifies:
- `piece_tracker.piece_id_at("e4") == "white_pawn_e"` (logical state correct)
- `chess_service.board.piece_at(chess.E4)` is a Pawn (board state committed)
- `result.accepted` and `result.physical_success` are both True

### `test_all_square_moves.py`

Physical reachability coverage: tests arm moves across a representative subset of src→dst board square pairs with `ModelEmbeddedController`. Verifies the arm can physically reach the sampled board positions without timing out or crashing.

---

## Testing Patterns

### Fake Physical Executor

Many chess logic tests inject a `FakePhysicalExecutor` to isolate chess rules from arm behaviour:

```python
class FakePhysicalExecutor:
    def __init__(self, success=True):
        self.success = success
        self.plans = []
        self.home_calls = 0

    def execute(self, plan):
        self.plans.append(plan)
        return FakePhysicalResult(self.success, None if self.success else "ROBOT_FAILED")

    def return_to_home(self):
        self.home_calls += 1
        return FakePhysicalResult(True)
```

This pattern lets tests verify that:
- Plans are generated correctly (`executor.plans[0].chess_move_uci == "e2e4"`)
- Home is called once per move (`executor.home_calls == 1`)
- Physical failure doesn't commit the board state

### `GameOrchestrator.create_headless()`

For pure chess logic tests that don't even need a fake executor:

```python
orchestrator = GameOrchestrator.create_headless()
result = orchestrator.submit_human_move("e2", "e4")
```

Uses `NoOpPhysicalExecutor` (always returns success) and no Stockfish engine.

### FEN-based Scenario Testing

Many `ChessService` and `MovePlanner` tests pre-load a specific FEN position to test edge cases without playing through a whole game:

```python
service = ChessService("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")  # Stalemate
board = chess.Board("4k3/8/8/3pPp2/8/8/8/4K3 w - f6 0 1")  # En passant available
```

### `LogicalPieceTracker.empty()`

For `MovePlanner` tests that only need a few pieces:

```python
tracker = LogicalPieceTracker.empty()
tracker.set_piece_at("e4", "white_pawn_e")
tracker.set_piece_at("d5", "black_pawn_d")
```

`LogicalPieceTracker.empty()` creates a tracker with no pieces (unlike the normal constructor which pre-populates all 32 starting positions).

---

## Running the Tests

```bash
# Run all tests
pytest

# Skip slow integration tests
pytest -m "not integration"

# Run only chess logic tests (no MuJoCo)
pytest tests/chess_game/

# Run with verbose output
pytest -v tests/chess_game/test_chess_service.py

# Run a specific test
pytest tests/chess_game/test_move_planner.py::test_promotion_emits_arm_and_two_teleports

# Run config schema validation
pytest tests/test_config_schema.py
```

Tests that require MuJoCo will fail if the environment is not installed. Tests that require Stockfish will fail if `stockfish` is not on the `PATH`.
