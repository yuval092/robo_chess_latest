# Testing

This document covers the automated test suite: its structure, what each test file covers, how to run it, and known gaps.

---

## 1. Test Structure

```
tests/
├── conftest.py                        # Shared fixtures
├── chess_game/
│   ├── test_board_mapper.py           # Square ↔ XYZ coordinate conversion
│   ├── test_chess_service.py          # Chess rules and engine
│   ├── test_game_orchestrator.py      # Orchestrator busy/error/flow logic
│   └── test_move_planner.py           # Physical plan generation for all move types
├── physical/
│   ├── test_piece_registry.py         # Piece ID naming, starting squares
│   ├── test_piece_teleport.py         # Slot index parsing, xyz computation
│   ├── test_piece_xml.py              # XML fragment generation for pieces
│   └── test_zone_alignment.py        # Graveyard/reserve zone alignment
├── ui/
│   ├── test_app.py                    # Flask route responses
│   └── test_run_chess_ui_backend.py   # QueuedUIBackend plumbing
├── integration/
│   ├── test_chess_piece_move.py       # Full arm move in live MuJoCo environment
│   └── test_chess_turn_flow.py        # Multi-turn game flow with live arm
└── chess_env/
    ├── test_environment_generation.py # Scene generation, XML validity
    ├── test_task_chaining.py          # Multi-step task scenarios
    └── test_waypoints.py              # Waypoint algorithm correctness
```

Run all tests:
```bash
python -m pytest tests/ -v
```

Run only fast (non-MuJoCo) tests:
```bash
python -m pytest tests/ -v --ignore=tests/integration --ignore=tests/chess_env
```

---

## 2. Unit Tests

### `tests/chess_game/test_chess_service.py`

Tests the `ChessService` layer over python-chess:

- `test_initial_board_fen_is_standard` — default board has standard starting FEN.
- `test_legal_move_e2e4_accepted_and_turn_alternates` — push e2e4; side-to-move becomes Black; SAN history updated.
- `test_illegal_move_rejected` — e2e5 raises `IllegalMoveError`.
- `test_square_move_validation` — `validate_square_move("e2", "e4")` returns the correct `chess.Move`.
- `test_check_and_checkmate_detection_fools_mate` — Fool's Mate produces checkmate with `outcome == "0-1"`.
- `test_stalemate_detection` — custom FEN stalemate position is detected.
- `test_castling_legal_move_appears_when_path_clear` — castling appears in legal moves when path is clear.
- `test_en_passant_legal_move` — en passant appears when en-passant square is set.
- `test_promotion_move_accepted` — pushing a pawn to the back rank with promotion is accepted.
- `test_draw_by_insufficient_material` — K vs K position is drawn.
- `test_engine_prefers_checkmate` — `choose_engine_move` picks checkmate over a capture.
- `test_save_and_load_preserves_fen_and_history` — round-trip through `save_to_file`/`load_from_file`.

### `tests/chess_game/test_move_planner.py`

Tests `MovePlanner` plan generation for all move types:

- `test_normal_move_emits_one_arm_command` — quiet move → single `ArmMoveCommand`.
- `test_capture_emits_remove_then_arm` — capture → `RemoveFromBoardCommand` + `ArmMoveCommand`.
- `test_castling_emits_king_then_rook_arm_moves` — kingside castle → king ArmMove + rook ArmMove.
- `test_en_passant_removes_passed_over_pawn` — en passant → `RemoveFromBoardCommand` at pawn's square (not destination) + `ArmMoveCommand`.
- `test_promotion_emits_arm_and_two_teleports` — promotion → `ArmMoveCommand` + `TeleportCommand(pawn, reserve)` + `TeleportCommand(queen, square)`.
- `test_no_physical_piece_at_source_raises` — missing tracker entry raises `ValueError`.
- `test_illegal_move_raises` — `MovePlanner.plan()` raises for illegal moves.

### `tests/chess_game/test_game_orchestrator.py`

Tests `GameOrchestrator` flow:

- `test_busy_guard_rejects_move` — submitting a move while `is_busy=True` returns rejected result.
- `test_busy_guard_rejects_new_game` — calling `new_game()` while busy returns error snapshot.
- `test_human_move_increments_history` — successful move appears in SAN history.
- `test_auto_computer_reply_fires` — with `auto_computer_reply=True`, computer move is triggered automatically.
- `test_error_cleared_on_success` — after a failed move, a subsequent successful move clears the error.

### `tests/chess_game/test_board_mapper.py`

Tests `BoardMapper` coordinate conversion:

- `test_e4_maps_to_expected_xy` — e4 → correct world XY.
- `test_a1_is_at_low_corner` — a1 is at the lowest X, lowest Y corner.
- `test_h8_is_at_high_corner` — h8 is at highest X, highest Y.
- `test_all_squares_have_valid_z` — all 64 squares produce Z at cube centre height.
- `test_board_width_equals_config` — span from a1 to h1 equals 7 × cell_size.

### `tests/physical/test_move_planner.py` (LogicalPieceTracker)

- `test_piece_id_at_initial_squares` — starting squares map correctly.
- `test_inverse_map_consistency_after_set_piece_at` — inverse map stays in sync.
- `test_captured_piece_goes_to_graveyard_list` — `_remove_piece` adds to `_captured`.
- `test_reserve_piece_for_returns_free_piece` — `reserve_piece_for` skips already-placed pieces.
- `test_next_graveyard_slot_increments` — each capture returns the next sequential slot.
- `test_next_promotion_reserve_slot_counts_promoted_pawns` — only promoted (not captured) pawns count.

### `tests/physical/test_piece_registry.py`

- `test_32_active_pieces` — exactly 32 pieces.
- `test_all_piece_ids_unique` — no duplicate IDs.
- `test_starting_square_map_covers_all_32_pieces` — all pieces have starting squares.
- `test_king_queen_ids_have_no_file` — `white_king`, `white_queen` (no file suffix).
- `test_pawn_ids_include_file` — `white_pawn_a` through `white_pawn_h`.

### `tests/physical/test_piece_teleport.py`

- `test_slot_index_valid` — `_slot_index("slot_07")` → 7.
- `test_slot_index_rejects_bad_format` — `"slot_"`, `"slot_abc"`, `"7"` all raise `ValueError`.
- `test_slot_xyz_first_slot` — slot 0 position matches `origin_xyz`.
- `test_slot_xyz_row_and_col` — slot 5 (row=1, col=1 with cols=4) offset is correct.
- `test_slot_out_of_bounds_raises` — slot beyond `rows*cols` raises `ValueError`.

### `tests/ui/test_app.py`

Flask route tests using a mock backend:

- `test_snapshot_returns_200` — `GET /api/snapshot` returns 200 with snapshot JSON.
- `test_new_game_returns_200` — `POST /api/new-game` calls `backend.new_game()`.
- `test_move_accepted_returns_200` — valid move payload returns 200.
- `test_move_rejected_returns_400` — backend rejection returns 400.
- `test_move_missing_fields_returns_400` — missing src/dst returns 400 with error.
- `test_promote_returns_501` — `POST /api/promote` returns 501.
- `test_undo_returns_501` — `POST /api/undo` returns 501.

---

## 3. Integration Tests

Integration tests require the full MuJoCo environment. They are slower (~5–30 seconds each) and are skipped in fast mode.

### `tests/integration/test_chess_piece_move.py`

Builds a live `ChessFetchTask-v0` environment and runs real arm moves:

- `test_move_piece_in_crowded_starting_position[white_pawn_e-e2-e4]` — arm moves pawn e2→e4 with all other pieces present. Verifies final position is within 2mm of expected and no other piece moved.
- `test_move_piece_in_crowded_starting_position[white_knight_g-g1-f3]` — knight g1→f3 in crowded opening position.
- `test_move_rejects_empty_source_and_occupied_destination_before_motion` — occupancy checks fail before any arm motion.

### `tests/integration/test_chess_turn_flow.py`

Multi-turn game flow test:

- Plays several moves through the `GameOrchestrator` against the full physical executor.
- Verifies board state, physical piece positions, and tracker state after each move.
- Tests that auto-computer-reply triggers correctly.

---

## 4. Chess Environment Tests

### `tests/chess_env/test_environment_generation.py`

- Tests that `ChessSimulationEnv` can be created and reset without errors.
- Verifies XML is valid after scene generation.
- Checks that all 32 piece bodies are present in the model.

### `tests/chess_env/test_task_chaining.py`

- Tests that multiple sequential `env.reset()` calls work correctly.
- Verifies the threading lock prevents race conditions in XML path substitution.

### `tests/chess_env/test_waypoints.py`

- Tests waypoint interpolation: TRANSIT, DESCEND, ASCEND.
- Verifies tube constraint is respected during descend/ascend.
- Checks SAFE_Z and HOVER_Z constants match `configs/env.yaml`.

---

## 5. Running Tests

### Full suite
```bash
python -m pytest tests/ -v
```

### Fast (no MuJoCo)
```bash
python -m pytest tests/ -v --ignore=tests/integration --ignore=tests/chess_env
```

### Single file
```bash
python -m pytest tests/chess_game/test_move_planner.py -v
```

### With coverage
```bash
python -m pytest tests/ --cov=src --cov-report=html
```

### Expected output (all tests passing)
```
=================== 81 passed in X.Xs ====================
```

---

## 6. Test Gaps

The following areas have limited or no automated test coverage:

| Area | Gap | Risk |
|---|---|---|
| JavaScript frontend | No browser-level tests; no test for promotion dialog flow | Medium |
| `PhysicalPlanExecutor.execute()` | No unit test for multi-command plans | Low (tested via integration) |
| `ScriptedController` controller stages | Only integration tests; no unit tests for individual stage transitions | Medium |
| Finger contact physics | `grasp_verify_finger_threshold` value is empirical; not regression-tested | High |
| Concurrent access to `QueuedUIBackend` | No load test; potential deadlock if queue never drains | Medium |
| Long games (50+ moves) | Graveyard slot overflow not tested | Low |
| Queenside castling | Only kingside tested in integration suite | Low |
| Promotion capture combinations | Unit test covers pure promotion; capture+promotion not integration-tested | Medium |
