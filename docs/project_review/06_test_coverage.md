# Test Coverage Review

## Overview

The project has **81 tests** across 9 test files, all passing as of this review. Coverage is strongest for the physics controller and chess logic layers; the UI, physical executor, and special chess rules have meaningful gaps.

---

## Test Inventory

### `tests/unit/test_scripted_controller.py` (inferred from eval output)

Tests the `ScriptedController` in isolation via mock environments. Covers:
- Transit phase convergence
- Descend/ascend Z-level transitions
- Grasp finger ramp (open→close)
- Step-budget enforcement

**Coverage: Good.** The unit tests mirror what the eval scripts confirm empirically.

---

### `tests/unit/test_waypoints.py`

Covers `validate_chain()`, `derive_goal_pos()`, and `exit_waypoint()`. All valid and invalid phase transitions are enumerated.

**Coverage: Good.** Every transition in the `VALID_TRANSITIONS` dict has a corresponding test case.

---

### `tests/unit/test_board_mapper.py`

Tests `square_to_xy()` for corner squares (a1, h1, a8, h8) and center (d4, e5). Tests `nearest_square()` round-trips.

**Gap:** The mapper is only tested at 6 of 64 squares. Floating-point edge cases at cell boundaries (exactly on the midpoint between two squares) are untested. `nearest_square()` has no test for coordinates outside the board boundary (should it clamp, raise, or return None?).

---

### `tests/unit/test_chess_service.py`

Tests `is_legal()`, `apply_move()`, `choose_engine_move()`, initial board state, and FEN round-trip.

**Gap:** `choose_engine_move()` priority ordering (checkmate > capture > promotion > check > lexicographic) is not tested with a constructed position that exercises each branch. The promotion branch in the engine heuristic is particularly untested — and promotion moves in the game are broken end-to-end (see `05_ui.md` Issue 1).

---

### `tests/unit/test_move_planner.py`

Tests `MovePlanner.plan()` for:
- Quiet move (no capture)
- Capture (generates RemoveFromBoardCommand)
- En passant
- Castling (both sides)

**Gap:** Promotion is not tested. `PromoteCommand` is dead code but the planner also never generates a `TeleportCommand` with a promoted piece type — this gap means promotion would silently produce a queen (or crash) at the physical layer without a test catching it.

---

### `tests/unit/test_logical_piece_tracker.py`

Tests `place()`, `remove()`, `piece_id_at()`, `all_pieces()`.

**Gap:** No test for:
- Castling state update (two pieces move atomically)
- Tracker state after `reset()` — does it return to the 32-piece starting layout?
- `piece_id_at()` called on an empty square returns `None` (boundary condition)

---

### `tests/unit/test_game_orchestrator.py`

Tests:
- Legal move accepted
- Illegal move rejected
- Physical failure returns `physical_success=False`
- `is_busy` lifecycle (set before, cleared after)

**Gap:** No test for the race condition where `new_game()` is called while `is_busy=True`. No test for the chess board state after physical failure (it should remain unchanged — this is the "only commit on success" invariant).

---

### `tests/unit/test_plan_executor.py`

Tests `PlanExecutor.execute()` for a normal plan and a plan where the physical move fails.

**Gap:**
- No test for `return_to_home()` — this is a separate code path that calls `run_chess_ui.py:arm_home_xy` (potential config divergence, see Bug #7 in `02_bugs.md`).
- No test for the `except Exception` blanket catch — confirming that a `RuntimeError` from `soft_reset` is correctly caught and surfaced as `physical_success=False`.

---

### `tests/ui/test_app.py`

7 tests covering:
- `GET /api/snapshot` initial state
- `GET /` HTML structure
- `POST /api/new-game` resets board
- Illegal move → 400
- Legal move → accepted snapshot
- Computer play → valid move UCI
- `is_busy=True` reflected in snapshot

**Gaps:**
- No test for `POST /api/promote` (just verifies it exists and returns 501 — even that is absent).
- No test for `POST /api/undo` (same).
- No test for `GET /api/snapshot` when `is_busy=True` AND a move is in progress — the snapshot should not reflect a half-committed board.
- No test that `POST /api/new-game` while `is_busy=True` is handled safely (currently it isn't — see `05_ui.md` Issue 2).

---

## Evaluation Script Coverage

The `scripts/eval_*` scripts supplement the unit tests with integration-level checks:

| Script | What it covers | Gap |
|--------|---------------|-----|
| `eval_stages.py` | Individual arm phases in isolation | Does not test phase transitions end-to-end |
| `eval_sequence.py` | Full pick-and-place sequence | Uses a fixed square pair; no randomization |
| `eval_chess_reachability.py` | 6 squares reachable by arm | 58/64 squares untested |
| `eval_stress.py` | Grid of 9 positions at arm-table level | **Grid extends outside chess board** — false failures |
| `eval_special_moves.py` | En passant, castling, (promotion stub) | Promotion test is explicitly skipped |
| `eval_draw_conditions.py` | Stalemate, 50-move, threefold | Correct; uses `python-chess` positions |
| `eval_chess_game_flow.py` | Full game with mock executor | Mock bypasses all physics — game-logic only |
| `debug_one_move.py` | Single move physics + JSONL log | **Hardcoded e2→e3** — no CLI arguments work |

---

## Missing Test Areas

### 1. Promotion (end-to-end)

No test exercises the complete promotion path: pawn reaches back rank → `chess_service` flags promotion required → `move_planner` generates plan with correct promoted piece → `plan_executor` places the correct piece type. This entire chain is untested and partially broken.

### 2. Concurrent / threading

`QueuedUIBackend` serializes requests, but no test verifies that two concurrent `/api/move` POSTs result in sequential execution rather than a race. A simple `threading.Thread` test sending two moves simultaneously would cover this.

### 3. Environment reset consistency

No test verifies that `soft_reset()` after a completed move leaves the MuJoCo state in a configuration valid for the next move (piece positions consistent with tracker state, arm at home position).

### 4. Config loading edge cases

No test for missing or malformed YAML configs. The app will crash at import time if `env.yaml` is absent — a `pytest.raises` test with a tmp config would catch regressions here.

### 5. STL generation / scene XML injection

`environment_generation.py` is entirely untested. `_replace_marked_fragment()` raises `ValueError` on missing markers with no test catching it.

### 6. Board mapper boundary conditions

`nearest_square()` behavior for out-of-board coordinates is undefined and untested.

### 7. `eval_stress.py` grid correctness

The grid test reports 66.7% success for arm-reachability but uses positions outside the chess board. This should either be fixed to use valid chess square centers (all 64 of them) or a separate "arm workspace" test that clearly documents it is testing arm capability, not chess-square reachability.

---

## Recommended Additions (Priority Order)

1. **`test_promotion_e2e.py`** — construct a board one move from promotion, execute the full chain through `GameOrchestrator`, assert the promoted piece type is correct in tracker and board state.

2. **`test_app_promotion_stub.py`** — extend `test_app.py` to assert `/api/promote` returns 501 and document the expected 200 response structure for when it is implemented.

3. **`test_new_game_busy.py`** — assert that `POST /api/new-game` while `is_busy=True` returns 409 (after the guard is added).

4. **`test_board_mapper_boundaries.py`** — test all 4 corners, midpoints between adjacent squares, and one out-of-board coordinate.

5. **`test_plan_executor_return_home.py`** — verify `return_to_home()` sends the arm to the correct home coordinate from `chess.yaml`.

6. **`test_scene_generation.py`** — call `regenerate_scene()` against a temp XML file with valid markers; assert output contains expected mesh paths. Assert `ValueError` on missing markers.

7. **Fix `eval_stress.py`** — replace the `--grid` bounds with the 64 chess-square centers and report per-square results.

8. **Fix `debug_one_move.py`** — add `argparse` for `--src`, `--dst`, `--piece-id` so the script is actually configurable.

---

## Test Quality Notes

- All 81 tests are deterministic and fast (no real MuJoCo physics in unit tests, mock executors used in UI tests).
- Fixtures are clean and minimal.
- `FakePhysicalExecutor` in `test_app.py` is a good pattern — reuse it in `test_game_orchestrator.py` rather than duplicating stub logic.
- No use of `unittest.mock.patch` on internal methods — tests go through public interfaces. This is good practice and makes refactoring safe.
- No flaky tests observed across multiple runs.

---

## Coverage Estimate

| Layer | Estimated line coverage | Confidence |
|-------|------------------------|------------|
| `chess_game/` (service, planner, orchestrator) | ~85% | High |
| `chess_env/` (controller, waypoints) | ~75% | High |
| `ui/app.py` | ~70% | High |
| `physical/` (executor, teleport, occupancy) | ~55% | Medium |
| `chess_env/environment_generation.py` | ~0% | High |
| `chess_env/simulation.py` | ~30% | Medium |
