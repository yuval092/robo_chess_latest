# Chess-Game Refactor Overview

Nickname: `chess-game refactor`

## Objective

Convert the current scripted Fetch pick-and-place project into a playable chess game simulation.

The tabletop itself becomes the chess board. Each chess piece is represented physically by the existing 30mm graspable cube shape, with a visual-only STL chess symbol mounted above the cube. The arm remains physically unchanged and continues to use the current waypoint movement algorithm for moving one graspable cube from one board cell to another.

The refactor must preserve the working parts of the current project:

- MuJoCo/Gymnasium Fetch environment registration.
- Mocap/weld-based arm control.
- Scripted `transit -> descend -> grasp -> ascend -> transit -> descend -> place -> ascend` movement.
- Evaluation scripts and physics verification workflow.
- Existing grasp parameters for the 30mm cube body.

The refactor adds:

- 8x8 chess board geometry on the existing table surface.
- 32 active chess pieces, graveyards, and promotion reserves.
- Chess state, legal move validation, turn management, and special move handling using `python-chess`.
- A translation layer from chess moves to physical movement commands.
- A human-facing UI for board display, move selection, status, and "let computer play for me".

## Current Baseline Findings

Read before implementation:

- `docs/current_status_new/` describes the current architecture, motion pipeline, physics, and evaluation tools.
- The code is in a scripted-only phase. RL artifacts remain but must not drive this refactor.
- The current checked-out working tree has user changes. Do not revert them as part of this plan.
- The current XML still contains a Fetch target marker:
  - `chess_env/assets/pick_and_place.xml` has `site name="target0"` with red sphere color. This is the red dot leftover and must be removed.
- The refreshed current-status docs establish a stronger chess-ready baseline:
  - Table center: `[0.88, 0.2641]`
  - Table half extents: `0.35 x 0.35`
  - Arm base: `robot0:base_link` at `[0.56, 0.2641, 0]`
  - Torso height: `0.3661`
  - `scripts/verify_physics.py` now reports all 64 status-doc chess grid centers reachable with max 2.9mm error.
- Important geometry caveat:
  - The refreshed reachability test computes its 64 grid centers from `edge_margin=0.04`, producing square centers around `x=0.609..1.151`.
  - The chess-game requirement says each chess cell is exactly `0.08m x 0.08m`; a true 64cm board centered at `x=0.88` produces centers `x=0.600..1.160`.
  - The plan therefore requires an exact-8cm reachability verification before board implementation. Do not assume the current `edge_margin` training grid proves the final chess board geometry.

For chess-game refactor stages, `configs/env.yaml` owns robot/table physics and `configs/chess.yaml` owns exact chess board geometry. `env.yaml` `edge_margin` is a movement sampling margin and must not define chess cell size.

## Architectural Principles

The refactor must keep three layers separate:

1. Chess layer
   - Owns chess rules and board state.
   - Uses `python-chess`.
   - Does not import MuJoCo, Gymnasium, or physical coordinate helpers.

2. Physical layer
   - Owns robot motion and MuJoCo state operations.
   - Knows about graspable piece bodies, coordinates, safe heights, teleportation primitives, and waypoint movement.
   - Does not know chess legality, check, castling rights, or FEN semantics.

3. Translation/integration layer
   - Converts legal chess moves into physical commands.
   - Converts board squares to world coordinates.
   - Expands complex chess moves into ordered physical and teleport operations.
   - Is the only layer allowed to depend on both chess and physical APIs.

This split is required for SOLID compliance:

- Single Responsibility: chess rules, physical movement, and orchestration are separate.
- Open/Closed: new move planners or UI clients can be added without rewriting chess rules or arm control.
- Liskov/Substitution: physical executors can be replaced by mock executors in tests.
- Interface Segregation: UI and tests depend on small service interfaces, not raw MuJoCo internals.
- Dependency Inversion: orchestration depends on abstract physical operations and chess service contracts.

## Target Directory Shape

Add these modules during the staged implementation:

```text
src/chess_game/
  __init__.py
  constants.py
  models.py
  board_mapper.py
  chess_service.py
  move_planner.py
  game_orchestrator.py

src/physical/
  __init__.py
  piece_registry.py
  piece_teleport.py
  movement_executor.py
  occupancy.py

src/ui/
  __init__.py
  app.py
  static/
  templates/

configs/
  chess.yaml           # board, pieces, reserves, game settings

scripts/
  generate_board_xml.py   # generates 64 square visual geoms
  generate_pieces_xml.py  # generates 32 piece body XML fragments

tests/chess_game/
tests/physical/
tests/integration/
```

This shape can be adjusted if implementation reveals a better local fit, but the layer boundaries must remain.

## Plan Files

- `01_stage0_baseline_reconciliation.md`
- `02_stage1_board_geometry.md`
- `03_stage2_piece_modeling.md`
- `04_stage3_physical_piece_api.md`
- `05_stage4_chess_logic.md`
- `06_stage5_move_planning_special_moves.md`
- `07_stage6_game_orchestration.md`
- `08_stage7_ui.md`
- `09_stage8_evaluation_and_validation.md`
- `10_stage9_cleanup_and_docs.md`

Each stage is self-contained and must be validated before the next stage begins.

## Deliberate Feature Omissions

The following features are explicitly out of scope for this refactor. Document them here so they are not accidentally re-added as requirements mid-implementation.

| Feature | Decision | Notes |
|---------|----------|-------|
| Move undo / take-back | Out of scope | `python-chess` supports `board.pop()`; wire it up only if user requests |
| Game clock / time controls | Out of scope | No timer on moves |
| Network / two-player over network | Out of scope | Local human vs. computer only |
| Stockfish or external engine | Out of scope | Internal heuristic selector in `ChessService.choose_engine_move`; document the integration point |
| Animated STL piece identity display during transit | Out of scope | Pieces are cubes for physics; STL is static visual only |
| Drag-and-drop on board (UI) | Out of scope | Click source, click destination |

## Cross-Cutting Risks

These risks span multiple stages. Each stage plan must address the specific part that falls within its scope.

### Arm Home Position Between Moves

The plan does not specify where the arm parks between moves. After completing a full pick-and-place the arm ends at SAFE_Z above the destination square. The arm should return to `HOME_POSITION_XY` (from `env.yaml`) before the next move is computed and the computer turn starts. This avoids long transit paths that start from unpredictable positions and prevents the arm from hovering over the board during the UI promotion dialog.

**Resolution**: `GameOrchestrator` must call a `return_to_home()` physical command at the end of every move execution before committing the turn and triggering the computer reply. Specify this explicitly in Stage 6.

### Piece Rotation After Grasp-Place

The current arm always uses `VERTICAL_QUAT` (straight-down gripper). After a pick-and-place, the piece cube is placed in a physically stable position but its yaw may differ from its starting orientation. Over successive moves, this rotation accumulates. The STL visual symbol will rotate accordingly. For cube-shaped collision bodies this is fine physically, but visually pieces will slowly rotate to random orientations.

**Resolution**: After `execute_place` succeeds, use `PieceTeleporter.set_piece_orientation(piece_id, identity_quat)` to snap the piece back to identity quaternion. This is safe because the gripper has already retracted. Specify this in Stage 3.

### XML Generator Scripts

The plan requires 64 board square visual geoms and 32 piece bodies. Writing these by hand in XML is error-prone and tedious. Both sets must be generated by Python scripts that read from `configs/chess.yaml` and `BoardMapper`. Do not write the XML manually.

**Resolution**: Add `scripts/generate_board_xml.py` in Stage 1 and `scripts/generate_pieces_xml.py` in Stage 2. Each script reads the relevant config and writes XML fragments to be included in `pick_and_place.xml`.

### Physics Drift Between Moves

The MuJoCo simulation continues to step even when the arm is between moves. Pieces on freejoints may drift slightly from vibrations in the simulation timestep. This is especially problematic if pieces are near the edge of the board visual geoms.

**Resolution**: Set freejoint damping high enough to prevent visible drift between moves. Use `damping="0.5"` minimum per degree of freedom for each piece freejoint. The existing `object0` uses `damping="0.1"`, which should be increased for 32 simultaneous pieces. Verify stability with a 5-second idle sim in `verify_physics.py`.

### MuJoCo Render Thread vs UI Server Thread

`render_mode="human"` opens a GLFW window that must be driven from the main thread. A Flask or FastAPI server running in the same process needs its own thread. These two cannot share the main thread.

**Resolution**: Run Flask in a background daemon thread. All MuJoCo `env.step()`, `env.render()`, and `mujoco_step()` calls remain in the main thread. Protect all env access with a single `threading.Lock`. The background Flask thread acquires this lock to read `GameSnapshot` (read-only, safe after a move completes) but never calls MuJoCo APIs directly. Specify this explicitly in Stage 6 and Stage 7.

## Non-Negotiable Constraints

- Keep the tuned Fetch arm setup unchanged: arm base `x=0.56`, table center `x=0.88`, torso height `0.3661`. Change it only if the exact-8cm chess-square reachability test fails and the failure cannot be solved by board placement within the 70cm table.
- Keep the waypoint algorithm as the core object-movement algorithm.
- Use 8cm x 8cm chess cells.
- Use the physical table surface as the board, not a separate board object placed on top.
- Chess piece collision must remain cube-based. STL chess symbols are visual only.
- Graveyards and promotion reserves are floor objects and are never arm-reachable. Moves to/from those locations use teleportation.
- Legal move validation must happen before physical movement begins.
- If physical execution fails after a legal move is accepted, the system must expose a recoverable error state instead of silently updating chess state.
- Evaluation scripts must stay first-class and must be updated as behavior changes.

## Implementation Order Summary

1. Freeze the tuned baseline, verify exact-8cm square reachability, and remove the Fetch leftover red dot.
2. Define chess board coordinates, table colors, and square materials.
3. Replace single object assumptions with 32 chess piece bodies.
4. Add physical piece registry, teleport API, and movement executor.
5. Add pure chess service around `python-chess`.
6. Add move planner for normal moves, captures, castling, en passant, and promotion.
7. Add orchestrator that performs turns transactionally.
8. Add UI.
9. Expand tests, physics verification, and evaluation scripts.
10. Cleanup docs, configs, and obsolete names.
