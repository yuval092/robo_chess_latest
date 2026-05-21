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
  chess.yaml

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
