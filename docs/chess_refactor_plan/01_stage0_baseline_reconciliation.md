# Stage 0: Tuned Baseline Freeze And Exact Board Verification

## Goal

Freeze the current tuned project baseline before implementing chess behavior, then verify that the requested exact 8cm chess cells are reachable.

This stage is still required even though the refreshed current-status docs now document a healthy reachability baseline. The reason is subtle but important: the current physics reachability test uses the environment sampling grid derived from `edge_margin=0.04`, while the chess requirement requires exact `0.08m` cells. These are not the same geometry.

## Inputs

Read:

- `docs/current_status_new/00_review_report.md`
- `docs/current_status_new/01_system_architecture.md`
- `docs/current_status_new/02_xml_environment.md`
- `docs/current_status_new/03_physics.md`
- `docs/current_status_new/07_waypoint_algorithm.md`
- `configs/env.yaml`
- `chess_env/assets/pick_and_place.xml`
- `chess_env/assets/robot.xml`
- `scripts/verify_physics.py`
- `scripts/eval_stress.py`

## Baseline To Freeze

Use this as the canonical physical baseline:

```yaml
table_surface_z: 0.400
table_center_xy: [0.88, 0.2641]
table_half_x: 0.35
table_half_y: 0.35
edge_margin: 0.04
torso_height: 0.3661
arm_base_xy: [0.56, 0.2641]
```

For chess-game refactor stages, `configs/env.yaml` owns robot/table physics and `configs/chess.yaml` owns exact chess board geometry. `env.yaml` `edge_margin` is a movement sampling margin and must not define chess cell size.

Reasoning:

- An 8x8 board with 8cm cells is exactly 64cm x 64cm.
- A 70cm table leaves 3cm margin per side.
- The refreshed current-status docs report that moving the arm base to `x=0.56` and raising the torso to `0.3661m` clears the near-row dead zone.
- `scripts/verify_physics.py` currently reports max 2.9mm error across its 64-square grid at `SAFE_Z` and `GRASP_Z`.
- The absolute near table edge around `x=0.57` remains unreachable near the arm centerline, but this is not a chess square center.

Do not revert the arm base to `x=0.60` or torso height to `0.25`; those older values are known to reintroduce the near-row dead zone.

## Geometry Issue To Resolve

Current status-doc reachability centers:

```text
x0 = table_center_x - table_half_x + edge_margin = 0.57
x1 = table_center_x + table_half_x - edge_margin = 1.19
cell_width = (x1 - x0) / 8 = 0.0775m
near center = 0.60875
far center = 1.15125
```

Required chess-game centers for exact 8cm cells:

```text
board_width = 8 * 0.08 = 0.64m
board_min_x = 0.88 - 0.32 = 0.56
board_max_x = 0.88 + 0.32 = 1.20
near center = 0.60
far center = 1.16
```

The current reachability test does not prove the exact-8cm board. Stage 0 must add or update tests so exact 8cm square centers are validated directly.

## Implementation Tasks

### 0.0 Verify Red Dot Is Already Fixed

The target0 and object0 red sites have already been addressed in a prior session:
- Both sites have `rgba="1 0 0 0"` (alpha=0, invisible) in `chess_env/assets/pick_and_place.xml`.
- `ChessSimulationEnv._render_callback` is overridden with `pass` in `simulation.py`.

Stage 0 must verify these changes are present and the red dot does not appear. Do not redo them, but do confirm they exist before proceeding. The formal task 0.2 below checks this.

### 0.1 Freeze Baseline Constants

Update `configs/env.yaml` comments to state the tuned physical geometry explicitly:

- `table_center_xy: [0.88, 0.2641]`
- `table_half_x: 0.35`
- `table_half_y: 0.35`
- `edge_margin: 0.04` remains a movement sampling margin, not chess board geometry.
- `torso_height: 0.3661`

Verify `chess_env/assets/robot.xml` has:

```xml
<body childclass="robot0:fetch" name="robot0:base_link" pos="0.56 0.2641 0">
```

Add `configs/chess.yaml` with chess-specific geometry:

```yaml
board:
  cell_size_m: 0.08
  board_size: 8
  center_xy: [0.88, 0.2641]
  width_m: 0.64
  margin_on_table_m: 0.03
  orientation:
    white_side: near_arm
    file_axis: y
    rank_axis: x
  use_env_edge_margin_for_board: false

pieces:
  cube_half_extent_m: 0.015
  cube_height_m: 0.030
  visual_stl_scale: 0.020
  freejoint_damping: 0.5        # higher than object0's 0.1; needed for 32 simultaneous pieces

reserves:
  graveyard_slot_spacing_m: 0.045
  promotion_slot_spacing_m: 0.045
  promotion_pieces_per_type: 8  # worst case: all 8 pawns promote to same type

game:
  human_color: white
  auto_computer_reply: true
  arm_home_xy: [0.680, 0.2641]  # arm returns here after each move, before computer turn
```

Use `configs/chess.yaml` for chess mapping and piece layout. Do not use `env.yaml.edge_margin` to derive chess square centers.

### 0.2 Remove Fetch Red Dot

In `chess_env/assets/pick_and_place.xml`, remove:

```xml
<site name="target0" ... rgba="1 0 0 1" type="sphere"></site>
```

Only remove this from the loaded chess scene file. Other archived Fetch task XML files can be ignored unless they are loaded by tests.

### 0.3 Document Baseline Relationship

Add a short note in the chess refactor docs and future chess-game docs:

Minimum required note:

```text
For chess-game refactor stages, configs/env.yaml owns robot/table physics and configs/chess.yaml owns exact chess board geometry. env.yaml edge_margin is a movement sampling margin and must not define chess cell size.
```

### 0.4 Exact-8cm Reachability Sweep For Chess Cell Centers

Add or update a script:

```text
scripts/eval_chess_reachability.py
```

The script must:

1. Load `configs/chess.yaml`.
2. Generate all 64 square centers using `BoardMapper` with exact `cell_size_m: 0.08`.
3. For every square center:
   - Reset env with `force_scenario="transit"`, `hide_object=True`.
   - Set `inner.force_start_pos = inner.HOME_POS.copy()`.
   - Run `ScriptedController.run_transit(square_xy)`.
   - Transition to descend and run `run_descend(square_xy)`.
   - Run `run_ascend(square_xy)`.
4. Report per-square success, error in mm, and crash reason.
5. Exit non-zero if any square fails.

Expected command:

```bash
python scripts/eval_chess_reachability.py --all-squares --n-episodes 1
```

Also update `scripts/verify_physics.py` so its kinematic reachability test uses `BoardMapper` or `configs/chess.yaml`, not `env.yaml.edge_margin`.

## Validation

Run:

```bash
pytest tests/ -v
python scripts/verify_physics.py
python scripts/eval_stress.py --chain pick --n-episodes 1
python scripts/eval_chess_reachability.py --all-squares --n-episodes 1
```

Pass criteria:

- Existing tests pass.
- Physics verification is healthy.
- Red dot no longer appears in the loaded scene.
- All 64 exact-8cm chess square centers are reachable for transit, descend, and ascend.
- Max reachability error is at or below 5mm at both `SAFE_Z` and `GRASP_Z`.

## Near-Arm Rank Proximity Warning

`board_min_x = 0.56` equals the robot base X coordinate. The rank-1 chess square center at `x = 0.600` is only 40mm from the arm base. The arm has been verified to reach this range with the high torso setting (`torso_height: 0.3661`), but it is the hardest reach in the entire board. Any arm parameter change that reduces torso height or shifts the arm base backwards will reintroduce the dead zone.

The exact-8cm reachability sweep in task 0.4 is the formal guard against this, but note it explicitly in `configs/chess.yaml` as a warning comment next to `board.center_xy`.

## Stop Conditions

Do not proceed if:

- Any chess square center is unreachable.
- Table constants differ between XML, `env.yaml`, and `chess.yaml`.
- The arm cannot descend/ascend at a square within the 10mm tube.

If a square is unreachable:

1. Try shifting `configs/chess.yaml.board.center_xy` within the 70cm tabletop while preserving exact 8cm cells and at least 3cm table margin.
2. Try a tiny torso-height adjustment within the known high-torso operating band.
3. Only then consider changing arm placement or movement behavior.
4. Do not shrink cell size below 8cm to make tests pass.
