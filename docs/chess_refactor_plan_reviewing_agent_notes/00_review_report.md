# Stage 10–12 Implementation Review Report

## Summary

The implementing agent's Stage 10–12 changes are **functionally correct**. All 77 tests pass,
all eval scripts pass, and verify_physics reports HEALTHY. No logic bugs were found.

## Verified Correct

| Area | Verdict |
|------|---------|
| `board_mapper.py` — file_axis=X swap | ✓ Correct |
| `test_board_mapper.py` — updated expectations | ✓ Correct |
| `eval_chess_reachability.py` — validate_board_geometry | ✓ Correct |
| Zone geoms in worldbody (not table0) | ✓ Correct |
| Piece positions in XML (white_king e1, black_king e8) | ✓ Correct |
| Knight euler in radians (π/2 white, 3π/2 black) | ✓ Correct |
| `visual_stl_scale` removed from chess.yaml | ✓ Done |
| Knight STL 184 triangles | ✓ Done |
| `test_each_piece_type_uses_correct_mesh` added | ✓ Done |

## Pre-existing Issues (addressed in user notes below)

### 1. object0 cube is visible in chess mode
The legacy Fetch object (`object0`) uses `material="block_mat"` with `rgba="0.2 0.2 0.2 1"`.
It is teleported to `[2.0, 2.0, 0.015]` in chess mode but remains visually present as a dark grey cube.
**Fix:** Change block_mat alpha to 0 in shared.xml (see note 5).

### 2. STL meshes do not wrap the cube
All STL meshes are small cylinders (radius 5–7mm) placed at `pos="0 0 0.017"` ON TOP of the
cube. The cube half-extent is 15mm, making pieces look like cubes with tiny markings rather
than chess pieces.
**Fix:** Redesign all STLs with 30×30mm base, place at `pos="0 0 -0.015"` (see note 6).

### 3. HOME_POSITION is not board center
`home_position_xy=[0.680, 0.2641]` is off-center. User wants board center `[0.88, 0.2641]`.
**Fix:** Update env.yaml and chess.yaml (see note 3).

### 4. Arm is too slow; cube vibrates during grasp
`MAX_STEP_SIZE_M=0.008`, `grasp_close_steps=150`, `grasp_hold_steps=50`. Release ramp hardcoded
to 80 steps. Cube mass=0.05kg, freejoint_damping=0.5 are too light/low causing vibration.
**Fix:** See notes 2 and 4.
