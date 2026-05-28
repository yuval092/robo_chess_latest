# Scene Assets

## Source of Truth

The MuJoCo scene is static source code. Runtime startup does not regenerate XML
or STL files.

Important assets:

```text
chess_env/assets/pick_and_place.xml
chess_env/assets/shared.xml
chess_env/assets/robot.xml
chess_env/textures/
chess_env/stls/chess/*.stl
chess_env/stls/fetch/*.stl
```

`ChessSimulationEnv` loads `pick_and_place.xml` through the Fetch XML path
override described in [03_mujoco_environment.md](03_mujoco_environment.md).

## Scene Content

The scene includes:

- Fetch robot meshes and collision geometry.
- A larger table sized for a 64 cm chess board.
- 64 visual board square geoms at 8 cm spacing.
- Active chess pieces in starting positions.
- Promotion reserve pieces.
- Graveyard and reserve visual zones.
- Legacy Fetch `object0` for training compatibility.
- Transparent collision cube geoms for pieces.
- Visual STL mesh geoms for chess silhouettes.

## Piece Design

Each chess piece has a physical collision cube and a visual mesh:

- Collision body: 30 mm cube.
- Freejoint: lets the piece be moved or teleported independently.
- Visual mesh: STL loaded from `chess_env/stls/chess`.
- Mesh geometry is visual-only; collision is the cube.

This simplifies grasping and keeps trained model assumptions stable.

## Board Geometry

Board geometry lives in `configs/chess.yaml`, not in `env.edge_margin`.

Key values:

| Config | Value |
|---|---:|
| `board.cell_size_m` | `0.08` |
| `board.board_size` | `8` |
| `board.width_m` | `0.64` |
| `board.center_xy` | `[0.88, 0.2641]` |
| `board.margin_on_table_m` | `0.03` |

Expected world extents:

| Square edge | Approx coordinate |
|---|---:|
| file a center X | `0.600` |
| file h center X | `1.160` |
| rank 1 center Y | `-0.0159` |
| rank 8 center Y | `0.5441` |

## Graveyards and Promotion Reserves

Captured pieces go to two 4x4 graveyard grids:

| Color | Origin |
|---|---|
| white | `[0.640, -0.300, 0.015]` |
| black | `[0.640, 0.680, 0.015]` |

Promotion reserves use two 8x4 grids:

| Color | Origin |
|---|---|
| white | `[0.820, -0.300, 0.015]` |
| black | `[0.820, 0.680, 0.015]` |

Both use 45 mm slot spacing.

## Editing Assets

When changing XML, STL scale, board geometry, grasp geometry, or zone layout:

1. Update the asset/config together.
2. Update tests that lock the invariant.
3. Run static asset and physics checks.
4. Re-evaluate trained model assumptions if Z levels, grasp geometry, or board
   reachability changed.

Recommended checks:

```bash
pytest tests/chess_env/test_static_assets.py tests/physical/test_scene_physics.py tests/physical/test_zone_alignment.py
```

