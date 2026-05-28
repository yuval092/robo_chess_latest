# Static Scene Assets

## Ownership

The MuJoCo scene is no longer generated at runtime. The checked-in files are the
source of truth:

```text
chess_env/assets/pick_and_place.xml
chess_env/assets/shared.xml
chess_env/assets/robot.xml
chess_env/stls/chess/*.stl
chess_env/stls/fetch/*.stl
chess_env/stls/hand/*.stl
```

`ChessSimulationEnv` loads `chess_env/assets/pick_and_place.xml` directly through
Gymnasium Robotics' Fetch XML path override. Startup does not rewrite XML or STL
files.

## Scene Invariants

`pick_and_place.xml` documents the major sections in place. The key invariants
covered by tests are:

- The table top is at z=0.400 m and has four legs.
- The board has 64 visual-only square geoms at 0.08 m spacing.
- Graveyard and promotion reserve markers are world-space, visual-only geoms.
- `object0` remains available for legacy Fetch/stage training and can be hidden.
- Active chess pieces start on their board squares.
- Reserve pieces start in the promotion banks.
- Each chess piece has a transparent colliding cube and a visual-only STL mesh.
- Grasp actuator parameters and `GRASP_Z` remain fixed for trained models.

## Editing Assets

Edit XML/STL assets directly and keep changes reviewed like normal source code.
When changing geometry, update the corresponding configs and pytest assertions in
the same change. There is no scene-generation command or runtime generation
module.

## Verification

Static and MuJoCo-backed checks are in pytest:

```bash
pytest tests/chess_env/test_static_assets.py tests/physical/test_scene_physics.py
```

These tests replace the old generation and physics-evaluation command coverage.
