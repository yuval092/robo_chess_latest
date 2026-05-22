# Stage 11: Graveyard and Promotion Reserve Visual Zones

## Goal

Add physical floor markers for the four off-board zones — white graveyard, black graveyard,
white promotion reserve, and black promotion reserve. Currently, `PieceTeleporter` correctly
places pieces at the configured floor coordinates, but there is nothing visible in the scene
to indicate where those zones are. After this stage each zone has a flat coloured geom on
the floor, the piece positions land inside the visible area, and the zone layout is
validated by a new script.

## Background: Zone Coordinate System

Zones are defined in `configs/chess.yaml` as `origin_xyz` + `rows` + `cols`. The
`PieceTeleporter._slot_xyz` formula is:

```python
return np.array([
    origin[0] + row * spacing,   # +X for each row
    origin[1] + col * spacing,   # +Y for each col
    origin[2],
])
```

`row = slot // cols`, `col = slot % cols`. Pieces are placed at z=origin[2] (floor height),
which puts the piece centre at z=0.030 (origin[2]=0.015 is the z of the piece _centre_ when
`cube_half_extent_m=0.015`).

After Stage 10 the zone positions are:

| Zone                    | origin_xyz              | rows | cols | X range            | Y range          |
|-------------------------|-------------------------|------|----- |--------------------|------------------|
| white graveyard         | [0.640, −0.300, 0.015]  | 4    | 4    | 0.640 → 0.775      | −0.300 → −0.165  |
| black graveyard         | [0.640, 0.680, 0.015]   | 4    | 4    | 0.640 → 0.775      | 0.680 → 0.815    |
| white promotion reserve | [0.820, −0.300, 0.015]  | 8    | 4    | 0.820 → 1.135      | −0.300 → −0.165  |
| black promotion reserve | [0.820, 0.680, 0.015]   | 8    | 4    | 0.820 → 1.135      | 0.680 → 0.815    |

All zones are outside the table footprint (`y ∈ [−0.086, 0.614]`).

## Implementation Tasks

### 11.1 Add Zone Materials to `chess_env/assets/shared.xml`

Inside the `<asset>` block, after the existing chess-piece materials, add:

```xml
<!-- Zone floor markers -->
<material name="white_graveyard_mat"  specular="0" shininess="0" rgba="0.80 0.76 0.62 0.85"/>
<material name="black_graveyard_mat"  specular="0" shininess="0" rgba="0.25 0.22 0.18 0.85"/>
<material name="white_reserve_mat"    specular="0" shininess="0" rgba="0.72 0.85 0.72 0.85"/>
<material name="black_reserve_mat"    specular="0" shininess="0" rgba="0.20 0.35 0.20 0.85"/>
```

Colour intent:
- White graveyard: warm beige (similar to white pieces, low saturation)
- Black graveyard: dark brown (similar to black pieces)
- White promotion reserve: muted green (distinct from graveyard)
- Black promotion reserve: dark green

### 11.2 Add `scripts/generate_zones_xml.py`

Create a new script that reads zone positions from `configs/chess.yaml` and generates a
flat box geom covering each zone's full extent. The geom sits on the floor at z=0 with a
0.002m half-thickness (4mm total height). Pieces placed at z=0.015 sit on top of it.

```python
"""Generate visual floor zone geoms for graveyard and promotion reserve areas."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

sys.path.append(os.getcwd())

from src.utils.config import load_config


START_MARKER = "\t\t<!-- generated zone markers start -->"
END_MARKER = "\t\t<!-- generated zone markers end -->"


ZONE_SPECS = [
    ("white_graveyard",  "graveyards",        "white", "white_graveyard_mat"),
    ("black_graveyard",  "graveyards",        "black", "black_graveyard_mat"),
    ("white_reserve",    "promotion_reserve", "white", "white_reserve_mat"),
    ("black_reserve",    "promotion_reserve", "black", "black_reserve_mat"),
]


def _zone_geom(name: str, cfg: dict, spacing: float, material: str) -> str:
    origin = cfg["origin_xyz"]
    rows = cfg["rows"]
    cols = cfg["cols"]
    # Extent covers all slots plus a small visual margin (5mm per side)
    margin = 0.005
    half_x = (rows * spacing) / 2.0 + margin
    half_y = (cols * spacing) / 2.0 + margin
    center_x = origin[0] + (rows - 1) * spacing / 2.0
    center_y = origin[1] + (cols - 1) * spacing / 2.0
    center_z = 0.002   # 4mm thick slab on the floor
    return (
        f'\t\t<geom name="zone_{name}" type="box" '
        f'size="{half_x:.4f} {half_y:.4f} 0.002" '
        f'pos="{center_x:.4f} {center_y:.4f} {center_z:.4f}" '
        f'material="{material}" contype="0" conaffinity="0" mass="0"/>'
    )


def build_fragment() -> str:
    cfg = load_config("chess")
    spacing = cfg["reserves"]["graveyard_slot_spacing_m"]
    promo_spacing = cfg["reserves"]["promotion_slot_spacing_m"]
    lines = [START_MARKER]
    for short_name, section_key, color, material in ZONE_SPECS:
        zone_cfg = cfg[section_key][color]
        sp = promo_spacing if "reserve" in short_name else spacing
        lines.append(_zone_geom(short_name, zone_cfg, sp, material))
    lines.append(END_MARKER)
    return "\n".join(lines)


def update_scene(scene_path: Path) -> None:
    text = scene_path.read_text()
    start = text.index(START_MARKER)
    end = text.index(END_MARKER) + len(END_MARKER)
    scene_path.write_text(text[:start] + build_fragment() + text[end:])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate zone visual floor geoms.")
    parser.add_argument("--write", action="store_true",
                        help="Update chess_env/assets/pick_and_place.xml in place.")
    args = parser.parse_args()
    if args.write:
        scene_path = Path("chess_env/assets/pick_and_place.xml")
        update_scene(scene_path)
        print(f"Updated {scene_path}")
    else:
        print(build_fragment())


if __name__ == "__main__":
    main()
```

### 11.3 Add Zone Marker Placeholders to `chess_env/assets/pick_and_place.xml`

The zone generator needs start/end markers in the XML. Insert them in the `<worldbody>`
section, **after** the `<body name="table0">` block and **before** the
`<!-- generated chess pieces start -->` comment. Do not place them inside the `table0`
body; zone geoms are in world coordinates, not table-local coordinates.

```xml
		<!-- generated zone markers start -->
		<!-- generated zone markers end -->
```

After adding the markers, run `scripts/generate_zones_xml.py --write` to populate them.

### 11.4 Verify Zone Teleport Alignment

Write a unit test in `tests/physical/test_piece_teleport.py` (or a new file
`tests/physical/test_zone_alignment.py`) that verifies each slot's XY position lands
inside the corresponding zone's visual geom extent.

```python
import numpy as np
import pytest
from src.physical.piece_teleport import PieceTeleporter
from src.utils.config import load_config


@pytest.mark.parametrize("color,zone_key,section", [
    ("white", "white_graveyard_mat", "graveyards"),
    ("black", "black_graveyard_mat", "graveyards"),
    ("white", "white_reserve_mat",   "promotion_reserve"),
    ("black", "black_reserve_mat",   "promotion_reserve"),
])
def test_all_slots_inside_zone_extent(color, zone_key, section):
    cfg = load_config("chess")
    zone_cfg = cfg[section][color]
    spacing = cfg["reserves"]["graveyard_slot_spacing_m"]
    if "reserve" in zone_key:
        spacing = cfg["reserves"]["promotion_slot_spacing_m"]

    origin = zone_cfg["origin_xyz"]
    rows = zone_cfg["rows"]
    cols = zone_cfg["cols"]

    margin = 0.005
    x_min = origin[0] - margin
    x_max = origin[0] + (rows - 1) * spacing + margin
    y_min = origin[1] - margin
    y_max = origin[1] + (cols - 1) * spacing + margin

    for slot in range(rows * cols):
        row = slot // cols
        col = slot % cols
        slot_x = origin[0] + row * spacing
        slot_y = origin[1] + col * spacing
        assert x_min <= slot_x <= x_max, f"slot {slot} x={slot_x:.4f} outside zone"
        assert y_min <= slot_y <= y_max, f"slot {slot} y={slot_y:.4f} outside zone"
```

This test is pure-config (no MuJoCo required) and runs in the normal test suite.

### 11.5 Verify `PieceTeleporter` Uses Updated Config Positions

The `PieceTeleporter.teleport_piece_to_graveyard` and `teleport_piece_to_promotion_reserve`
read directly from `load_config("chess")` at call time. After updating `chess.yaml`
(Stage 10.1), these methods automatically use the new positions. No code changes needed.

Run a manual sanity check:

```python
python -c "
import sys; sys.path.insert(0,'.')
from src.utils.config import load_config
cfg = load_config('chess')

def slot_xyz(section, color, slot):
    zone = cfg[section][color]
    sp = cfg['reserves']['graveyard_slot_spacing_m']
    row, col = divmod(slot, zone['cols'])
    o = zone['origin_xyz']
    return o[0]+row*sp, o[1]+col*sp, o[2]

print('White graveyard slot 0:', slot_xyz('graveyards','white',0))
print('White graveyard slot 15:', slot_xyz('graveyards','white',15))
print('Black graveyard slot 0:', slot_xyz('graveyards','black',0))
print('White promo slot 0:', slot_xyz('promotion_reserve','white',0))
print('Black promo slot 31:', slot_xyz('promotion_reserve','black',31))
"
```

Expected: white graveyard slot 0 = (0.640, -0.300, 0.015), white graveyard slot 15 =
(0.775, -0.165, 0.015).

## Validation

```bash
# 1. Unit tests including new zone alignment test
pytest tests/ -v --ignore=tests/integration -q

# 2. Visual verification — run the scene and confirm zones appear on the floor
#    (requires MuJoCo viewer; skip in CI)
python -c "
import sys; sys.path.insert(0,'.')
import gymnasium as gym, src.chess_env
env = gym.make('ChessFetchTask-v0', render_mode='human', show_chess_pieces=True)
env.reset()
input('Check zones visible on floor. Press Enter to continue.')
env.close()
"

# 3. Zone slot positions
python -c "
import sys; sys.path.insert(0,'.')
from src.utils.config import load_config
cfg = load_config('chess')
for s in ['graveyards','promotion_reserve']:
    for c in ['white','black']:
        o = cfg[s][c]['origin_xyz']
        print(f'{s}/{c}: origin={o}')
"

# 4. Generate zones and confirm XML is updated
python scripts/generate_zones_xml.py --write
grep -c 'zone_' chess_env/assets/pick_and_place.xml   # expect 4
```

## Stop Conditions

Do not proceed to Stage 12 if:

- The zone geoms are inside the `table0` body (they must be in world coordinates, not
  table-local; a piece at y=-0.300 world must land at the correct zone geom).
- Any slot XY is outside its zone geom extent.
- The zone markers overlap with the board (board y range: [−0.056, 0.584]; zone y ranges
  are outside this).
- Unit tests fail after adding the zone materials.
