# Stage 8 — Environment Generation Hardening and Documentation

**Objective**: Ensure the chess scene (XML fragments and STL meshes) is always auto-generated at startup. Clean up `environment_generation.py`: add docstrings, name all geometry constants, and document every generation function.

---

## 8.1 Auto-Generation at Startup

**Current situation**: `run_chess_ui.py` calls `regenerate_environment()` before creating the gymnasium environment. However, if a developer runs eval scripts or tests directly, they might use a stale XML. The scene XML must always be up-to-date before any simulation runs.

**Action**: Add a `regenerate_environment()` call at the top of `src/chess_env/__init__.py`, so it fires automatically whenever the package is imported:

```python
"""Gymnasium environment registration for ChessFetchTask-v0."""
from gymnasium.envs.registration import register

from src.chess_env.environment_generation import regenerate_environment

# Regenerate scene XML and STL meshes every time the environment package is imported.
# This is fast (<100ms) and ensures the scene always matches the current configuration.
regenerate_environment()

register(
    id="ChessFetchTask-v0",
    entry_point="src.chess_env.task:ChessTaskEnv",
    max_episode_steps=200,
)
```

**Why**: This guarantees that no script or test can accidentally run against a stale scene. The regeneration is idempotent and cheap.

**Validation**: Modify `chess.yaml` temporarily (e.g., change a graveyard origin), `import src.chess_env`, verify XML is updated, revert change.

---

## 8.2 Document `environment_generation.py` — Module Docstring

Add to the top of the file:

```python
"""
Chess scene asset generator for RoboChess.

This module generates all runtime assets for the MuJoCo chess simulation:

1.  **STL meshes** (``regenerate_stls``): Binary STL files for each of the 6
    chess piece types.  Meshes are procedurally generated from parametric
    geometry (frustums, boxes, spheres) and written to
    ``chess_env/stls/chess/``.

2.  **Board XML** (``build_board_fragment`` / ``update_board_scene``): A
    ``<worldbody>`` fragment containing the flat visual geom for the 8×8 board,
    injected between ``<!-- BOARD_START -->`` and ``<!-- BOARD_END -->`` markers
    in ``pick_and_place.xml``.

3.  **Pieces XML** (``build_pieces_fragment`` / ``update_pieces_scene``): A
    ``<worldbody>`` fragment containing one free-jointed body per chess piece
    (active + reserve), injected between ``<!-- PIECES_START -->`` and
    ``<!-- PIECES_END -->`` markers.

4.  **Zones XML** (``build_zones_fragment`` / ``update_zones_scene``): A
    ``<worldbody>`` fragment containing floor geoms for the graveyard and
    promotion-reserve zones, injected between ``<!-- ZONES_START -->`` and
    ``<!-- ZONES_END -->`` markers.

All generation is driven by ``configs/chess.yaml`` and ``configs/env.yaml``
so that changing a config value automatically propagates to the scene on the
next import of ``src.chess_env``.
"""
```

---

## 8.3 Name All STL Geometry Constants

**Current**: The six piece-geometry functions contain raw floating-point literals.

**Action**: Define named module-level constants for every piece geometry value. Group them by piece type with a comment block. All dimensions are in **metres**.

```python
# ---------------------------------------------------------------------------
# STL geometry constants — all dimensions in metres
# These define the procedural mesh geometry for each piece type.
# Changing these values requires re-running regenerate_stls() (done automatically
# on import via regenerate_environment()).
# ---------------------------------------------------------------------------

# Shared
_STL_SEGS = 32  # Circular cross-section triangulation segments

# Pawn
_PAWN_BASE_R      = 0.0115  # Bottom frustum outer radius
_PAWN_WAIST_R     = 0.0085  # Frustum top / waist radius
_PAWN_NECK_R      = 0.0070  # Neck radius below head sphere
_PAWN_HEAD_R      = 0.0100  # Head sphere radius
_PAWN_BASE_H      = 0.0080  # Bottom frustum height
_PAWN_WAIST_H     = 0.0060  # Waist cylinder height
_PAWN_NECK_H      = 0.0030  # Neck height
_PAWN_HEAD_CENTRE = 0.0155  # Z of head sphere centre

# Rook
_ROOK_BASE_R      = 0.0120
_ROOK_WAIST_R     = 0.0090
_ROOK_TOP_R       = 0.0110
_ROOK_BASE_H      = 0.0100
_ROOK_SHAFT_H     = 0.0080
_ROOK_BATTLEMENT_OUTER = 0.0045   # Half-width of battlements
_ROOK_BATTLEMENT_H     = 0.0040

# Knight
_KNIGHT_BASE_R    = 0.0110
_KNIGHT_WAIST_R   = 0.0080
_KNIGHT_NECK_R    = 0.0065
_KNIGHT_HEAD_W    = 0.0090  # Head box half-width (X)
_KNIGHT_HEAD_D    = 0.0050  # Head box half-depth (Y)
_KNIGHT_HEAD_H    = 0.0115  # Head box half-height (Z)
_KNIGHT_BASE_H    = 0.0100
_KNIGHT_SHAFT_H   = 0.0060
_KNIGHT_NECK_H    = 0.0025
_KNIGHT_HEAD_TILT = 25.0    # Head tilt in degrees (forward lean)

# Bishop
_BISHOP_BASE_R    = 0.0110
_BISHOP_WAIST_R   = 0.0080
_BISHOP_NECK_R    = 0.0060
_BISHOP_HEAD_R    = 0.0085
_BISHOP_TIP_R     = 0.0025
_BISHOP_BASE_H    = 0.0100
_BISHOP_SHAFT_H   = 0.0070
_BISHOP_NECK_H    = 0.0040
_BISHOP_HEAD_CENTRE = 0.0185
_BISHOP_TIP_CENTRE  = 0.0240

# Queen
_QUEEN_BASE_R     = 0.0125
_QUEEN_WAIST_R    = 0.0090
_QUEEN_NECK_R     = 0.0070
_QUEEN_HEAD_R     = 0.0105
_QUEEN_CROWN_R    = 0.0040
_QUEEN_BASE_H     = 0.0100
_QUEEN_SHAFT_H    = 0.0080
_QUEEN_NECK_H     = 0.0035
_QUEEN_HEAD_CENTRE = 0.0205
_QUEEN_CROWN_COUNT = 5       # Number of crown points

# King
_KING_BASE_R      = 0.0125
_KING_WAIST_R     = 0.0090
_KING_NECK_R      = 0.0070
_KING_HEAD_R      = 0.0105
_KING_CROSS_W     = 0.0040  # Cross bar half-width
_KING_CROSS_H     = 0.0070  # Cross bar half-height
_KING_CROSS_D     = 0.0015  # Cross bar half-depth
_KING_BASE_H      = 0.0100
_KING_SHAFT_H     = 0.0080
_KING_NECK_H      = 0.0035
_KING_HEAD_CENTRE = 0.0205
```

Replace all raw literals in the six geometry functions with the corresponding constant names.

---

## 8.4 Add Docstrings to All Generation Functions

Every function in `environment_generation.py` must have a one-line docstring:

```python
def _replace_marked_fragment(xml: str, marker: str, fragment: str) -> str:
    """Replace the content between <!-- {marker}_START --> and <!-- {marker}_END --> with fragment."""

def build_board_fragment() -> str:
    """Generate the XML worldbody fragment for the 8×8 board visual geom."""

def knight_visual_euler() -> str:
    """Return the Euler angle string that orients the knight STL mesh correctly."""

def piece_body_xml(piece: PhysicalPiece) -> str:
    """Generate the XML for a single chess piece body with freejoint and visual geom."""

def reserve_position(color: str, piece_type: str, index: int) -> tuple[float, float, float]:
    """Return the XYZ world position of a reserve piece slot for the given colour and type."""

def build_pieces_fragment() -> str:
    """Generate the XML worldbody fragment for all active and reserve chess piece bodies."""

def _zone_geom(name: str, origin: list, rows: int, cols: int, spacing: float) -> str:
    """Generate a floor geom XML element for one zone (graveyard or promotion reserve)."""

def build_zones_fragment() -> str:
    """Generate the XML worldbody fragment for all graveyard and promotion-reserve floor geoms."""

def update_board_scene(scene_path: Path) -> None:
    """Inject the board XML fragment into the scene XML file in place."""

def update_pieces_scene(scene_path: Path) -> None:
    """Inject the pieces XML fragment into the scene XML file in place."""

def update_zones_scene(scene_path: Path) -> None:
    """Inject the zones XML fragment into the scene XML file in place."""

def regenerate_stls(out_dir: Path) -> Path:
    """Regenerate all STL mesh files in the output directory and return the directory path."""

def regenerate_environment() -> None:
    """Regenerate all scene assets: STL meshes, board XML, pieces XML, and zones XML."""

# STL geometry helpers
def normal(v0, v1, v2) -> np.ndarray:
    """Compute the unit outward normal of a triangle given three vertices."""

def box_triangles(cx, cy, cz, hx, hy, hz) -> list:
    """Return the 12 triangles (2 per face) for an axis-aligned box."""

def frustum_triangles(r0, r1, h, z0, segs) -> list:
    """Return the triangles for a truncated cone (frustum) from radius r0 to r1 over height h."""

def cone_triangles(r, h, z0, segs) -> list:
    """Return the triangles for a solid cone of radius r and height h."""

def sphere_triangles(r, cx, cy, cz, segs) -> list:
    """Return the triangles approximating a sphere of radius r centred at (cx, cy, cz)."""

def offset_tris(tris, dz) -> list:
    """Translate all triangles vertically by dz."""

def pawn_triangles() -> list:
    """Return the triangle mesh for a pawn piece."""

def rook_triangles() -> list:
    """Return the triangle mesh for a rook piece."""

def knight_triangles() -> list:
    """Return the triangle mesh for a knight piece."""

def bishop_triangles() -> list:
    """Return the triangle mesh for a bishop piece."""

def queen_triangles() -> list:
    """Return the triangle mesh for a queen piece."""

def king_triangles() -> list:
    """Return the triangle mesh for a king piece."""
```

---

## 8.5 Document `pick_and_place.xml` Markers

Add a comment near the top of `chess_env/assets/pick_and_place.xml` (after the XML declaration) explaining the injection markers:

```xml
<!--
  RoboChess scene file for Fetch Pick-and-Place environment.

  The following marker pairs are automatically managed by
  src/chess_env/environment_generation.py — do NOT edit content
  between them manually, as it will be overwritten on the next run:

    <!-- BOARD_START --> ... <!-- BOARD_END -->
      Chess board visual geom (8x8 grid of dark/light squares)

    <!-- PIECES_START --> ... <!-- PIECES_END -->
      All chess piece bodies (32 active + 64 reserve), each with a freejoint,
      collision box, and visual mesh.

    <!-- ZONES_START --> ... <!-- ZONES_END -->
      Floor geoms for graveyard and promotion-reserve zones.

  To update manually: python scripts/generate_scene.py all
-->
```

---

## 8.6 Verify `regenerate_environment()` is Idempotent

Running `regenerate_environment()` twice must produce identical XML output. Add an assertion in the test suite:

```python
# tests/chess_env/test_environment_generation.py
def test_regenerate_is_idempotent(tmp_path):
    """Running regenerate_environment twice produces identical XML."""
    # Already tested implicitly; add explicit check:
    from src.chess_env.environment_generation import regenerate_environment, DEFAULT_SCENE_PATH
    regenerate_environment()
    xml_first = DEFAULT_SCENE_PATH.read_text()
    regenerate_environment()
    xml_second = DEFAULT_SCENE_PATH.read_text()
    assert xml_first == xml_second
```

---

## Stage 8 — Full Validation Checklist

```bash
# 1. Auto-generation fires on import
python -c "
import src.chess_env
from pathlib import Path
xml = Path('chess_env/assets/pick_and_place.xml').read_text()
assert '<!-- BOARD_START -->' in xml
assert '<!-- PIECES_START -->' in xml
assert '<!-- ZONES_START -->' in xml
print('Auto-generation markers present')
"

# 2. generate_scene.py all works
python scripts/generate_scene.py all

# 3. No raw float literals in geometry functions
python -c "
import ast, pathlib
source = pathlib.Path('src/chess_env/environment_generation.py').read_text()
tree = ast.parse(source)
# Count float literals inside function bodies (rough check)
in_fn = False
suspicious = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name.endswith('_triangles'):
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, float):
                suspicious.append((node.name, child.value))
if suspicious:
    print('Remaining float literals in triangle functions:')
    for fn, val in suspicious[:10]: print(f'  {fn}: {val}')
else:
    print('No raw float literals in triangle functions')
"

# 4. Idempotency test
python -m pytest tests/chess_env/test_environment_generation.py -v

# 5. Full test suite
python -m pytest tests/ -v
```
