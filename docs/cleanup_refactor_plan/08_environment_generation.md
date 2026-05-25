# Stage 8 — Environment Generation Hardening and Documentation

**Objective**: Ensure the chess scene (XML fragments and STL meshes) is always auto-generated at startup. Clean up `environment_generation.py`: add docstrings, name all geometry constants, and document every generation function.

---

## 8.1 Auto-Generation at Environment Construction

**Why NOT on import**: Calling `regenerate_environment()` inside `__init__.py` fires on every `import src.chess_env`, which causes:
- Static analyzers, linters, and package metadata tools to write files on import.
- Tests that only exercise game logic (no MuJoCo) to trigger XML writes unnecessarily.
- Relative-path writes that break when the import happens outside the repo root.
- Read-only deployments or installed wheels that cannot write package asset files.

**Correct approach**: Call generation lazily, from `ChessSimulationEnv.__init__`, just before MuJoCo loads the XML. This fires exactly when a MuJoCo env is actually constructed — never during pure-logic tests or imports.

**Action**: Add `ensure_environment_generated()` to `environment_generation.py`:

```python
_PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../src/chess_env → .../robo_chess_latest
DEFAULT_SCENE_PATH = _PROJECT_ROOT / "chess_env/assets/pick_and_place.xml"
DEFAULT_STL_DIR    = _PROJECT_ROOT / "chess_env/stls/chess"


def ensure_environment_generated(scene_path: Path = DEFAULT_SCENE_PATH) -> None:
    """Regenerate board/pieces/zones XML if the scene XML is stale or missing.

    Uses absolute paths so this is safe to call from any working directory.
    No-op when content already matches the current config.
    """
    _update_scene_if_stale(scene_path)
```

Then in `src/chess_env/simulation.py`, `ChessSimulationEnv.__init__`:
```python
from src.chess_env.environment_generation import ensure_environment_generated

class ChessSimulationEnv(...):
    def __init__(self, ...):
        ensure_environment_generated()   # idempotent; writes only if stale
        super().__init__(...)            # MuJoCo loads XML after this
```

`src/chess_env/__init__.py` must not call `regenerate_environment()`. Remove any such call if present.

**Validation**: Modify `chess.yaml` temporarily (e.g., change a graveyard origin), construct `gym.make("ChessFetchTask-v0")`, verify XML is updated, revert change. Verify that `import src.chess_env` alone (without constructing an env) does NOT write any files.

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
    injected between ``<!-- generated board squares start -->`` and
    ``<!-- generated board squares end -->`` markers in ``pick_and_place.xml``.

3.  **Pieces XML** (``build_pieces_fragment`` / ``update_pieces_scene``): A
    ``<worldbody>`` fragment containing one free-jointed body per chess piece
    (active + reserve), injected between ``<!-- generated chess pieces start -->``
    and ``<!-- generated chess pieces end -->`` markers.

4.  **Zones XML** (``build_zones_fragment`` / ``update_zones_scene``): A
    ``<worldbody>`` fragment containing floor geoms for the graveyard and
    promotion-reserve zones, injected between ``<!-- generated zone markers start -->``
    and ``<!-- generated zone markers end -->`` markers.

All generation is driven by ``configs/chess.yaml`` and ``configs/env.yaml``
so that changing a config value automatically propagates to the scene on the
next env construction.
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
# Changing these values requires re-running STL generation explicitly:
#   python scripts/generate_scene.py stls
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
  src/chess_env/environment_generation.py. Do NOT edit generated content
  between them manually, as it will be overwritten on the next run:

    generated board squares start/end
      Chess board visual geom (8x8 grid of dark/light squares)

    generated chess pieces start/end
      All chess piece bodies (32 active + 64 reserve), each with a freejoint,
      collision box, and visual mesh.

    generated zone markers start/end
      Floor geoms for graveyard and promotion-reserve zones.

  To update manually: python scripts/generate_scene.py all
-->
```

Do not place literal `<!-- ... -->` marker comments inside this explanatory XML comment; XML comments cannot contain nested comment delimiters. The actual marker comments remain only at the generated section boundaries.

---

## 8.6 Verify `ensure_environment_generated()` is Idempotent

Calling `ensure_environment_generated()` twice must produce identical XML output. Add an assertion in the test suite:

```python
# tests/chess_env/test_environment_generation.py
def test_ensure_generated_is_idempotent():
    """Calling ensure_environment_generated twice produces identical XML."""
    from src.chess_env.environment_generation import ensure_environment_generated, DEFAULT_SCENE_PATH
    ensure_environment_generated()
    xml_first = DEFAULT_SCENE_PATH.read_text()
    ensure_environment_generated()
    xml_second = DEFAULT_SCENE_PATH.read_text()
    assert xml_first == xml_second


def test_import_src_chess_env_does_not_write_files():
    """Importing src.chess_env must not write any files (no import-time side effects)."""
    import sys
    from pathlib import Path
    tracked = [
        Path("chess_env/assets/pick_and_place.xml"),
        Path("chess_env/stls/chess"),
    ]
    # Snapshot mtimes before
    before = {p: p.stat().st_mtime_ns for p in tracked if p.exists()}
    # Flush cached module
    for key in list(sys.modules):
        if "chess_env" in key:
            del sys.modules[key]
    import src.chess_env  # noqa: F401
    # Check nothing changed
    after = {p: p.stat().st_mtime_ns for p in tracked if p.exists()}
    for p in before:
        assert before[p] == after.get(p), f"Import wrote {p}"
```

---

## Stage 8 — Full Validation Checklist

```bash
# 1. Import does NOT write files (no import-time generation)
python -c "
import sys
for k in list(sys.modules):
    if 'chess_env' in k:
        del sys.modules[k]
import src.chess_env
print('Import OK — no files written (verify via mtime test below)')
"

# 2. Auto-generation fires on env construction (not import)
python -c "
import gymnasium as gym, src.chess_env
from pathlib import Path
env = gym.make('ChessFetchTask-v0')
xml = Path('chess_env/assets/pick_and_place.xml').read_text()
assert '<!-- generated board squares start -->' in xml
assert '<!-- generated chess pieces start -->' in xml
assert '<!-- generated zone markers start -->' in xml
print('Generation markers present after env construction')
env.close()
"

# 3. generate_scene.py all works
python scripts/generate_scene.py all

# 4. No raw float literals in geometry functions
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

# 5. Idempotency test and import-side-effect test
python -m pytest tests/chess_env/test_environment_generation.py -v

# 6. Full test suite
python -m pytest tests/ -v
```
