"""Generate MuJoCo chess-board XML fragments and visual STL meshes."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import chess

from src.chess_game.board_mapper import BoardMapper
from src.physical.piece_registry import PieceRegistry, reserve_piece_ids
from src.utils.io import load_config

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_PATH = _PROJECT_ROOT / "chess_env/assets/pick_and_place.xml"
DEFAULT_STL_DIR = _PROJECT_ROOT / "chess_env/stls/chess"

BOARD_START_MARKER = "\t\t\t<!-- generated board squares start -->"
BOARD_END_MARKER = "\t\t\t<!-- generated board squares end -->"
PIECES_START_MARKER = "\t\t<!-- generated chess pieces start -->"
PIECES_END_MARKER = "\t\t<!-- generated chess pieces end -->"
ZONES_START_MARKER = "\t\t<!-- generated zone markers start -->"
ZONES_END_MARKER = "\t\t<!-- generated zone markers end -->"

BASE_R = 0.022
WAIST_R = 0.010
BASE_TOP = 0.032
SEGS = 32

# STL geometry constants — all dimensions in metres.
# Changing any value requires re-running: python scripts/generate_scene.py stls

# Pawn
_PAWN_NECK_R        = 0.007    # narrow neck radius below head sphere
_PAWN_NECK_Z        = 0.038    # z where neck section begins
_PAWN_HEAD_COLLAR_R = 0.010    # radius at base of head sphere
_PAWN_HEAD_COLLAR_Z = 0.041    # z at base of head sphere
_PAWN_HEAD_Z        = 0.047    # z of head sphere centre
_PAWN_HEAD_R        = 0.0115   # head sphere radius
_PAWN_HEAD_Z_SCALE  = 0.9      # head sphere z compression (slightly flattened)

# Rook
_ROOK_SHAFT_R           = 0.013    # shaft radius above base
_ROOK_SHAFT_Z           = 0.041    # z at top of shaft / bottom of collar
_ROOK_COLLAR_R          = 0.017    # collar outer radius
_ROOK_COLLAR_Z          = 0.044    # z at bottom of parapet cylinder
_ROOK_PARAPET_Z         = 0.047    # z at top of parapet cylinder
_ROOK_INNER_R           = 0.014    # inner parapet radius
_ROOK_TOP_Z             = 0.050    # z at top of rook body (battlement floor)
_ROOK_BATTLEMENT_OFFSET = 0.0125   # cardinal offset distance for battlement boxes
_ROOK_BATTLEMENT_Z      = 0.053    # z centre of battlement boxes
_ROOK_BATTLEMENT_HX     = 0.009    # battlement box half-length (radial)
_ROOK_BATTLEMENT_HY     = 0.011    # battlement box half-width (tangential)
_ROOK_BATTLEMENT_HZ     = 0.006    # battlement box half-height
_ROOK_CORNER_OFFSET     = 0.0088   # diagonal offset for corner battlement boxes
_ROOK_CORNER_Z          = 0.052    # z centre of corner battlement boxes
_ROOK_CORNER_HXY        = 0.006    # corner battlement box half-size in x and y
_ROOK_CORNER_HZ         = 0.004    # corner battlement box half-height

# Knight
_KNIGHT_SHAFT_NECK_R = 0.008    # shaft radius at top (narrow part)
_KNIGHT_SHAFT_Z      = 0.040    # z where upper body begins
_KNIGHT_BODY_X       = 0.004    # body block x offset (forward lean)
_KNIGHT_BODY_Z       = 0.043    # z centre of body block
_KNIGHT_BODY_HX      = 0.012    # body block half-length (x)
_KNIGHT_BODY_HY      = 0.012    # body block half-width (y)
_KNIGHT_BODY_HZ      = 0.010    # body block half-height (z)
_KNIGHT_NECK_X       = 0.012    # neck/snout block x offset
_KNIGHT_NECK_Z       = 0.050    # z centre of neck/snout block
_KNIGHT_NECK_HX      = 0.021    # neck block half-length (x)
_KNIGHT_NECK_HY      = 0.014    # neck block half-width (y)
_KNIGHT_NECK_HZ      = 0.014    # neck block half-height (z)
_KNIGHT_EAR_X        = 0.023    # ear block x offset
_KNIGHT_EAR_Z        = 0.045    # z centre of ear blocks
_KNIGHT_EAR_HX       = 0.010    # ear block half-length (x)
_KNIGHT_EAR_HY       = 0.009    # ear block half-depth (y)
_KNIGHT_EAR_HZ       = 0.008    # ear block half-height (z)
_KNIGHT_SNOUT_Z      = 0.055    # z centre of snout block
_KNIGHT_SNOUT_HX     = 0.006    # snout block half-length (x)
_KNIGHT_SNOUT_HY     = 0.010    # snout block half-width (y)
_KNIGHT_SNOUT_HZ     = 0.006    # snout block half-height (z)
_KNIGHT_NOSTRIL_X    = 0.015    # nostril block x offset
_KNIGHT_NOSTRIL_Y    = 0.004    # nostril block y offset (±)
_KNIGHT_NOSTRIL_Z    = 0.058    # z centre of nostril blocks
_KNIGHT_NOSTRIL_HXY  = 0.004    # nostril block half-size in x and y (square)
_KNIGHT_NOSTRIL_HZ   = 0.005    # nostril block half-height (z)
_KNIGHT_MANE_X       = 0.018    # mane strip x offset
_KNIGHT_MANE_Z       = 0.051    # z centre of mane strip
_KNIGHT_MANE_HX      = 0.003    # mane strip half-length (x)
_KNIGHT_MANE_HY      = 0.016    # mane strip half-width (y)
_KNIGHT_MANE_HZ      = 0.003    # mane strip half-height (z)

# Bishop
_BISHOP_SHAFT_NECK_R = 0.008    # shaft radius at top (narrow part)
_BISHOP_SHAFT_Z      = 0.040    # z where body begins
_BISHOP_BODY_R       = 0.012    # widest body radius
_BISHOP_BODY_Z       = 0.043    # z at widest body point
_BISHOP_SHOULDER_R   = 0.010    # shoulder radius (narrowing above body)
_BISHOP_SHOULDER_Z   = 0.046    # z at shoulder
_BISHOP_HEAD_Z       = 0.049    # z of head sphere centre
_BISHOP_HEAD_R       = 0.0085   # head sphere radius
_BISHOP_HEAD_Z_SCALE = 1.1      # head sphere z elongation
_BISHOP_CROSS_X      = 0.005    # cross notch x offset
_BISHOP_CROSS_Z      = 0.051    # z centre of cross notch
_BISHOP_CROSS_HX     = 0.004    # cross notch half-length (x)
_BISHOP_CROSS_HY     = 0.020    # cross notch half-width (y)
_BISHOP_CROSS_HZ     = 0.012    # cross notch half-height (z)
_BISHOP_TIP_R        = 0.004    # finial cone base radius
_BISHOP_TIP_Z        = 0.056    # z base of finial cone
_BISHOP_TIP_APEX_Z   = 0.059    # z apex of finial cone

# Queen
_QUEEN_SHAFT_NECK_R  = 0.008    # shaft radius at top (narrow part)
_QUEEN_SHAFT_Z       = 0.038    # z where shaft section starts
_QUEEN_BODY_R        = 0.014    # widest body radius
_QUEEN_BODY_Z        = 0.046    # z at widest body point
_QUEEN_SHOULDER_R    = 0.017    # shoulder radius (bell shape)
_QUEEN_SHOULDER_Z    = 0.048    # z at shoulder
_QUEEN_NECK_R        = 0.015    # neck radius below crown
_QUEEN_NECK_Z        = 0.050    # z at neck top / crown base
_QUEEN_CROWN_ORBIT_R = 0.014    # orbit radius for crown spike positions
_QUEEN_CROWN_SPIKE_R = 0.0045   # cone base radius for each crown spike
_QUEEN_CROWN_TIP_Z   = 0.057    # z tip of crown spikes
_QUEEN_ORB_Z         = 0.054    # z of central orb sphere centre
_QUEEN_ORB_R         = 0.0055   # central orb sphere radius
_QUEEN_ORB_Z_SCALE   = 0.9      # orb z compression

# King
_KING_SHAFT_NECK_R = 0.008    # shaft radius at top (narrow part)
_KING_SHAFT_Z      = 0.038    # z where shaft section starts
_KING_BODY_R       = 0.014    # widest body radius
_KING_BODY_Z       = 0.048    # z at widest body point
_KING_SHOULDER_R   = 0.011    # shoulder radius
_KING_SHOULDER_Z   = 0.051    # z at shoulder top
_KING_SPIRE_R      = 0.006    # spire base radius (narrows to tip)
_KING_SPIRE_Z      = 0.053    # z at top of spire section
_KING_CROSS_V_Z    = 0.056    # z centre of vertical cross bar
_KING_CROSS_V_H    = 0.005    # vertical bar square cross-section half-size
_KING_CROSS_V_HZ   = 0.008    # vertical bar half-height
_KING_CROSS_H_Z    = 0.055    # z centre of horizontal cross bar
_KING_CROSS_H_HX   = 0.020    # horizontal bar half-width
_KING_CROSS_H_HY   = 0.005    # horizontal bar half-depth
_KING_CROSS_H_HZ   = 0.004    # horizontal bar half-height
_KING_TOP_Z        = 0.0585   # z centre of top cross arm
_KING_TOP_HX       = 0.012    # top cross arm half-width
_KING_TOP_HY       = 0.004    # top cross arm half-depth
_KING_TOP_HZ       = 0.003    # top cross arm half-height


def _replace_marked_fragment(
    text: str, start_marker: str, end_marker: str, fragment: str
) -> str:
    """Run  replace marked fragment logic."""
    start = text.index(start_marker)
    end = text.index(end_marker) + len(end_marker)
    return text[:start] + fragment + text[end:]


def build_board_fragment() -> str:
    """Run build board fragment logic."""
    mapper = BoardMapper.from_configs()
    env_cfg = load_config("env")
    table_cx, table_cy = env_cfg["table_center_xy"]
    square_half = mapper.geometry.cell_size_m / 2.0
    z = mapper.geometry.table_surface_z + 0.0006

    lines = [BOARD_START_MARKER]
    for square in chess.SQUARES:
        name = chess.square_name(square)
        xy = mapper.square_to_xy(square)
        local_x = xy[0] - table_cx
        local_y = xy[1] - table_cy
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        material = (
            "chess_dark_square_mat"
            if (rank + file) % 2 == 0
            else "chess_light_square_mat"
        )
        lines.append(
            "\t\t\t"
            f'<geom name="board_{name}_visual" type="box" '
            f'size="{square_half:.3f} {square_half:.3f} 0.0005" '
            f'pos="{local_x:.4f} {local_y:.4f} {z:.4f}" '
            f'material="{material}" contype="0" conaffinity="0" mass="0"/>'
        )
    lines.append(BOARD_END_MARKER)
    return "\n".join(lines)


def knight_visual_euler(color: str) -> str:
    """Run knight visual euler logic."""
    return "0 0 1.5707963268" if color == "white" else "0 0 4.7123889804"


def piece_body_xml(
    piece_id: str,
    color: str,
    piece_type: str,
    xyz,
    *,
    initial_square: str | None = None,
    visual_euler: str | None = None,
) -> list[str]:
    """Run piece body xml logic."""
    body_name = f"piece_{piece_id}"
    material_prefix = "white" if color == "white" else "black"
    damping = load_config("chess")["pieces"]["freejoint_damping"]
    comment = f" <!-- {initial_square} -->" if initial_square else ""
    euler_attr = f' euler="{visual_euler}"' if visual_euler else ""
    return [
        f'\t\t<body name="{body_name}" pos="{xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}">{comment}',
        f'\t\t\t<joint name="{body_name}:joint" type="free" damping="{damping}"/>',
        f'\t\t\t<geom name="{body_name}_cube" type="box" size="0.015 0.015 0.015"',
        f'\t\t\t      material="{material_prefix}_piece_cube_mat" rgba="0 0 0 0" mass="0.35" condim="6"',
        '\t\t\t      friction="2.0 0.005 0.0001" solref="0.002 1" solimp="0.99 0.999 0.001"/>',
        f'\t\t\t<geom name="{body_name}_visual" type="mesh" mesh="chess_{piece_type}_mesh"',
        f'\t\t\t      pos="0 0 -0.016" material="{material_prefix}_piece_visual_mat"',
        f'\t\t\t      contype="0" conaffinity="0" mass="0"{euler_attr}/>',
        f'\t\t\t<site name="{body_name}_site" pos="0 0 0" size="0.005"/>',
        "\t\t</body>",
    ]


def reserve_position(
    index: int, color: str, piece_type: str
) -> tuple[float, float, float]:
    """Run reserve position logic."""
    cfg = load_config("chess")
    reserve_cfg = cfg["promotion_reserve"][color]
    spacing = cfg["reserves"]["promotion_slot_spacing_m"]
    origin = reserve_cfg["origin_xyz"]
    piece_type_offset = {"queen": 0, "rook": 8, "bishop": 16, "knight": 24}[piece_type]
    slot = piece_type_offset + index
    row = slot // reserve_cfg["cols"]
    col = slot % reserve_cfg["cols"]
    return origin[0] + row * spacing, origin[1] + col * spacing, origin[2]


def build_pieces_fragment() -> str:
    """Run build pieces fragment logic."""
    mapper = BoardMapper.from_configs()
    registry = PieceRegistry()
    lines = [PIECES_START_MARKER]

    for piece in registry.all_pieces():
        xyz = mapper.square_to_piece_xyz(chess.parse_square(piece.initial_square))
        visual_euler = (
            knight_visual_euler(piece.color) if piece.piece_type == "knight" else None
        )
        lines.extend(
            piece_body_xml(
                piece.piece_id,
                piece.color,
                piece.piece_type,
                xyz,
                initial_square=piece.initial_square,
                visual_euler=visual_euler,
            )
        )

    for piece_id, color, piece_type in reserve_piece_ids():
        reserve_index = int(piece_id.rsplit("_", 1)[1]) - 1
        xyz = reserve_position(reserve_index, color, piece_type)
        visual_euler = knight_visual_euler(color) if piece_type == "knight" else None
        lines.extend(
            piece_body_xml(piece_id, color, piece_type, xyz, visual_euler=visual_euler)
        )

    lines.append(PIECES_END_MARKER)
    return "\n".join(lines)


ZONE_SPECS = [
    ("white_graveyard", "graveyards", "white", "white_graveyard_mat"),
    ("black_graveyard", "graveyards", "black", "black_graveyard_mat"),
    ("white_reserve", "promotion_reserve", "white", "white_reserve_mat"),
    ("black_reserve", "promotion_reserve", "black", "black_reserve_mat"),
]


def _zone_geom(name: str, cfg: dict, spacing: float, material: str) -> str:
    """Run  zone geom logic."""
    origin = cfg["origin_xyz"]
    rows = cfg["rows"]
    cols = cfg["cols"]
    margin = 0.005
    half_x = (rows * spacing) / 2.0 + margin
    half_y = (cols * spacing) / 2.0 + margin
    center_x = origin[0] + (rows - 1) * spacing / 2.0
    center_y = origin[1] + (cols - 1) * spacing / 2.0
    center_z = 0.002
    return (
        f'\t\t<geom name="zone_{name}" type="box" '
        f'size="{half_x:.4f} {half_y:.4f} 0.002" '
        f'pos="{center_x:.4f} {center_y:.4f} {center_z:.4f}" '
        f'material="{material}" contype="0" conaffinity="0" mass="0"/>'
    )


def build_zones_fragment() -> str:
    """Run build zones fragment logic."""
    cfg = load_config("chess")
    graveyard_spacing = cfg["reserves"]["graveyard_slot_spacing_m"]
    promotion_spacing = cfg["reserves"]["promotion_slot_spacing_m"]
    lines = [ZONES_START_MARKER]
    for short_name, section_key, color, material in ZONE_SPECS:
        zone_cfg = cfg[section_key][color]
        spacing = promotion_spacing if "reserve" in short_name else graveyard_spacing
        lines.append(_zone_geom(short_name, zone_cfg, spacing, material))
    lines.append(ZONES_END_MARKER)
    return "\n".join(lines)


def regenerate_scene(scene_path: Path = DEFAULT_SCENE_PATH) -> Path:
    """Regenerate board, zone, and piece XML fragments in a scene file."""
    text = scene_path.read_text()
    text = _replace_marked_fragment(
        text, BOARD_START_MARKER, BOARD_END_MARKER, build_board_fragment()
    )
    text = _replace_marked_fragment(
        text, ZONES_START_MARKER, ZONES_END_MARKER, build_zones_fragment()
    )
    text = _replace_marked_fragment(
        text, PIECES_START_MARKER, PIECES_END_MARKER, build_pieces_fragment()
    )
    scene_path.write_text(text)
    return scene_path


def ensure_environment_generated(scene_path: Path = DEFAULT_SCENE_PATH) -> None:
    """Regenerate scene XML if generated content differs from current config."""
    scene_path = Path(scene_path).resolve()
    before = scene_path.read_text()
    after = _replace_marked_fragment(
        before, BOARD_START_MARKER, BOARD_END_MARKER, build_board_fragment()
    )
    after = _replace_marked_fragment(
        after, ZONES_START_MARKER, ZONES_END_MARKER, build_zones_fragment()
    )
    after = _replace_marked_fragment(
        after, PIECES_START_MARKER, PIECES_END_MARKER, build_pieces_fragment()
    )
    if after != before:
        scene_path.write_text(after)


def update_board_scene(scene_path: Path = DEFAULT_SCENE_PATH) -> Path:
    """Run update board scene logic."""
    text = scene_path.read_text()
    scene_path.write_text(
        _replace_marked_fragment(
            text, BOARD_START_MARKER, BOARD_END_MARKER, build_board_fragment()
        )
    )
    return scene_path


def update_pieces_scene(scene_path: Path = DEFAULT_SCENE_PATH) -> Path:
    """Run update pieces scene logic."""
    text = scene_path.read_text()
    scene_path.write_text(
        _replace_marked_fragment(
            text, PIECES_START_MARKER, PIECES_END_MARKER, build_pieces_fragment()
        )
    )
    return scene_path


def update_zones_scene(scene_path: Path = DEFAULT_SCENE_PATH) -> Path:
    """Run update zones scene logic."""
    text = scene_path.read_text()
    scene_path.write_text(
        _replace_marked_fragment(
            text, ZONES_START_MARKER, ZONES_END_MARKER, build_zones_fragment()
        )
    )
    return scene_path


def normal(a, b, c):
    """Run normal logic."""
    ux, uy, uz = (b[i] - a[i] for i in range(3))
    vx, vy, vz = (c[i] - a[i] for i in range(3))
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / length, ny / length, nz / length


def box_triangles(center, size):
    """Run box triangles logic."""
    cx, cy, cz = center
    sx, sy, sz = (dim / 2.0 for dim in size)
    v = {
        "000": (cx - sx, cy - sy, cz - sz),
        "001": (cx - sx, cy - sy, cz + sz),
        "010": (cx - sx, cy + sy, cz - sz),
        "011": (cx - sx, cy + sy, cz + sz),
        "100": (cx + sx, cy - sy, cz - sz),
        "101": (cx + sx, cy - sy, cz + sz),
        "110": (cx + sx, cy + sy, cz - sz),
        "111": (cx + sx, cy + sy, cz + sz),
    }
    return [
        (v["000"], v["100"], v["110"]),
        (v["000"], v["110"], v["010"]),
        (v["001"], v["011"], v["111"]),
        (v["001"], v["111"], v["101"]),
        (v["000"], v["001"], v["101"]),
        (v["000"], v["101"], v["100"]),
        (v["010"], v["110"], v["111"]),
        (v["010"], v["111"], v["011"]),
        (v["000"], v["010"], v["011"]),
        (v["000"], v["011"], v["001"]),
        (v["100"], v["101"], v["111"]),
        (v["100"], v["111"], v["110"]),
    ]


def frustum_triangles(r_bottom, r_top, z_bottom, z_top, segments):
    """Run frustum triangles logic."""
    angles = [2 * math.pi * i / segments for i in range(segments)]
    bot = [(r_bottom * math.cos(a), r_bottom * math.sin(a), z_bottom) for a in angles]
    top = [(r_top * math.cos(a), r_top * math.sin(a), z_top) for a in angles]
    bc = (0.0, 0.0, z_bottom)
    tc = (0.0, 0.0, z_top)
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        tris.append((bc, bot[j], bot[i]))
        tris.append((tc, top[i], top[j]))
        tris.append((bot[i], bot[j], top[j]))
        tris.append((bot[i], top[j], top[i]))
    return tris


def cone_triangles(r_base, z_base, z_tip, segments):
    """Run cone triangles logic."""
    tip = (0.0, 0.0, z_tip)
    angles = [2 * math.pi * i / segments for i in range(segments)]
    base = [(r_base * math.cos(a), r_base * math.sin(a), z_base) for a in angles]
    bc = (0.0, 0.0, z_base)
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        tris.append((bc, base[j], base[i]))
        tris.append((base[i], base[j], tip))
    return tris


def offset_tris(tris, dx, dy, dz=0.0):
    """Run offset tris logic."""
    return [tuple((x + dx, y + dy, z + dz) for x, y, z in tri) for tri in tris]


def sphere_triangles(center, radius, rings=5, segments=20, z_scale=1.0):
    """Run sphere triangles logic."""
    cx, cy, cz = center
    vertices = []
    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        rr = radius * math.sin(phi)
        z = cz + radius * math.cos(phi) * z_scale
        vertices.append(
            [
                (
                    cx + rr * math.cos(2 * math.pi * i / segments),
                    cy + rr * math.sin(2 * math.pi * i / segments),
                    z,
                )
                for i in range(segments)
            ]
        )

    tris = []
    for ring in range(rings):
        for i in range(segments):
            j = (i + 1) % segments
            a, b = vertices[ring][i], vertices[ring][j]
            c, d = vertices[ring + 1][i], vertices[ring + 1][j]
            if ring == 0:
                tris.append((a, d, c))
            elif ring == rings - 1:
                tris.append((a, b, c))
            else:
                tris.append((a, b, d))
                tris.append((a, d, c))
    return tris


def piece_base():
    """Run piece base logic."""
    tris = frustum_triangles(0.019, BASE_R, 0.000, 0.003, SEGS)
    tris += frustum_triangles(BASE_R, BASE_R, 0.003, 0.006, SEGS)
    tris += frustum_triangles(BASE_R, 0.018, 0.006, 0.009, SEGS)
    tris += frustum_triangles(0.018, 0.013, 0.009, 0.024, SEGS)
    tris += frustum_triangles(0.013, 0.015, 0.024, 0.026, SEGS)
    tris += frustum_triangles(0.015, 0.012, 0.026, 0.029, SEGS)
    tris += frustum_triangles(0.012, WAIST_R, 0.029, BASE_TOP, SEGS)
    return tris


def pawn_triangles():
    """Return the triangle mesh for a pawn piece."""
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, _PAWN_NECK_R, BASE_TOP, _PAWN_NECK_Z, 24)
    tris += frustum_triangles(
        _PAWN_NECK_R, _PAWN_HEAD_COLLAR_R, _PAWN_NECK_Z, _PAWN_HEAD_COLLAR_Z, 24
    )
    tris += sphere_triangles(
        (0.0, 0.0, _PAWN_HEAD_Z),
        _PAWN_HEAD_R,
        rings=6,
        segments=24,
        z_scale=_PAWN_HEAD_Z_SCALE,
    )
    return tris


def rook_triangles():
    """Return the triangle mesh for a rook piece."""
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, _ROOK_SHAFT_R, BASE_TOP, _ROOK_SHAFT_Z, 24)
    tris += frustum_triangles(_ROOK_SHAFT_R, _ROOK_COLLAR_R, _ROOK_SHAFT_Z, _ROOK_COLLAR_Z, 24)
    tris += frustum_triangles(_ROOK_COLLAR_R, _ROOK_COLLAR_R, _ROOK_COLLAR_Z, _ROOK_PARAPET_Z, 24)
    tris += frustum_triangles(_ROOK_INNER_R, _ROOK_INNER_R, _ROOK_PARAPET_Z, _ROOK_TOP_Z, 24)
    for dx, dy in [
        (_ROOK_BATTLEMENT_OFFSET, 0.0),
        (-_ROOK_BATTLEMENT_OFFSET, 0.0),
        (0.0, _ROOK_BATTLEMENT_OFFSET),
        (0.0, -_ROOK_BATTLEMENT_OFFSET),
    ]:
        tris += offset_tris(
            box_triangles(
                (0, 0, _ROOK_BATTLEMENT_Z),
                (_ROOK_BATTLEMENT_HX, _ROOK_BATTLEMENT_HY, _ROOK_BATTLEMENT_HZ),
            ),
            dx,
            dy,
        )
    for dx, dy in [
        (_ROOK_CORNER_OFFSET, _ROOK_CORNER_OFFSET),
        (_ROOK_CORNER_OFFSET, -_ROOK_CORNER_OFFSET),
        (-_ROOK_CORNER_OFFSET, _ROOK_CORNER_OFFSET),
        (-_ROOK_CORNER_OFFSET, -_ROOK_CORNER_OFFSET),
    ]:
        tris += offset_tris(
            box_triangles(
                (0, 0, _ROOK_CORNER_Z),
                (_ROOK_CORNER_HXY, _ROOK_CORNER_HXY, _ROOK_CORNER_HZ),
            ),
            dx,
            dy,
        )
    return tris


def knight_triangles():
    """Return the triangle mesh for a knight piece."""
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, _KNIGHT_SHAFT_NECK_R, BASE_TOP, _KNIGHT_SHAFT_Z, 20)
    tris += box_triangles(
        (_KNIGHT_BODY_X, 0.0, _KNIGHT_BODY_Z),
        (_KNIGHT_BODY_HX, _KNIGHT_BODY_HY, _KNIGHT_BODY_HZ),
    )
    tris += box_triangles(
        (_KNIGHT_NECK_X, 0.0, _KNIGHT_NECK_Z),
        (_KNIGHT_NECK_HX, _KNIGHT_NECK_HY, _KNIGHT_NECK_HZ),
    )
    tris += box_triangles(
        (_KNIGHT_EAR_X, 0.0, _KNIGHT_EAR_Z),
        (_KNIGHT_EAR_HX, _KNIGHT_EAR_HY, _KNIGHT_EAR_HZ),
    )
    tris += box_triangles(
        (_KNIGHT_BODY_X, 0.0, _KNIGHT_SNOUT_Z),
        (_KNIGHT_SNOUT_HX, _KNIGHT_SNOUT_HY, _KNIGHT_SNOUT_HZ),
    )
    tris += box_triangles(
        (_KNIGHT_NOSTRIL_X, _KNIGHT_NOSTRIL_Y, _KNIGHT_NOSTRIL_Z),
        (_KNIGHT_NOSTRIL_HXY, _KNIGHT_NOSTRIL_HXY, _KNIGHT_NOSTRIL_HZ),
    )
    tris += box_triangles(
        (_KNIGHT_NOSTRIL_X, -_KNIGHT_NOSTRIL_Y, _KNIGHT_NOSTRIL_Z),
        (_KNIGHT_NOSTRIL_HXY, _KNIGHT_NOSTRIL_HXY, _KNIGHT_NOSTRIL_HZ),
    )
    tris += box_triangles(
        (_KNIGHT_MANE_X, 0.0, _KNIGHT_MANE_Z),
        (_KNIGHT_MANE_HX, _KNIGHT_MANE_HY, _KNIGHT_MANE_HZ),
    )
    return tris


def bishop_triangles():
    """Return the triangle mesh for a bishop piece."""
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, _BISHOP_SHAFT_NECK_R, BASE_TOP, _BISHOP_SHAFT_Z, 24)
    tris += frustum_triangles(
        _BISHOP_SHAFT_NECK_R, _BISHOP_BODY_R, _BISHOP_SHAFT_Z, _BISHOP_BODY_Z, 24
    )
    tris += frustum_triangles(
        _BISHOP_BODY_R, _BISHOP_SHOULDER_R, _BISHOP_BODY_Z, _BISHOP_SHOULDER_Z, 24
    )
    tris += sphere_triangles(
        (0.0, 0.0, _BISHOP_HEAD_Z),
        _BISHOP_HEAD_R,
        rings=5,
        segments=24,
        z_scale=_BISHOP_HEAD_Z_SCALE,
    )
    tris += box_triangles(
        (_BISHOP_CROSS_X, 0.0, _BISHOP_CROSS_Z),
        (_BISHOP_CROSS_HX, _BISHOP_CROSS_HY, _BISHOP_CROSS_HZ),
    )
    tris += cone_triangles(_BISHOP_TIP_R, _BISHOP_TIP_Z, _BISHOP_TIP_APEX_Z, 16)
    return tris


def queen_triangles():
    """Return the triangle mesh for a queen piece."""
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, _QUEEN_SHAFT_NECK_R, BASE_TOP, _QUEEN_SHAFT_Z, 24)
    tris += frustum_triangles(
        _QUEEN_SHAFT_NECK_R, _QUEEN_BODY_R, _QUEEN_SHAFT_Z, _QUEEN_BODY_Z, 24
    )
    tris += frustum_triangles(
        _QUEEN_BODY_R, _QUEEN_SHOULDER_R, _QUEEN_BODY_Z, _QUEEN_SHOULDER_Z, 24
    )
    tris += frustum_triangles(
        _QUEEN_SHOULDER_R, _QUEEN_NECK_R, _QUEEN_SHOULDER_Z, _QUEEN_NECK_Z, 24
    )
    for i in range(5):
        angle = 2 * math.pi * i / 5
        cx = _QUEEN_CROWN_ORBIT_R * math.cos(angle)
        cy = _QUEEN_CROWN_ORBIT_R * math.sin(angle)
        tris += offset_tris(
            cone_triangles(_QUEEN_CROWN_SPIKE_R, _QUEEN_NECK_Z, _QUEEN_CROWN_TIP_Z, 12),
            cx,
            cy,
        )
    tris += sphere_triangles(
        (0.0, 0.0, _QUEEN_ORB_Z),
        _QUEEN_ORB_R,
        rings=4,
        segments=16,
        z_scale=_QUEEN_ORB_Z_SCALE,
    )
    return tris


def king_triangles():
    """Return the triangle mesh for a king piece."""
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, _KING_SHAFT_NECK_R, BASE_TOP, _KING_SHAFT_Z, 24)
    tris += frustum_triangles(
        _KING_SHAFT_NECK_R, _KING_BODY_R, _KING_SHAFT_Z, _KING_BODY_Z, 24
    )
    tris += frustum_triangles(
        _KING_BODY_R, _KING_SHOULDER_R, _KING_BODY_Z, _KING_SHOULDER_Z, 24
    )
    tris += frustum_triangles(
        _KING_SHOULDER_R, _KING_SPIRE_R, _KING_SHOULDER_Z, _KING_SPIRE_Z, 20
    )
    tris += box_triangles(
        (0.0, 0.0, _KING_CROSS_V_Z),
        (_KING_CROSS_V_H, _KING_CROSS_V_H, _KING_CROSS_V_HZ),
    )
    tris += box_triangles(
        (0.0, 0.0, _KING_CROSS_H_Z),
        (_KING_CROSS_H_HX, _KING_CROSS_H_HY, _KING_CROSS_H_HZ),
    )
    tris += box_triangles(
        (0.0, 0.0, _KING_TOP_Z),
        (_KING_TOP_HX, _KING_TOP_HY, _KING_TOP_HZ),
    )
    return tris


PIECE_BUILDERS = {
    "pawn": pawn_triangles,
    "rook": rook_triangles,
    "knight": knight_triangles,
    "bishop": bishop_triangles,
    "queen": queen_triangles,
    "king": king_triangles,
}


def write_binary_stl(path: Path, name: str, triangles) -> None:
    """Run write binary stl logic."""
    header = f"RoboChess {name} visual mesh".encode("ascii")[:80].ljust(80, b"\0")
    with path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            nx, ny, nz = normal(*tri)
            values = [nx, ny, nz]
            for vertex in tri:
                values.extend(vertex)
            f.write(struct.pack("<12fH", *values, 0))


def regenerate_stls(out_dir: Path = DEFAULT_STL_DIR) -> Path:
    """Run regenerate stls logic."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in PIECE_BUILDERS.items():
        write_binary_stl(out_dir / f"{name}.stl", name, builder())
    return out_dir


def regenerate_environment(
    scene_path: Path = DEFAULT_SCENE_PATH,
    stl_dir: Path = DEFAULT_STL_DIR,
    *,
    include_stls: bool = False,
) -> None:
    """Run regenerate environment logic."""
    if include_stls:
        regenerate_stls(stl_dir)
    regenerate_scene(scene_path)
