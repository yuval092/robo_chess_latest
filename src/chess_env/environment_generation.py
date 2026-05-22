"""Generate MuJoCo chess-board XML fragments and visual STL meshes."""
from __future__ import annotations

import math
import struct
from pathlib import Path

import chess

from src.chess_game.board_mapper import BoardMapper
from src.physical.piece_registry import PieceRegistry, reserve_piece_ids
from src.utils.config import load_config


DEFAULT_SCENE_PATH = Path("chess_env/assets/pick_and_place.xml")
DEFAULT_STL_DIR = Path("chess_env/stls/chess")

BOARD_START_MARKER = "\t\t\t<!-- generated board squares start -->"
BOARD_END_MARKER = "\t\t\t<!-- generated board squares end -->"
PIECES_START_MARKER = "\t\t<!-- generated chess pieces start -->"
PIECES_END_MARKER = "\t\t<!-- generated chess pieces end -->"
ZONES_START_MARKER = "\t\t<!-- generated zone markers start -->"
ZONES_END_MARKER = "\t\t<!-- generated zone markers end -->"

BASE_R = 0.022
DISC_TOP = 0.006
TAPER_TOP = 0.030
WAIST_R = 0.010
BASE_TOP = 0.032
SEGS = 24


def _replace_marked_fragment(text: str, start_marker: str, end_marker: str, fragment: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker) + len(end_marker)
    return text[:start] + fragment + text[end:]


def build_board_fragment() -> str:
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
        material = "chess_dark_square_mat" if (rank + file) % 2 == 0 else "chess_light_square_mat"
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
        f'\t\t\t      friction="2.0 0.005 0.0001" solref="0.002 1" solimp="0.99 0.999 0.001"/>',
        f'\t\t\t<geom name="{body_name}_visual" type="mesh" mesh="chess_{piece_type}_mesh"',
        f'\t\t\t      pos="0 0 -0.016" material="{material_prefix}_piece_visual_mat"',
        f'\t\t\t      contype="0" conaffinity="0" mass="0"{euler_attr}/>',
        f'\t\t\t<site name="{body_name}_site" pos="0 0 0" size="0.005"/>',
        "\t\t</body>",
    ]


def reserve_position(index: int, color: str, piece_type: str) -> tuple[float, float, float]:
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
    mapper = BoardMapper.from_configs()
    registry = PieceRegistry()
    lines = [PIECES_START_MARKER]

    for piece in registry.all_pieces():
        xyz = mapper.square_to_piece_xyz(chess.parse_square(piece.initial_square))
        visual_euler = knight_visual_euler(piece.color) if piece.piece_type == "knight" else None
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
        lines.extend(piece_body_xml(piece_id, color, piece_type, xyz, visual_euler=visual_euler))

    lines.append(PIECES_END_MARKER)
    return "\n".join(lines)


ZONE_SPECS = [
    ("white_graveyard", "graveyards", "white", "white_graveyard_mat"),
    ("black_graveyard", "graveyards", "black", "black_graveyard_mat"),
    ("white_reserve", "promotion_reserve", "white", "white_reserve_mat"),
    ("black_reserve", "promotion_reserve", "black", "black_reserve_mat"),
]


def _zone_geom(name: str, cfg: dict, spacing: float, material: str) -> str:
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
    text = scene_path.read_text()
    text = _replace_marked_fragment(text, BOARD_START_MARKER, BOARD_END_MARKER, build_board_fragment())
    text = _replace_marked_fragment(text, ZONES_START_MARKER, ZONES_END_MARKER, build_zones_fragment())
    text = _replace_marked_fragment(text, PIECES_START_MARKER, PIECES_END_MARKER, build_pieces_fragment())
    scene_path.write_text(text)
    return scene_path


def update_board_scene(scene_path: Path = DEFAULT_SCENE_PATH) -> Path:
    text = scene_path.read_text()
    scene_path.write_text(
        _replace_marked_fragment(text, BOARD_START_MARKER, BOARD_END_MARKER, build_board_fragment())
    )
    return scene_path


def update_pieces_scene(scene_path: Path = DEFAULT_SCENE_PATH) -> Path:
    text = scene_path.read_text()
    scene_path.write_text(
        _replace_marked_fragment(text, PIECES_START_MARKER, PIECES_END_MARKER, build_pieces_fragment())
    )
    return scene_path


def update_zones_scene(scene_path: Path = DEFAULT_SCENE_PATH) -> Path:
    text = scene_path.read_text()
    scene_path.write_text(
        _replace_marked_fragment(text, ZONES_START_MARKER, ZONES_END_MARKER, build_zones_fragment())
    )
    return scene_path


def normal(a, b, c):
    ux, uy, uz = (b[i] - a[i] for i in range(3))
    vx, vy, vz = (c[i] - a[i] for i in range(3))
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / length, ny / length, nz / length


def box_triangles(center, size):
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
        (v["000"], v["100"], v["110"]), (v["000"], v["110"], v["010"]),
        (v["001"], v["011"], v["111"]), (v["001"], v["111"], v["101"]),
        (v["000"], v["001"], v["101"]), (v["000"], v["101"], v["100"]),
        (v["010"], v["110"], v["111"]), (v["010"], v["111"], v["011"]),
        (v["000"], v["010"], v["011"]), (v["000"], v["011"], v["001"]),
        (v["100"], v["101"], v["111"]), (v["100"], v["111"], v["110"]),
    ]


def frustum_triangles(r_bottom, r_top, z_bottom, z_top, segments):
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
    return [tuple((x + dx, y + dy, z + dz) for x, y, z in tri) for tri in tris]


def piece_base():
    tris = frustum_triangles(BASE_R, BASE_R, 0.000, DISC_TOP, SEGS)
    tris += frustum_triangles(BASE_R, WAIST_R, DISC_TOP, TAPER_TOP, SEGS)
    tris += frustum_triangles(WAIST_R, WAIST_R, TAPER_TOP, BASE_TOP, SEGS)
    return tris


def pawn_triangles():
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, 0.007, BASE_TOP, 0.038, 16)
    tris += frustum_triangles(0.007, 0.015, 0.038, 0.044, 14)
    tris += frustum_triangles(0.015, 0.015, 0.044, 0.047, 14)
    tris += cone_triangles(0.015, 0.047, 0.050, 14)
    return tris


def rook_triangles():
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, 0.012, BASE_TOP, 0.042, 16)
    tris += frustum_triangles(0.012, 0.018, 0.042, 0.044, 16)
    tris += frustum_triangles(0.018, 0.018, 0.044, 0.046, 16)
    for dx, dy in [(0.013, 0.0), (-0.013, 0.0), (0.0, 0.013), (0.0, -0.013)]:
        tris += offset_tris(box_triangles((0, 0, 0.049), (0.010, 0.010, 0.006)), dx, dy)
    return tris


def knight_triangles():
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, 0.007, BASE_TOP, 0.040, 16)
    tris += box_triangles((0.009, 0.0, 0.047), (0.026, 0.016, 0.014))
    tris += box_triangles((0.018, 0.0, 0.042), (0.010, 0.009, 0.006))
    tris += box_triangles((0.006, 0.0, 0.054), (0.006, 0.006, 0.004))
    return tris


def bishop_triangles():
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, 0.008, BASE_TOP, 0.040, 16)
    tris += frustum_triangles(0.008, 0.011, 0.040, 0.042, 14)
    tris += frustum_triangles(0.011, 0.008, 0.042, 0.044, 14)
    tris += frustum_triangles(0.008, 0.004, 0.044, 0.052, 16)
    tris += cone_triangles(0.004, 0.052, 0.056, 12)
    return tris


def queen_triangles():
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, 0.008, BASE_TOP, 0.038, 16)
    tris += frustum_triangles(0.008, 0.014, 0.038, 0.046, 16)
    tris += frustum_triangles(0.014, 0.017, 0.046, 0.048, 16)
    for i in range(5):
        angle = 2 * math.pi * i / 5
        cx = 0.014 * math.cos(angle)
        cy = 0.014 * math.sin(angle)
        tris += offset_tris(cone_triangles(0.007, 0.048, 0.054, 10), cx, cy)
    tris += frustum_triangles(0.004, 0.006, 0.048, 0.051, 10)
    tris += cone_triangles(0.006, 0.051, 0.054, 10)
    return tris


def king_triangles():
    tris = piece_base()
    tris += frustum_triangles(WAIST_R, 0.008, BASE_TOP, 0.038, 16)
    tris += frustum_triangles(0.008, 0.013, 0.038, 0.048, 16)
    tris += frustum_triangles(0.013, 0.008, 0.048, 0.050, 12)
    tris += box_triangles((0.0, 0.0, 0.054), (0.006, 0.006, 0.008))
    tris += box_triangles((0.0, 0.0, 0.052), (0.022, 0.006, 0.006))
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
    if include_stls:
        regenerate_stls(stl_dir)
    regenerate_scene(scene_path)
