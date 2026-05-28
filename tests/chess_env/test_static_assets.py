import struct
import xml.etree.ElementTree as ET
from pathlib import Path


SCENE_PATH = Path("chess_env/assets/pick_and_place.xml")
CHESS_STL_DIR = Path("chess_env/stls/chess")


def test_static_scene_contains_canonical_chess_sections() -> None:
    root = ET.parse(SCENE_PATH).getroot()

    board_geoms = [
        geom
        for geom in root.findall(".//geom")
        if geom.attrib.get("name", "").startswith("board_")
    ]
    zone_geoms = [
        geom
        for geom in root.findall(".//geom")
        if geom.attrib.get("name", "").startswith("zone_")
    ]
    piece_bodies = [
        body
        for body in root.findall(".//body")
        if body.attrib.get("name", "").startswith("piece_")
    ]

    assert len(board_geoms) == 64
    assert len(zone_geoms) == 4
    assert len(piece_bodies) == 96


def test_static_chess_stls_are_detailed_and_below_hover_clearance() -> None:
    for path in CHESS_STL_DIR.glob("*.stl"):
        data = path.read_bytes()
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        z_values = []
        offset = 84
        for _ in range(triangle_count):
            values = struct.unpack_from("<12fH", data, offset)
            z_values.extend(values[5:12:3])
            offset += 50

        assert triangle_count >= 1000
        assert max(z_values) <= 0.0605
