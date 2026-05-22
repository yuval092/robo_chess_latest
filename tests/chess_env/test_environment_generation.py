from pathlib import Path
import struct

from src.chess_env.environment_generation import regenerate_scene, regenerate_stls


def test_regenerate_scene_updates_all_chess_fragments(tmp_path):
    scene_copy = tmp_path / "pick_and_place.xml"
    scene_copy.write_text(Path("chess_env/assets/pick_and_place.xml").read_text())

    regenerate_scene(scene_copy)
    text = scene_copy.read_text()

    assert text.count('name="board_') == 64
    assert text.count('name="zone_') == 4
    assert text.count('<body name="piece_') == 96


def test_generated_stls_are_detailed_and_below_hover_clearance(tmp_path):
    out_dir = tmp_path / "stls"
    regenerate_stls(out_dir)

    for path in out_dir.glob("*.stl"):
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
