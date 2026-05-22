from pathlib import Path

from src.chess_env.environment_generation import regenerate_scene


def test_regenerate_scene_updates_all_chess_fragments(tmp_path):
    scene_copy = tmp_path / "pick_and_place.xml"
    scene_copy.write_text(Path("chess_env/assets/pick_and_place.xml").read_text())

    regenerate_scene(scene_copy)
    text = scene_copy.read_text()

    assert text.count('name="board_') == 64
    assert text.count('name="zone_') == 4
    assert text.count('<body name="piece_') == 96
