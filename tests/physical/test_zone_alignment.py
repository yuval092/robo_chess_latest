import pytest

from src.utils.config import load_config


@pytest.mark.parametrize(
    "color,section,is_reserve",
    [
        ("white", "graveyards", False),
        ("black", "graveyards", False),
        ("white", "promotion_reserve", True),
        ("black", "promotion_reserve", True),
    ],
)
def test_all_slots_inside_zone_extent(color, section, is_reserve):
    cfg = load_config("chess")
    zone_cfg = cfg[section][color]
    spacing = (
        cfg["reserves"]["promotion_slot_spacing_m"]
        if is_reserve
        else cfg["reserves"]["graveyard_slot_spacing_m"]
    )
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


def test_zones_do_not_overlap_board_y_range():
    cfg = load_config("chess")
    board_center_y = cfg["board"]["center_xy"][1]
    board_half = cfg["board"]["cell_size_m"] * cfg["board"]["board_size"] / 2.0
    board_min_y = board_center_y - board_half
    board_max_y = board_center_y + board_half

    for section in ("graveyards", "promotion_reserve"):
        for color in ("white", "black"):
            zone_cfg = cfg[section][color]
            spacing = (
                cfg["reserves"]["promotion_slot_spacing_m"]
                if section == "promotion_reserve"
                else cfg["reserves"]["graveyard_slot_spacing_m"]
            )
            origin_y = zone_cfg["origin_xyz"][1]
            zone_min_y = origin_y
            zone_max_y = origin_y + (zone_cfg["cols"] - 1) * spacing
            assert zone_max_y < board_min_y or zone_min_y > board_max_y
