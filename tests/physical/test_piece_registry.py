from src.physical.occupancy import PhysicalOccupancy
from src.physical.piece_registry import PieceRegistry, reserve_piece_ids


def test_active_registry_has_32_unique_pieces():
    registry = PieceRegistry()
    pieces = registry.all_pieces()

    assert len(pieces) == 32
    assert len({piece.piece_id for piece in pieces}) == 32
    assert len({piece.initial_square for piece in pieces}) == 32


def test_starting_square_map_contains_expected_pieces():
    registry = PieceRegistry()
    starts = registry.starting_square_map()

    assert starts["white_king"] == "e1"
    assert starts["white_queen"] == "d1"
    assert starts["white_pawn_e"] == "e2"
    assert starts["black_king"] == "e8"
    assert starts["black_pawn_e"] == "e7"


def test_ids_for_color():
    registry = PieceRegistry()

    assert len(registry.ids_for_color("white")) == 16
    assert len(registry.ids_for_color("black")) == 16


def test_reserve_piece_ids_cover_64_reserves():
    reserves = reserve_piece_ids()

    assert len(reserves) == 64
    assert len({piece_id for piece_id, _, _ in reserves}) == 64


def test_physical_occupancy_reset_restores_starting_map():
    starting = PieceRegistry().starting_square_map()
    occupancy = PhysicalOccupancy(starting)

    occupancy.set_piece_square("white_pawn_e", "e4")
    occupancy.reset(starting)

    assert occupancy.square_of_piece("white_pawn_e") == "e2"
    assert occupancy.piece_at_square("e4") is None
