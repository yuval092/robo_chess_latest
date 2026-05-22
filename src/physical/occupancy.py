from __future__ import annotations


class PhysicalOccupancy:
    """Expected physical square occupancy for active board pieces."""

    def __init__(self, starting_square_map: dict[str, str] | None = None):
        self._piece_to_square: dict[str, str | None] = dict(starting_square_map or {})

    def piece_at_square(self, square: str) -> str | None:
        for piece_id, piece_square in self._piece_to_square.items():
            if piece_square == square:
                return piece_id
        return None

    def square_of_piece(self, piece_id: str) -> str | None:
        return self._piece_to_square.get(piece_id)

    def set_piece_square(self, piece_id: str, square: str | None) -> None:
        if square is not None:
            current = self.piece_at_square(square)
            if current is not None and current != piece_id:
                raise ValueError(f"Square {square} is already occupied by {current}")
        self._piece_to_square[piece_id] = square

    def reset(self, starting_square_map: dict[str, str] | None = None) -> None:
        self._piece_to_square = dict(starting_square_map or {})

    def assert_square_empty(self, square: str) -> None:
        current = self.piece_at_square(square)
        if current is not None:
            raise ValueError(f"Square {square} is occupied by {current}")

    def assert_piece_at(self, piece_id: str, square: str) -> None:
        actual = self.square_of_piece(piece_id)
        if actual != square:
            raise ValueError(f"Expected {piece_id} at {square}, found {actual}")
