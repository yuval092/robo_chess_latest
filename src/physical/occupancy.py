"""Expected physical occupancy tracker for chess pieces."""

from __future__ import annotations


class PhysicalOccupancy:
    """Expected physical square occupancy for active board pieces."""

    def __init__(self, starting_square_map: dict[str, str] | None = None):
        """Initialise this object."""
        self._piece_to_square: dict[str, str | None] = dict(starting_square_map or {})
        self._square_to_piece: dict[str, str] = {
            sq: pid for pid, sq in self._piece_to_square.items() if sq is not None
        }

    def piece_at_square(self, square: str) -> str | None:
        """Run piece at square logic."""
        return self._square_to_piece.get(square)

    def square_of_piece(self, piece_id: str) -> str | None:
        """Run square of piece logic."""
        return self._piece_to_square.get(piece_id)

    def set_piece_square(self, piece_id: str, square: str | None) -> None:
        """Run set piece square logic."""
        if square is not None:
            current = self._square_to_piece.get(square)
            if current is not None and current != piece_id:
                raise ValueError(f"Square {square} is already occupied by {current}")
        old_sq = self._piece_to_square.get(piece_id)
        if old_sq is not None:
            self._square_to_piece.pop(old_sq, None)
        self._piece_to_square[piece_id] = square
        if square is not None:
            self._square_to_piece[square] = piece_id

    def reset(self, starting_square_map: dict[str, str] | None = None) -> None:
        """Run reset logic."""
        self._piece_to_square = dict(starting_square_map or {})
        self._square_to_piece = {
            sq: pid for pid, sq in self._piece_to_square.items() if sq is not None
        }

    def assert_square_empty(self, square: str) -> None:
        """Run assert square empty logic."""
        current = self.piece_at_square(square)
        if current is not None:
            raise ValueError(f"Square {square} is occupied by {current}")

    def assert_piece_at(self, piece_id: str, square: str) -> None:
        """Run assert piece at logic."""
        actual = self.square_of_piece(piece_id)
        if actual != square:
            raise ValueError(f"Expected {piece_id} at {square}, found {actual}")
