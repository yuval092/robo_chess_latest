from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalPiece:
    piece_id: str
    color: str
    piece_type: str
    body_name: str
    joint_name: str
    cube_geom_name: str
    visual_geom_name: str
    initial_square: str | None


class PieceRegistry:
    """Deterministic registry for the 32 active chess pieces."""

    BACK_RANK = (
        ("a", "rook"),
        ("b", "knight"),
        ("c", "bishop"),
        ("d", "queen"),
        ("e", "king"),
        ("f", "bishop"),
        ("g", "knight"),
        ("h", "rook"),
    )
    FILES = "abcdefgh"

    def __init__(self):
        self._pieces = self._build_active_pieces()
        self._by_id = {piece.piece_id: piece for piece in self._pieces}

    def all_pieces(self) -> list[PhysicalPiece]:
        return list(self._pieces)

    def by_id(self, piece_id: str) -> PhysicalPiece:
        try:
            return self._by_id[piece_id]
        except KeyError as exc:
            raise KeyError(f"Unknown physical piece id: {piece_id}") from exc

    def starting_square_map(self) -> dict[str, str]:
        return {piece.piece_id: piece.initial_square for piece in self._pieces if piece.initial_square is not None}

    def ids_for_color(self, color: str) -> list[str]:
        return [piece.piece_id for piece in self._pieces if piece.color == color]

    @classmethod
    def _build_active_pieces(cls) -> list[PhysicalPiece]:
        pieces: list[PhysicalPiece] = []
        for color, back_rank, pawn_rank in (("white", "1", "2"), ("black", "8", "7")):
            for file_name, piece_type in cls.BACK_RANK:
                piece_id = cls._piece_id(color, piece_type, file_name)
                pieces.append(cls._make_piece(piece_id, color, piece_type, f"{file_name}{back_rank}"))
            for file_name in cls.FILES:
                piece_id = f"{color}_pawn_{file_name}"
                pieces.append(cls._make_piece(piece_id, color, "pawn", f"{file_name}{pawn_rank}"))
        return pieces

    @staticmethod
    def _piece_id(color: str, piece_type: str, file_name: str) -> str:
        if piece_type in {"king", "queen"}:
            return f"{color}_{piece_type}"
        return f"{color}_{piece_type}_{file_name}"

    @staticmethod
    def _make_piece(piece_id: str, color: str, piece_type: str, initial_square: str | None) -> PhysicalPiece:
        body_name = f"piece_{piece_id}"
        return PhysicalPiece(
            piece_id=piece_id,
            color=color,
            piece_type=piece_type,
            body_name=body_name,
            joint_name=f"{body_name}:joint",
            cube_geom_name=f"{body_name}_cube",
            visual_geom_name=f"{body_name}_visual",
            initial_square=initial_square,
        )


def reserve_piece_ids() -> list[tuple[str, str, str]]:
    """Return reserve ids as (piece_id, color, piece_type)."""
    ids: list[tuple[str, str, str]] = []
    for color in ("white", "black"):
        for piece_type in ("queen", "rook", "bishop", "knight"):
            for index in range(1, 9):
                ids.append((f"{color}_reserve_{piece_type}_{index}", color, piece_type))
    return ids
