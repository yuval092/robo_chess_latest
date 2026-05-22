from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameStatus:
    turn: str
    is_check: bool
    is_game_over: bool
    is_checkmate: bool
    is_stalemate: bool
    is_insufficient_material: bool
    is_seventyfive_moves: bool
    is_fivefold_repetition: bool
    can_claim_fifty_moves: bool
    can_claim_threefold_repetition: bool
    outcome: str | None
    fen: str
    legal_moves: list[str]
