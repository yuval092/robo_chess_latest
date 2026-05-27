"""Chess rules engine wrapper and game status model."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine


class IllegalMoveError(ValueError):
    pass


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


class ChessService:
    def __init__(
        self,
        starting_fen: str | None = None,
        engine_cfg: dict[str, Any] | None = None,
    ):
        """Initialise the chess service, optionally starting a Stockfish engine.

        engine_cfg keys (all optional):
          stockfish_path  — executable name or full path (default: "stockfish")
          skill_level     — 0–20 (default: 5)
          think_time_s    — seconds per move (default: 0.5)
        """
        self._board = chess.Board(starting_fen) if starting_fen else chess.Board()
        self._san_history: list[str] = []
        self._engine: chess.engine.SimpleEngine | None = None
        self._think_time: float = 0.5
        self._logger = logging.getLogger(__name__)

        if engine_cfg:
            self._think_time = float(engine_cfg.get("think_time_s", 0.5))
            path = engine_cfg.get("stockfish_path", "stockfish")
            if path:
                self._engine = chess.engine.SimpleEngine.popen_uci(path)
                skill = int(engine_cfg.get("skill_level", 5))
                self._engine.configure({"Skill Level": skill})

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Shut down the Stockfish process if one is running."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    # ── Board queries ─────────────────────────────────────────────────────────

    @property
    def board(self) -> chess.Board:
        """Return the underlying python-chess Board."""
        return self._board

    def legal_moves(self) -> list[str]:
        """Return all legal moves as UCI strings."""
        return [move.uci() for move in self._board.legal_moves]

    def validate_uci(self, uci: str) -> chess.Move:
        """Parse and validate a UCI move string."""
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise IllegalMoveError(f"Invalid UCI move: {uci}") from exc
        return self._validate_move(move)

    def validate_square_move(
        self, src: str, dst: str, promotion: str | None = None
    ) -> chess.Move:
        """Construct and validate a move from source and destination squares."""
        try:
            from_square = chess.parse_square(src)
            to_square = chess.parse_square(dst)
        except ValueError as exc:
            raise IllegalMoveError(f"Invalid square move: {src}->{dst}") from exc
        move = chess.Move(
            from_square, to_square, promotion=self._parse_promotion(promotion)
        )
        return self._validate_move(move)

    def push(self, move: chess.Move) -> None:
        """Commit a validated move to the board."""
        move = self._validate_move(move)
        self._san_history.append(self._board.san(move))
        self._board.push(move)

    def pop(self) -> chess.Move:
        """Undo the latest move."""
        move = self._board.pop()
        if self._san_history:
            self._san_history.pop()
        return move

    def status(self) -> GameStatus:
        """Return the current game status snapshot."""
        outcome = self._board.outcome(claim_draw=True)
        return GameStatus(
            turn="white" if self._board.turn == chess.WHITE else "black",
            is_check=self._board.is_check(),
            is_game_over=self._board.is_game_over(claim_draw=True),
            is_checkmate=self._board.is_checkmate(),
            is_stalemate=self._board.is_stalemate(),
            is_insufficient_material=self._board.is_insufficient_material(),
            is_seventyfive_moves=self._board.is_seventyfive_moves(),
            is_fivefold_repetition=self._board.is_fivefold_repetition(),
            can_claim_fifty_moves=self._board.can_claim_fifty_moves(),
            can_claim_threefold_repetition=self._board.can_claim_threefold_repetition(),
            outcome=outcome.result() if outcome else None,
            fen=self._board.fen(),
            legal_moves=self.legal_moves(),
        )

    def fen(self) -> str:
        """Return the current board FEN."""
        return self._board.fen()

    def san_history(self) -> list[str]:
        """Return the SAN move history."""
        return list(self._san_history)

    def piece_at(self, square: str) -> chess.Piece | None:
        """Return the piece at a square, if present."""
        return self._board.piece_at(chess.parse_square(square))

    def side_to_move(self) -> chess.Color:
        """Return the side whose turn it is."""
        return self._board.turn

    # ── Engine ────────────────────────────────────────────────────────────────

    def choose_engine_move(self) -> chess.Move:
        """Choose the computer's move using Stockfish."""
        if not list(self._board.legal_moves):
            raise IllegalMoveError("No legal moves available.")
        if self._engine is None:
            raise RuntimeError(
                "No chess engine configured. Pass engine_cfg when creating ChessService."
            )
        result = self._engine.play(
            self._board,
            chess.engine.Limit(time=self._think_time),
        )
        if result.move is None or result.move not in self._board.legal_moves:
            raise RuntimeError("Stockfish returned an invalid move.")
        return result.move

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_to_file(self, path: str) -> None:
        """Save the current game state to JSON."""
        data = {
            "fen": self.fen(),
            "san_history": self.san_history(),
            "half_move_clock": self._board.halfmove_clock,
            "full_move_number": self._board.fullmove_number,
        }
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=True))

    @classmethod
    def load_from_file(cls, path: str) -> ChessService:
        """Load a game state from JSON."""
        data = json.loads(Path(path).read_text())
        service = cls(data["fen"])
        service._san_history = list(data.get("san_history", []))
        return service

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_move(self, move: chess.Move) -> chess.Move:
        """Raise IllegalMoveError if the move is not legal in the current position."""
        if move not in self._board.legal_moves:
            raise IllegalMoveError(
                f"Illegal move {move.uci()} for position {self._board.fen()}"
            )
        return move

    @staticmethod
    def _parse_promotion(promotion: str | None) -> int | None:
        """Convert a promotion letter to a python-chess piece type constant."""
        if promotion is None:
            return None
        promotion_map = {
            "q": chess.QUEEN,
            "r": chess.ROOK,
            "b": chess.BISHOP,
            "n": chess.KNIGHT,
        }
        try:
            return promotion_map[promotion.lower()]
        except KeyError as exc:
            raise IllegalMoveError(f"Unsupported promotion piece: {promotion}") from exc
