"""Chess rules engine wrapper and game status model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import chess


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


PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


class ChessService:
    def __init__(self, starting_fen: str | None = None):
        """Initialise this object."""
        self._board = chess.Board(starting_fen) if starting_fen else chess.Board()
        self._san_history: list[str] = []

    @property
    def board(self) -> chess.Board:
        """Run board logic."""
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

    def choose_engine_move(self) -> chess.Move:
        """Choose a legal move using the built-in heuristic."""
        legal_moves = list(self._board.legal_moves)
        if not legal_moves:
            raise IllegalMoveError("No legal moves available.")

        def score(move: chess.Move) -> tuple[int, int, int, int, str]:
            """Score a candidate engine move."""
            board = self._board
            captured = board.piece_at(move.to_square)
            if board.is_en_passant(move):
                captured = chess.Piece(chess.PAWN, not board.turn)
            captured_value = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0

            board.push(move)
            is_checkmate = board.is_checkmate()
            board.pop()

            gives_check = board.gives_check(move)
            promotion_value = (
                PIECE_VALUES.get(move.promotion, 0) if move.promotion else 0
            )
            return (
                1 if is_checkmate else 0,
                captured_value,
                promotion_value,
                1 if gives_check else 0,
                self._stable_move_tiebreak(move),
            )

        return max(legal_moves, key=score)

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

    def _validate_move(self, move: chess.Move) -> chess.Move:
        """Run  validate move logic."""
        if move not in self._board.legal_moves:
            raise IllegalMoveError(
                f"Illegal move {move.uci()} for position {self._board.fen()}"
            )
        return move

    @staticmethod
    def _parse_promotion(promotion: str | None) -> int | None:
        """Run  parse promotion logic."""
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

    @staticmethod
    def _stable_move_tiebreak(move: chess.Move) -> str:
        # max() picks lexicographically greatest as the final deterministic tiebreak.
        """Run  stable move tiebreak logic."""
        return move.uci()
