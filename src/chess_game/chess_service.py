from __future__ import annotations

import json
from pathlib import Path

import chess

from src.chess_game.models import GameStatus


class IllegalMoveError(ValueError):
    pass


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
        self._board = chess.Board(starting_fen) if starting_fen else chess.Board()
        self._san_history: list[str] = []

    @property
    def board(self) -> chess.Board:
        return self._board

    def legal_moves(self) -> list[str]:
        return [move.uci() for move in self._board.legal_moves]

    def validate_uci(self, uci: str) -> chess.Move:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise IllegalMoveError(f"Invalid UCI move: {uci}") from exc
        return self._validate_move(move)

    def validate_square_move(self, src: str, dst: str, promotion: str | None = None) -> chess.Move:
        try:
            from_square = chess.parse_square(src)
            to_square = chess.parse_square(dst)
        except ValueError as exc:
            raise IllegalMoveError(f"Invalid square move: {src}->{dst}") from exc
        move = chess.Move(from_square, to_square, promotion=self._parse_promotion(promotion))
        return self._validate_move(move)

    def push(self, move: chess.Move) -> None:
        move = self._validate_move(move)
        self._san_history.append(self._board.san(move))
        self._board.push(move)

    def pop(self) -> chess.Move:
        move = self._board.pop()
        if self._san_history:
            self._san_history.pop()
        return move

    def status(self) -> GameStatus:
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
        return self._board.fen()

    def san_history(self) -> list[str]:
        return list(self._san_history)

    def piece_at(self, square: str) -> chess.Piece | None:
        return self._board.piece_at(chess.parse_square(square))

    def side_to_move(self) -> chess.Color:
        return self._board.turn

    def choose_engine_move(self) -> chess.Move:
        legal_moves = list(self._board.legal_moves)
        if not legal_moves:
            raise IllegalMoveError("No legal moves available.")

        def score(move: chess.Move) -> tuple[int, int, int, int, str]:
            board = self._board
            captured = board.piece_at(move.to_square)
            if board.is_en_passant(move):
                captured = chess.Piece(chess.PAWN, not board.turn)
            captured_value = PIECE_VALUES.get(captured.piece_type, 0) if captured else 0

            board.push(move)
            is_checkmate = board.is_checkmate()
            board.pop()

            gives_check = board.gives_check(move)
            promotion_value = PIECE_VALUES.get(move.promotion, 0) if move.promotion else 0
            return (
                1 if is_checkmate else 0,
                captured_value,
                promotion_value,
                1 if gives_check else 0,
                self._stable_move_tiebreak(move),
            )

        return max(legal_moves, key=score)

    def save_to_file(self, path: str) -> None:
        data = {
            "fen": self.fen(),
            "san_history": self.san_history(),
            "half_move_clock": self._board.halfmove_clock,
            "full_move_number": self._board.fullmove_number,
        }
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=True))

    @classmethod
    def load_from_file(cls, path: str) -> "ChessService":
        data = json.loads(Path(path).read_text())
        service = cls(data["fen"])
        service._san_history = list(data.get("san_history", []))
        return service

    def _validate_move(self, move: chess.Move) -> chess.Move:
        if move not in self._board.legal_moves:
            raise IllegalMoveError(f"Illegal move {move.uci()} for position {self._board.fen()}")
        return move

    @staticmethod
    def _parse_promotion(promotion: str | None) -> int | None:
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
        return move.uci()
