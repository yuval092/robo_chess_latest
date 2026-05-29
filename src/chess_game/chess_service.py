"""Chess rules engine wrapper and game status model."""

from __future__ import annotations

import logging
import os
import select
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import chess


class IllegalMoveError(ValueError):
    pass


class UciEngine:
    """Minimal synchronous UCI wrapper for Stockfish-style engines."""

    def __init__(self, command: str, *, skill_level: int | None = None, timeout: float = 5.0):
        self._timeout = timeout
        self._stdout_buffer = b""
        self._proc = subprocess.Popen(
            [command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self._send("uci")
        self._read_until("uciok")
        if skill_level is not None:
            self._send(f"setoption name Skill Level value {skill_level}")
        self._send("isready")
        self._read_until("readyok")

    def choose_move(self, board: chess.Board, think_time: float) -> chess.Move:
        self._send(f"position fen {board.fen()}")
        self._send(f"go movetime {max(1, int(think_time * 1000))}")
        lines = self._read_until_prefix("bestmove ")
        bestmove = lines[-1].split()[1]
        if bestmove == "(none)":
            raise RuntimeError("Stockfish returned no move.")
        move = chess.Move.from_uci(bestmove)
        if move not in board.legal_moves:
            raise RuntimeError("Stockfish returned an invalid move.")
        return move

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        try:
            self._send("quit")
            self._proc.wait(timeout=1.0)
        except Exception:
            self._proc.kill()
        finally:
            self._proc.stdin and self._proc.stdin.close()
            self._proc.stdout and self._proc.stdout.close()

    def _send(self, command: str) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("Chess engine stdin is closed.")
        self._proc.stdin.write((command + "\n").encode("utf-8"))
        self._proc.stdin.flush()

    def _read_until(self, marker: str) -> list[str]:
        lines = []
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            line = self._read_line(deadline)
            if line is None:
                continue
            lines.append(line)
            if line == marker:
                return lines
        raise TimeoutError(f"Timed out waiting for UCI marker {marker!r}.")

    def _read_until_prefix(self, prefix: str) -> list[str]:
        lines = []
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            line = self._read_line(deadline)
            if line is None:
                continue
            lines.append(line)
            if line.startswith(prefix):
                return lines
        raise TimeoutError(f"Timed out waiting for UCI line prefix {prefix!r}.")

    def _read_line(self, deadline: float) -> str | None:
        if self._proc.stdout is None:
            raise RuntimeError("Chess engine stdout is closed.")
        if b"\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
            return line.decode("utf-8", errors="replace").strip()

        remaining = max(0.0, deadline - time.monotonic())
        ready, _, _ = select.select([self._proc.stdout], [], [], remaining)
        if not ready:
            return None
        chunk = os.read(self._proc.stdout.fileno(), 4096)
        if chunk == b"":
            raise RuntimeError("Chess engine exited unexpectedly.")
        self._stdout_buffer += chunk
        if b"\n" not in self._stdout_buffer:
            return None
        line, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
        return line.decode("utf-8", errors="replace").strip()


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
        starting_fen: str | None = None, # is the state of the chess board at the start of the game in a string.
        engine_cfg: dict[str, Any] | None = None,
    ):
        """Initialise the chess service, optionally starting a Stockfish engine.

        engine_cfg keys (all optional):
          stockfish_path  — executable name or full path (default: "stockfish")
          skill_level     — 0–20 (default: 5)
          think_time_s    — seconds per move (default: 0.5)
        """
        self._board = chess.Board(starting_fen) if starting_fen else chess.Board()
        self._san_history: list[str] = [] # move history in Standard Algebraic Notation (SAN)
        self._engine: UciEngine | None = None
        self._think_time: float = 0.5
        self._logger = logging.getLogger(__name__)

        if engine_cfg:
            self._think_time = float(engine_cfg.get("think_time_s", 0.5))
            path = engine_cfg.get("stockfish_path", "stockfish")
            if path:
                skill = int(engine_cfg.get("skill_level", 5))
                self._engine = UciEngine(path, skill_level=skill)

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Shut down the Stockfish process if one is running."""
        if self._engine is not None:
            try:
                self._engine.close()
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

    def parse_uci(self, uci: str) -> chess.Move:
        """Parse and validate a UCI move string."""
        try:
            move = chess.Move.from_uci(uci)
        except ValueError as exc:
            raise IllegalMoveError(f"Invalid UCI move: {uci}") from exc
        self._validate_move(move)
        return move

    def construct_move_from_squares(
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
        self._validate_move(move)
        return move

    def push(self, move: chess.Move) -> None:
        """Commit a validated move to the board."""
        self._validate_move(move)
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
        return self._engine.choose_move(self._board, self._think_time)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _validate_move(self, move: chess.Move) -> None:
        """Raise IllegalMoveError if the move is not legal in the current position."""
        if move not in self._board.legal_moves:
            raise IllegalMoveError(
                f"Illegal move {move.uci()} for position {self._board.fen()}"
            )

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
