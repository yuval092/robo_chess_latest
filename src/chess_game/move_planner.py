"""Physical plan generator and logical piece tracker for chess moves."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from src.physical.piece_registry import PieceRegistry, reserve_piece_ids


@dataclass(frozen=True)
class ArmMoveCommand:
    piece_id: str
    src_square: str
    dst_square: str


@dataclass(frozen=True)
class TeleportCommand:
    piece_id: str
    destination_kind: str
    destination_id: str


@dataclass(frozen=True)
class RemoveFromBoardCommand:
    piece_id: str
    graveyard_slot: str


class LogicalPieceTracker:
    """Tracks stable physical piece ids against logical chess squares."""

    def __init__(self, starting_square_map: dict[str, str] | None = None):
        if starting_square_map is None:
            starting_square_map = PieceRegistry().starting_square_map()
        self._piece_to_square: dict[str, str | None] = dict(starting_square_map)
        self._captured: dict[str, list[str]] = {"white": [], "black": []}
        self._reserve_to_square: dict[str, str | None] = {
            piece_id: None for piece_id, _, _ in reserve_piece_ids()
        }
        # Inverse map for O(1) square → piece_id lookups
        self._square_to_piece: dict[str, str] = {
            sq: pid for pid, sq in self._piece_to_square.items() if sq is not None
        }

    @classmethod
    def empty(cls) -> LogicalPieceTracker:
        """Return an empty logical piece tracker."""
        return cls({})

    def piece_id_at(self, square: str) -> str | None:
        """Return the piece ID occupying a square."""
        return self._square_to_piece.get(square)

    def _get_piece_map(self, piece_id: str) -> dict:
        """Return the forward map (piece→square) that owns this piece_id."""
        return self._reserve_to_square if piece_id in self._reserve_to_square else self._piece_to_square

    def _evict_square(self, square: str) -> None:
        """Remove any piece currently on this square."""
        old_occupant = self._square_to_piece.pop(square, None)
        if old_occupant is not None:
            self._get_piece_map(old_occupant)[old_occupant] = None

    def place_piece_at(self, square: str, piece_id: str) -> None:
        """Place a piece on a square, evicting any current occupant."""
        self._evict_square(square)
        old_sq = self._piece_to_square.get(piece_id) or self._reserve_to_square.get(piece_id)
        if old_sq is not None:
            self._square_to_piece.pop(old_sq, None)
        self._get_piece_map(piece_id)[piece_id] = square
        self._square_to_piece[square] = piece_id

    def find_reserve_piece(self, color: str, piece_type: str) -> str:
        """Reserve a promotion piece for a colour and type."""
        prefix = f"{color}_reserve_{piece_type}_"
        for piece_id in sorted(self._reserve_to_square):
            if (
                piece_id.startswith(prefix)
                and self._reserve_to_square[piece_id] is None
            ):
                return piece_id
        raise ValueError(f"No reserve {color} {piece_type} piece available")

    def next_graveyard_slot(self, color: str) -> str:
        """Return the next graveyard slot ID for a colour."""
        return f"slot_{len(self._captured[color]):02d}"

    def next_promotion_reserve_slot(self, color: str) -> str:
        """Return the next promotion reserve slot ID for a colour."""
        # Count only pawns that are off-board AND not captured (i.e., they were promoted out)
        captured_set = set(self._captured[color])
        promoted_out = sum(
            1
            for piece_id, square in self._piece_to_square.items()
            if piece_id.startswith(f"{color}_pawn_")
            and square is None
            and piece_id not in captured_set
        )
        return f"slot_{promoted_out:02d}"

    def apply_plan(self, plan: list) -> None:
        """Apply a committed physical plan to logical occupancy."""
        for command in plan:
            if isinstance(command, RemoveFromBoardCommand):
                self._remove_piece(command.piece_id)
            elif isinstance(command, ArmMoveCommand):
                self.place_piece_at(command.dst_square, command.piece_id)
            elif isinstance(command, TeleportCommand):
                if command.destination_kind == "square":
                    self.place_piece_at(command.destination_id, command.piece_id)
                else:
                    self._remove_piece_from_board(command.piece_id)

    def _remove_piece(self, piece_id: str) -> None:
        """Mark a piece as removed from the board."""
        self._remove_piece_from_board(piece_id)
        color = "white" if piece_id.startswith("white_") else "black"
        if piece_id not in self._captured[color]:
            self._captured[color].append(piece_id)

    def _remove_piece_from_board(self, piece_id: str) -> None:
        """Clear a piece from board occupancy."""
        piece_map = self._get_piece_map(piece_id)
        old_sq = piece_map[piece_id]
        piece_map[piece_id] = None
        if old_sq is not None:
            self._square_to_piece.pop(old_sq, None)


class MovePlanner:
    """Builds physical command plans from already legal chess moves."""

    PROMOTION_NAMES = {
        chess.QUEEN: "queen",
        chess.ROOK: "rook",
        chess.BISHOP: "bishop",
        chess.KNIGHT: "knight",
    }

    def __init__(self, board: chess.Board, tracker: LogicalPieceTracker):
        self.board = board
        self.tracker = tracker

    def plan(self, move: chess.Move) -> list:
        """Translate a chess move into a physical plan."""
        if move not in self.board.legal_moves:
            raise ValueError(f"Cannot plan illegal move {move.uci()}")

        src = chess.square_name(move.from_square)
        dst = chess.square_name(move.to_square)
        moving_piece_id = self.tracker.piece_id_at(src)
        if moving_piece_id is None:
            raise ValueError(f"No physical piece tracked at source square {src}")

        commands: list = []

        if self.board.is_castling(move):
            commands.extend(self._build_castle_commands(move, moving_piece_id, src, dst))
            return commands

        captured_piece_id = self._captured_piece_id(move)
        if captured_piece_id is not None:
            commands.append(self._build_capture_command(move, captured_piece_id))

        commands.append(ArmMoveCommand(moving_piece_id, src, dst))

        if move.promotion is not None:
            commands.extend(self._build_promotion_commands(move, moving_piece_id, dst))

        return commands
    
    def _build_promotion_commands(self, move: chess.Move, pawn_id: str, dst: str) -> list[TeleportCommand]:
        """Build teleport commands to swap the pawn out for a reserve promotion piece."""
        color = "white" if self.board.turn == chess.WHITE else "black"
        promoted_type = self.PROMOTION_NAMES[move.promotion]
        promoted_piece_id = self.tracker.find_reserve_piece(color, promoted_type)
        reserve_slot = self.tracker.next_promotion_reserve_slot(color)
        return [
            TeleportCommand(pawn_id, "promotion_reserve", reserve_slot),
            TeleportCommand(promoted_piece_id, "square", dst),
        ]

    def _build_capture_command(self, move: chess.Move, captured_piece_id: str) -> RemoveFromBoardCommand:
        """Build the remove command for a captured piece."""
        captured_piece = self.board.piece_at(self._captured_square(move))
        color = "white" if captured_piece.color == chess.WHITE else "black"
        return RemoveFromBoardCommand(captured_piece_id, self.tracker.next_graveyard_slot(color))

    def _build_castle_commands(
        self, move: chess.Move, king_id: str, king_src: str, king_dst: str
    ) -> list[ArmMoveCommand]:
        """Build physical commands for a castle move."""
        rank = chess.square_rank(move.from_square)
        if move.to_square > move.from_square:
            rook_src_square = chess.square(7, rank)
            rook_dst_square = chess.square(5, rank)
        else:
            rook_src_square = chess.square(0, rank)
            rook_dst_square = chess.square(3, rank)
        rook_src = chess.square_name(rook_src_square)
        rook_dst = chess.square_name(rook_dst_square)
        rook_id = self.tracker.piece_id_at(rook_src)
        if rook_id is None:
            raise ValueError(
                f"No physical rook tracked at castling source square {rook_src}"
            )
        return [
            ArmMoveCommand(king_id, king_src, king_dst),
            ArmMoveCommand(rook_id, rook_src, rook_dst),
        ]

    def _captured_square(self, move: chess.Move) -> chess.Square:
        """Return the square of a captured piece."""
        if self.board.is_en_passant(move):
            return (
                move.to_square - 8
                if self.board.turn == chess.WHITE
                else move.to_square + 8
            )
        return move.to_square

    def _captured_piece_id(self, move: chess.Move) -> str | None:
        """Return the ID of the captured piece."""
        if not self.board.is_capture(move):
            return None
        square = chess.square_name(self._captured_square(move))
        piece_id = self.tracker.piece_id_at(square)
        if piece_id is None:
            raise ValueError(f"No physical captured piece tracked at {square}")
        return piece_id
