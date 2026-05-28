"""Top-level game coordinator for logical and physical chess moves."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from src.chess_game.chess_service import ChessService, GameStatus, IllegalMoveError
from src.chess_game.move_planner import LogicalPieceTracker, MovePlanner
from src.utils.io import load_config


@dataclass(frozen=True)
class GameSnapshot:
    fen: str
    turn: str
    board: dict[str, str | None]
    physical_piece_ids: dict[str, str | None]
    legal_moves: list[str]
    status: GameStatus
    last_move: str | None
    move_history_san: list[str]
    is_busy: bool
    error: str | None


@dataclass(frozen=True)
class MoveExecutionResult:
    accepted: bool
    physical_success: bool
    move_uci: str | None
    error: str | None
    snapshot: GameSnapshot
    awaiting_promotion: bool = False
    promotion_square: str | None = None


class GameOrchestrator:
    def __init__(
        self,
        chess_service: ChessService,
        physical_executor,
        piece_tracker: LogicalPieceTracker,
        *,
        human_color: str | None = None,
        auto_computer_reply: bool | None = None,
        engine_cfg: dict | None = None,
    ):
        """Initialise this object."""
        game_cfg = load_config("chess")["game"]
        self.chess_service = chess_service
        self.physical_executor = physical_executor
        self.piece_tracker = piece_tracker
        self.human_color = human_color or game_cfg["human_color"]
        self.auto_computer_reply = (
            game_cfg["auto_computer_reply"]
            if auto_computer_reply is None
            else auto_computer_reply
        )
        self._engine_cfg = engine_cfg
        self.is_busy = False
        self.error: str | None = None
        self.last_move: str | None = None

    def new_game(self) -> GameSnapshot:
        """Reset the game and physical state, restarting the chess engine."""
        if self.is_busy:
            self.error = "Cannot start a new game while the arm is moving."
            return self.snapshot()
        self.chess_service.close()
        self.chess_service = ChessService(engine_cfg=self._engine_cfg)
        self.piece_tracker = LogicalPieceTracker()
        self.error = None
        self.last_move = None
        return self.snapshot()

    def snapshot(self) -> GameSnapshot:
        """Return a frozen game snapshot."""
        status = self.chess_service.status()
        return GameSnapshot(
            fen=self.chess_service.fen(),
            turn=status.turn,
            board=self._logical_board_symbols(),
            physical_piece_ids={
                square: self.piece_tracker.piece_id_at(square)
                for square in chess.SQUARE_NAMES
            },
            legal_moves=self.chess_service.legal_moves(),
            status=status,
            last_move=self.last_move,
            move_history_san=self.chess_service.san_history(),
            is_busy=self.is_busy,
            error=self.error,
        )

    def submit_human_move(
        self, src: str, dst: str, promotion: str | None = None
    ) -> MoveExecutionResult:
        """Validate and execute a human move request."""
        if self.human_color != "both" and self._turn_color_name() != self.human_color:
            return self._rejected("It is not the human side's turn.")
        result = self._submit_move(
            lambda: self.chess_service.validate_square_move(src, dst, promotion)
        )
        if (
            result.accepted
            and result.physical_success
            and self.auto_computer_reply
            and not self.chess_service.board.is_game_over()
        ):
            return self.let_computer_play_current_turn()
        return result

    def let_computer_play_current_turn(self) -> MoveExecutionResult:
        """Execute one computer-selected move."""
        if self.chess_service.board.is_game_over(claim_draw=True):
            return self._rejected(self._game_over_message())
        result = self._submit_move(self.chess_service.choose_engine_move)
        # If the opponent's turn is also computer-controlled, chain immediately.
        # This handles pressing "Let computer play" on the human's own turn — the
        # computer plays that turn, then the opponent's reply follows automatically.
        if (
            result.accepted
            and result.physical_success
            and self.auto_computer_reply
            and not self.chess_service.board.is_game_over()
            and self.human_color != "both"
            and self._turn_color_name() != self.human_color
        ):
            return self.let_computer_play_current_turn()
        return result

    def _submit_move(self, move_factory) -> MoveExecutionResult:
        """Execute a move through validation, planning, and physical execution."""
        if self.is_busy:
            return self._rejected("Game is busy.")
        try:
            move = move_factory()
            plan = MovePlanner(self.chess_service.board, self.piece_tracker).plan(move)
        except (IllegalMoveError, ValueError) as exc:
            return self._rejected(str(exc))

        self.is_busy = True
        try:
            physical_result = self.physical_executor.execute(plan)
            if not physical_result.success:
                self.error = physical_result.error
                self.is_busy = False
                return MoveExecutionResult(
                    True, False, move.uci(), physical_result.error, self.snapshot()
                )
            home_result = self.physical_executor.return_to_home()
            home_failed = not home_result.success
            if home_failed:
                self.error = home_result.error
            self.chess_service.push(move)
            self.piece_tracker.apply_committed_move(move, plan)
            self.last_move = move.uci()
            if home_failed:
                # Home failure is a non-fatal warning; keep it as the current error.
                pass
            else:
                self.error = None
            self.is_busy = False
            return MoveExecutionResult(
                True, True, move.uci(), self.error, self.snapshot()
            )
        except Exception as exc:
            self.error = f"INTERNAL_ERROR: {exc}"
            raise
        finally:
            self.is_busy = False

    def _rejected(self, error: str) -> MoveExecutionResult:
        """Return a rejected move result with a snapshot."""
        self.error = error
        return MoveExecutionResult(False, False, None, error, self.snapshot())

    def _turn_color_name(self) -> str:
        """Return the current turn as a colour name."""
        return "white" if self.chess_service.side_to_move() == chess.WHITE else "black"

    def _logical_board_symbols(self) -> dict[str, str | None]:
        """Return board symbols keyed by square name."""
        board = {}
        for square in chess.SQUARES:
            piece = self.chess_service.board.piece_at(square)
            board[chess.square_name(square)] = piece.symbol() if piece else None
        return board

    def _game_over_message(self) -> str:
        """Return a human-readable terminal game message."""
        status = self.chess_service.status()
        if status.outcome == "1-0":
            result = "White wins"
        elif status.outcome == "0-1":
            result = "Black wins"
        elif status.outcome == "1/2-1/2":
            result = "Draw"
        else:
            result = "Game over"

        if status.is_checkmate:
            reason = " by checkmate"
        elif status.is_stalemate:
            reason = " by stalemate"
        elif status.is_insufficient_material:
            reason = " by insufficient material"
        elif status.is_seventyfive_moves:
            reason = " by the 75-move rule"
        elif status.is_fivefold_repetition:
            reason = " by fivefold repetition"
        elif status.can_claim_fifty_moves:
            reason = " by the 50-move rule"
        elif status.can_claim_threefold_repetition:
            reason = " by threefold repetition"
        else:
            reason = ""
        return f"Game over: {result}{reason}."
