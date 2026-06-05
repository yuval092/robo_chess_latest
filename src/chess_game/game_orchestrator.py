"""Top-level game coordinator for logical and physical chess moves."""

from __future__ import annotations

from dataclasses import dataclass

import chess

from src.chess_game.chess_service import ChessService, GameStatus, IllegalMoveError
from src.chess_game.move_planner import LogicalPieceTracker, MovePlanner
from src.physical.plan_executor import PhysicalPlanExecutor
from src.utils.io import load_config


READY = "READY"
BUSY = "BUSY"
FAULTED = "FAULTED"


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
    error: str | None
    state: str


@dataclass(frozen=True)
class MoveExecutionResult:
    accepted: bool
    physical_success: bool
    move_uci: str | None
    error: str | None
    snapshot: GameSnapshot


class GameOrchestrator:
    def __init__(
        self,
        chess_service: ChessService,
        physical_executor: PhysicalPlanExecutor,
        piece_tracker: LogicalPieceTracker,
        *,
        human_color: str | None = None,
        auto_computer_reply: bool | None = None,
        engine_cfg: dict | None = None,
    ):
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
        self.error: str | None = None
        self.last_move: str | None = None
        self.state = READY

    def new_game(self) -> GameSnapshot:
        """Reset the game and physical state, restarting the chess engine."""
        new_service = ChessService(engine_cfg=self._engine_cfg)
        self.chess_service.close()
        self.chess_service = new_service
        self.piece_tracker = LogicalPieceTracker()
        self.physical_executor.reset_board_state()
        self.error = None
        self.last_move = None
        self.state = READY
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
            error=self.error,
            state=self.state,
        )

    def submit_human_move(
        self, src: str, dst: str, promotion: str | None = None
    ) -> MoveExecutionResult:
        """Validate and execute a human move request."""
        if self.state == FAULTED:
            return self._rejected("System is faulted. Start a new game to recover.")
        if self.state == BUSY:
            return self._rejected("System is busy.")
        if self.human_color != "both" and self._turn_color_name() != self.human_color:
            return self._rejected("It is not the human side's turn.")
        try:
            move = self.chess_service.construct_move_from_squares(src, dst, promotion)
        except IllegalMoveError as exc:
            return self._rejected(str(exc))
        result = self._execute_move(move)
        return self._auto_play_if_computer_turn(result)

    def let_computer_play_current_turn(self) -> MoveExecutionResult:
        """Execute one computer-selected move."""
        if self.state == FAULTED:
            return self._rejected("System is faulted. Start a new game to recover.")
        if self.state == BUSY:
            return self._rejected("System is busy.")
        if self.chess_service.board.is_game_over(claim_draw=True):
            return self._rejected(self._game_over_message())
        try:
            move = self.chess_service.choose_engine_move()
        except (IllegalMoveError, RuntimeError, TimeoutError, OSError) as exc:
            return self._fault(f"ENGINE_ERROR: {exc}")
        result = self._execute_move(move)
        return self._auto_play_if_computer_turn(result)

    def _execute_move(self, move: chess.Move) -> MoveExecutionResult:
        """Plan, execute physically, and commit the move."""
        try:
            plan = MovePlanner(self.chess_service.board, self.piece_tracker).plan(move)
        except ValueError as exc:
            return self._rejected(str(exc))
        try:
            self.state = BUSY
            physical_success, error = self._run_plan(move, plan)
        except Exception as exc:
            return self._fault(f"INTERNAL_ERROR: {exc}", move.uci())
        if physical_success and error is None:
            self.state = READY
            self.error = None
        else:
            error = error or "MOVE_FAILED"
            self._fault(error)
        return MoveExecutionResult(True, physical_success, move.uci(), error, self.snapshot())

    def _run_plan(self, move: chess.Move, plan: list) -> tuple[bool, str | None]:
        """Execute the physical plan; commit the move on success.

        Returns (physical_success, error_message). The error message is None on full success,
        or the home-return error string when the move committed but the arm failed to return home.
        """
        physical_result = self.physical_executor.execute(plan)
        if not physical_result.success:
            return False, physical_result.error
        home_result = self.physical_executor.return_to_home()
        self.chess_service.push(move)
        self.piece_tracker.apply_plan(plan)
        self.last_move = move.uci()
        if not home_result.success:
            return True, home_result.error or "HOME_RETURN_FAILED"
        return True, home_result.error

    def _auto_play_if_computer_turn(self, result: MoveExecutionResult) -> MoveExecutionResult:
        """If the last move succeeded and it's the computer's turn, play it automatically."""
        if (
            result.accepted
            and result.physical_success
            and result.error is None
            and self.state == READY
            and self.auto_computer_reply
            and self.human_color != "both"
            and self._turn_color_name() != self.human_color
            and not self.chess_service.board.is_game_over()
        ):
            return self.let_computer_play_current_turn()
        return result

    def _rejected(self, error: str) -> MoveExecutionResult:
        """Return a rejected move result with a snapshot."""
        self.error = error
        return MoveExecutionResult(False, False, None, error, self.snapshot())

    def _fault(self, error: str, move_uci: str | None = None) -> MoveExecutionResult:
        """Enter the fault state and return a failed result."""
        self.error = error
        self.state = FAULTED
        return MoveExecutionResult(False, False, move_uci, error, self.snapshot())

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
