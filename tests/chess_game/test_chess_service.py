import chess
import pytest

from src.chess_game.chess_service import ChessService, IllegalMoveError


def test_initial_board_fen_is_standard():
    assert ChessService().fen() == chess.STARTING_FEN


def test_legal_move_e2e4_accepted_and_turn_alternates():
    service = ChessService()
    move = service.validate_uci("e2e4")
    service.push(move)

    assert service.side_to_move() == chess.BLACK
    assert service.san_history() == ["e4"]


def test_illegal_move_rejected():
    with pytest.raises(IllegalMoveError):
        ChessService().validate_uci("e2e5")


def test_square_move_validation():
    service = ChessService()
    move = service.validate_square_move("e2", "e4")

    assert move == chess.Move.from_uci("e2e4")


def test_check_and_checkmate_detection_fools_mate():
    service = ChessService()
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        service.push(service.validate_uci(uci))

    status = service.status()
    assert status.is_check
    assert status.is_checkmate
    assert status.outcome == "0-1"


def test_stalemate_detection():
    service = ChessService("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")

    assert service.status().is_stalemate


def test_castling_legal_move_appears_when_path_clear():
    service = ChessService("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")

    assert "e1g1" in service.legal_moves()


def test_en_passant_legal_move_appears():
    service = ChessService("4k3/8/8/3pPp2/8/8/8/4K3 w - f6 0 1")

    assert "e5f6" in service.legal_moves()


def test_promotion_requires_promotion_piece():
    service = ChessService("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")

    with pytest.raises(IllegalMoveError):
        service.validate_square_move("a7", "a8")
    assert service.validate_square_move("a7", "a8", "q") == chess.Move.from_uci("a7a8q")


def test_choose_engine_move_returns_legal_move():
    service = ChessService()
    move = service.choose_engine_move()

    assert move in service.board.legal_moves


def test_save_and_load_roundtrip(tmp_path):
    service = ChessService()
    service.push(service.validate_uci("e2e4"))
    path = tmp_path / "game.json"

    service.save_to_file(str(path))
    loaded = ChessService.load_from_file(str(path))

    assert loaded.fen() == service.fen()
    assert loaded.san_history() == ["e4"]
