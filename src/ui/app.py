"""Flask application factory and REST API routes for the RoboChess web UI."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from typing import Protocol

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException


class UIBackend(Protocol):
    """Protocol implemented by UI-compatible game backends."""

    def snapshot(self):
        """Return a frozen game snapshot."""
        ...

    def new_game(self):
        """Reset the game and physical state."""
        ...

    def submit_human_move(self, src: str, dst: str, promotion: str | None = None):
        """Validate and execute a human move request."""
        ...

    def let_computer_play_current_turn(self):
        """Execute one computer-selected move."""
        ...


def to_jsonable(value):
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def create_app(backend: UIBackend) -> Flask:
    app = Flask(__name__)

    @app.errorhandler(Exception)
    def handle_error(exc: Exception):
        if isinstance(exc, HTTPException):
            status = exc.code or HTTPStatus.INTERNAL_SERVER_ERROR
            error = exc.description
        else:
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            error = str(exc) or exc.__class__.__name__
        return jsonify({"accepted": False, "error": error}), status

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/snapshot")
    def api_snapshot():
        return jsonify(to_jsonable(backend.snapshot()))

    @app.post("/api/new-game")
    def api_new_game():
        return jsonify(to_jsonable(backend.new_game()))

    @app.post("/api/move")
    def api_move():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify(
                {"accepted": False, "error": "JSON body must be an object"}
            ), HTTPStatus.BAD_REQUEST
        src = payload.get("src")
        dst = payload.get("dst")
        promotion = payload.get("promotion")
        if not isinstance(src, str) or not isinstance(dst, str) or not src or not dst:
            return jsonify(
                {"accepted": False, "error": "src and dst are required"}
            ), HTTPStatus.BAD_REQUEST
        if promotion is not None and not isinstance(promotion, str):
            return jsonify(
                {"accepted": False, "error": "promotion must be a string"}
            ), HTTPStatus.BAD_REQUEST
        result = backend.submit_human_move(src, dst, promotion)
        status = HTTPStatus.OK if result.accepted else HTTPStatus.BAD_REQUEST
        return jsonify(to_jsonable(result)), status

    @app.post("/api/let-computer-play")
    def api_let_computer_play():
        result = backend.let_computer_play_current_turn()
        status = HTTPStatus.OK if result.accepted else HTTPStatus.BAD_REQUEST
        return jsonify(to_jsonable(result)), status

    return app
