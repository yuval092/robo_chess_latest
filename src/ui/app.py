from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Protocol

from flask import Flask, jsonify, render_template, request


class UIBackend(Protocol):
    def snapshot(self): ...
    def new_game(self): ...
    def submit_human_move(self, src: str, dst: str, promotion: str | None = None): ...
    def let_computer_play_current_turn(self): ...


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
        src = payload.get("src")
        dst = payload.get("dst")
        if not src or not dst:
            return jsonify({"accepted": False, "error": "src and dst are required"}), 400
        result = backend.submit_human_move(src, dst, payload.get("promotion"))
        status = 200 if result.accepted else 400
        return jsonify(to_jsonable(result)), status

    @app.post("/api/promote")
    def api_promote():
        return jsonify({"accepted": False, "error": "Promotion continuation is not implemented yet."}), 501

    @app.post("/api/let-computer-play")
    def api_let_computer_play():
        result = backend.let_computer_play_current_turn()
        status = 200 if result.accepted else 400
        return jsonify(to_jsonable(result)), status

    @app.post("/api/undo")
    def api_undo():
        return jsonify({"accepted": False, "error": "Undo is not implemented yet."}), 501

    return app
