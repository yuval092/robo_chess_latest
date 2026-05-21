# Stage 4: Chess Logic Layer

## Goal

Add a pure chess layer using `python-chess` that owns rules, legal move validation, turn management, move history, game result, and board serialization.

This layer must not import MuJoCo, Gymnasium, NumPy, or any physical movement module.

## Dependency

Update `requirements.txt`:

```text
python-chess
```

Validate import:

```bash
python -c "import chess; print(chess.__version__)"
```

## ChessService API

Create `src/chess_game/chess_service.py`.

```python
import chess

class ChessService:
    def __init__(self, starting_fen: str | None = None): ...
    @property
    def board(self) -> chess.Board: ...

    def legal_moves(self) -> list[str]: ...
    def validate_uci(self, uci: str) -> chess.Move: ...
    def validate_square_move(self, src: str, dst: str, promotion: str | None = None) -> chess.Move: ...
    def push(self, move: chess.Move) -> None: ...
    def pop(self) -> chess.Move: ...
    def status(self) -> GameStatus: ...
    def fen(self) -> str: ...
    def san_history(self) -> list[str]: ...
    def piece_at(self, square: str) -> chess.Piece | None: ...
    def side_to_move(self) -> chess.Color: ...
    def choose_engine_move(self) -> chess.Move: ...
```

Important note:

`python-chess` validates moves and represents board state. It does not provide a strong chess engine by itself. The user requested that the computer move be determined by the `python-chess` package. Implement this as a deterministic legal move selector at first:

1. Prefer checkmate moves.
2. Prefer captures by highest captured value.
3. Prefer promotions.
4. Prefer checks.
5. Otherwise choose a legal move using a stable seed or simple material heuristic.

Document that replacing this selector with Stockfish is a future enhancement, not part of this requirement.

## GameStatus

Create `src/chess_game/models.py`.

```python
@dataclass(frozen=True)
class GameStatus:
    turn: str
    is_check: bool
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
```

Use `python-chess` APIs:

- `board.is_check()`
- `board.is_checkmate()`
- `board.is_stalemate()`
- `board.is_insufficient_material()`
- `board.is_seventyfive_moves()`
- `board.is_fivefold_repetition()`
- `board.can_claim_fifty_moves()`
- `board.can_claim_threefold_repetition()`
- `board.outcome()`
- `board.fen()`

## Move Validation

Move input from UI:

- Source square selected by click.
- Destination square selected by click.
- Promotion piece selected when needed.

Implementation:

```python
def validate_square_move(self, src, dst, promotion=None):
    from_square = chess.parse_square(src)
    to_square = chess.parse_square(dst)
    promotion_piece_type = parse_promotion(promotion)
    move = chess.Move(from_square, to_square, promotion=promotion_piece_type)
    if move not in self.board.legal_moves:
        raise IllegalMoveError(...)
    return move
```

Promotion parsing:

```text
q -> chess.QUEEN
r -> chess.ROOK
b -> chess.BISHOP
n -> chess.KNIGHT
```

Default promotion should be queen only when the UI explicitly requests "auto queen" or the computer selector chooses it. Human promotion should prompt in the UI.

## Tests

Add `tests/chess_game/test_chess_service.py`.

Required tests:

- Initial board FEN is standard chess.
- Legal move `e2e4` accepted.
- Illegal move `e2e5` rejected.
- Turn alternates after push.
- Move history records SAN.
- Check detection works.
- Checkmate detection works with a known short mate.
- Stalemate detection works with a known FEN.
- Castling legal move appears when path is clear in a constructed FEN.
- En passant legal move appears in a constructed FEN.
- Promotion requires promotion piece.
- `choose_engine_move()` returns a legal move.

## Validation

Run:

```bash
pytest tests/chess_game/test_chess_service.py -v
python -c "from src.chess_game.chess_service import ChessService; s=ChessService(); print(s.fen())"
```

Pass criteria:

- All chess logic tests pass without MuJoCo.
- The service can serialize and restore game state through FEN.
- Every move returned by computer selector is legal.
