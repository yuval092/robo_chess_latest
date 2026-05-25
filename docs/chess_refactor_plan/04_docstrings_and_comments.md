# Stage 4 — Docstrings and Comments

**Objective**: Every Python file must have a module-level docstring. Every function and method must have a single-line docstring explaining its purpose. Comments must explain *why*, not *what* — remove any comment that just restates the code.

---

## 4.1 Docstring Standards

**Module docstring** — first string literal in the file, one sentence describing what the module is:
```python
"""Board-to-world coordinate mapper for the RoboChess chess board."""
```

**Function/method docstring** — one sentence, imperative mood, immediately after the `def` line:
```python
def piece_id_at(self, square: str) -> str | None:
    """Return the piece ID occupying the given square, or None if empty."""
```

**Class docstring** — one sentence naming the responsibility of the class. Already present in most classes; audit and fix where absent or vague.

**What NOT to write**:
- Multi-paragraph docstrings (one line is enough for internal methods).
- Docstrings that copy the function signature: `"""Sets the piece at the given square."""` for a method named `set_piece_at` adds nothing.
- Args/Returns sections for trivial single-argument functions.

---

## 4.2 Files Requiring Module Docstrings

Add a module docstring to the **first line** of each of these files:

| File | Suggested docstring |
|---|---|
| `src/chess_env/__init__.py` | `"""Gymnasium environment registration for ChessFetchTask-v0."""` |
| `src/chess_env/simulation.py` | `"""Base MuJoCo simulation environment wrapping the Fetch Pick-and-Place robot."""` |
| `src/chess_env/task.py` | `"""Chess task environment: scenario management, grasp/place phases, and rewards."""` |
| `src/chess_game/board_mapper.py` | `"""Board-to-world coordinate mapper for the RoboChess chess board."""` |
| `src/chess_game/chess_service.py` | `"""Chess rules engine wrapper: move validation, state queries, and built-in heuristic engine."""` |
| `src/chess_game/game_orchestrator.py` | `"""Top-level game coordinator: accepts moves, drives the arm, and maintains game state."""` |
| `src/chess_game/move_planner.py` | `"""Physical plan generator and logical piece tracker for chess moves."""` |
| `src/physical/movement_executor.py` | `"""Board-to-board arm movement executor: occupancy checks, arm control, and reconciliation."""` |
| `src/physical/occupancy.py` | `"""Expected physical occupancy tracker: bidirectional square-to-piece mapping."""` |
| `src/physical/piece_registry.py` | `"""Deterministic registry of the 32 active chess pieces and their physical IDs."""` |
| `src/physical/piece_teleport.py` | `"""Free-joint teleport helpers for instantaneous chess piece repositioning."""` |
| `src/physical/plan_executor.py` | `"""Physical plan executor: routes ArmMove/Teleport/Remove commands to the correct handler."""` |
| `src/ui/app.py` | `"""Flask application factory and REST API routes for the RoboChess web UI."""` |
| `src/utils/io.py` (merged) | `"""Shared I/O utilities: YAML config loading and file-backed logging setup."""` |
| `src/chess_game/__init__.py` | `"""Chess game logic: service, orchestrator, move planner, and board mapper."""` |
| `src/physical/__init__.py` | `"""Physical execution layer: arm movement, teleporter, occupancy, and plan executor."""` |
| `src/ui/__init__.py` | `"""Web UI: Flask application and REST API."""` |
| `src/utils/__init__.py` | `"""Shared utilities: I/O, config loading, and logging."""` |

---

## 4.3 Functions Requiring Docstrings

Below is the complete list of functions/methods that currently have no docstring, grouped by file. Each must receive exactly one sentence.

### `src/chess_game/chess_service.py`

```python
def __init__(self, starting_fen: str | None = None):
    """Initialise a new chess board, optionally from a FEN string."""

def legal_moves(self) -> list[str]:
    """Return all legal moves from the current position as UCI strings."""

def validate_uci(self, uci: str) -> chess.Move:
    """Parse and validate a UCI move string; raise IllegalMoveError if illegal."""

def validate_square_move(self, src: str, dst: str, promotion: str | None = None) -> chess.Move:
    """Construct and validate a move from algebraic square names; raise IllegalMoveError if illegal."""

def push(self, move: chess.Move) -> None:
    """Commit a move to the board and append its SAN representation to the history."""

def pop(self) -> chess.Move:
    """Undo the last move and remove it from SAN history."""

def status(self) -> GameStatus:
    """Return a complete snapshot of the current game status including all terminal conditions."""

def fen(self) -> str:
    """Return the FEN string for the current board position."""

def san_history(self) -> list[str]:
    """Return a copy of the SAN move history."""

def piece_at(self, square: str) -> chess.Piece | None:
    """Return the piece at the given square name, or None if empty."""

def side_to_move(self) -> chess.Color:
    """Return the colour whose turn it is to move."""

def choose_engine_move(self) -> chess.Move:
    """Select the best legal move using a greedy heuristic scorer."""

def save_to_file(self, path: str) -> None:
    """Serialise the current game state (FEN + SAN history) to a JSON file."""

def load_from_file(cls, path: str) -> "ChessService":
    """Deserialise a game state from a JSON file and return a new ChessService."""

def _validate_move(self, move: chess.Move) -> chess.Move:
    """Raise IllegalMoveError if the move is not in the current legal moves set."""

def _parse_promotion(promotion: str | None) -> int | None:
    """Convert a promotion letter ('q', 'r', 'b', 'n') to a chess piece type constant."""

def _stable_move_tiebreak(move: chess.Move) -> str:
    """Return the UCI string of the move for use as a deterministic sort tiebreak."""
```

### `src/chess_game/board_mapper.py`

```python
def __init__(self, geometry: BoardGeometry, ...):
    """Initialise with board geometry and validation thresholds."""

def from_configs(cls) -> "BoardMapper":
    """Construct a BoardMapper from chess.yaml and env.yaml configuration."""

def board_width_m(self) -> float:
    """Return the total board width in metres (board_size × cell_size_m)."""

def board_min_xy(self) -> np.ndarray:
    """Return the [x, y] world coordinates of the board's lower-left corner."""

def board_max_xy(self) -> np.ndarray:
    """Return the [x, y] world coordinates of the board's upper-right corner."""

def square_to_xy(self, square: chess.Square) -> np.ndarray:
    """Return the [x, y] world centre of the given chess square index."""

def square_name_to_xy(self, square_name: str) -> np.ndarray:
    """Return the [x, y] world centre for a square given by algebraic name."""

def square_to_piece_xyz(self, square: chess.Square) -> np.ndarray:
    """Return the [x, y, z] world position of a piece's centre on the given square."""

def all_square_centers(self) -> dict[str, np.ndarray]:
    """Return a mapping of all 64 square names to their [x, y] world centres."""

def nearest_square(self, xy: np.ndarray) -> chess.Square:
    """Return the chess square whose centre is nearest to the given [x, y] world position."""

def assert_on_board(self, xy: np.ndarray) -> None:
    """Raise ValueError if the given [x, y] position is outside the board boundaries."""

def _validate_geometry(self) -> None:
    """Raise ValueError if the loaded geometry fails consistency checks."""
```

### `src/chess_game/game_orchestrator.py`

```python
def __init__(self, ...):
    """Initialise with a chess service, physical executor, and piece tracker."""

def new_game(self) -> GameSnapshot:
    """Reset to a new game; refuse if the arm is currently busy."""

def snapshot(self) -> GameSnapshot:
    """Return a frozen snapshot of the current game state."""

def submit_human_move(self, src: str, dst: str, promotion: str | None = None) -> MoveExecutionResult:
    """Validate and execute a human move, then trigger the computer reply if configured."""

def let_computer_play_current_turn(self) -> MoveExecutionResult:
    """Ask the engine to choose and execute a move for the current side."""

def run_computer_turn_if_needed(self) -> MoveExecutionResult | None:
    """Execute a computer move if it is the computer's turn; return None otherwise."""

def _submit_move(self, move_factory) -> MoveExecutionResult:
    """Execute the move returned by move_factory through the full physical pipeline."""

def _rejected(self, error: str) -> MoveExecutionResult:
    """Record an error and return a rejected MoveExecutionResult."""

def _turn_color_name(self) -> str:
    """Return 'white' or 'black' for the side currently to move."""

def _logical_board_symbols(self) -> dict[str, str | None]:
    """Build a square-to-symbol mapping for all 64 squares from the current board."""
```

### `src/chess_game/move_planner.py` — LogicalPieceTracker

```python
def __init__(self, starting_square_map: dict[str, str] | None = None):
    """Initialise with the standard starting position or a custom square map."""

def empty(cls) -> "LogicalPieceTracker":
    """Return a tracker with no pieces registered."""

def piece_id_at(self, square: str) -> str | None:
    """Return the piece ID at the given square in O(1), or None if empty."""

def set_piece_at(self, square: str, piece_id: str | None) -> None:
    """Move a piece to a square, updating both forward and inverse maps atomically."""

def captured_pieces(self, color: str) -> list[str]:
    """Return the list of captured piece IDs for the given colour."""

def reserve_piece_for(self, color: str, piece_type: str) -> str:
    """Return the ID of the first free reserve piece of the given colour and type."""

def next_graveyard_slot(self, color: str) -> str:
    """Return the next unused graveyard slot ID for the given colour."""

def next_promotion_reserve_slot(self, color: str) -> str:
    """Return the next unused promotion-reserve slot ID for the given colour."""

def apply_committed_move(self, move: chess.Move, plan: PhysicalPlan) -> None:
    """Apply all commands in a committed PhysicalPlan to update the internal maps."""

def _remove_piece(self, piece_id: str) -> None:
    """Mark a piece as captured: set it off-board and add to the captured list."""

def _set_off_board(self, piece_id: str) -> None:
    """Clear a piece's square in both maps without adding it to the captured list."""
```

### `src/chess_game/move_planner.py` — MovePlanner

```python
def __init__(self, board: chess.Board, tracker: LogicalPieceTracker):
    """Initialise with the current board position and the piece tracker."""

def plan(self, move: chess.Move) -> PhysicalPlan:
    """Translate a validated chess move into a sequence of physical commands."""

def _castle_commands(self, move, king_id, king_src, king_dst) -> list:
    """Return the two ArmMoveCommands needed for a castling move."""

def _captured_square(self, move: chess.Move) -> chess.Square:
    """Return the square of the piece being captured, handling en passant."""

def _captured_piece_id(self, move: chess.Move) -> str | None:
    """Return the physical piece ID of the piece being captured, or None."""
```

### `src/physical/occupancy.py`

```python
def __init__(self, starting_square_map: dict[str, str] | None = None):
    """Initialise with a piece-to-square map and build the inverse square-to-piece map."""

def piece_at_square(self, square: str) -> str | None:
    """Return the piece ID at the given square in O(1), or None if empty."""

def square_of_piece(self, piece_id: str) -> str | None:
    """Return the current square of the given piece, or None if off-board."""

def set_piece_square(self, piece_id: str, square: str | None) -> None:
    """Update the piece's location, maintaining both maps; raise on double-occupancy."""

def reset(self, starting_square_map: dict[str, str] | None = None) -> None:
    """Reset all occupancy to the given map, discarding previous state."""

def assert_square_empty(self, square: str) -> None:
    """Raise ValueError if the square is occupied by any piece."""

def assert_piece_at(self, piece_id: str, square: str) -> None:
    """Raise ValueError if the piece is not found at the expected square."""
```

### `src/physical/piece_registry.py`

```python
def __init__(self):
    """Build the registry of all 32 active pieces from the canonical layout."""

def all_pieces(self) -> list[PhysicalPiece]:
    """Return the list of all 32 active chess pieces."""

def by_id(self, piece_id: str) -> PhysicalPiece:
    """Return the PhysicalPiece for the given ID; raise KeyError if unknown."""

def starting_square_map(self) -> dict[str, str]:
    """Return a mapping of piece_id → initial_square for all active pieces."""

def ids_for_color(self, color: str) -> list[str]:
    """Return all piece IDs for the given colour."""

def _build_active_pieces(cls) -> list[PhysicalPiece]:
    """Construct the 32-piece list from the canonical back-rank and pawn layouts."""

def _piece_id(color: str, piece_type: str, file_name: str) -> str:
    """Compute the canonical piece ID from colour, type, and file."""

def _make_piece(piece_id, color, piece_type, initial_square) -> PhysicalPiece:
    """Construct a PhysicalPiece dataclass from its identifying components."""
```

### `src/physical/movement_executor.py`

```python
def __init__(self, env, controller, board_mapper, occupancy):
    """Initialise with environment, controller, board geometry, and occupancy tracker."""

def move_piece_between_squares(self, piece_id, src_square, dst_square) -> PhysicalMoveResult:
    """Execute a full arm move between two named squares, with occupancy pre-checks."""

def move_piece_xy(self, piece_id, src_xy, dst_xy, *, src_square=None, dst_square=None) -> PhysicalMoveResult:
    """Execute a full arm move given XY world coordinates; reconcile and snap after place."""
```

### `src/physical/plan_executor.py`

```python
def execute(self, plan: PhysicalPlan) -> PhysicalExecutionResult:
    """Execute all commands in a PhysicalPlan in order; stop on first ArmMove failure."""

def return_to_home(self) -> PhysicalExecutionResult:
    """Move the arm back to the home position above the board centre."""

def reset_board_state(self) -> None:
    """Teleport all pieces to starting squares and reset occupancy tracking."""

def reset_occupancy(self) -> None:
    """Alias for reset_board_state; kept for backward compatibility."""
```

### `src/physical/piece_teleport.py`

```python
def __init__(self, env, board_mapper: BoardMapper | None = None):
    """Initialise with the simulation environment and an optional board mapper."""

def teleport_piece_to_xyz(self, piece_id: str, xyz: np.ndarray, quat: np.ndarray | None = None) -> None:
    """Write the piece's freejoint position directly into qpos, zeroing velocity."""

def teleport_piece_to_square(self, piece_id: str, square: str) -> None:
    """Teleport a piece to the exact grid centre of the given chess square."""

def teleport_piece_to_graveyard(self, piece_id: str, slot_id: str) -> None:
    """Teleport a captured piece to the numbered graveyard slot for its colour."""

def teleport_piece_to_promotion_reserve(self, piece_id: str, slot_id: str) -> None:
    """Teleport a piece to the numbered promotion-reserve slot for its colour."""

def _piece_color(piece_id: str) -> str:
    """Infer 'white' or 'black' from the piece_id prefix."""

def _slot_index(slot_id: str) -> int:
    """Parse a slot ID of the form 'slot_NN' and return the integer index."""

def _slot_xyz(self, cfg: dict, spacing: float, slot_id: str) -> np.ndarray:
    """Compute the world XYZ of a slot from the zone config and spacing."""
```

---

## 4.4 Remove Stale / Redundant Comments

Go through all `src/` files and remove comments that merely restate what the code does. Keep comments that explain *why* something is non-obvious.

**Examples of comments to delete**:
```python
# Load configurations from YAML files     ← obvious from the next line
# Alias for backward compatibility         ← keep this one; explains a WHY
# We use a larger table (64cm x 64cm) than the standard Fetch task.  ← keep; explains a decision
```

**Specific locations to clean**:
- `task.py` lines 28–30: Remove `# Consume legacy params to avoid gym warnings` block comment if the params are already deleted; keep if they are still being consumed.
- `simulation.py`: `# --- Table Geometry ---`, `# --- Physics Parameters ---` section headers are acceptable since `task.py` is long; keep them.

---

## Stage 4 — Full Validation Checklist

```bash
# 1. Every source file has a module docstring
python -c "
import ast, pathlib
missing = []
for f in pathlib.Path('src').rglob('*.py'):
    tree = ast.parse(f.read_text())
    if not (tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)):
        missing.append(str(f))
if missing:
    print('Missing module docstrings:')
    for m in missing: print(' ', m)
else:
    print('All source files have module docstrings')
"

# 2. Full test suite
python -m pytest tests/ -v

# 3. Smoke-import all modules
python -c "
import src.chess_game.chess_service
import src.chess_game.board_mapper
import src.chess_game.game_orchestrator
import src.chess_game.move_planner
import src.physical.occupancy
import src.physical.piece_registry
import src.physical.movement_executor
import src.physical.plan_executor
import src.physical.piece_teleport
import src.ui.app
import src.utils.io
print('All imports OK')
"
```
