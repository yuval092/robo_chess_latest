# Stage 6: Game Orchestration

## Goal

Add a service that coordinates UI requests, chess validation, physical planning, physical execution, and turn progression.

The orchestrator is the application-level boundary. It owns the "game turn" workflow but delegates rules and motion to lower layers.

## GameOrchestrator API

Create `src/chess_game/game_orchestrator.py`.

```python
class GameOrchestrator:
    def __init__(
        self,
        chess_service: ChessService,
        move_planner: MovePlanner,
        physical_executor: PhysicalPlanExecutor,
        piece_tracker: LogicalPieceTracker,
    ): ...

    def new_game(self) -> GameSnapshot: ...
    def snapshot(self) -> GameSnapshot: ...
    def submit_human_move(self, src: str, dst: str, promotion: str | None = None) -> MoveExecutionResult: ...
    def let_computer_play_current_turn(self) -> MoveExecutionResult: ...
    def run_computer_turn_if_needed(self) -> MoveExecutionResult | None: ...
```

## GameSnapshot

```python
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
```

The UI should render from `GameSnapshot`, not from MuJoCo internals.

## Execution Result

```python
@dataclass(frozen=True)
class MoveExecutionResult:
    accepted: bool
    physical_success: bool
    move_uci: str | None
    error: str | None
    snapshot: GameSnapshot
```

Distinguish:

- `accepted=False`: illegal move or wrong turn; no physical movement attempted.
- `accepted=True, physical_success=False`: legal move but robot execution failed; recovery needed.
- `accepted=True, physical_success=True`: move complete and committed.

## Turn Model

Recommended default:

- Human is White.
- Computer is Black.
- Human move is requested through UI.
- After successful human move, orchestrator automatically runs one computer move unless game is over.
- "Let computer play for me" runs the computer selector for the current side, including human side.

Make side assignment configurable:

```yaml
game:
  human_color: white
  auto_computer_reply: true
```

## Arm Home Position Between Moves

After each completed physical move (arm move + orientation snap), the orchestrator must send the arm to `arm_home_xy` from `configs/chess.yaml` before committing the turn and triggering the next move.

Add to `PhysicalPlanExecutor`:

```python
def return_to_home(self) -> PhysicalMoveResult:
    """Transit arm to home_xy at SAFE_Z. Used between turns."""
```

The commit protocol becomes:

```python
physical_result = physical_executor.execute(plan)
home_result = physical_executor.return_to_home()
# now safe to push chess board and start computer turn
chess_service.push(move)
```

If `return_to_home` fails, treat it as a non-fatal warning: log and continue. The arm's last-known position will be its actual location. This is not a reason to reject the move.

**Settle delay**: After returning home, add a 0.5 second simulation settle before starting the computer turn. This allows any physics disturbances (piece wobble from the move) to settle so the board is stable when the computer selects its move.

## Concurrency And Threading Model

Physical movement is slow and must be serialized. MuJoCo is not thread-safe.

**Threading model:**

- The main Python thread owns MuJoCo: all `env.step()`, `execute_grasp()`, `_mujoco_step()`, and `env.render()` calls.
- Flask / FastAPI server runs in a background daemon thread via `threading.Thread(target=app.run, daemon=True)`.
- A single `threading.Lock()` protects all MuJoCo env access.
- The background server thread acquires the lock only to read a frozen `GameSnapshot` copy after a move completes. It never calls MuJoCo APIs.

**Move execution flow:**

```
Background thread (Flask) → submits move request to a Queue
Main thread → dequeues request → acquires lock → runs physical plan → releases lock → posts result
Background thread → reads result from result Queue → responds to HTTP client
```

Use `queue.Queue` for request/response handoff between threads.

**GLFW constraint**: `render_mode="human"` opens a GLFW window that calls OpenGL. GLFW must be driven from the same thread that created it (the main thread). Never call `env.render()` from the Flask background thread.

Requirements:

- Orchestrator has a `is_busy` flag (set in main thread, read by background thread with lock).
- UI cannot submit another move while busy.
- Every result updates a single authoritative in-memory `GameSnapshot` that the server thread reads as a frozen copy.
- No MuJoCo call is ever made from the Flask thread.

## PhysicalPlanExecutor

Create:

```python
class PhysicalPlanExecutor:
    def execute(self, plan: PhysicalPlan) -> PhysicalExecutionResult: ...
```

It dispatches:

- `ArmMoveCommand` -> `MovementExecutor.move_piece_between_squares`
- `RemoveFromBoardCommand` -> `PieceTeleporter.teleport_piece_to_graveyard`
- `TeleportCommand` -> `PieceTeleporter`
- `PromoteCommand` can be represented as two teleport commands or handled explicitly.

## Commit Protocol

Pseudo-code:

```python
def submit_human_move(src, dst, promotion=None):
    if self.is_busy:
        return rejected("Game is busy")

    try:
        move = chess_service.validate_square_move(src, dst, promotion)
        plan = move_planner.plan(move)
    except IllegalMoveError as exc:
        return rejected(str(exc))

    self.is_busy = True
    try:
        physical_result = physical_executor.execute(plan)
        if not physical_result.success:
            return physical_failed(physical_result.error)

        chess_service.push(move)
        piece_tracker.apply_committed_move(move, plan)
        return success()
    finally:
        self.is_busy = False
```

Important:

- `chess_service.push(move)` happens after physical execution.
- `piece_tracker.apply_committed_move` happens after chess push.
- For computer moves, the same path is used after move selection.

## Promotion Intermediate State

When a human pawn move reaches the back rank, the orchestrator must return an intermediate result to the UI BEFORE the chess board is pushed:

```python
@dataclass(frozen=True)
class MoveExecutionResult:
    accepted: bool
    physical_success: bool
    awaiting_promotion: bool = False  # new field
    promotion_square: str | None = None
    ...
```

Workflow:
1. Human submits `e7e8` (no promotion piece specified).
2. Orchestrator detects `move.promotion is None` and the move is a promotion candidate.
3. Orchestrator executes the arm pawn move (pawn goes to e8, arm retracts to home).
4. Returns `awaiting_promotion=True` to UI.
5. UI displays promotion choice: queen, rook, bishop, knight.
6. Human selects queen → UI POSTs `api/promote?piece=q&square=e8`.
7. Orchestrator executes the two teleport commands and pushes the move with promotion piece.

If the arm pawn move fails before promotion choice, return `physical_success=False`. Do not show promotion dialog.

## Recovery State

If physical execution fails:

- UI shows an error.
- Chess board remains unchanged.
- Orchestrator exposes current physical debug state.
- Operator may retry the same move or reset physical board to logical board.

Add recovery command later:

```python
def resync_physical_to_logical(self) -> RecoveryResult:
    # Teleport all active pieces to squares from tracker/chess board.
```

This is acceptable because recovery is explicit and not normal game movement.

## Tests

Use fake physical executor for pure orchestration tests.

Required tests:

- Illegal move rejects without calling physical executor.
- Legal move calls physical executor.
- Chess board is pushed only after physical success.
- Physical failure leaves chess FEN unchanged.
- Computer move is legal and uses same execution path.
- Busy state rejects overlapping move requests.
- Let-computer-play works on either side.

## Validation

Run:

```bash
pytest tests/chess_game/test_game_orchestrator.py -v
pytest tests/integration/test_chess_turn_flow.py -v
```

Pass criteria:

- A complete `e2e4` human move followed by a computer reply works in a mocked physical test.
- A real physical smoke test can perform a single legal move and update FEN only after success.
