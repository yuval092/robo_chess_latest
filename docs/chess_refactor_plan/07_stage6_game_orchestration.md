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

## Concurrency

Physical movement is slow and should be serialized.

Requirements:

- Orchestrator has a busy flag.
- UI cannot submit another move while busy.
- Move execution should run in a background worker thread or async task if UI server would otherwise block.
- Every result updates a single authoritative in-memory game state.

Do not allow concurrent MuJoCo access from multiple threads without a lock.

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
