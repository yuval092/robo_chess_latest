# Stage 6 — SOLID Principles and Code Duplication

**Objective**: Eliminate all code duplication. Ensure every class follows the Single Responsibility Principle. Identify and fix any Interface Segregation or Dependency Inversion violations.

---

## 6.1 Current Duplication Patterns

### 6.1.1 Phase control flow in `task.py` (execute_grasp / execute_place)

`execute_grasp()` and `execute_place()` in `src/chess_env/task.py` follow the same structural pattern:
1. Halt the arm.
2. Align to target XY.
3. Plunge vertically.
4. Execute finger action (close or open).
5. Hold/settle.
6. Verify outcome.
7. Retract.
8. Return result dict.

Each phase has an identical abort-check structure:
```python
if not phase_result.success:
    return {"success": False, "error": phase_result.crash_reason, ...}
```

**Action**: Extract a `_run_phase(name, fn)` helper that executes a phase callable and returns early on failure:

```python
def _run_phase(
    self,
    phase_name: str,
    phase_fn: Callable[[], StageResult],
    result: dict,
) -> StageResult | None:
    """Run a named phase; populate result and return None on failure, StageResult on success."""
    stage = phase_fn()
    result["stages"][phase_name] = stage
    if not stage.success:
        result["success"] = False
        result["error"] = stage.crash_reason
        return None
    return stage
```

Then both `execute_grasp` and `execute_place` become:
```python
def execute_grasp(self, target_xy: np.ndarray) -> dict:
    """Execute the full grasp pipeline: halt, align, plunge, close, hold, verify, retract."""
    result = {"success": True, "stages": {}, "error": None}

    if self._run_phase("halt",   lambda: self._halt_arm(), result) is None:
        return result
    if self._run_phase("align",  lambda: self._align_xy(target_xy), result) is None:
        return result
    if self._run_phase("plunge", lambda: self._plunge_to_z(self.GRASP_Z), result) is None:
        return result
    # ... etc.
    return result
```

---

### 6.1.2 Gripper site position query repeated throughout `task.py`

The expression `self._utils.get_site_xpos(self.model, self.data, "robot0:grip")` appears many times throughout `task.py`. Extract to a single method:

```python
def _grip_pos(self) -> np.ndarray:
    """Return the current world position of the gripper site."""
    return self._utils.get_site_xpos(self.model, self.data, "robot0:grip").copy()
```

All call sites become `self._grip_pos()`.

---

### 6.1.3 `env` unwrapping pattern repeated in four classes

All of `PieceTeleporter`, `MovementExecutor`, `ScriptedController`, and `PhysicalPlanExecutor` contain the same unwrapping loop:
```python
inner = env
while hasattr(inner, "env"):
    inner = inner.env
self.env = inner
```

**Action**: Extract to a module-level utility function in `src/chess_env/simulation.py` (or a new `src/utils/env_utils.py`):

```python
def unwrap_env(env):
    """Unwrap a gymnasium environment to its innermost ChessTaskEnv."""
    inner = env
    while hasattr(inner, "env"):
        inner = inner.env
    return inner
```

All four classes import and call `unwrap_env(env)` instead of repeating the loop.

---

### 6.1.4 Mock physical executor duplicated across scripts

`eval_special_moves.py`, `eval_draw_conditions.py`, `eval_chess_game_flow.py`, and `tests/chess_game/test_game_orchestrator.py` all define a `MockPhysicalExecutor` class that looks like:
```python
class MockPhysicalExecutor:
    def execute(self, plan):
        return PhysicalExecutionResult(True, [(cmd, True) for cmd in plan.commands])
    def return_to_home(self):
        return PhysicalExecutionResult(True, [])
```

**Action**: Move this to `src/physical/plan_executor.py` as a public helper class:

```python
class NoOpPhysicalExecutor:
    """Physical executor stub that accepts all plans without simulation; for testing and logic-only evaluation."""

    def execute(self, plan: PhysicalPlan) -> PhysicalExecutionResult:
        """Accept the plan and report success for every command."""
        return PhysicalExecutionResult(
            success=True,
            command_results=[(cmd, True) for cmd in plan.commands],
        )

    def return_to_home(self) -> PhysicalExecutionResult:
        """Report success without moving the arm."""
        return PhysicalExecutionResult(success=True, command_results=[])
```

Update all 4 files to import `NoOpPhysicalExecutor` instead of defining their own.

---

### 6.1.5 Orchestrator construction boilerplate in scripts

`eval_chess_game_flow.py`, `eval_special_moves.py`, `eval_draw_conditions.py`, and `scripts/run_chess_ui.py` all construct a `GameOrchestrator` with slightly different parameters. Extract a factory to `src/chess_game/game_orchestrator.py`:

```python
@classmethod
def create_headless(
    cls,
    human_color: str | None = None,
    auto_computer_reply: bool | None = None,
) -> "GameOrchestrator":
    """Create a GameOrchestrator backed by NoOpPhysicalExecutor for testing and evaluation."""
    from src.physical.plan_executor import NoOpPhysicalExecutor
    return cls(
        chess_service=ChessService(),
        physical_executor=NoOpPhysicalExecutor(),
        piece_tracker=LogicalPieceTracker(),
        human_color=human_color,
        auto_computer_reply=auto_computer_reply,
    )
```

Scripts that only need logic-level testing use `GameOrchestrator.create_headless()` instead of wiring everything manually.

---

## 6.2 SOLID Analysis and Fixes

### Single Responsibility

**`src/chess_env/task.py` — 1040 lines**: This file has two distinct responsibilities:
1. **Observation/action/reward** — the gymnasium environment interface (`step`, `reset`, `_get_obs`, etc.)
2. **Grasp/place execution** — the high-level manipulation pipeline (`execute_grasp`, `execute_place`, helper methods)

**Action**: Extract the grasp/place execution into `src/chess_env/task_execution.py` as a `GraspExecutor` mixin class:

```python
# src/chess_env/task_execution.py
"""Grasp and place execution pipeline for ChessTaskEnv."""
class GraspPlaceMixin:
    """Provides execute_grasp() and execute_place() to ChessTaskEnv."""

    def execute_grasp(self, target_xy: np.ndarray) -> dict:
        """Execute the full grasp pipeline: halt, align, plunge, close, verify, retract."""
        ...

    def execute_place(self, target_xy: np.ndarray) -> dict:
        """Execute the full place pipeline: halt, align, plunge, open, settle, retract."""
        ...

    # All private phase helpers (_halt_arm, _align_xy, _plunge_to_z, etc.)
```

`ChessTaskEnv` in `task.py` inherits from both `ChessSimulationEnv` and `GraspPlaceMixin`:
```python
class ChessTaskEnv(ChessSimulationEnv, GraspPlaceMixin):
    ...
```

This brings `task.py` down to ~600 lines (environment interface) and creates a focused ~400-line `task_execution.py`.

---

### Open/Closed Principle

The `PhysicalPlanExecutor.execute()` method uses `isinstance()` checks for each command type:
```python
if isinstance(command, RemoveFromBoardCommand):
    ...
elif isinstance(command, ArmMoveCommand):
    ...
elif isinstance(command, TeleportCommand):
    ...
```

This is **closed to extension** — adding a new command type requires modifying `execute()`.

**Action**: Add a `execute_command(command)` dispatch method and a registry:

```python
class PhysicalPlanExecutor:

    def __init__(self, ...):
        ...
        self._handlers: dict[type, Callable] = {
            RemoveFromBoardCommand: self._handle_remove,
            ArmMoveCommand:         self._handle_arm_move,
            TeleportCommand:        self._handle_teleport,
        }

    def execute(self, plan: PhysicalPlan) -> PhysicalExecutionResult:
        """Execute all commands in a plan; stop on the first arm-move failure."""
        results = []
        try:
            for command in plan.commands:
                handler = self._handlers.get(type(command))
                if handler is None:
                    return PhysicalExecutionResult(
                        False, results, f"Unsupported command type: {type(command).__name__}"
                    )
                result = handler(command)
                results.append((command, result))
                if isinstance(result, PhysicalMoveResult) and not result.success:
                    return PhysicalExecutionResult(False, results, result.error)
        except Exception as exc:
            return PhysicalExecutionResult(False, results, str(exc))
        return PhysicalExecutionResult(True, results)

    def _handle_remove(self, command: RemoveFromBoardCommand) -> bool:
        """Teleport a captured piece to its graveyard slot and clear occupancy."""
        self.piece_teleporter.teleport_piece_to_graveyard(command.piece_id, command.graveyard_slot)
        self.occupancy.set_piece_square(command.piece_id, None)
        return True

    def _handle_arm_move(self, command: ArmMoveCommand) -> PhysicalMoveResult:
        """Execute a board-to-board arm move."""
        return self.movement_executor.move_piece_between_squares(
            command.piece_id, command.src_square, command.dst_square
        )

    def _handle_teleport(self, command: TeleportCommand) -> bool:
        """Teleport a piece to the specified destination kind."""
        if command.destination_kind == "promotion_reserve":
            self.piece_teleporter.teleport_piece_to_promotion_reserve(
                command.piece_id, command.destination_id
            )
            self.occupancy.set_piece_square(command.piece_id, None)
        elif command.destination_kind == "graveyard":
            self.piece_teleporter.teleport_piece_to_graveyard(
                command.piece_id, command.destination_id
            )
            self.occupancy.set_piece_square(command.piece_id, None)
        elif command.destination_kind == "square":
            self.piece_teleporter.teleport_piece_to_square(
                command.piece_id, command.destination_id
            )
            self.occupancy.set_piece_square(command.piece_id, command.destination_id)
        else:
            raise ValueError(f"Unknown teleport destination kind: {command.destination_kind!r}")
        return True
```

New command types can now be added by registering a handler without touching `execute()`.

---

### Interface Segregation

`UIBackend` protocol in `src/ui/app.py` is already minimal and correctly segregated.

`QueuedUIBackend` in `scripts/run_chess_ui.py` should be moved to `src/ui/queued_backend.py` since it is infrastructure, not a script. Scripts import it from there:

```python
# src/ui/queued_backend.py
"""Thread-safe UIBackend wrapper that serialises all calls through a queue."""
...
class QueuedUIBackend:
    ...
```

This makes `run_chess_ui.py` a thin orchestration script rather than a file that defines important infrastructure.

---

### Dependency Inversion

`GameOrchestrator.__init__` already accepts interfaces (not concrete classes) via constructor injection. No changes needed.

`ChessTaskEnv` loads config directly (`load_config("env")`) rather than accepting it as a parameter. This is acceptable for a simulation environment where the config is always loaded from disk.

---

## 6.3 `src/chess_env/task.py` — Legacy Parameter Cleanup

Lines 28–36 of `task.py` consume legacy kwargs to suppress gymnasium warnings:
```python
kwargs.pop('drift_curriculum_steps', None)
kwargs.pop('force_drift_limit', None)
kwargs.pop('fixed_drift', None)
kwargs.pop('sample_debug_freq', None)
kwargs.pop('total_curriculum_steps', None)
kwargs.pop('num_envs', None)
kwargs.pop('curriculum_progress_override', None)
```

**Action**: Verify that none of these keys are passed anywhere in the current codebase:
```bash
grep -rn "drift_curriculum\|force_drift\|fixed_drift\|sample_debug_freq\|total_curriculum\|num_envs\|curriculum_progress" src/ scripts/ tests/
```

If no callers pass these keys, delete the `kwargs.pop` lines entirely. If any caller still passes them, fix the caller first, then delete the pops.

---

## Stage 6 — Full Validation Checklist

```bash
# 1. No duplicate MockPhysicalExecutor definitions
grep -rn "class Mock.*Executor\|class NoOp.*Executor" src/ scripts/ tests/
# Should show only the one definition in src/physical/plan_executor.py

# 2. No inline env-unwrap loops
grep -rn "while hasattr.*env" src/ scripts/
# Should show only the one definition in src/utils/env_utils.py (or wherever placed)

# 3. task.py line count reduced
wc -l src/chess_env/task.py
# Should be <= 650 after extraction to task_execution.py

# 4. QueuedUIBackend in src/ui/
python -c "from src.ui.queued_backend import QueuedUIBackend; print('OK')"

# 5. NoOpPhysicalExecutor importable
python -c "from src.physical.plan_executor import NoOpPhysicalExecutor; print('OK')"

# 6. GameOrchestrator.create_headless works
python -c "
from src.chess_game.game_orchestrator import GameOrchestrator
o = GameOrchestrator.create_headless()
snap = o.snapshot()
print('legal_moves:', len(snap.legal_moves))
"

# 7. Full test suite
python -m pytest tests/ -v

# 8. run_chess_ui.py is importable
python -c "import scripts.run_chess_ui; print('OK')"
```
