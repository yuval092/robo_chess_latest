# RoboChess — Bug Report

All confirmed bugs with file locations, reproduction steps, and fixes.

---

## CRITICAL BUGS

### Bug 1 — `debug_one_move.py` ignores all CLI arguments

**File:** `scripts/debug_one_move.py:37-39`  
**Severity:** Critical (tool is non-functional for any move other than e2→e3)

**Description:**  
The script hardcodes the move at the module level:
```python
SRC_SQUARE = "e2"
DST_SQUARE = "e3"
SRC_PIECE_ID = "white_pawn_e"
```
The script has no `argparse` block. Any `--piece`, `--src`, `--dst` arguments passed by the user are silently ignored.

**Reproduction:**
```bash
python scripts/debug_one_move.py --piece white_knight_g --src g1 --dst f3
# Output: "Running white_pawn_e: e2 → e3"   ← wrong piece, wrong squares
```

**Impact:** Debugging any move other than e2→e3 requires editing the source file manually.

**Fix:**
```python
# Add to debug_one_move.py:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--piece", default="white_pawn_e")
parser.add_argument("--src", default="e2")
parser.add_argument("--dst", default="e3")
args = parser.parse_args()

SRC_SQUARE = args.src
DST_SQUARE = args.dst
SRC_PIECE_ID = args.piece
```

---

### Bug 2 — `chess_env/__init__.py` contains broken legacy registrations

**File:** `chess_env/__init__.py:1-13`  
**Severity:** Critical (crash on instantiation)

**Description:**  
The root-level `chess_env/__init__.py` registers two environments pointing to non-existent entry points:
```python
register(
    id='ChessFetch-v0',
    entry_point='chess_env.chess_fetch_env:ChessFetchEnv',  # file does not exist
    max_episode_steps=175,
)
register(
    id='ChessFetchDense-v0',
    entry_point='chess_env.chess_fetch_dense_env:ChessFetchDenseEnv',  # file does not exist
    max_episode_steps=250,
)
```
The files `chess_env/chess_fetch_env.py` and `chess_env/chess_fetch_dense_env.py` do not exist. Calling `gym.make("ChessFetch-v0")` raises `ModuleNotFoundError`.

Note: The active environment `ChessFetchTask-v0` is correctly registered in `src/chess_env/__init__.py` and is unaffected.

**Reproduction:**
```bash
python -c "import chess_env; import gymnasium as gym; gym.make('ChessFetch-v0')"
# ModuleNotFoundError: No module named 'chess_env.chess_fetch_env'
```

**Fix:** Delete `chess_env/__init__.py` entirely (it serves no purpose) or remove the two stale registrations.

---

### Bug 3 — `PromoteCommand` is dead code that silently causes failures

**File:** `src/chess_game/move_planner.py:31-35`  
**Severity:** Critical (silent failure if somehow invoked)

**Description:**  
`PromoteCommand` is defined:
```python
@dataclass(frozen=True)
class PromoteCommand:
    pawn_piece_id: str
    promoted_piece_id: str
    square: str
    reserve_slot: str
```
But `MovePlanner.plan()` never generates a `PromoteCommand` — promotions are handled via two `TeleportCommand` objects. Additionally, `PhysicalPlanExecutor.execute()` has no `isinstance(command, PromoteCommand)` branch. If a `PromoteCommand` were ever introduced by future code, it would fall into:
```python
else:
    return PhysicalExecutionResult(False, results, f"Unsupported command {command}")
```
This fails silently with no indication that the class exists but is unhandled.

**Fix:** Delete `PromoteCommand` from `move_planner.py`. If a dedicated promote step is ever needed, add both the generation logic in `MovePlanner.plan()` and the handling in `PhysicalPlanExecutor.execute()` at the same time.

---

## LOGIC / DESIGN BUGS

### Bug 4 — `eval_stress.py --grid` tests positions outside the chess board

**File:** `scripts/eval_stress.py:27-35`  
**Severity:** Medium (misleading success rate)

**Description:**  
The `--grid` mode computes positions using `table_center_xy ± table_half_xy ± edge_margin`:
```python
min_x, max_x = cx - hx + margin, cx + hx - margin  # 0.57 to 1.19
min_y, max_y = cy - hy + margin, cy + hy - margin   # -0.046 to 0.574
```
With `linspace(0.57, 1.19, 3)`, the grid positions are [0.570, 0.880, 1.190]. But the chess board spans only [0.600, 1.160] in X and [-0.016, 0.544] in Y.

Result: positions at x=0.570 and x=1.190 are 30mm outside the board, and the arm cannot reach them. The stress test reports 0% success for these, but this is not a genuine arm failure — it is an out-of-range test.

**Observed output:**
```
grid_0_1   (0.570, 0.264)   0%  LOW   ← 30mm outside board; TIMEOUT during transit
grid_2_0   (1.190, -0.046)  0%  LOW   ← 30mm outside board
grid_2_2   (1.190, 0.574)   0%  LOW   ← 30mm outside board
Overall success rate: 66.7%           ← misleading
```

**Fix:** Clamp grid positions to chess board boundaries, or document that `--grid` intentionally probes beyond-board workspace limits. The default corner test (no `--grid`) correctly uses actual chess square centers and reports 100%.

---

### Bug 5 — Duplicated home position creates divergence risk

**Files:** `configs/env.yaml:22`, `configs/chess.yaml:58`  
**Severity:** Medium

**Description:**
- `env.yaml:home_position_xy = [0.88, 0.2641]` — used by `ChessTaskEnv` for arm start position
- `chess.yaml:game.arm_home_xy = [0.88, 0.2641]` — used by `PhysicalPlanExecutor.return_to_home()`

Both values are identical today. If the table is repositioned and one config is updated but not the other, the arm will transit between two different "home" positions on different operations.

**Fix:** Remove `game.arm_home_xy` from `chess.yaml`. Update `plan_executor.py:74`:
```python
# Before:
home_xy = np.array(load_config("chess")["game"]["arm_home_xy"])
# After:
home_xy = np.array(load_config("env")["home_position_xy"])
```

---

### Bug 6 — `GRASP_VERIFY_FINGER_THRESHOLD` config key loaded but never used

**Files:** `src/chess_env/task.py:127`, `configs/env.yaml:36`  
**Severity:** Low (silent dead config)

**Description:**  
The config key is loaded:
```python
self.GRASP_VERIFY_FINGER_THRESHOLD = self.env_cfg.get("grasp_verify_finger_threshold", 0.012)
```
But `GRASP_VERIFY_FINGER_THRESHOLD` is never referenced in `execute_grasp()`. The grasp verification uses `empty_detect_threshold` (= `EMPTY_GRASP_THRESHOLD` = 0.011) instead. The config comment in `env.yaml` makes it appear this value controls grasp verification, but it does nothing.

**Fix:** Either use `GRASP_VERIFY_FINGER_THRESHOLD` in `execute_grasp()` for the final finger check (replacing or augmenting `empty_detect_threshold`), or remove the config key and the attribute entirely.

---

### Bug 7 — `physics.yaml` stores unnormalized quaternion

**File:** `configs/physics.yaml:4`  
**Severity:** Low (masked by normalization in code)

**Description:**
```yaml
vertical_quat: [1.0, 0.0, 1.0, 0.0]
```
This vector has norm = √2 ≈ 1.414, not 1. It is not a valid unit quaternion. The code normalizes it at load time (`simulation.py:53-54`), so there is no runtime error, but:
- The config file is misleading — a reader unfamiliar with the code would not know normalization occurs.
- The resulting quaternion [1/√2, 0, 1/√2, 0] represents a 90° rotation around the Y axis; the intended orientation should be documented.

**Fix:**
```yaml
vertical_quat: [0.7071068, 0.0, 0.7071068, 0.0]   # normalized; 90° rotation about Y axis
```
And optionally remove the normalization step from code (or keep it as a safety check with a warning if the input is already non-normalized).

---

## PERFORMANCE ISSUES

### Bug 8 — O(n) linear scan in `piece_id_at()` and `piece_at_square()`

**Files:** `src/chess_game/move_planner.py:60-67`, `src/physical/occupancy.py:11-14`  
**Severity:** Low (not yet a bottleneck, but scales poorly)

**Description:**  
Both `LogicalPieceTracker.piece_id_at()` and `PhysicalOccupancy.piece_at_square()` iterate over all pieces to find a piece at a given square:
```python
def piece_id_at(self, square: str) -> str | None:
    for piece_id, piece_square in self._piece_to_square.items():  # up to 32
        if piece_square == square:
            return piece_id
    for piece_id, piece_square in self._reserve_to_square.items():  # up to 64
        if piece_square == square:
            return piece_id
    return None
```
With 32 + 64 = 96 pieces, each lookup is O(96). During `GameOrchestrator.snapshot()`, this is called 64 times (once per square) to build `physical_piece_ids`, costing ~6144 comparisons per snapshot.

**Fix:** Maintain an inverse map `_square_to_piece: dict[str, str]` updated alongside `_piece_to_square`. Lookups become O(1).

---

### Bug 9 — Non-thread-safe global XML path mutation

**File:** `src/chess_env/simulation.py:61-70`  
**Severity:** Low (only relevant for parallel environments)

**Description:**
```python
_fpp_module.MODEL_XML_PATH = asset_path
try:
    super().__init__(**kwargs)
finally:
    _fpp_module.MODEL_XML_PATH = _original_path
```
If two `ChessSimulationEnv` objects are constructed concurrently in separate threads (parallel training), both threads modify the same module-level `MODEL_XML_PATH` variable. One environment may load with the other's XML path.

**Fix:** Use a threading lock around the mutation, or pass the XML path through an alternative mechanism (subclassing the underlying env differently).

---

## CODE QUALITY ISSUES

### Bug 10 — Hardcoded approximate step counts in `controller.py`

**File:** `src/chess_env/controller.py:226-229, 246-248`

```python
steps=result_dict.get("close_steps_used", 0) + 15 + 100, # Approx steps
# ...
steps=15 + 100 + 80 + 30 + 50, # Approx steps based on task.py
```
These are rough estimates used only for reporting. The comment "Approx steps based on task.py" will silently diverge if the grasp/place pipeline budgets change. These values should either be accurately computed from `result_dict` or clearly labeled as informational-only estimates with no functional impact.

---

### Bug 11 — Logger creates empty log files per environment instance

**File:** `src/chess_env/task.py:144-152`

Every `ChessTaskEnv` instance creates a log file named `env_{pid}.log`. During testing and evaluation, this creates many empty (0-byte) log files in `logs/env_debug/`. Over time these accumulate.

**Observed:**
```
logs/env_debug/env_489279.log  (0 bytes)
logs/env_debug/env_489354.log  (0 bytes)
... (10+ empty files per test run)
```

**Fix:** Only create the file handler when the first log message is actually written, or use a `NullHandler` by default that is replaced when `debug=True`.

---

### Bug 12 — Indentation inconsistency in `simulation.py`

**File:** `src/chess_env/simulation.py:39`

```python
        self.EDGE_MARGIN     = self.env_cfg["edge_margin"]
  # Safety margin to avoid edge collisions        ← 2-space indent (should be 8)
        self.MIN_GOAL_DIST   = self.env_cfg["min_goal_dist"]
```
This comment is at 2-space indentation instead of the surrounding 8-space method body indentation. Not a runtime issue, but inconsistent.
