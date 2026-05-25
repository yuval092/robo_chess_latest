# Stage 5 — PEP8 Compliance and Style Consistency

**Objective**: All Python source files must pass `flake8` (or `ruff`) at the project-level config. The primary violations are: lines exceeding 88 characters, inconsistent import ordering, and a few formatting issues.

---

## 5.1 Tooling Setup

Add a `pyproject.toml` section for `ruff` (preferred over flake8 — faster, auto-fixable):

```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "UP",   # pyupgrade
]
ignore = [
    "E501",  # line length — handled separately with explicit wrapping decisions
]

[tool.ruff.lint.isort]
known-first-party = ["src", "scripts", "chess_env"]
```

Run auto-fix for all fixable issues:
```bash
ruff check --fix src/ scripts/ tests/
ruff format src/ scripts/ tests/
```

Then manually address the remaining violations listed below.

---

## 5.2 Import Order

**Standard**: `from __future__ import annotations` → stdlib → third-party → local.

Every file must follow this order. `ruff --fix` handles this automatically.

**Files that combine `import sys, os` on one line** (style violation): Split to one import per line.

```python
# Before
import sys, os, argparse, time

# After
import argparse
import os
import sys
import time
```

---

## 5.3 Long Lines — Hotspot Files

After `ruff format`, manually review and wrap remaining long lines in these high-count files.

### `scripts/analyze_debug_log.py`

This script has 94+ lines exceeding 88 characters, primarily in string formatting and print statements. Wrap using implicit line continuation inside parentheses:

```python
# Before (131 chars)
print(f"  Phase budget overruns: transit={transit_over}/{total} ({transit_over/total*100:.1f}%), descend={desc_over}/{total} ({desc_over/total*100:.1f}%)")

# After
print(
    f"  Phase budget overruns: "
    f"transit={transit_over}/{total} ({transit_over/total*100:.1f}%), "
    f"descend={desc_over}/{total} ({desc_over/total*100:.1f}%)"
)
```

### `src/chess_env/task.py`

The 89+ long lines are mostly in multi-condition `if` statements and string formatting inside grasp/place phase methods. Wrap conditions using `(`:

```python
# Before
if self.GRASP_VERIFY_XY_THRESHOLD is not None and xy_error > self.GRASP_VERIFY_XY_THRESHOLD and z_error > self.GRASP_VERIFY_Z_THRESHOLD:

# After
if (
    self.GRASP_VERIFY_XY_THRESHOLD is not None
    and xy_error > self.GRASP_VERIFY_XY_THRESHOLD
    and z_error > self.GRASP_VERIFY_Z_THRESHOLD
):
```

### `src/chess_env/controller.py`

The 50+ long lines are mostly in log/print statements and `f"..."` format strings. Apply the same pattern.

### `src/chess_env/environment_generation.py`

Long lines are in `xml_fragment` multi-line string construction. These can be broken with `\` continuation or by building strings incrementally.

---

## 5.4 Naming Conventions

All names are already consistent. Confirm with:

```bash
ruff check --select N src/ scripts/
```

Specific items to verify:
- Class names: `CamelCase` ✓
- Function/method names: `snake_case` ✓
- Constants: `UPPER_SNAKE_CASE` for module-level constants; `UPPER_SNAKE_CASE` for class attributes loaded from config ✓
- Private methods: single leading underscore `_method` ✓
- Private "module-private" helpers: single leading underscore `_helper` ✓

---

## 5.5 Type Annotations

All public function signatures in `src/` must have full type annotations (return type + parameter types). Private methods should also have them for clarity.

Audit with:
```bash
python -m mypy src/ --ignore-missing-imports --no-error-summary 2>&1 | grep "error:" | head -40
```

Common issues to fix:

**Missing return type on `__init__`**: These are `-> None` implicitly; no annotation needed (mypy understands this).

**`list` vs `list[str]`**: Use the parametrised form in all type annotations (Python 3.10+, so `list[str]` without importing from `typing`).

**`Optional[X]` → `X | None`**: All annotations must use the modern union syntax:
```python
# Before
from typing import Optional
def foo(x: Optional[str]) -> Optional[int]: ...

# After
def foo(x: str | None) -> int | None: ...
```

**`Dict`, `List`, `Tuple` from `typing`**: Replace with lowercase builtins:
```python
# Before
from typing import Dict, List
def foo(x: Dict[str, int]) -> List[str]: ...

# After
def foo(x: dict[str, int]) -> list[str]: ...
```

---

## 5.6 Remove Unused Imports

`ruff check --select F401` finds unused imports. Remove all flagged instances.

Known candidates:
- Some scripts import `os` solely for the path hack — after removing the hack (Stage 2), `os` may become unused.
- Some scripts import `sys` solely for `sys.path` — after Stage 2, `sys` may become unused.

---

## 5.7 Blank Lines

PEP8 requires:
- Two blank lines before and after top-level class and function definitions.
- One blank line between methods within a class.
- No trailing whitespace.

`ruff format` handles all of these automatically.

---

## 5.8 String Quotes

The project uses double quotes consistently. `ruff format` enforces double quotes.

---

## 5.9 Dataclass Field Ordering

In frozen dataclasses, fields with defaults must come after fields without. Verify:

```bash
python -c "
from src.chess_game.game_orchestrator import GameSnapshot, MoveExecutionResult
from src.chess_game.move_planner import ArmMoveCommand, TeleportCommand, RemoveFromBoardCommand, PhysicalPlan
from src.physical.movement_executor import PhysicalMoveResult
from src.physical.plan_executor import PhysicalExecutionResult
print('All dataclasses importable')
"
```

---

## Stage 5 — Full Validation Checklist

```bash
# 1. Install ruff if not present
pip install ruff

# 2. Auto-fix all fixable issues
ruff check --fix src/ scripts/ tests/
ruff format src/ scripts/ tests/

# 3. Check remaining violations (should be 0 errors)
ruff check src/ scripts/ tests/

# 4. Check no legacy typing imports remain
grep -rn "from typing import Dict\|from typing import List\|from typing import Optional\|from typing import Tuple" src/ scripts/

# 5. Check no uppercase typing aliases in annotations
grep -rn ": Dict\[\\|: List\[\\|: Optional\[\\|: Tuple\[" src/ scripts/

# 6. Full test suite
python -m pytest tests/ -v

# 7. mypy type check (informational; failures are not blockers for this stage)
python -m mypy src/ --ignore-missing-imports --no-error-summary 2>&1 | grep "error:" | wc -l
```
