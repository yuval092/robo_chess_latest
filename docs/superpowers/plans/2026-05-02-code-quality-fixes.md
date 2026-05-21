# Code Quality and Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve code quality, add missing type hints, add robust error handling for unknown scenarios, and fix bugs in environment reset and evaluation scripts.

**Architecture:** Surgical updates to `src/chess_env/waypoints.py`, `src/chess_env/task.py`, and `scripts/eval_sequence.py`, followed by expanded testing.

**Tech Stack:** Python, NumPy, Gymnasium, Pytest.

---

### Task 1: Refactor `src/chess_env/waypoints.py`

**Files:**
- Modify: `src/chess_env/waypoints.py`

- [ ] **Step 1: Add docstrings and type hints**

Add module and function docstrings. Add return type hints `-> np.ndarray` to `derive_goal_pos` and `exit_waypoint`.

- [ ] **Step 2: Add robust error handling**

Explicitly check for unknown scenarios in `derive_goal_pos` and `exit_waypoint` and raise `ValueError`.

```python
def derive_goal_pos(scenario: str, cell_xy: np.ndarray) -> np.ndarray:
    """
    Returns the 3D target position for a given scenario and 2D cell coordinate.
    """
    z_map = {"transit": SAFE_Z, "descend": GRASP_Z, "ascend": SAFE_Z}
    if scenario not in z_map:
        raise ValueError(f"Unknown scenario: {scenario}")
    return np.array([cell_xy[0], cell_xy[1], z_map[scenario]])

def exit_waypoint(scenario: str, cell_xy: np.ndarray) -> np.ndarray:
    """
    Returns the 3D exit waypoint for a scenario, used as the entry point for the next.
    """
    if scenario not in SCENARIO_EXIT_Z:
        raise ValueError(f"Unknown scenario: {scenario}")
    return np.array([cell_xy[0], cell_xy[1], SCENARIO_EXIT_Z[scenario]])
```

- [ ] **Step 3: Verify constants are present**

Ensure `SCENARIO_EXIT_Z`, `SCENARIO_ENTRY_Z`, `VALID_TRANSITIONS`, `CHAIN_SHORTCUTS` are correctly defined (verified in research).

- [ ] **Step 4: Commit**

```bash
git add src/chess_env/waypoints.py
git commit -m "refactor: add docstrings, type hints, and error handling to waypoints.py"
```

### Task 2: Fix `src/chess_env/task.py`

**Files:**
- Modify: `src/chess_env/task.py`

- [ ] **Step 1: Clean up `soft_reset`**

Remove the broken `while hasattr(curr_env, \"env\"):` block from `soft_reset`. This logic belongs in the caller or should be handled by the environment stack correctly.

- [ ] **Step 2: Update `soft_reset` docstring**

State that the caller is responsible for resetting environment wrappers (like `TimeLimit`).

- [ ] **Step 3: Commit**

```bash
git add src/chess_env/task.py
git commit -m "fix: remove broken wrapper reset from soft_reset and update docstring"
```

### Task 3: Fix `scripts/eval_sequence.py`

**Files:**
- Modify: `scripts/eval_sequence.py`

- [ ] **Step 1: Fix drift limit attribute**

Change `uw.eval_drift_limit = drift_limit` to `uw.force_drift_limit = drift_limit`.

- [ ] **Step 2: Verify TimeLimit wrapper reset logic**

Ensure the `TimeLimit` wrapper reset logic is present and correct (it was already there, but double-check during implementation).

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_sequence.py
git commit -m "fix: use force_drift_limit in eval_sequence.py"
```

### Task 4: Expand Tests and Verify

**Files:**
- Modify: `tests/chess_env/test_waypoints.py`
- Run: `tests/chess_env/test_task_chaining.py`

- [ ] **Step 1: Expand `test_waypoints.py`**

Cover all scenarios for `derive_goal_pos` and `exit_waypoint`, test `CHAIN_SHORTCUTS` validity, and test `ValueError` for unknown scenarios.

```python
def test_derive_goal_pos_all_scenarios():
    cell = np.array([0.4, 0.4])
    assert np.allclose(derive_goal_pos("transit", cell), [0.4, 0.4, 0.550])
    assert np.allclose(derive_goal_pos("descend", cell), [0.4, 0.4, 0.430])
    assert np.allclose(derive_goal_pos("ascend", cell), [0.4, 0.4, 0.550])
    with pytest.raises(ValueError, match="Unknown scenario"):
        derive_goal_pos("invalid", cell)

def test_exit_waypoint_all_scenarios():
    cell = np.array([0.5, 0.5])
    assert np.allclose(exit_waypoint("transit", cell), [0.5, 0.5, 0.550])
    assert np.allclose(exit_waypoint("descend", cell), [0.5, 0.5, 0.430])
    assert np.allclose(exit_waypoint("ascend", cell), [0.5, 0.5, 0.550])
    with pytest.raises(ValueError, match="Unknown scenario"):
        exit_waypoint("invalid", cell)

def test_chain_shortcuts_validity():
    from src.chess_env.waypoints import CHAIN_SHORTCUTS, validate_chain
    for chain in CHAIN_SHORTCUTS.values():
        validate_chain(chain)
```

- [ ] **Step 2: Run all tests**

Run: `PYTHONPATH=. pytest`
Expected: All tests pass, including `tests/chess_env/test_task_chaining.py`.

- [ ] **Step 3: Commit tests**

```bash
git add tests/chess_env/test_waypoints.py
git commit -m "test: expand waypoints tests and verify all pass"
```
