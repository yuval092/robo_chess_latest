# Selected Hardening Changes

This document summarizes the minimal hardening changes implemented for the selected mentor-check findings.

## Implemented

- Added a runtime game state in `GameOrchestrator`: `READY`, `BUSY`, and `FAULTED`.
- Actual execution failures now enter `FAULTED`, including engine failures, internal runtime errors, physical execution failures, and failed return-to-home.
- While `FAULTED`, move actions and computer actions are rejected. `new_game` remains the recovery path.
- Added structured Flask error handling so unexpected backend exceptions return JSON with `accepted: false` and `error`.
- Added request-shape validation for `/api/move`, including non-object JSON and non-string promotion values.
- Updated the browser UI to display server error messages, handle non-JSON failed responses, and report server communication failures.
- Disabled board and computer actions in the UI while the backend reports `FAULTED`.
- Added startup config validation in play mode.
- Hardened deployed model path resolution so relative model paths are resolved from the project root, not the current working directory.
- Wrapped model loading errors with a message that names the failed stage and path.
- Reconciled Stockfish think time with the UCI read timeout and cleaned up partially-started engine processes on startup failure.
- Added finite-value checks for gripper position, gripper velocity, finger angle, active piece pose, model observations, model actions, teleport targets, and key arm-control targets.
- Checked all `_move_grip_to()` return values and converted ignored failures into explicit runtime errors.
- Checked the final place retract `_move_z()` result instead of ignoring it.
- Wrapped `return_to_home()` so exceptions become physical failure results.

## Intentional Limits

- Illegal user input still returns a normal rejection and does not fault the system, because no engine or physical move was attempted.
- No complex physical recovery sequence was added. A failed move faults the system and requires `new_game`.
- No large UI redesign was done. The UI changes are limited to error display, communication failure reporting, and fault-state disabling.
- Training startup was left unchanged after review; the requested hardening targets the product runtime path.

## Review Follow-up

- Reviewed the working tree with `git status` and `git diff`.
- Removed the training-entrypoint validation hook to reduce scope.
- Added focused unit tests for fault-state lockout/recovery, engine failure faulting, home-return failure faulting, HTTP JSON errors, config validation failure, project-root model path resolution, and finite-value validation.

## Verification

- `python -m py_compile ...` passed for the changed Python modules.
- `python -c "from src.utils.config_validation import validate_config; validate_config(); print('config ok')"` passed.
- `PYTHONPATH=. pytest tests/chess_game tests/test_config_schema.py tests/test_validation.py -q` passed: 38 tests.
- `PYTHONPATH=. .venv/bin/python -m pytest tests/ui tests/physical/test_piece_registry.py tests/physical/test_piece_teleport.py -q` passed: 19 tests.
