# Final University Mentor Readiness Review

## Purpose

This document merges two independent robustness reviews of RoboChess into one
final submission-readiness audit. The review assumes the mentor will treat the
project like a product and will intentionally try unusual inputs, broken
dependencies, modified configuration, dirty sensor data, and perturbed physics.

The target behavior is not "always complete the move." The target behavior is:

```text
success OR controlled rejection/fault
never crash, hang, silently corrupt state, or continue uncontrolled
```

## Review Sources

This final review merges:

- The existing internal audit in `docs/12_production_hardening_audit.md`.
- The second AI agent's mentor-check review supplied by the user.
- Additional validation probes against the current codebase.

The second review was helpful and mostly directionally correct. It correctly
identified major risks around Stockfish failures, physical-state corruption
after failed moves, missing runtime config validation, lack of Flask error
handling, unbounded request queueing, and model-load friendliness.

It also missed or understated several important issues:

- JSON scalar and array bodies can crash `/api/move` with HTTP 500.
- Numeric promotion values can crash `/api/move` with HTTP 500.
- Physics perturbation is not fully defended because non-finite sensor values
  can bypass safety checks.
- Missing MuJoCo IDs can silently index `-1` and write to the wrong actuator.
- Packaging is not product-ready: the wheel omits runtime configs, models,
  textures, UI templates/static files, and play-mode entry point.
- Return-home and reset paths have their own failure modes.

## Current Baseline

Targeted non-integration and MuJoCo tests previously passed:

```text
72 passed, 2 warnings in 49.53s
```

The full `pytest -q` command previously failed during collection because two
integration tests import `tests.integration...` while `tests/` is not reliably
importable as a package in the pytest environment.

This must be fixed before final submission.

## Validation Of The Second Review

| Claim from second review | Validation | Corrected interpretation |
|---|---|---|
| Malformed or empty JSON becomes `{}` and does not crash. | Partly true. Non-JSON text and JSON `null` return HTTP 400. JSON string and JSON array crash because `payload.get(...)` is called on a non-dict. | Add payload-shape validation before reading fields. |
| Missing `src` / `dst` returns clean 400. | True. | Keep current behavior. |
| Bad square strings return clean rejection. | True. | Keep current behavior. |
| Bad promotion letters are caught. | True for string values such as `"x"`. False for non-string values such as `123`, which crash at `.lower()`. | Validate promotion type and enum. |
| Moving when not your turn is handled. | True. | Keep current behavior. |
| Moving after game-over is handled. | Mostly true. Engine path explicitly checks game-over. Human path rejects because there are no legal moves, but does not produce a terminal-specific message. | Add explicit terminal-state handling for human requests. |
| Stockfish runtime failures are not caught gracefully. | True. `let_computer_play_current_turn()` catches only `IllegalMoveError`, not `RuntimeError`, `TimeoutError`, or subprocess errors. | Convert engine faults to structured rejected results or engine-fault state. |
| `think_time_s > 5` can exceed the hard UCI read timeout. | True. `UciEngine` has a fixed default timeout of 5 seconds while `think_time_s` is independent. | Validate or derive timeout from think time. |
| Missing Stockfish fails at startup. | True when engine is configured and executable is absent. | Fail clearly before MuJoCo startup or degrade to human-only mode. |
| `new_game()` can fail if Stockfish disappears. | True. The old engine is closed before constructing the replacement service. If construction fails, the orchestrator can be left with a closed old service. | Construct replacement first or enter explicit fault state. |
| Failed physical moves can leave dirty simulation state. | True. | Add rollback or reset-required fault state. |
| `grasp_mode=True` can remain after failure. | True for failures after grasp mode is enabled. | Cleanup in `finally` or fault requiring reset. |
| Config validation exists but is not run at startup. | True. | Call expanded validation from play and training entry points. |
| Physics perturbation is generally bounded by stage checks. | Partly true. Many finite motion failures become `TIMEOUT`, `TUBE_BREACH`, `FLOOR_HIT`, etc. But `NaN`, missing IDs, scene/config drift, and actual-piece displacement are not safely handled. | Do not rely only on stage timeout checks; add finite-state validation and scene schema validation. |
| Corrupt model files produce unfriendly startup errors. | True. | Wrap model loading and add startup inference smoke test. |
| No Flask error handler means raw 500 responses. | True. | Add stable JSON error responses. |
| Queue is unbounded. | True. | Add capacity, timeout, and busy handling. |

Observed HTTP probe:

```text
non_json_text  -> 400
json_null      -> 400
json_string    -> 500
json_array     -> 500
missing_dst    -> 400
bad_square     -> 400
same_square    -> 400
promotion_x    -> 400
promotion_int  -> 500
```

## Critical Findings

### P0-1: Non-finite sensor data can fail open

Many safety checks use comparisons without `np.isfinite(...)`. In Python and
NumPy, comparisons involving `NaN` are false:

```text
abs(nan) > limit: False
nan < floor: False
norm([nan, 0]) > tube: False
landing nan > tolerance: False
```

Affected areas include:

- Gripper position and velocity.
- Active-piece position and quaternion.
- Finger joint angle.
- SAC observations and actions.
- Tube-breach checks.
- Drop checks.
- Landing checks.
- Placement checks.
- Success checks.

Mentor check:

- Modify sensors to return `NaN`, `Inf`, wrong shape, or dirty values.

Current behavior:

- `NaN` can bypass safety comparisons or propagate into MuJoCo control state.

Required fix:

- Add centralized finite-value validation and fail closed with
  `INVALID_SENSOR_DATA` or `INVALID_SIMULATION_STATE`.

### P0-2: Physical plans are not transactional

`PhysicalPlanExecutor.execute()` runs command lists sequentially. It does not
roll back successful commands when a later command fails.

Risk cases:

- Capture removes captured piece, then attacker fails.
- Castling moves king, then rook fails.
- Promotion moves pawn, then reserve swap fails.
- En passant removes pawn, then attacker fails.

Current behavior:

- Chess board may remain uncommitted, but MuJoCo and `PhysicalOccupancy` may
  already be changed.

Required fix:

- Snapshot touched piece poses and occupancy before plan execution.
- Roll back on failure or enter `FAULTED_REQUIRES_RESET`.

### P0-3: No explicit runtime fault state

After many physical failures, the project returns an error but remains available
for the next move. That is not enough for product-style robustness.

Failure examples:

- Empty grasp.
- Dropped piece.
- Landing outside tolerance.
- Soft-reset failure.
- Partial capture/castle/promotion.
- Stale `grasp_mode`.

Required fix:

- Add `READY`, `BUSY`, `FAULTED`, and optionally `RECOVERING`.
- Reject moves while faulted.
- Allow snapshots and reset while faulted.
- Show the fault in the UI.

### P0-4: Packaging is not product-ready

The wheel builds but omits runtime resources:

- `configs/*.yaml`
- deployed `models/*.zip`
- `chess_env/textures/*.png`
- Flask templates and static files
- `main.py`

It also exposes only `robo-chess-train`, not a play-mode command.

Mentor check:

- Install the package cleanly and run it as a product.

Current behavior:

- The installed wheel is not a runnable RoboChess product.

Required fix:

- Package all runtime resources or define an external app-data layout.
- Add `robo-chess-play`.
- Resolve resources independent of current working directory.
- Add clean-install smoke tests.

### P0-5: Runtime paths depend on current working directory

Config files are loaded from the source tree, but model paths in
`configs/deployed_models.yaml` are checked relative to the current process
directory.

Mentor check:

- Run the project from another working directory.

Observed behavior:

```text
FileNotFoundError: Model file(s) not found for: transit, descend, ascend.
```

Required fix:

- Resolve model paths relative to the config file or project root, not the
  shell's current working directory.

## Final Check Matrix

### HTTP And UI Boundary

| Check | Current result | Severity | Fix |
|---|---|---|---|
| Missing `src` or `dst` | Controlled 400. | Low | Keep. |
| Invalid square string | Controlled 400. | Low | Keep. |
| Illegal move | Controlled rejection. | Low | Keep. |
| JSON `null` body | Controlled 400. | Low | Keep. |
| Non-JSON text body | Controlled 400. | Low | Keep. |
| JSON string body | HTTP 500. | High | Require object payload. |
| JSON array body | HTTP 500. | High | Require object payload. |
| Numeric `promotion` | HTTP 500. | High | Validate promotion type. |
| Very large request body | No app-level limit. | Medium | Set `MAX_CONTENT_LENGTH`. |
| 100 concurrent move requests | Unbounded queue; can grow and wait indefinitely. | High | Bound queue and return busy/429. |
| Main loop dies while request waits | Flask handler can block forever. | High | Add request timeout. |
| Browser fetch never completes | UI can stay busy indefinitely. | Medium | Add browser timeout. |
| Unexpected route exception | HTML/default 500. | Medium | Add JSON error handler. |
| Server port already in use | Server thread can fail while simulation continues. | High | Confirm server startup. |
| Running on `0.0.0.0` | No auth or CSRF protection. | Medium | Keep loopback default or add access control. |

### Chess And Planning Boundary

| Check | Current result | Severity | Fix |
|---|---|---|---|
| Wrong turn | Controlled rejection. | Low | Keep. |
| Game over, engine path | Controlled rejection. | Low | Keep. |
| Game over, human path | Rejected as illegal move, less clear. | Low | Add explicit terminal message. |
| Normal capture with later arm failure | Partial physical side effect remains. | Critical | Rollback or fault. |
| En passant with later arm failure | Partial physical side effect remains. | Critical | Rollback or fault. |
| Castling with rook failure | King may already be moved. | Critical | Transaction boundary. |
| Promotion with swap failure | Pawn/reserve state may diverge. | Critical | Transaction boundary. |
| Promotion reserve exhausted | Planner rejects before execution. | Low | Keep and test. |
| Graveyard slot exhausted | Can fail during physical command. | Medium | Prevalidate capacity. |
| Invalid `human_color` config | Bad turn logic / auto-reply risk. | High | Validate enum. |
| Auto-reply recursion with bad config | Possible repeated recursive play. | High | Validate config and use bounded loop. |

### Stockfish Boundary

| Check | Current result | Severity | Fix |
|---|---|---|---|
| Stockfish missing on startup | Raw startup failure. | Medium | Friendly fail-fast or human-only mode. |
| Engine disabled, click computer | HTTP 500. | High | Structured rejected result. |
| Kill Stockfish mid-game | Runtime exception / 500. | High | Catch engine faults. |
| Engine hangs during move | Timeout exception / weak UI handling. | High | Structured timeout result. |
| Engine handshake fails | Possible subprocess cleanup gap. | Medium | Cleanup in constructor failure. |
| `think_time_s > engine timeout` | Timeout likely. | Medium | Derive timeout from think time. |
| Invalid skill level | Depends on engine behavior. | Medium | Validate 0-20. |
| New game after engine disappears | Can leave old closed engine in orchestrator. | High | Construct replacement before swapping. |
| Application exit | Stockfish not explicitly closed in `main.py`. | Medium | Close chess service on shutdown. |

### Physical Execution Boundary

| Check | Current result | Severity | Fix |
|---|---|---|---|
| Stage cannot reach target | Usually `TIMEOUT`. | Low | Keep, log stage data. |
| Floor/table hit | Usually explicit error. | Low | Keep. |
| Tube breach | Usually explicit error. | Low | Keep. |
| Empty grasp | Explicit error, but recovery weak. | High | Fault state and cleanup. |
| Piece dropped | Explicit error, but recovery weak. | High | Fault state and cleanup. |
| Placement outside tolerance | Error, but piece remains displaced. | High | Rollback or reset-required fault. |
| Return-home failure | Move commits anyway. | High | Commit policy plus block auto-reply/recovery. |
| Return-home exception | Can bubble out as 500. | High | Catch and fault. |
| Grasp failure after `grasp_mode=True` | `grasp_mode` may stay true. | Critical | `finally` cleanup or fault. |
| Place retract failure | Return value ignored. | Medium | Report degraded state. |
| Reset arm cannot settle | Return value ignored. | High | Fail reset explicitly. |
| Finger transition cannot settle | Return value ignored. | High | Fail reset explicitly. |
| New game after failed physical move | Usually can reset, but no formal recovery state. | Medium | Make reset recovery explicit. |

### MuJoCo / Sensor / Physics Boundary

| Check | Current result | Severity | Fix |
|---|---|---|---|
| Add unused sensors | Usually ignored. | Low | Keep. |
| Change observation tuple shape | Exact unpacking can raise. | High | Adapter validation. |
| `NaN` grip position | Can fail open. | Critical | Finite validation. |
| `NaN` grip velocity | Can fail open / timeout weirdly. | Critical | Finite validation. |
| `NaN` piece position | Can corrupt scripted target. | Critical | Finite validation. |
| `NaN` piece quaternion | Yaw check may fail open. | Critical | Quaternion validation. |
| `Inf` model action | Can corrupt control. | Critical | Validate model action. |
| Wrong-shape model action | Exception or wrong behavior. | High | Validate shape. |
| Out-of-range model action | Runtime does not clip. | High | Clip or reject. |
| Missing actuator name | `mj_name2id` can return `-1`; code writes `ctrl[-1]`. | Critical | Check all MuJoCo IDs. |
| Reordered robot DOFs | Hard-coded first-15 DOF assumption breaks. | High | Resolve named DOFs. |
| Source piece moved slightly | May still work. | Low | Define allowed correction radius. |
| Source piece moved far away | Gripper may chase it. | Critical | Preflight source-pose tolerance. |
| Destination physically occupied but occupancy stale | Possible collision/incorrect placement. | Critical | Actual occupancy reconciliation. |
| Unrelated piece bumped | Runtime likely misses it. | High | Full-board reconciliation. |
| Piece tipped over | Yaw-only check may miss roll/pitch. | High | Uprightness check. |
| Shift table/board XML without config | Config/scene divergence. | Critical | Startup scene validation. |
| Change gravity/friction/mass moderately | Often returns explicit stage failure. | Medium | Add fault-state tests. |
| Add obstacle | May timeout, no collision watchdog. | Medium | Contact and joint-limit watchdog. |

### Configuration Boundary

| Check | Current result | Severity | Fix |
|---|---|---|---|
| Delete config key | Deep `KeyError`/`TypeError`. | High | Run expanded startup validation. |
| Wrong type in config | Deep runtime failure. | High | Schema validation. |
| Malformed YAML | Raw parser error. | Medium | Friendly config error. |
| Negative tolerance | Incorrect semantics. | High | Range validation. |
| Zero step count | Division by zero risk. | High | Positive integer validation. |
| Zero vertical quaternion | Division by zero / invalid orientation. | Critical | Nonzero quaternion validation. |
| Board size not 8 | Chess and mapper disagree. | High | Enforce standard board. |
| Piece height mismatch between configs | Placement math disagreement. | High | Cross-file validation. |
| Graveyard overlaps board | Captures can interfere. | High | Zone validation. |
| Home position unreachable | Runtime timeout later. | Medium | Workspace validation. |
| Config edited while runtime active | Components may hold inconsistent snapshots. | Medium | Immutable loaded config. |
| `validate_config()` not called by runtime | Bad config reaches runtime. | High | Call at entry points. |

### Model Boundary

| Check | Current result | Severity | Fix |
|---|---|---|---|
| Missing model path | Clear `FileNotFoundError`. | Low | Keep. |
| Corrupt ZIP | Startup traceback. | Medium | Friendly model-load error. |
| Wrong policy architecture | Load/predict failure. | High | Compatibility smoke test. |
| Model returns bad action | Not validated. | Critical | Runtime action validation. |
| Model file from different scene/config version | May behave badly. | High | Model manifest and hashes. |
| Relative model path from different CWD | File not found. | High | Resolve relative to config/project root. |

### Packaging / Deployment Boundary

| Check | Current result | Severity | Fix |
|---|---|---|---|
| Build wheel | Wheel builds. | Low | Keep. |
| Install wheel and run play mode | Not runnable. | Critical | Package resources and play entry point. |
| Install wheel and render UI | Templates/static missing. | Critical | Include UI package data. |
| Install wheel and load XML texture | Texture PNG missing. | Critical | Include textures. |
| Install wheel and load configs | Configs missing. | Critical | Package or externalize configs. |
| Install wheel and load models | Models missing. | Critical | Package or model-install docs. |
| Run outside repo root | Model paths fail. | High | CWD-independent resources. |
| Fresh install later | Dependencies unpinned. | Medium | Add constraints/lock. |
| Windows training run | `fork` start method risk. | Medium | Declare platform or branch by OS. |
| Headless machine with viewer default | Viewer may fail. | Medium | Detect headless or document `--no-visualize`. |

### Tests And Release Process

| Check | Current result | Severity | Fix |
|---|---|---|---|
| `pytest -q` | Collection failure observed previously. | High | Fix imports/package init. |
| Robustness tests for HTTP bad payloads | Missing scalar/array/numeric promotion tests. | High | Add `tests/robustness`. |
| Robustness tests for engine failures | Missing. | High | Mock/kill engine tests. |
| Robustness tests for `NaN` sensors | Missing. | Critical | Add fault injection. |
| Robustness tests for partial capture/castle/promotion | Missing. | Critical | Add rollback/fault tests. |
| Robustness tests for post-failure next move | Missing. | Critical | Add wedged-state regression. |
| Clean wheel install test | Missing. | Critical | Add package smoke test. |
| Launch-from-other-CWD test | Missing. | High | Add CWD smoke test. |
| Exhaustive 64x63 sweep | Exists, opt-in. | Low | Run and record final result. |
| Docs examples match code | Currently drifted. | Medium | Update docs before release. |

## Final Prioritized Fix Plan

### Phase 1: Make failures controlled

1. Add runtime state: `READY`, `BUSY`, `FAULTED`, `RECOVERING`.
2. Reject new moves while `BUSY` or `FAULTED`.
3. Add structured fault object to snapshots.
4. Add JSON Flask error handler.
5. Add request timeout and bounded queue.
6. Disable auto-reply on any degraded home-return or physical fault.

### Phase 2: Fail closed on dirty data

1. Add finite scalar/vector validators.
2. Validate sensor reads, piece poses, quaternions, model observations, and
   actions.
3. Validate MuJoCo object IDs before indexing.
4. Clip or reject runtime SAC actions.
5. Add workspace limits to scripted mocap targets.

### Phase 3: Make physical execution transactional

1. Snapshot touched piece poses and occupancy before each plan.
2. Roll back touched pieces on failure, or force reset-required fault.
3. Preflight source piece pose and destination physical emptiness.
4. Reconcile all active board pieces before and after each move.
5. Add special-move partial-failure tests.

### Phase 4: Harden engine and config startup

1. Catch `RuntimeError`, `TimeoutError`, `OSError`, and engine protocol errors.
2. Convert engine failures to rejected results or engine-fault state.
3. Construct replacement engine before swapping during `new_game()`.
4. Close Stockfish on app shutdown and constructor failure.
5. Call expanded `validate_config()` from `main.py` and training entry point.
6. Validate type, range, cross-file geometry, model paths, engine settings, and
   scene compatibility.

### Phase 5: Harden reset and lifecycle

1. Check every `_move_grip_to()` result.
2. Report place retract failure.
3. Guarantee `grasp_mode` cleanup or mark reset-required fault.
4. Replace hard-coded DOF slices with named joint resolution.
5. Detect server startup failure.
6. Add `/api/health`.

### Phase 6: Make packaging product-ready

1. Add `robo-chess-play`.
2. Package configs, textures, UI templates/static files, and required assets.
3. Decide whether deployed models are packaged or externally installed.
4. Resolve all resource paths independent of CWD.
5. Pin tested dependency versions.
6. Add clean-install and CWD-independent smoke tests.

### Phase 7: Add robustness test suite

Create `tests/robustness/` covering:

- JSON scalar/array/null/non-JSON/missing fields/numeric promotion.
- Engine disabled, missing, killed, hanging, invalid response.
- Bad configs: missing keys, wrong types, invalid ranges, malformed YAML.
- Corrupt and incompatible model files.
- `NaN`, `Inf`, wrong-shape sensor and action data.
- Source piece displaced slightly/far; piece tipped over.
- Destination physically occupied with stale occupancy.
- Partial capture/castle/promotion failures.
- Failed move followed by another move.
- Queue flood, request timeout, and busy rejection.
- Wheel installation, UI rendering, texture loading, and play command.

## Release Gate For Final Submission

Before final submission, require:

1. `pytest -q` collects and passes.
2. Robustness tests pass for HTTP, engine, config, model, sensor, and physical
   fault injection.
3. Any failed physical plan either rolls back or blocks further movement.
4. No non-finite value can reach MuJoCo control state.
5. Auto-reply cannot start after any degraded physical/home-return result.
6. `New Game` recovers from any defined fault state.
7. A clean installed package or documented source bundle starts successfully.
8. Play mode works from a different current working directory.
9. Stockfish lifecycle is closed cleanly.
10. Final docs match commands, configs, and file paths.

## Final Assessment

The second review correctly identified several practical robustness gaps,
especially Stockfish failures and post-failure physical state. The larger merged
audit shows that the final submission risk is broader: dirty numeric data,
transaction boundaries, package installation, resource paths, queue behavior,
and MuJoCo schema assumptions all need attention.

The project should be presented as architecturally strong but not yet
production-hardened. The core engineering goal before submission is:

```text
detect bad state -> halt or reject -> preserve/restore consistency -> expose a clear fault -> allow controlled reset
```

That directly matches the mentor's "do not freak out" criterion.
