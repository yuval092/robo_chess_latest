# Production Hardening and Adversarial Test Audit

## Purpose

This report evaluates RoboChess as a product-like submission rather than only as
a working demonstration. The main question is:

> When an unexpected condition occurs, does the system remain controlled,
> preserve a coherent state, and report a useful error?

The system does not need to complete every requested move. A safe rejection,
explicit fault state, or controlled reset is acceptable. Crashing, hanging,
silently accepting corrupt data, or continuing from a partially modified board
is not acceptable.

This audit is based on:

- Reading the runtime, controller, environment, UI, configuration, training, and
  test code.
- Running the standard targeted test groups.
- Sending malformed HTTP payloads to the Flask application.
- Checking Python `NaN` comparison behavior against the current safety checks.
- Launching model-path resolution outside the repository root.
- Building and inspecting the Python wheel.

No production-hardening fixes were applied as part of this report.

## Current Test Baseline

The targeted configuration, chess logic, UI, physical, and MuJoCo suites pass:

```text
72 passed, 2 warnings in 49.53s
```

The full command below currently fails during collection:

```bash
pytest -q
```

Two integration modules import `tests.integration...`, but `tests/` is not
consistently importable as a package in the pytest environment:

```text
ModuleNotFoundError: No module named 'tests'
```

The full-suite collection issue should be fixed before submission.

## Executive Summary

RoboChess already has several good safety-oriented design decisions:

- MuJoCo mutations are serialized onto the main thread through
  `QueuedUIBackend`.
- Chess legality is validated before arm movement.
- The robot controller uses bounded stage loops and returns explicit failure
  reasons such as `TIMEOUT`, `TUBE_BREACH`, `FINGER_CLOSED_EMPTY`,
  `PIECE_DROPPED_Z`, and `PLACE_XY_FAILED`.
- The moved piece is checked after placement and snapped to the exact square
  center only when it is within a configured tolerance.
- Finger state, hover height, yaw rotation, drop offsets, and home posture are
  checked at several points.

However, the project is not yet production-ready under adversarial
perturbations. The most important gaps are:

1. Non-finite sensor values such as `NaN` can bypass safety comparisons.
2. Physical command plans are not transactional. A failed capture, castle, or
   promotion may leave partial side effects behind.
3. A failed move does not place the system into a blocked fault state.
4. The runtime trusts expected occupancy without reconciling all actual MuJoCo
   piece poses.
5. Scripted grasp may chase an externally displaced piece without workspace
   bounds.
6. Return-home failure still commits the chess move and can allow automatic
   continuation.
7. The API, queue, engine lifecycle, and startup paths have error cases that
   become HTTP 500 errors, indefinite waits, or silent degraded operation.
8. The built wheel omits files required to run the product.

## Classification Legend

| Result | Meaning |
|---|---|
| `GOOD` | The project should continue or reject the operation with a controlled error. |
| `PARTIAL` | The issue is detected or bounded, but state recovery, diagnostics, or availability is incomplete. |
| `FAIL` | The project can crash, hang, silently accept corrupt state, or continue from an unsafe state. |
| `VERIFY` | Static inspection suggests a risk, but a dedicated fault-injection test is still needed. |

## Highest-Priority Problems

### P0-1: `NaN` sensor values bypass safety checks

Several safety paths use numeric comparisons without first checking
`np.isfinite(...)`.

Examples:

- `src/chess_env/model_controller.py`: tube breach and floor checks.
- `src/chess_env/production_env.py`: held-piece, hover, yaw, grasp, and placement
  checks.
- `src/physical/movement_executor.py`: landing tolerance checks.
- `src/chess_env/base_env.py`: success checks and mocap target calculations.

Observed Python behavior:

```text
abs(nan) > limit: False
nan < floor: False
norm([nan, 0]) > tube: False
landing nan > tolerance: False
```

A non-finite value can therefore bypass checks that are intended to fail closed.
It may also propagate into model observations, actions, mocap positions, or
MuJoCo state.

Required fix:

- Add centralized finite-value validation for observations, actions, gripper
  pose, gripper velocity, finger joints, active-piece pose, active-piece
  quaternion, goals, mocap state, and placement measurements.
- Return an explicit error such as `INVALID_SENSOR_DATA` or
  `INVALID_SIMULATION_STATE`.
- Enter a blocked fault state until reset.

### P0-2: Physical plans are not transactional

`PhysicalPlanExecutor.execute()` performs commands sequentially and returns on
the first failure. It does not undo commands that already succeeded.

This affects:

- Capture: captured piece is teleported to graveyard, then the attacking arm
  move fails.
- Castling: king moves successfully, then rook movement fails.
- Promotion: pawn moves, pawn is removed, or reserve piece teleports before a
  later command fails.
- Any future multi-command plan.

The chess board is intentionally not committed when the physical plan fails,
but MuJoCo and `PhysicalOccupancy` may already be changed. The next move can then
start from inconsistent state.

Required fix:

- Snapshot physical occupancy and the relevant MuJoCo free-joint poses before a
  plan.
- On failure, either roll back all touched pieces or enter a fault state that
  requires a full new-game reset.
- Do not accept another move while recovery is pending.

### P0-3: No system-level fault state after motion failure

The controller can correctly report a local failure but the runtime remains
available for more move requests. Examples:

- Empty grasp.
- Dropped piece.
- Placement failure.
- Landing tolerance failure.
- Soft-reset alignment failure.
- Partial capture or castling failure.

There is no state such as:

```text
READY
BUSY
FAULTED_REQUIRES_RESET
```

Required fix:

- Add an explicit runtime state to `GameOrchestrator` or the physical executor.
- Halt the arm on fault.
- Reject all new move and computer-play requests while faulted.
- Continue serving snapshots.
- Allow `New Game` or an explicit recovery endpoint.
- Include the fault state and fault reason in the UI snapshot.

### P0-4: Scripted grasp may chase displaced or corrupted piece coordinates

After learned transit and descend stages target the logical source square,
`execute_grasp()` reads the live piece position and calls `_align_over_xy()`.
The finger-closing and hold loops continue to track the live piece XY.

This is useful for small drift, but there is no maximum allowed distance from
the expected source square and no workspace-bound check before scripted mocap
movement. If a piece is manually moved far away, or its pose becomes dirty but
finite, the gripper may chase it.

Required fix:

- Before grasp, compare the live active-piece pose with the expected source
  square.
- Permit only a small correction radius.
- Reject with `SOURCE_PIECE_OUT_OF_POSITION` if outside tolerance.
- Apply hard workspace limits before every scripted target and mocap update.

### P0-5: Built wheel is not a runnable product

The wheel builds successfully, but inspection shows it omits:

- `configs/*.yaml`
- `models/*.zip`
- `chess_env/textures/*.png`
- `src/ui/templates/index.html`
- `src/ui/static/app.js`
- `src/ui/static/styles.css`
- `main.py`

The wheel exposes console scripts for play and training:

```text
robo-chess-play
robo-chess-train
```

Consequences:

- Installed configuration loading fails because `src/utils/io.py` expects a
  repository-level `configs/` directory.
- MuJoCo XML texture loading can fail because `block.png` is absent.
- The browser UI cannot render because Flask templates and static files are
  absent.
- Deployed controller models are absent.
- Runtime resources still need clean-install verification.

Required fix:

- Decide whether submission is a source bundle or an installable package.
- For a product-like package, include configs, textures, UI files, and an
  explicit model acquisition strategy.
- Use package-resource APIs rather than repository-relative paths.
- Add a clean-install smoke test in an isolated temporary environment.

## Adversarial Check Matrix

### Chess Rules and Command Planning

| Mentor check | Expected current behavior | Result | Hardening action |
|---|---|---|---|
| Submit an illegal move such as `e2e5` | Rejected before arm movement. | `GOOD` | Keep existing tests. |
| Submit an invalid square such as `z9` | Rejected with HTTP 400 and error snapshot. | `GOOD` | Keep existing tests. |
| Move when it is not the configured human side's turn | Rejected before planning. | `GOOD` | Keep existing behavior. |
| Submit move after checkmate | Human path relies on legal move rejection; engine path explicitly rejects game over. | `GOOD` | Add explicit human-path game-over message for consistency. |
| Trigger castling | Planner emits king move then rook move. | `PARTIAL` | Add rollback or reset-required fault handling if rook movement fails. |
| Trigger en passant | Planner removes passed-over pawn then moves attacker. | `PARTIAL` | Add rollback if attacker movement fails. |
| Trigger capture | Captured piece is removed first, then attacker moves. | `PARTIAL` | Add rollback if attacker movement fails. |
| Trigger promotion | Pawn moves, is removed, and reserve piece appears. | `PARTIAL` | Add rollback across the entire promotion plan. |
| Trigger capture-promotion | Multiple removals and teleport occur. | `PARTIAL` | Add dedicated rollback fault-injection test. |
| Exhaust promotion reserve pieces | Planner raises `ValueError` before execution. | `GOOD` | Add explicit user-facing error test. |
| Exhaust graveyard slots | Teleport may raise after prior physical side effects. | `PARTIAL` | Validate plan capacity before execution. |
| Corrupt logical tracker so source piece is missing | Planner rejects before execution. | `GOOD` | Add consistency preflight and fault state. |
| Corrupt physical occupancy so source is wrong | Movement executor rejects before controller motion. | `GOOD` | Add full-board reconciliation against MuJoCo poses. |
| Corrupt physical occupancy so destination appears occupied | Movement executor rejects before controller motion. | `GOOD` | Add consistency preflight. |
| Configure invalid `human_color` | Human requests may always reject; auto-reply recursion risk exists. | `FAIL` | Validate enum: `white`, `black`, or `both`. |
| Enable auto-reply with invalid human color | Computer recursion can continue across turns until terminal state or recursion failure. | `FAIL` | Validate config and replace recursion with bounded orchestration. |

### HTTP API and Browser UI

| Mentor check | Expected current behavior | Result | Hardening action |
|---|---|---|---|
| Omit `src` or `dst` | Structured HTTP 400. | `GOOD` | Keep existing behavior. |
| Send invalid square string | Structured HTTP 400. | `GOOD` | Keep existing behavior. |
| Send numeric `src` or list `src` | `python-chess` currently rejects these through the existing error path. | `GOOD` | Add explicit schema validation anyway. |
| Send JSON string instead of JSON object | `payload.get(...)` raises; Flask returns HTTP 500. | `FAIL` | Require JSON object before reading fields. |
| Send JSON array instead of JSON object | Same shape error risk. | `FAIL` | Return structured HTTP 400. |
| Send numeric promotion field | `.lower()` raises; Flask returns HTTP 500. | `FAIL` | Validate promotion type and enum. |
| Send very large request body | No application-level body-size limit is configured. | `PARTIAL` | Set `MAX_CONTENT_LENGTH`. |
| Send repeated move requests while one move runs | Requests queue and execute sequentially. | `PARTIAL` | Reject duplicate/busy requests or cap queue length. |
| Flood mutating endpoints | Queue is unbounded. Memory use and wait time can grow. | `FAIL` | Use bounded queue and `BUSY` or HTTP 429 response. |
| Main loop stops while a caller waits | `_enqueue()` blocks forever on `response.get()`. | `FAIL` | Add request timeout and cancellation response. |
| Browser request never completes | `fetch()` has no timeout; UI stays busy indefinitely. | `FAIL` | Use `AbortController` timeout and recovery UI. |
| Flask returns HTML error page | Client calls `response.json()` and reports misleading network error. | `PARTIAL` | Handle non-JSON responses explicitly. |
| Click `New Game` during long move through direct API | Reset is queued behind move; it is not an emergency stop. | `PARTIAL` | Add an abort/reset control path or clearly document queued reset semantics. |
| Close browser during long request | Server-side request still waits and motion continues. | `PARTIAL` | Add cancellation policy or accept this explicitly. |
| Bind to `0.0.0.0` on shared network | No authentication or CSRF protection. | `PARTIAL` | Keep default loopback; document exposure risk or add access control. |
| Port is already occupied | Flask server thread can fail while simulation loop continues without UI. | `FAIL` | Detect server startup success and terminate with clear error. |
| Flask daemon thread crashes later | Main loop does not monitor server health. | `FAIL` | Add thread-health monitoring and shutdown path. |
| Long move is running | UI shows cached snapshot and busy state, but it does not poll progress. | `PARTIAL` | Add periodic snapshot polling or stage progress updates. |
| API raises unexpected exception | Runtime queue returns exception to Flask; response is usually HTTP 500 without structured JSON. | `PARTIAL` | Add Flask error handler and stable error response schema. |

Observed malformed-payload probe:

```text
JSON string payload -> HTTP 500
numeric promotion   -> HTTP 500
missing src         -> HTTP 400
invalid square      -> HTTP 400
```

### Sensor Data, Observations, and Model Outputs

| Mentor check | Expected current behavior | Result | Hardening action |
|---|---|---|---|
| Add unused MuJoCo sensors to XML | Usually ignored by runtime. | `GOOD` | Add smoke test if sensor extension is expected. |
| Add extra values to `generate_mujoco_observations()` return tuple | Exact tuple unpacking can raise. | `FAIL` | Isolate upstream observation adapter and validate interface. |
| Add finite noise to grip position | May still work or return timeout/tube failure. | `GOOD` | Add noise-envelope tests. |
| Add finite bias to grip position | May falsely fail or command wrong movement. | `PARTIAL` | Add sanity range checks and expected-state cross-checks. |
| Add finite noise to finger joint | Can trigger precondition error. | `GOOD` | Add hysteresis if realistic sensor noise is expected. |
| Set grip position to `NaN` | Safety checks can fail open and propagate corruption. | `FAIL` | Validate finiteness immediately. |
| Set grip velocity to `NaN` | Stability comparisons fail; stage may timeout or propagate inconsistent behavior. | `FAIL` | Validate finiteness immediately. |
| Set piece pose to `NaN` | Drop, yaw, alignment, and landing checks can fail open or corrupt mocap targets. | `FAIL` | Validate active-piece pose before use. |
| Set piece quaternion to `NaN` | Yaw check may not reject safely. | `FAIL` | Validate and normalize quaternion. |
| Set home mocap quaternion opposite saved quaternion | Linear interpolation can reach zero norm and generate `NaN`. | `VERIFY` | Use robust quaternion interpolation and finite check. |
| Return model action with `NaN` | Runtime passes it to MuJoCo without validation. | `FAIL` | Validate shape and finiteness before `_set_action()`. |
| Return model action with `Inf` | Can corrupt mocap or MuJoCo state. | `FAIL` | Validate finiteness. |
| Return model action outside `[-1, 1]` | Runtime does not clip actions; training does clip. | `FAIL` | Clip or reject runtime actions. |
| Return model action with wrong shape | Indexing or assertion error can bubble into generic failure. | `PARTIAL` | Validate shape and return explicit `INVALID_MODEL_ACTION`. |
| Modify observation dimension | SAC prediction may raise. | `PARTIAL` | Validate model/environment compatibility during startup. |
| Modify model ZIP to incompatible policy | Load or prediction exception. | `PARTIAL` | Add startup inference smoke test. |
| Set goal or target to `NaN` through config | Success checks and mocap updates can fail open. | `FAIL` | Validate config finiteness and runtime targets. |
| Set vertical quaternion to all zeros | Division by zero creates invalid orientation. | `FAIL` | Validate nonzero norm before normalization. |

### MuJoCo Scene and Physics Perturbations

| Mentor check | Expected current behavior | Result | Hardening action |
|---|---|---|---|
| Increase chess-piece mass moderately | Grasp may work or fail with explicit reason. | `GOOD` | Add parameterized tolerance tests. |
| Reduce piece friction moderately | Piece may drop or drift; checks often report failure. | `GOOD` | Add fault-injection test and enforce fault state. |
| Increase gravity moderately | Likely drop, placement, or timeout error. | `GOOD` | Add fault-state recovery test. |
| Change MuJoCo timestep moderately | Policy quality may degrade into timeout. | `PARTIAL` | Validate supported timestep range at startup. |
| Lower finger actuator strength | Empty grasp or held-piece failure likely reported. | `GOOD` | Verify both finger sensors and fault state. |
| Raise finger actuator strength greatly | Contact instability can occur; no force limit is monitored. | `PARTIAL` | Add contact-force and control-range limits. |
| Add obstacle on arm path | May time out, but no collision-force rule guarantees containment. | `PARTIAL` | Monitor contacts, excessive speed, and joint limits. |
| Add obstacle that moves another piece | Runtime does not monitor non-moving-piece displacement. | `FAIL` | Reconcile every active board piece after motion. |
| Manually move source piece slightly | Scripted alignment may compensate. | `GOOD` | Define and test allowed correction radius. |
| Manually move source piece far away | Scripted alignment may chase it without workspace bounds. | `FAIL` | Reject out-of-position source before scripted alignment. |
| Manually move destination occupant into target square without updating occupancy | Arm may attempt placement into a physically occupied square. | `FAIL` | Compare actual board occupancy to expected occupancy pre-move. |
| Manually move unrelated board piece | Runtime usually does not notice. | `FAIL` | Full-board reconciliation pre-move and post-move. |
| Rotate source piece above 25 degrees | Rejected before plunge with `PIECE_ROTATED`. | `GOOD` | Validate quaternion first. |
| Tip piece onto its side | Yaw-only logic may not detect roll/pitch reliably. | `FAIL` | Validate upright orientation and resting Z. |
| Change collision cube size | Grasp assumptions may break; startup validation does not run. | `PARTIAL` | Validate scene geometry at startup. |
| Shift table in XML only | Config and world geometry diverge. | `FAIL` | Validate XML/config alignment at startup. |
| Shift board config only | Visual board, table, policy assumptions, and reset placement diverge. | `FAIL` | Centralize geometry and validate scene compatibility. |
| Change square size in config only | Mapped positions differ from XML visuals and trained model assumptions. | `FAIL` | Validate and version model/scene compatibility. |
| Change table height without updating grasp constants | Plunge and placement checks become wrong. | `FAIL` | Derive heights or validate cross-field invariants. |
| Change piece height in one config only | Environment and board mapper can disagree. | `FAIL` | Validate `env.yaml` and `chess.yaml` agreement. |
| Remove required piece body or joint | Environment or request raises lookup exception. | `PARTIAL` | Run scene schema validation before serving UI. |
| Rename finger actuator | `mj_name2id()` may return `-1`; indexing `ctrl[-1]` can silently drive the wrong actuator. | `FAIL` | Check every MuJoCo ID lookup for `-1`. |
| Add robot DOFs before existing DOFs | `_zero_robot_dynamics()` assumes first 15 DOFs belong to robot. | `FAIL` | Resolve DOF addresses by joint name. |
| Reorder robot DOFs | Same hard-coded DOF slice risk. | `FAIL` | Resolve named DOFs. |
| Make table or link collisions severe | No general collision, force, or joint-limit watchdog exists. | `PARTIAL` | Add runtime watchdog and emergency halt. |
| Close viewer window or cause render exception | Main loop can exit while server requests remain blocked. | `FAIL` | Handle viewer close and coordinate server shutdown. |
| Run on a headless machine | Viewer startup may fail depending on machine. | `PARTIAL` | Detect headless environment or document supported display requirements. |

### Reset, Recovery, and State Consistency

| Mentor check | Expected current behavior | Result | Hardening action |
|---|---|---|---|
| Press `New Game` after ordinary move | Environment reset and logical reset usually restore starting state. | `GOOD` | Add end-to-end reset test with real environment. |
| Fail during grasp after `grasp_mode=True` | Method returns early without guaranteed release or state cleanup. | `FAIL` | Add cleanup in `finally` or enter fault state. |
| Fail during place retract | Retract error is ignored after placement. | `PARTIAL` | Report degraded state and require recovery if needed. |
| Fail during soft reset halt | Exception may propagate from controller path. | `PARTIAL` | Convert to controlled fault and halt. |
| Fail during soft reset alignment | Exception may propagate from controller path. | `PARTIAL` | Convert to controlled fault and halt. |
| Fail during home transit | Error returned, but recovery policy is incomplete. | `PARTIAL` | Enter degraded/fault state. |
| Fail during exact home-posture reset | Chess move still commits. | `PARTIAL` | Commit board if desired, but block auto-reply and require recovery. |
| Return-home fails after a human move with auto-reply enabled | Move result still has `physical_success=True`; auto-reply can start another move. | `FAIL` | Disable auto-reply on any home-return error. |
| Reset arm cannot reach requested start pose | `_settle_arm_to_start()` ignores `_move_grip_to()` result. | `FAIL` | Check result and fail reset explicitly. |
| Finger transition helper cannot reach pose | `_set_fingers_and_move_to()` ignores movement result. | `FAIL` | Check result and fail reset explicitly. |
| `env.reset()` fails or retries unexpectedly | Runtime ignores reset return value. | `PARTIAL` | Validate returned state and home posture before serving requests. |
| Directly call orchestrator `new_game()` without queued backend | Board bodies reset without the same full environment-reset sequence. | `PARTIAL` | Make reset ownership explicit in one component. |
| Modify active piece after occupancy update | Occupancy remains stale until later failure, if any. | `FAIL` | Reconcile actual poses continuously or before commands. |
| Non-moving piece is bumped during motion | Existing runtime does not verify it. | `FAIL` | Snapshot and reconcile all board pieces after every plan. |
| Piece lands within tolerance | Runtime snaps it to exact destination. | `GOOD` | Keep reconciliation, but log snap distance. |
| Piece lands outside tolerance | Error returned, but physical board remains displaced. | `PARTIAL` | Enter fault state and require reset or rollback. |

### Configuration and Startup Validation

| Mentor check | Expected current behavior | Result | Hardening action |
|---|---|---|---|
| Remove deployed model path | `resolve_model_paths()` gives clear error. | `GOOD` | Call comprehensive validation before environment startup. |
| Point model path to missing file | Clear `FileNotFoundError`. | `GOOD` | Resolve paths relative to project/package resources. |
| Launch from `/tmp` after editable install | Model files are not found because configured paths are CWD-relative. | `FAIL` | Resolve relative paths against repository or config directory. |
| Corrupt YAML syntax | Startup traceback. | `PARTIAL` | Wrap YAML parse errors with config filename and friendly message. |
| Remove required YAML key | Often `KeyError` during startup. | `PARTIAL` | Expand startup schema validation. |
| Set negative tolerance | Semantics become incorrect. | `FAIL` | Validate ranges. |
| Set zero grasp or release steps | Division-by-zero runtime error. | `FAIL` | Validate positive integer steps. |
| Set huge control scale | Unsafe mocap movement. | `FAIL` | Validate supported control range. |
| Set impossible home position | Bounded stage timeout likely, but startup still succeeds. | `PARTIAL` | Validate workspace bounds and home reachability. |
| Set graveyard overlapping board | Captured pieces can interfere with play. | `FAIL` | Validate zone separation at startup. |
| Set reserve zone outside floor or scene | Teleported pieces can behave unexpectedly. | `PARTIAL` | Validate zone bounds. |
| Set board size other than 8 | Chess logic remains 8x8 while mapper changes geometry. | `FAIL` | Enforce `board_size == 8`. |
| Set engine think time above UCI timeout | Engine may time out before expected result. | `PARTIAL` | Validate timeout relationship or configure timeout explicitly. |
| Set unsupported Stockfish skill | Behavior depends on engine. | `PARTIAL` | Validate `0 <= skill_level <= 20`. |
| Change config while runtime is active | Components load separate snapshots and may disagree if reconstructed later. | `PARTIAL` | Load one validated immutable application config. |
| Rely on `validate_config()` at startup | Runtime never calls it. | `FAIL` | Call validation from play and training entry points. |

Current `validate_config()` is too narrow. Its docstring mentions geometry and
model checks, but implementation mainly checks three environment keys, deployed
path types, and the base training model file. It should validate all runtime
invariants.

### Stockfish and External Process Behavior

| Mentor check | Expected current behavior | Result | Hardening action |
|---|---|---|---|
| Remove Stockfish executable while configured | Application construction fails. | `PARTIAL` | Either fail clearly before MuJoCo startup or degrade to human-only mode. |
| Configure `stockfish_path: null` | Human-only startup works. | `GOOD` | Make UI disable computer-play button. |
| Click computer-play with engine disabled | `RuntimeError` becomes HTTP 500. | `FAIL` | Convert to rejected move with user-facing message. |
| Kill Stockfish during game | Next computer request raises runtime error. | `PARTIAL` | Catch, report engine fault, and keep human-only mode available. |
| Stockfish hangs during handshake | Constructor times out, but subprocess cleanup is not guaranteed. | `PARTIAL` | Close or kill process on constructor failure. |
| Stockfish hangs during move search | Request fails after timeout; UI error handling is weak. | `PARTIAL` | Return structured engine timeout error. |
| Stockfish returns invalid move | Engine wrapper rejects it. | `GOOD` | Convert wrapper error to structured application result. |
| Stockfish emits excessive output | Read buffers have no explicit maximum. | `PARTIAL` | Add output limits. |
| Exit application normally | Main loop closes MuJoCo environment but does not explicitly close chess service. | `PARTIAL` | Close Stockfish in application shutdown. |
| Start a new game | Existing engine closes and restarts. | `GOOD` | Handle restart failure without leaving half-reset state. |
| Engine restart fails during `New Game` | Environment may already reset while orchestrator engine replacement fails. | `FAIL` | Construct replacement first or enter explicit fault state. |

### Packaging, Installation, and Deployment

| Mentor check | Expected current behavior | Result | Hardening action |
|---|---|---|---|
| Run from repository root with editable install | Main runtime works under expected setup. | `GOOD` | Keep as development workflow. |
| Run from another working directory | Relative deployed-model paths fail. | `FAIL` | Resolve paths against config directory. |
| Build wheel | Wheel builds successfully. | `GOOD` | Add wheel-content test. |
| Install wheel and start play mode | Required runtime resources and play entry point are missing. | `FAIL` | Package required resources and entry point. |
| Install wheel and render UI | Templates and static files are missing. | `FAIL` | Add `src.ui` package data. |
| Install wheel and load MuJoCo XML | Texture PNG files are missing. | `FAIL` | Add textures to package data. |
| Install wheel and load config | Repository-level configs are missing. | `FAIL` | Package config resources or support external config directory. |
| Install wheel and load controller | Deployed model ZIPs are missing. | `FAIL` | Package models or add documented download/install step. |
| Install fresh dependencies later | Dependencies are mostly unpinned and can drift. | `PARTIAL` | Add lock file or tested constraints file. |
| Reproduce Gymnasium Robotics behavior | Environment version warning indicates behavior drift is possible. | `PARTIAL` | Pin known-good versions. |
| Run on Windows | Training uses `SubprocVecEnv(..., start_method="fork")`; unsupported platform risk. | `PARTIAL` | Declare supported platform or select start method by OS. |
| Run on machine without writable Matplotlib config directory | Library import falls back to temporary cache with warning. | `PARTIAL` | Set writable runtime cache directory in deployment docs. |
| Run product command after installation | Play console command exists; resource packaging still needs smoke testing. | `PARTIAL` | Add clean-install smoke test. |
| Use Flask development server as store product | Suitable for local demo, not hardened deployment. | `PARTIAL` | State local-only scope or use production server wrapper. |

Built wheel inspection confirmed that only XML and STL assets were included from
`chess_env`; textures, configs, UI resources, models, and play entry point were
absent.

### Logging, Diagnostics, and Operational Control

| Mentor check | Expected current behavior | Result | Hardening action |
|---|---|---|---|
| Cause stage timeout | Error reason reaches move result. | `GOOD` | Include stage list and metrics in snapshot or log. |
| Cause unexpected exception | Physical executor often reduces exception to string; some paths become HTTP 500. | `PARTIAL` | Add structured fault records and traceback logging. |
| Run normal production mode | Environment logger uses `NullHandler` unless debug enabled. | `PARTIAL` | Keep rotating operational logs for warnings and faults. |
| Diagnose partial plan failure | Command result list exists internally but UI only sees final error. | `PARTIAL` | Persist plan ID, command history, and touched pieces. |
| Determine if system is healthy | No readiness or health endpoint exists. | `FAIL` | Add `/api/health` with runtime, engine, model, and fault status. |
| Determine current stage while move runs | Cached snapshots do not expose stage progress. | `PARTIAL` | Publish current stage and elapsed steps. |
| Stop unsafe motion manually | No emergency stop endpoint or UI action exists. | `FAIL` | Add halt/abort control. |
| Restart safely after fault | New game is the only practical recovery path and is not always immediate. | `PARTIAL` | Define recovery state machine. |
| Audit model versions | Model paths exist, but no checksums or compatibility metadata are exposed. | `PARTIAL` | Add model manifest with hashes and scene/config version. |

### Test Suite and Release Process

| Mentor check | Expected current behavior | Result | Hardening action |
|---|---|---|---|
| Run targeted non-integration groups | 72 tests pass. | `GOOD` | Preserve baseline. |
| Run `pytest -q` | Collection currently fails for two integration modules. | `FAIL` | Fix test-package imports and add CI. |
| Run exhaustive `64 x 63` sweep | Opt-in runner exists. | `GOOD` | Record results for final release artifacts. |
| Run fault injection with `NaN` sensors | No dedicated tests exist. | `FAIL` | Add tests. |
| Run partial-plan rollback tests | No dedicated tests exist. | `FAIL` | Add tests. |
| Run API malformed-payload tests | Missing JSON-shape and promotion-type cases. | `FAIL` | Add parameterized API tests. |
| Run package clean-install test | No test exists. | `FAIL` | Build wheel, install into temp venv, launch smoke test. |
| Run outside repository root | No test exists. | `FAIL` | Add CWD-independent smoke test. |
| Check dependency reproducibility | No lock or constraints file. | `PARTIAL` | Add tested dependency versions. |
| Verify startup validation | Existing schema test is narrow. | `PARTIAL` | Expand validator and test invalid configs. |
| Verify Stockfish unavailable behavior | No graceful-degradation test. | `FAIL` | Add engine-disabled and engine-crash tests. |
| Verify viewer-close behavior | No test exists. | `VERIFY` | Add lifecycle/manual test. |
| Verify server-port collision behavior | No test exists. | `VERIFY` | Add startup failure test. |

## Additional Code-Level Findings

### Missing MuJoCo ID validation

`src/chess_env/simulation.py` calls `mujoco.mj_name2id()` for finger actuators and
uses the returned value directly:

```python
actuator_id = mujoco.mj_name2id(...)
self.data.ctrl[actuator_id] = target
```

MuJoCo returns `-1` when a name is missing. Python then writes to the last array
element instead of raising. This is a silent wrong-actuator bug.

### Hard-coded robot DOF slice

`src/chess_env/base_env.py` assumes the first 15 DOFs are robot DOFs:

```python
robot_dof = 15
self.data.qvel[:robot_dof] = 0.0
self.data.qacc[:robot_dof] = 0.0
```

Adding or reordering joints can invalidate this assumption. Named joint address
lookup is safer.

### Reset ignores movement failures

`_settle_arm_to_start()` and `_set_fingers_and_move_to()` call `_move_grip_to()`
but ignore the returned boolean. Reset may appear successful even if the arm did
not converge.

### Place retract failure is ignored

After opening the fingers and verifying placement, `execute_place()` calls
`_move_z(...)` for retract but does not inspect its returned error. The sequence
can report success even when the arm did not return to hover height.

### Grasp failure can leave `grasp_mode` enabled

After `grasp_mode = True`, several early returns are possible. Cleanup is not
guaranteed. The next operation may run with stale held-piece assumptions.

### Return-home failure does not block auto-reply

`GameOrchestrator._run_plan()` commits the chess move even if
`return_to_home()` reports an error. Committing the board may be a reasonable
policy because the piece movement succeeded, but the next automatic engine move
must not start until the arm is recovered.

### Engine constructor can leak subprocess on handshake failure

`UciEngine.__init__()` starts the process and performs handshake reads. If the
handshake times out or raises, no constructor-level cleanup runs.

### Main shutdown does not close the chess engine

`main.py` closes the MuJoCo environment in `finally`, but it does not explicitly
close `orchestrator.chess_service`.

### Runtime path handling depends on current working directory

Configuration loading uses a source-tree-derived absolute path, but model paths
from `configs/deployed_models.yaml` are checked as relative paths against the
current process directory. Running the same installed source from `/tmp`
produced:

```text
FileNotFoundError: Model file(s) not found for: transit, descend, ascend.
```

## Existing Documentation Drift

The maintained documentation is detailed and useful, but several examples no
longer match the checked-in implementation. This matters for a product-like
submission because reviewers may follow the documented commands or use the
documented values as compatibility assumptions.

| Document | Stale documentation | Current implementation |
|---|---|---|
| `docs/01_runtime_usage.md` | Model examples use `models/transit/model.zip`, `models/descend/model.zip`, and `models/ascend/model.zip`. | `configs/deployed_models.yaml` uses `models/transit.zip`, `models/descend.zip`, and `models/ascend.zip`. |
| `docs/09_training.md` | Deployed model examples use nested model directories. | Active config uses ZIP files directly under `models/`. |
| `docs/10_configuration.md` | Deployed model examples use nested model directories. | Active config uses ZIP files directly under `models/`. |
| `docs/07_scene_assets.md` | Graveyard origins are documented as X `0.640` with a `4 x 4` grid. | `configs/chess.yaml` uses X `0.550` with a `6 x 4` grid. |
| `docs/02_architecture.md` | Main-loop example calls `backend.process_request(env=env, timeout=0.05)`. | Current method is `backend.process_request(timeout=0.05)`. |
| `docs/08_web_ui.md` | Documents `process_request(env, timeout)`. | Current backend stores the environment during construction and accepts only `timeout`. |
| `docs/11_testing.md` | Documents `run_all_square_moves(stop_on_failure=True)` as the default. | Function default is `stop_on_failure=False`; the pytest wrapper explicitly passes `True`. |
| Untracked presentation notes | Use the older `backend.process_one(env=env)` name. | Runtime uses `backend.process_request(timeout=0.05)`. |

Required fix:

- Update documentation after the runtime hardening work.
- Add a small documentation verification checklist to the release process.
- Keep sample commands executable in CI where practical.

## Recommended One-Pass Hardening Plan

The following order minimizes rework.

### Phase 1: Establish Fault Semantics

1. Add runtime states: `READY`, `BUSY`, `FAULTED`, and optionally `RECOVERING`.
2. Add a structured fault model with code, message, stage, command, timestamp,
   and whether reset is required.
3. Reject new moves while `BUSY` or `FAULTED`.
4. Keep snapshot and health endpoints available while faulted.
5. Disable automatic computer reply after any degraded result.
6. Add explicit `halt` and `new-game reset` recovery paths.

### Phase 2: Fail Closed on Dirty Data

1. Add reusable validators:

   ```python
   require_finite_vector(name, value, expected_shape)
   require_finite_scalar(name, value)
   require_workspace_xy(name, xy)
   require_normalized_quaternion(name, quat)
   require_model_action(action)
   ```

2. Validate observations before model inference.
3. Validate model actions before applying them.
4. Validate MuJoCo pose and velocity state after each simulation step.
5. Validate active-piece pose before grasp alignment.
6. Clip actions to supported bounds or reject them explicitly.

### Phase 3: Add Transaction and Reconciliation Boundaries

1. Snapshot touched free-joint poses and occupancy before physical plan
   execution.
2. Roll back touched pieces on command failure, or force a reset-required fault.
3. Verify all active board pieces before starting a move.
4. Verify all active board pieces after completing a plan.
5. Detect destination-square physical occupancy independently of expected
   occupancy.
6. Log every reconciliation snap distance.

### Phase 4: Harden Reset and Motion Bounds

1. Check every `_move_grip_to()` return value.
2. Add hard workspace bounds to scripted targets and mocap updates.
3. Add upright-piece checks for roll, pitch, yaw, and resting Z.
4. Replace hard-coded DOF slices with named-joint address resolution.
5. Validate all MuJoCo object IDs before indexing arrays.
6. Report retract failure instead of discarding it.
7. Guarantee `grasp_mode` cleanup or require reset after grasp fault.
8. Add a general watchdog for non-finite state, excessive speed, joint limit
   violations, and severe contacts.

### Phase 5: Harden API, Queue, and Engine Lifecycle

1. Validate HTTP payload object shape and field types.
2. Add stable JSON error handlers for unexpected exceptions.
3. Bound the request queue and add request timeouts.
4. Add browser fetch timeouts and fault rendering.
5. Add server startup-health detection.
6. Convert engine-disabled, engine-timeout, and engine-crash conditions into
   controlled application results.
7. Close Stockfish on startup failure, restart failure, and application exit.
8. Add `/api/health`.

### Phase 6: Fix Packaging and Reproducibility

1. Add package data for textures, templates, and static browser files.
2. Package configs or support an explicit external config directory.
3. Add model manifest and documented model installation path.
4. Resolve resources with `importlib.resources` or a deliberate application
   data directory.
5. Pin tested dependency versions or add a constraints file.
6. Add clean-wheel installation smoke test.
7. Add launch-from-different-CWD test.

### Phase 7: Expand Fault-Injection Tests

Add automated tests for:

- `NaN`, `Inf`, and wrong-shape grip position.
- `NaN`, `Inf`, and wrong-shape model actions.
- Piece manually displaced slightly and far away.
- Piece tipped onto its side.
- Unrelated piece bumped during movement.
- Capture failure after captured-piece removal.
- Castling failure after king movement.
- Promotion failure after pawn removal.
- Grasp failure after `grasp_mode=True`.
- Return-home failure with auto-reply enabled.
- Engine disabled, missing, killed, hanging, and invalid response.
- JSON scalar, array, oversized body, and invalid promotion type.
- Queue flood, busy rejection, and request timeout.
- Missing actuator, missing body, reordered robot joints, and invalid XML/config
  geometry.
- Wheel build, wheel installation, UI rendering, MuJoCo texture loading, and
  play entry point.
- Full `pytest -q` collection.

## Suggested Release Gate

Before final submission, require all of the following:

1. `pytest -q` collects and passes.
2. Fault-injection tests pass for dirty sensor data, partial plans, reset
   failures, and engine errors.
3. Exhaustive `64 x 63` physical sweep result is recorded.
4. A built wheel or source release starts from a clean environment using the
   documented command.
5. Play mode starts from a working directory outside the repository root.
6. UI remains available during faults and clearly requires reset when needed.
7. Every fault path either rolls back or blocks further movement.
8. No non-finite value reaches MuJoCo control state.
9. Automatic reply never starts after a degraded home-return result.
10. Runtime shutdown closes MuJoCo, Flask serving, and Stockfish cleanly.

## Practical Submission Position

The current implementation is a strong functional simulation with a thoughtful
layered architecture. The biggest improvement is not adding more successful
paths. It is making every unsuccessful path explicit and controlled:

```text
detect -> halt -> report -> block unsafe continuation -> reset or recover
```

That behavior directly matches the mentor's stated emphasis on handling
unexpected scenarios without losing control.
