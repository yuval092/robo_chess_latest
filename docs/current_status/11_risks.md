# Risks and Mitigations

This document catalogues known risks in the RoboChess system, their severity, and how the code mitigates them.

---

## 1. Physical Arm Risks

### 1.1 Piece Drop During Transit

**Risk**: The arm grasps a piece, lifts it to SAFE_Z, and the piece slips out before reaching the destination.

**Detection**: `cube_held_xy_limit` (30mm) and `cube_held_z_limit` (20mm) are checked continuously during ascent and transit. If the cube's position diverges beyond these thresholds from the gripper, the move is flagged as failed.

**Mitigation**: Finger-ramp-close actuator drives fingers to a secure grip (`grasp_ramp_end = 0.010`). `grasp_hold_steps = 2` lets physics settle after close. Hold verification checks `finger_joint >= grasp_verify_finger_threshold` (finger stalls against cube, proving contact) and XY/Z proximity checks.

**Residual risk**: Physics simulation is not perfectly elastic. A dropped piece leaves the graveyard empty and the board in a wrong state. The physical executor detects this and returns `success=False`, which propagates to the UI as an error. The game is not committed. The user must reset or reload.

---

### 1.2 Piece Knock During Transit

**Risk**: The arm's wrist/gripper hits a neighbouring piece while transiting at SAFE_Z.

**Mitigation**: SAFE_Z = 0.550m is 120mm above the table surface (table at 0.400m) and 80mm above the tallest piece (king at ~26mm height + table Z ≈ 0.426m). Transit happens in a straight line at SAFE_Z only after the arm has ascended fully. The gripper-down `vertical_quat` minimises lateral footprint.

**Residual risk**: In very crowded endgames with pieces at the board edge, the gripper body (not just the fingers) could theoretically clip a piece. No explicit collision avoidance planning is done; the straight-line path assumes SAFE_Z is clear.

---

### 1.3 Failed Home Return

**Risk**: After placing a piece, the arm's `return_to_home()` call fails (timeout, crash).

**Mitigation**: Home return failure is treated as **non-fatal**. The move has already been committed to the chess board and piece tracker. The error is stored in `GameOrchestrator.error` and surfaced to the UI as a warning, but the game continues. The arm is likely parked somewhere over the board.

**Residual risk**: Subsequent moves may fail because the arm is not at home position. The user should use the "Refresh" button and verify the arm position before continuing.

---

### 1.4 Post-Place Position Drift

**Risk**: After the arm releases a piece, physics simulation bounces the piece slightly, leaving it off-centre.

**Mitigation**: After every successful place, `MovementExecutor` teleports the piece to the exact grid position (`placed_xyz = board_mapper.square_to_piece_xyz(dst_square)`). This overrides any physics drift and ensures the next move finds the piece exactly where expected.

**Residual risk**: The teleport happens after a reconciliation check (XY error < 20mm, Z error < 10mm). If the piece drifted beyond 20mm (severe physics failure), the move is rejected before teleport.

---

## 2. Chess Logic Risks

### 2.1 Tracker Divergence from Board State

**Risk**: `LogicalPieceTracker` and `ChessService.board` get out of sync — the tracker thinks a piece is at square X but chess board says different.

**Mitigation**: Both are updated atomically in `_submit_move()`: chess board is pushed (`chess_service.push(move)`) and tracker is updated (`piece_tracker.apply_committed_move(move, plan)`) in the same call, with `is_busy` preventing concurrent modifications. If physical execution fails, neither the board nor the tracker is updated.

**Residual risk**: If an exception is thrown between `chess_service.push()` and `piece_tracker.apply_committed_move()`, the two states diverge. This is a narrow window; `_submit_move` has a `finally: is_busy = False` block but no transaction rollback. A diverged tracker would cause wrong piece IDs in future moves, which `PhysicalOccupancy.assert_piece_at()` would catch (raising an error before arm motion).

---

### 2.2 Reserve Piece Exhaustion

**Risk**: All 8 reserve queens are already on the board when a 9th pawn promotes to queen.

**Mitigation**: `LogicalPieceTracker.reserve_piece_for()` raises `ValueError` if no free reserve piece is found. This propagates as a rejected move to the UI.

**Residual risk**: 8 reserve pieces per type per colour is far more than any real game could use (only 8 pawns per colour, so at most 8 promotions). This is effectively impossible in practice.

---

### 2.3 Graveyard Slot Overflow

**Risk**: More pieces are captured than there are graveyard slots (4×4 = 16 slots, but maximum captures is 15 per colour).

**Mitigation**: 16 slots > 15 maximum captures. There is exactly one slot of buffer. The slot index is computed by `len(self._captured[color])`, which increments monotonically.

**Residual risk**: If a bug causes a piece to be captured twice (which would indicate a tracker bug), the slot index would eventually exceed `rows * cols`, and `PieceTeleporter._slot_xyz()` would raise `ValueError: Slot is outside configured slot grid`.

---

## 3. Concurrency Risks

### 3.1 MuJoCo Thread Safety

**Risk**: MuJoCo's data structures (`model`, `data`) are not thread-safe. Flask runs in a separate thread. Simultaneous access would corrupt simulation state.

**Mitigation**: `QueuedUIBackend` (in `scripts/run_chess_ui.py`) serializes all game API calls through a `queue.Queue`. The main thread (running the MuJoCo viewer loop) drains the queue on every step, processes the request, and puts the result back on a response queue. Flask threads block waiting for their result.

**Residual risk**: If the main thread's viewer loop stalls (e.g., user closes viewer), the queue never drains and Flask requests block forever. No timeout is implemented on the request side.

---

### 3.2 XML Path Race Condition During Env Construction

**Risk**: Two threads constructing `ChessSimulationEnv` simultaneously could race on the `MODEL_XML_PATH` module-level variable.

**Mitigation**: `_xml_path_lock` (a `threading.Lock`) wraps the critical section that swaps `MODEL_XML_PATH`, calls `super().__init__()`, and restores the original path. Only one environment can be under construction at a time.

**Residual risk**: The lock protects construction only. If two environments are used simultaneously in different threads after construction, MuJoCo's underlying C library may have global state. In practice, only one environment is constructed per process.

---

### 3.3 is_busy Flag Race

**Risk**: `is_busy` is a plain Python bool. Two HTTP requests arriving simultaneously could both see `is_busy == False` and both proceed.

**Mitigation**: The `QueuedUIBackend` queue serializes all calls — only one runs at a time. Without the queue (e.g., in testing with direct orchestrator calls), there is no thread safety guarantee.

**Residual risk**: In direct multi-threaded usage without `QueuedUIBackend`, simultaneous moves would corrupt game state. This is not a normal deployment scenario.

---

## 4. Physics Simulation Risks

### 4.1 Finger Threshold Calibration Drift

**Risk**: The `grasp_verify_finger_threshold` value (0.016) was empirically calibrated for a specific cube size. If piece dimensions or physics parameters change, this threshold may be wrong — either causing false "empty grasp" failures or accepting a dropped piece.

**Mitigation**: The value is in `configs/env.yaml` and can be recalibrated with `scripts/test_grasp_physics.py`.

**Residual risk**: No automated regression test verifies this threshold. A change to `pieces.cube_height_m` or `freejoint_damping` could silently invalidate it.

---

### 4.2 Non-Deterministic Physics

**Risk**: MuJoCo physics is deterministic for a given seed but may produce different results across MuJoCo versions or hardware (floating-point order of operations).

**Mitigation**: The scripted controller uses conservative thresholds (MIN_STEP = 2mm, HOVER_Z = 30mm above grasp). Post-place teleport normalises final position regardless of physics noise.

**Residual risk**: Performance metrics (step counts, success rates) may differ slightly across environments. Eval scripts report statistics rather than pass/fail on exact counts.

---

## 5. Configuration Risks

### 5.1 Board Centre Mismatch

**Risk**: `env.yaml: table_center_xy` and `chess.yaml: board.center_xy` must match. If they diverge, the arm will target wrong positions.

**Mitigation**: Both are currently set to `[0.88, 0.2641]`. `eval_chess_reachability.py` validates computed square coordinates against hardcoded expected values, which would catch a mismatch.

**Residual risk**: No automated alert if one is changed without updating the other.

---

### 5.2 Config Not Cached

**Risk**: `load_config()` reads YAML from disk on every call. If the file is modified at runtime, the new values take effect immediately and inconsistently.

**Mitigation**: All configs are loaded at class construction time and stored as instance attributes, so mid-session changes don't affect running objects.

**Residual risk**: If a config is loaded inside a hot loop, it creates unnecessary file I/O. Currently this doesn't happen.

---

## 6. Summary Table

| Risk | Severity | Detected? | Mitigated? |
|---|---|---|---|
| Piece drop during transit | High | Yes (position monitor) | Yes (hold verify + teleport) |
| Piece knock during transit | Medium | No | Partial (SAFE_Z clearance) |
| Failed home return | Low | Yes | Yes (non-fatal, game continues) |
| Tracker/board divergence | High | Yes (occupancy assertions) | Partial (no rollback) |
| Reserve piece exhaustion | Low | Yes (ValueError) | Yes (8 reserves >> max needed) |
| Graveyard overflow | Low | Yes (ValueError) | Yes (16 slots > 15 max) |
| MuJoCo thread safety | High | No | Yes (QueuedUIBackend) |
| XML path race condition | Medium | No | Yes (threading.Lock) |
| is_busy race | Medium | No | Yes (queue serialization in prod) |
| Finger threshold drift | Medium | No | Partial (recalibration script exists) |
| Board centre mismatch | High | Partial (reachability eval) | Partial (no auto-check) |
