# Automation and Evaluation

## Scripts Overview

| Script | Purpose | Requires RL Model? |
|--------|---------|-------------------|
| `scripts/verify_physics.py` | Physics sanity checks | No |
| `scripts/eval.py` | Per-scenario RL evaluation | Yes |
| `scripts/eval_grasp.py` | Full pick sequence evaluation | Yes |
| `scripts/eval_sequence.py` | Multi-scenario chain evaluation | Yes |
| `scripts/test_grasp_physics.py` | Grasp physics unit tests | No |
| `scripts/test_corners.py` | Corner/edge reachability stress test | Yes |
| `scripts/visualize.py` | Human-render observation loop | Optional |
| `scripts/record_move.py` | Offscreen frame capture | Yes |
| `scripts/train.py` | Training entry point | N/A |

---

## `scripts/verify_physics.py`

Standalone physics test suite. No RL model required. Tests run sequentially and print PASS/FAIL:

```bash
python scripts/verify_physics.py [--debug]
```

**Tests**:
1. `test_xml_integrity`: Load `ChessTaskEnv`, verify `robot0:grip` site exists.
2. `verify_grasp_xml_changes`: Check `cube_mass=0.05`, `Kp=150000`, `ctrlrange=[0, 0.05]`, `GRASP_Z=0.425`.
3. `test_static_stability`: Reset with `hide_object=False`, take 100 zero-action steps, check cube drift < 1mm.
4. `test_kinematic_reachability`: Sample 2 random board positions; `_settle_arm_to_start` at both `safe_z` and `grasp_z`; check arm error < 5mm.
5. `test_teleport_verification`: Reset with `hide_object=True`; check cube at `hidden_object_pos ± 1mm`.

---

## `scripts/eval.py`

Evaluates SAC model performance per scenario:

```bash
python scripts/eval.py --model path/to/model.zip [--scenarios transit,descend,ascend] \
    [--n-episodes 100] [--visualize] [--delay 0.02] [--debug]
```

**Logic**:
1. Load model and temp env.
2. For each scenario, create a new env with `force_scenario=scenario, force_drift_limit=eval_drift_limit`.
3. Run N episodes: `model.predict(obs, deterministic=True)` → `env.step()` → track success/crash/timeout.
4. Print summary table.

**Output metrics**: `success_rate`, `crash_rate`, `timeout_rate`, `avg_reward`.

**Key parameter**: `eval_drift_limit=0.010` (1cm) from config. This is stricter than the training curriculum's end value of 1cm — same strict final constraint, applied from episode 0.

---

## `scripts/eval_grasp.py`

Evaluates the full pick sequence with scripted grasp:

```bash
python scripts/eval_grasp.py --model path/to/model.zip \
    [--n-episodes 50] [--drift-limit 0.010] [--debug] [--visualize] [--wait]
```

**Flow** (per episode):
1. Reset with `force_start_pos=home_pos, force_cube_pos=src_xy, force_scenario=transit`.
2. Run transit RL loop → expect "success".
3. `soft_reset("descend", ...)`.
4. Run descend RL loop → expect "success".
5. `execute_grasp()`.
6. `soft_reset("ascend", ..., grasp_mode=True preserved)`.
7. Run ascend RL loop → expect "success" (cube held).
8. `soft_reset("transit", ..., toward home)`.
9. Run transit-home RL loop → expect "success".
10. Evaluate grasp quality (cube drift, Z error, rotation).

**Output**: Per-step success rates, grasp quality metrics (xy_drift_mm, z_error_mm, max_rotation_deg), failure breakdown.

**Helper**: `reset_episode_timelimit(env)` manually zeroes the TimeLimit wrapper's `_elapsed_steps` counter between scenarios. This prevents episode timeout across scenario boundaries (since `soft_reset` doesn't trigger a new Gymnasium episode).

---

## `scripts/eval_sequence.py`

Evaluates arbitrary scenario chains using `ChainEpisodeRunner`:

```bash
python scripts/eval_sequence.py --model path/to/model.zip \
    --chain full_move [or pick, place, vertical, or comma-separated] \
    [--n-episodes 10] [--drift-limit 0.010] [--debug] [--visualize] [--delay 0.02]
```

### `ChainEpisodeRunner`

The core class for chain evaluation:

```python
runner = ChainEpisodeRunner(env, model, chain, drift_limit, debug, delay, wait)
results = runner.run_one_chain(waypoints)
```

**Internal loop**:
1. Reset env at scenario 0.
2. For each scenario:
   a. If not first scenario: call `soft_reset` with `nom_exit`.
   b. Run RL loop until success/crash/timeout.
   c. Record metrics.
   d. If not last: compute `nom_exit = exit_waypoint(scenario, cell_xy)`.
3. Record transition diagnostics (halt steps, align steps, alignment error).

**Waypoint assignment** for each scenario in the chain:
- `full_move`: `[src, src, src, dst, dst, dst]`
- `pick`: `[src, src, src]`
- `place`: `[dst, dst, dst]`
- Custom: position-based assignment tracking transit/non-transit pattern

**Summary output** (`print_summary`):
- Per-scenario success rates, avg distance, speed, steps.
- Full chain success rate (all scenarios in chain succeeded).
- Failure breakdown (failed at which scenario, crash reasons).
- Transition diagnostics (avg halt/align steps, alignment error).

---

## `scripts/test_grasp_physics.py`

Physics unit tests for the grasp stage (no RL model):

```bash
python scripts/test_grasp_physics.py [--n-trials 10] [--visualize] [--debug]
```

**Tests** (run N trials each):

1. **Static Grasp** (`test_static_grasp`):
   - Teleport cube to valid position
   - Move arm to HOVER_Z above cube
   - Execute `execute_grasp()`
   - Report alignment error, cube displacement, finger position

2. **Lift** (`test_lift`):
   - Grasp cube, then lift arm 3mm/step to SAFE_Z
   - Monitor cube Z vs. grip Z offset (should be ~15mm)
   - Detect drop if XY error >30mm or Z deviation >40mm

3. **Transit Held** (`test_transit_held`):
   - Grasp cube, lift to SAFE_Z
   - Translate 100mm horizontally
   - Monitor for cube drop, measure max yaw rotation

**Summary**: Per-test success rates and failure reasons.

---

## `scripts/test_corners.py`

Corner/edge stress test (requires RL model):

```bash
python scripts/test_corners.py  # hardcoded model path
```

Tests 5 positions:
- `[0.6, -0.3]`, `[0.6, 0.3]`, `[1.2, -0.3]`, `[1.2, 0.3]` (corners)
- `[0.9, 0.0]` (center)

For each: runs `["transit", "descend", "ascend"]` chain with that position as source/destination. Uses `ChainEpisodeRunner`. Prints chain summary.

**Limitation**: The hardcoded corner coordinates are approximate — they don't perfectly match the actual board corners defined in the config.

---

## `scripts/visualize.py`

Continuous human-render loop:

```bash
python scripts/visualize.py [--model path] [--scenario transit|descend|ascend|random] \
    [--delay 0.02] [--episodes N] [--wait] [--debug]
```

If no model file is found, uses random actions. Useful for visual debugging of environment setup, arm behavior, and reset quality.

---

## `scripts/record_move.py`

Offscreen frame capture for a full 6-scenario chain:

```bash
python scripts/record_move.py  # hardcoded model/positions
```

Saves PNG frames to `logs/visual_recording/frame_NNNN.png` for later compositing into a video.

---

## Argparse Conventions

Current scripts each define their own argparse. Common flags across scripts:
- `--visualize`: Enable human rendering (`render_mode="human"`)
- `--delay FLOAT`: Sleep seconds between steps during visualization
- `--debug`: Enable environment debug logging
- `--n-episodes INT`: Number of evaluation episodes
- `--drift-limit FLOAT`: Override tube drift limit

There is **no shared argparse module** — each script redeclares these.
