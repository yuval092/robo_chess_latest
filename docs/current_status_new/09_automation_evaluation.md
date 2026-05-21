# Automation and Evaluation

All evaluation scripts live in `scripts/`. They use the `ScriptedController` and require no RL model. All scripts share a common argparse setup from `src/utils/args.py`.

---

## Common Arguments (`src/utils/args.py`)

```python
def add_common_args(parser) -> ArgumentParser:
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--drift-limit", type=float, default=0.010)  # 10mm tube
    parser.add_argument("--visualize", action="store_true")           # Human render
    parser.add_argument("--delay", type=float, default=0.0)           # Sleep per step
    parser.add_argument("--debug", action="store_true")               # Verbose logging
    return parser

def make_env(args, force_scenario=None, hide_object=True):
    render_mode = "human" if args.visualize else None
    return gym.make("ChessFetchTask-v0",
                    render_mode=render_mode,
                    force_scenario=force_scenario,
                    hide_object=hide_object,
                    debug=args.debug)
```

Passing `--visualize` opens the MuJoCo viewer and enables per-step rendering. `--delay N` inserts N seconds of `time.sleep` between each rendered step (slows animation for human viewing). Pass `render_fn=env.render` to `ScriptedController` to enable rendering in movement stages.

---

## `scripts/visualize.py` — Human Render Loop

**Purpose**: Watch the arm execute movement scenarios in real time.

```bash
python scripts/visualize.py --scenario transit
python scripts/visualize.py --scenario pick --src-xy "0.88 0.2641" --delay 0.02
python scripts/visualize.py --scenario full_move --src-xy "0.88 0.2641" --dst-xy "1.10 0.40"
python scripts/visualize.py --scenario full_move --episodes 3 --wait
```

**Supported scenarios**: `transit`, `descend`, `ascend`, `pick`, `full_move`

**Notes**:
- `args.visualize` is forced to `True` (always renders).
- `render_fn=env.render` is passed to `ScriptedController` so transit/descend/ascend stages render frame-by-frame.
- `--wait` pauses between episodes (press Enter to continue).
- `--delay 0.02` slows the animation to 20ms per physics step (50 FPS max).

---

## `scripts/eval_stages.py` — Per-Stage Accuracy

**Purpose**: Measure transit, descend, and ascend success rates in isolation.

```bash
python scripts/eval_stages.py --stages transit,descend,ascend --n-episodes 20
python scripts/eval_stages.py --stages descend --n-episodes 50 --debug
```

**What it does**:
- For each stage, creates a fresh env with `force_scenario=stage`
- Runs N episodes with random goal positions sampled from the full board area
- Records: success rate, crash rate, timeout rate, avg/p95 error_mm, crash reasons

**Output example**:
```
======================================================================
Stage         Success    Crash   Timeout    AvgErr    P95Err
----------------------------------------------------------------------
transit       100.0%    0.0%     0.0%      2.7mm      3.9mm
descend       100.0%    0.0%     0.0%      0.8mm      1.5mm
ascend         95.0%    5.0%     0.0%      3.5mm      3.8mm
======================================================================
```

**Interpretation note**: Descend/ascend tube breach rate in isolated tests (~5–20%) is higher than in the pick sequence (~0%) because isolated tests start from a fresh reset with 3mm settle tolerance. When run as part of a pick sequence (after transit aligns the arm to < 4mm XY), breach rate is 0%. For end-to-end quality assessment, use `eval_sequence.py` instead.

---

## `scripts/eval_sequence.py` — Full Chain Evaluation

**Purpose**: Measure success of complete movement chains across multiple episodes.

```bash
python scripts/eval_sequence.py --chain full_move --n-episodes 20
python scripts/eval_sequence.py --chain pick --n-episodes 10 --visualize --delay 0.02
python scripts/eval_sequence.py --chain vertical --src-xy "1.19 0.004" --n-episodes 5
```

**Available chains**:
- `full_move`: Transit → descend → grasp → ascend → transit → descend → place → ascend
- `pick`: Transit → descend → grasp → ascend
- `vertical`: Descend → ascend (no transit, tests tube-following from given XY)

**Output example**:
```
=================================================================
Chain: full_move  |  Episodes: 5  |  Full success: 5/5 (100.0%)

Stage          Success    Fail
-----------------------------------
transit         10/10    
descend         10/10    
grasp            5/5    
ascend          10/10    
place            5/5    
=================================================================
```

The per-stage breakdown helps identify which stage is failing in a complex chain.

---

## `scripts/eval_stress.py` — Corner & Grid Stress Test

**Purpose**: Verify the arm can pick/place from all positions on the board, not just the center.

```bash
python scripts/eval_stress.py --chain pick --n-episodes 3
python scripts/eval_stress.py --chain vertical --n-episodes 5
python scripts/eval_stress.py --chain pick --grid --grid-size 4 --n-episodes 2
```

**Default positions (5)**: The 4 corner chess squares + center, computed from `env.yaml`.
Chess squares are centered on an 8×8 grid within the usable board area (table extents minus `edge_margin`):
```
near_right: row 0, col 0 → (0.609, -0.007)
near_left:  row 0, col 7 → (0.609,  0.535)
far_right:  row 7, col 0 → (1.151, -0.007)
far_left:   row 7, col 7 → (1.151,  0.535)
center:     (cx, cy)     → (0.880,  0.264)
```
Note: these are actual chess square centers, not the board edge (x=0.570 / x=1.190). The board edge
is reachable in transit but is NOT a chess square position, so it is not tested here.

**Grid mode** (`--grid --grid-size N`): Creates an N×N grid of positions across the usable board area. Useful for identifying systematic dead zones.

**Output example**:
```
=======================================================
Position                             XY     Rate
-------------------------------------------------------
near_right              (0.609, -0.007)    100%  OK
near_left                (0.609, 0.535)   100%  OK
far_right               (1.151, -0.007)    100%  OK
far_left                 (1.151, 0.535)   100%  OK
center                   (0.880,  0.264)   100%  OK

Overall success rate: 100.0%
=======================================================
```

Positions with rate < 80% are flagged as `LOW`.

---

## `scripts/verify_physics.py` — Physics Sanity Checks

**Purpose**: Run after any environment change (XML, config, arm position) to confirm the simulation is correctly set up.

```bash
python scripts/verify_physics.py
# Expected: SYSTEM HEALTHY (6/6 tests passed)

python scripts/verify_physics.py --debug
# Shows detailed output for each test
```

See the Testing and Validation document for detailed test descriptions.

**When to run**:
- After modifying any `.xml` file
- After changing `env.yaml` geometry constants
- After changing `torso_height`
- After any code change to `_reset_sim` or `_env_setup`

---

## `scripts/test_grasp_physics.py` — Grasp Quality Tests

**Purpose**: Validate the full grasp pipeline under specific scenarios.

```bash
python scripts/test_grasp_physics.py --n-trials 5
```

**Tests**:
1. **Static grasp**: Place cube at GRASP_Z, close fingers, verify grip
2. **Lift test**: Grasp, then ascend 10cm, verify cube stays within limits
3. **Transit-held test**: Grasp, transit to 3 different squares, verify cube never drops

**Precondition assertions** (run at start, before any trial):
- Cube mass = 0.05 kg (within 0.001)
- Actuator Kp = 20000 (within 1%)
- `GRASP_VERIFY_FINGER_THRESHOLD = 0.016` (within 0.001)
- `grasp_z = 0.430m`

If any precondition fails, the script aborts with a clear message explaining the mismatch and what to fix.

---

## Recommended Evaluation Workflow

Run these in order after any code change:

```bash
# 1. Unit tests (fast, < 5 seconds)
pytest tests/ -v

# 2. Physics verification
python scripts/verify_physics.py

# 3. Stage accuracy
python scripts/eval_stages.py --stages transit,descend,ascend --n-episodes 10

# 4. Full sequence
python scripts/eval_sequence.py --chain full_move --n-episodes 5

# 5. Stress test (all corners)
python scripts/eval_stress.py --chain pick --n-episodes 3

# 6. Grasp quality
python scripts/test_grasp_physics.py --n-trials 3

# 7. Visual check (optional but recommended for any motion-related change)
python scripts/visualize.py --scenario full_move --episodes 1 --delay 0.02
```

### Performance Benchmarks (Current Configuration)

| Metric | Value | Notes |
|--------|-------|-------|
| Transit success | 100% | avg 2.7mm, p95 3.9mm |
| Descend success (in sequence) | ~100% | 0% breach when preceded by transit |
| Ascend success (in sequence) | ~100% | 0% breach when in sequence |
| Full pick success | 100% | 5/5 at center |
| Full full_move success | 100% | 5/5 center→offset |
| Stress test (5 positions) | 100% | All corners + center |
| pytest pass rate | 12/12 | 100% |
