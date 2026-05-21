# Stage 4: Evaluation Scripts

## Goal

Rewrite all evaluation scripts to use the `ScriptedController` instead of an RL model. Introduce a shared argparse module so all scripts share `--visualize`, `--delay`, `--debug`, `--n-episodes`, and `--drift-limit` flags. Create a physics test script, a per-stage accuracy script, a full sequence script, and a stress test script. Remove or archive RL-dependent scripts.

---

## Preconditions

- Stage 3 is complete: `ScriptedController` passes all validation checks.
- `src/utils/args.py` exists (created in Stage 3).
- `scripts/verify_physics.py` still passes.

---

## 4.1 — Scripts to Remove / Archive

| Script | Action | Reason |
|--------|--------|--------|
| `scripts/train.py` | **DELETE** | No RL training |
| `scripts/record_move.py` | **DELETE** | RL-dependent, hardcoded model path |
| `scripts/eval.py` | **DELETE** | RL per-scenario eval, replaced by `eval_stages.py` |
| `scripts/eval_grasp.py` | **DELETE** | RL full pick eval, replaced by `eval_sequence.py` |
| `scripts/test_corners.py` | **REPLACE** | Hardcoded corners; replaced by `eval_stress.py` |

Before deleting, confirm there are no imports from these scripts in other files:
```bash
grep -r "eval\.py\|train\.py\|record_move\|eval_grasp\|test_corners" scripts/ src/ tests/
```

---

## 4.2 — Scripts to Keep / Update

| Script | Action |
|--------|--------|
| `scripts/verify_physics.py` | **UPDATE**: Remove `drift_curriculum_steps` param, use new `__init__` signature |
| `scripts/visualize.py` | **REWRITE**: Remove SAC import, use ScriptedController |
| `scripts/eval_sequence.py` | **REWRITE**: Remove SAC, use ScriptedController |
| `scripts/test_grasp_physics.py` | **UPDATE**: Minor cleanup, no RL dependency |

---

## 4.3 — New Script: `scripts/eval_stages.py`

Tests each stage independently (transit, descend, ascend) over N episodes, reports accuracy statistics. No RL model required.

```python
"""
eval_stages.py — Per-stage accuracy evaluation using ScriptedController.

Usage:
    python scripts/eval_stages.py [--stages transit,descend,ascend]
        [--n-episodes 50] [--drift-limit 0.010] [--visualize] [--delay 0.02] [--debug]
"""
import sys, os, argparse, time
import numpy as np
sys.path.append(os.getcwd())

import src.chess_env
from src.chess_env.controller import ScriptedController
from src.utils.args import add_common_args, make_env

STAGE_CHOICES = ["transit", "descend", "ascend"]


def evaluate_stage(stage: str, args) -> dict:
    """Run N episodes of a single stage and collect accuracy statistics."""
    env = make_env(args, force_scenario=stage)
    ctrl = ScriptedController(env, drift_limit=args.drift_limit)
    inner = env.unwrapped

    successes = 0
    crashes = 0
    timeouts = 0
    errors_mm = []
    step_counts = []
    crash_reasons = {}

    for ep in range(args.n_episodes):
        obs, info = env.reset()
        target_xy = inner.goal_pos[:2].copy()

        if stage == "transit":
            result = ctrl.run_transit(target_xy)
        elif stage == "descend":
            result = ctrl.run_descend(target_xy)
        elif stage == "ascend":
            result = ctrl.run_ascend(target_xy)

        if result.success:
            successes += 1
            errors_mm.append(result.error_mm)
        elif result.crash_reason == "TIMEOUT":
            timeouts += 1
        else:
            crashes += 1
            reason = result.crash_reason or "UNKNOWN"
            crash_reasons[reason] = crash_reasons.get(reason, 0) + 1
        step_counts.append(result.steps)

        if args.visualize:
            env.render()
            if args.delay > 0:
                time.sleep(args.delay)

    env.close()
    return {
        "stage": stage,
        "n_episodes": args.n_episodes,
        "success_rate": successes / args.n_episodes,
        "crash_rate": crashes / args.n_episodes,
        "timeout_rate": timeouts / args.n_episodes,
        "avg_error_mm": np.mean(errors_mm) if errors_mm else 0.0,
        "p95_error_mm": float(np.percentile(errors_mm, 95)) if errors_mm else 0.0,
        "avg_steps": np.mean(step_counts),
        "crash_reasons": crash_reasons,
    }


def print_summary(results: list):
    print("\n" + "=" * 70)
    print(f"{'Stage':<12} {'Success':>8} {'Crash':>8} {'Timeout':>9} {'AvgErr':>9} {'P95Err':>9}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['stage']:<12} "
            f"{r['success_rate']:>7.1%} "
            f"{r['crash_rate']:>7.1%} "
            f"{r['timeout_rate']:>8.1%} "
            f"{r['avg_error_mm']:>8.1f}mm "
            f"{r['p95_error_mm']:>8.1f}mm"
        )
        if r["crash_reasons"]:
            for reason, count in r["crash_reasons"].items():
                print(f"  Crash breakdown: {reason}: {count}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Per-stage accuracy evaluation.")
    parser.add_argument(
        "--stages", type=str, default="transit,descend,ascend",
        help="Comma-separated list of stages to evaluate"
    )
    parser = add_common_args(parser)
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip() in STAGE_CHOICES]
    if not stages:
        print(f"No valid stages specified. Choose from: {STAGE_CHOICES}")
        sys.exit(1)

    results = []
    for stage in stages:
        print(f"\nEvaluating stage: {stage} ({args.n_episodes} episodes)...")
        result = evaluate_stage(stage, args)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()
```

**Expected outputs:**
- Success rate > 95% for all stages at 1cm drift limit
- Average error < 5mm for transit, < 5mm for descend/ascend
- P95 error < 10mm

---

## 4.4 — Rewrite: `scripts/eval_sequence.py`

Replace the current RL-dependent `eval_sequence.py` with a scripted version:

```python
"""
eval_sequence.py — Full scenario chain evaluation using ScriptedController.

Usage:
    python scripts/eval_sequence.py
        [--chain full_move|pick|place|vertical]
        [--src-xy "0.88 0.2641"] [--dst-xy "1.00 0.40"]
        [--n-episodes 20] [--drift-limit 0.010]
        [--visualize] [--delay 0.02] [--debug]
"""
import sys, os, argparse, time
import numpy as np
sys.path.append(os.getcwd())

import src.chess_env
from src.chess_env.controller import ScriptedController, SequenceResult
from src.utils.args import add_common_args, make_env

CHAIN_CHOICES = ["full_move", "pick", "place", "vertical"]


def run_sequence_episodes(args, src_xy, dst_xy) -> list:
    """Run N episodes of the specified chain. Returns list of SequenceResult."""
    env = make_env(args, force_scenario="transit")
    ctrl = ScriptedController(env, drift_limit=args.drift_limit)
    inner = env.unwrapped
    results = []

    for ep in range(args.n_episodes):
        # Force start near home_pos
        inner.force_start_pos = inner.HOME_POS.copy()
        if args.chain in ("full_move", "pick"):
            inner.force_cube_pos = np.array([src_xy[0], src_xy[1],
                                              inner.TABLE_SURFACE_Z + inner.CUBE_HEIGHT / 2])
        obs, _ = env.reset()

        if args.chain == "full_move":
            result = ctrl.run_full_move(src_xy, dst_xy)
        elif args.chain == "pick":
            result = ctrl.run_pick_sequence(src_xy)
        elif args.chain == "place":
            result = ctrl.run_place_sequence(dst_xy)
        elif args.chain == "vertical":
            from src.chess_env.waypoints import SAFE_Z, HOVER_Z
            nom_exit = np.array([src_xy[0], src_xy[1], SAFE_Z])
            goal_descend = np.array([src_xy[0], src_xy[1], HOVER_Z])
            ctrl.transition("descend", goal_descend, nom_exit, src_xy)
            d = ctrl.run_descend(src_xy)
            a = ctrl.run_ascend(src_xy) if d.success else None
            from src.chess_env.controller import SequenceResult
            result = SequenceResult(
                success=d.success and (a is not None and a.success),
                stage_results=[("descend", d)] + ([("ascend", a)] if a else []),
                failed_at=None if (d.success and a and a.success) else ("descend" if not d.success else "ascend"),
                grasp_quality=None
            )
        results.append(result)

        if args.visualize:
            env.render()
        if args.delay > 0:
            time.sleep(args.delay)

    env.close()
    return results


def print_summary(results: list, chain: str):
    n = len(results)
    successes = sum(r.success for r in results)
    print(f"\n{'='*65}")
    print(f"Chain: {chain}  |  Episodes: {n}  |  Full success: {successes}/{n} ({successes/n:.1%})")

    # Per-stage breakdown
    stage_stats = {}
    for r in results:
        for name, sr in r.stage_results:
            if name not in stage_stats:
                stage_stats[name] = {"ok": 0, "fail": 0, "reasons": {}}
            if sr.success:
                stage_stats[name]["ok"] += 1
            else:
                stage_stats[name]["fail"] += 1
                reason = sr.crash_reason or "UNKNOWN"
                stage_stats[name]["reasons"][reason] = stage_stats[name]["reasons"].get(reason, 0) + 1

    print(f"\n{'Stage':<12} {'Success':>9} {'Fail':>7}")
    print("-" * 35)
    for name, s in stage_stats.items():
        total = s["ok"] + s["fail"]
        print(f"{name:<12} {s['ok']:>5}/{total}    ", end="")
        if s["reasons"]:
            print(", ".join(f"{r}×{c}" for r, c in s["reasons"].items()))
        else:
            print()
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Full scenario chain evaluation.")
    parser.add_argument("--chain", type=str, default="full_move", choices=CHAIN_CHOICES)
    parser.add_argument("--src-xy", type=str, default="0.88 0.2641",
                        help="Source XY: two space-separated floats")
    parser.add_argument("--dst-xy", type=str, default="1.00 0.40",
                        help="Destination XY: two space-separated floats")
    parser = add_common_args(parser)
    args = parser.parse_args()

    src_xy = np.array([float(x) for x in args.src_xy.split()])
    dst_xy = np.array([float(x) for x in args.dst_xy.split()])

    print(f"Chain: {args.chain}, src={src_xy}, dst={dst_xy}, n={args.n_episodes}")
    results = run_sequence_episodes(args, src_xy, dst_xy)
    print_summary(results, args.chain)


if __name__ == "__main__":
    main()
```

---

## 4.5 — New Script: `scripts/eval_stress.py`

Stress test: run the specified chain from all board corners and center, report per-position statistics.

```python
"""
eval_stress.py — Corner/position stress test using ScriptedController.

Usage:
    python scripts/eval_stress.py
        [--chain full_move|pick|vertical]
        [--n-episodes 10] [--drift-limit 0.010]
        [--grid] [--grid-size 3]
        [--visualize] [--delay 0.0] [--debug]

--grid: Test a grid of NxN positions across the board (instead of just corners+center)
"""
import sys, os, argparse, time
import numpy as np
sys.path.append(os.getcwd())

import src.chess_env
from src.chess_env.controller import ScriptedController
from src.utils.args import add_common_args, make_env
from src.utils.config import load_config


def get_test_positions(args) -> list:
    """Return list of (name, src_xy, dst_xy) test cases."""
    cfg = load_config("env")
    cx, cy = cfg["table_center_xy"]
    hx, hy = cfg["table_half_x"], cfg["table_half_y"]
    margin = cfg.get("edge_margin", 0.02)

    min_x, max_x = cx - hx + margin, cx + hx - margin
    min_y, max_y = cy - hy + margin, cy + hy - margin

    if args.grid:
        n = args.grid_size
        xs = np.linspace(min_x, max_x, n)
        ys = np.linspace(min_y, max_y, n)
        positions = []
        for xi, x in enumerate(xs):
            for yi, y in enumerate(ys):
                positions.append((f"grid_{xi}_{yi}", np.array([x, y])))
        # Create (name, src, dst) pairs: use center as destination for all
        center = np.array([cx, cy])
        return [(name, src, center) for name, src in positions]
    else:
        # 4 corners + center
        corners = [
            ("near_right", np.array([min_x, min_y])),
            ("near_left",  np.array([min_x, max_y])),
            ("far_right",  np.array([max_x, min_y])),
            ("far_left",   np.array([max_x, max_y])),
            ("center",     np.array([cx, cy])),
        ]
        center = np.array([cx, cy])
        return [(name, src, center) for name, src in corners]


def run_position(name, src_xy, dst_xy, args) -> dict:
    """Run N episodes at a specific source position."""
    env = make_env(args, force_scenario="transit")
    ctrl = ScriptedController(env, drift_limit=args.drift_limit)
    inner = env.unwrapped

    successes = 0
    failures = []

    for _ in range(args.n_episodes):
        inner.force_start_pos = inner.HOME_POS.copy()
        if args.chain in ("full_move", "pick"):
            inner.force_cube_pos = np.array([
                src_xy[0], src_xy[1],
                inner.TABLE_SURFACE_Z + inner.CUBE_HEIGHT / 2
            ])
        obs, _ = env.reset()

        if args.chain == "full_move":
            result = ctrl.run_full_move(src_xy, dst_xy)
        elif args.chain == "pick":
            result = ctrl.run_pick_sequence(src_xy)
        elif args.chain == "vertical":
            from src.chess_env.waypoints import SAFE_Z, HOVER_Z
            ctrl.transition("descend", np.array([*src_xy, HOVER_Z]),
                            np.array([*src_xy, SAFE_Z]), src_xy)
            d = ctrl.run_descend(src_xy)
            a = ctrl.run_ascend(src_xy) if d.success else None
            from src.chess_env.controller import SequenceResult
            result = SequenceResult(
                success=d.success and a is not None and a.success,
                stage_results=[("descend", d)] + ([("ascend", a)] if a else []),
                failed_at=None, grasp_quality=None
            )

        if result.success:
            successes += 1
        else:
            failures.append(result.failed_at or "unknown")

        if args.visualize:
            env.render()
        if args.delay > 0:
            time.sleep(args.delay)

    env.close()
    return {
        "name": name,
        "src_xy": src_xy,
        "successes": successes,
        "n": args.n_episodes,
        "rate": successes / args.n_episodes,
        "failure_stages": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Corner/grid stress test.")
    parser.add_argument("--chain", type=str, default="pick",
                        choices=["full_move", "pick", "vertical"])
    parser.add_argument("--grid", action="store_true",
                        help="Test a grid of positions instead of corners+center")
    parser.add_argument("--grid-size", type=int, default=3,
                        help="Grid dimension NxN (default: 3)")
    parser = add_common_args(parser)
    args = parser.parse_args()

    test_positions = get_test_positions(args)
    print(f"Stress test: {len(test_positions)} positions × {args.n_episodes} episodes = "
          f"{len(test_positions) * args.n_episodes} total")

    all_results = []
    for name, src_xy, dst_xy in test_positions:
        print(f"  Testing {name} {src_xy}...")
        r = run_position(name, src_xy, dst_xy, args)
        all_results.append(r)
        print(f"    {r['successes']}/{r['n']} ({r['rate']:.0%})")

    # Summary
    print(f"\n{'='*55}")
    print(f"{'Position':<18} {'XY':>20} {'Rate':>8}")
    print("-" * 55)
    for r in all_results:
        xy_str = f"({r['src_xy'][0]:.3f}, {r['src_xy'][1]:.3f})"
        status = "OK" if r["rate"] >= 0.8 else "LOW"
        print(f"{r['name']:<18} {xy_str:>20} {r['rate']:>7.0%}  {status}")
    overall = sum(r["successes"] for r in all_results) / sum(r["n"] for r in all_results)
    print(f"\nOverall success rate: {overall:.1%}")
    print("=" * 55)


if __name__ == "__main__":
    main()
```

---

## 4.6 — Update: `scripts/verify_physics.py`

Three changes needed:

1. **Remove RL-model-specific import**: The script currently imports `from src.chess_env.task import ChessTaskEnv`. This still works after Stage 2, but the constructor signature changed. Remove `drift_curriculum_steps` from any instantiation calls.

2. **Update table size assertion**: Add a test that the board half-extents are 0.30 (new 60×60cm table):
```python
def test_table_geometry():
    print("Testing Table Geometry (60x60cm, 4 legs)...")
    from src.utils.config import load_config
    cfg = load_config("env")
    assert cfg["table_half_x"] == 0.30, f"Expected 0.30, got {cfg['table_half_x']}"
    assert cfg["table_half_y"] == 0.30, f"Expected 0.30, got {cfg['table_half_y']}"
    
    env = ChessTaskEnv()
    import mujoco
    model = env.model
    # Verify 4 leg geoms exist
    for leg_name in ["table0_leg_far_plus", "table0_leg_far_minus",
                     "table0_leg_near_plus", "table0_leg_near_minus"]:
        try:
            model.geom(leg_name)
            print(f"  - Leg '{leg_name}' found.")
        except Exception:
            print(f"  - ERROR: Leg '{leg_name}' NOT found.")
            env.close()
            return False
    
    # Verify surface top Z = 0.400
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table0_surface")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table0")
    surface_top = model.body_pos[body_id][2] + model.geom_pos[geom_id][2] + model.geom_size[geom_id][2]
    assert abs(surface_top - 0.400) < 0.001, f"Surface Z = {surface_top:.4f}, expected 0.400"
    print(f"  - Surface top Z = {surface_top:.4f} (correct)")
    env.close()
    return True
```

3. **Update `test_kinematic_reachability`**: Expand sampled positions to include corners:
```python
def test_kinematic_reachability(debug=False):
    print("Testing Kinematic Reachability (arm at new position)...")
    from src.utils.config import load_config
    cfg = load_config("env")
    cx, cy = cfg["table_center_xy"]
    hx, hy = cfg["table_half_x"], cfg["table_half_y"]
    margin = cfg.get("edge_margin", 0.02)
    
    test_positions = [
        np.array([cx - hx + margin, cy]),               # near edge center
        np.array([cx + hx - margin, cy]),               # far edge center
        np.array([cx, cy - hy + margin]),               # right edge center
        np.array([cx, cy + hy - margin]),               # left edge center
        np.array([cx - hx + margin, cy - hy + margin]), # near-right corner
        np.array([cx + hx - margin, cy + hy - margin]), # far-left corner
    ]
    # ... rest of reachability test using _settle_arm_to_start
```

---

## 4.7 — Rewrite: `scripts/visualize.py`

Replace the SAC-dependent visualize.py with a scripted version:

```python
"""
visualize.py — Human-render loop using ScriptedController.

Usage:
    python scripts/visualize.py
        [--scenario transit|descend|ascend|pick|full_move]
        [--src-xy "0.88 0.2641"] [--dst-xy "1.00 0.40"]
        [--episodes N] [--delay 0.03] [--wait] [--debug]
"""
import sys, os, argparse, time
sys.path.append(os.getcwd())

import gymnasium as gym
import numpy as np
import src.chess_env
from src.chess_env.controller import ScriptedController
from src.utils.args import add_common_args


def main():
    parser = argparse.ArgumentParser(description="Human-render visualization.")
    parser.add_argument("--scenario", type=str, default="transit",
                        choices=["transit", "descend", "ascend", "pick", "full_move"])
    parser.add_argument("--src-xy", type=str, default="0.88 0.2641")
    parser.add_argument("--dst-xy", type=str, default="1.00 0.40")
    parser.add_argument("--episodes", type=int, default=0, help="0 = infinite")
    parser.add_argument("--wait", action="store_true", help="Wait for Enter between episodes")
    parser = add_common_args(parser)
    args = parser.parse_args()
    args.visualize = True  # Always visualize in this script

    src_xy = np.array([float(x) for x in args.src_xy.split()])
    dst_xy = np.array([float(x) for x in args.dst_xy.split()])

    force_scenario = args.scenario if args.scenario in ("transit", "descend", "ascend") else "transit"
    env = gym.make("ChessFetchTask-v0", render_mode="human",
                   force_scenario=force_scenario, debug=args.debug)
    ctrl = ScriptedController(env, drift_limit=args.drift_limit)
    inner = env.unwrapped

    ep = 0
    try:
        while args.episodes == 0 or ep < args.episodes:
            ep += 1
            inner.force_start_pos = inner.HOME_POS.copy()
            inner.force_cube_pos = np.array([
                src_xy[0], src_xy[1],
                inner.TABLE_SURFACE_Z + inner.CUBE_HEIGHT / 2
            ])
            obs, _ = env.reset()
            env.render()

            if args.scenario == "transit":
                target_xy = inner.goal_pos[:2].copy()
                result = ctrl.run_transit(target_xy)
            elif args.scenario == "descend":
                target_xy = inner.goal_pos[:2].copy()
                result = ctrl.run_descend(target_xy)
            elif args.scenario == "ascend":
                target_xy = inner.goal_pos[:2].copy()
                result = ctrl.run_ascend(target_xy)
            elif args.scenario == "pick":
                result = ctrl.run_pick_sequence(src_xy)
            elif args.scenario == "full_move":
                result = ctrl.run_full_move(src_xy, dst_xy)

            print(f"Ep {ep}: {'SUCCESS' if result.success else f'FAIL ({result.failed_at})'}")
            env.render()

            if args.wait:
                input("Press Enter for next episode...")
            elif args.delay > 0:
                time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
```

---

## 4.8 — Update: `scripts/test_grasp_physics.py`

This script already has no RL dependency. The only update needed:

1. Remove `drift_curriculum_steps` param from `ChessTaskEnv(...)` instantiation.
2. Update table corner coordinates to use config values rather than hardcoded numbers.

---

## 4.9 — Validation Steps

### Check 1: `eval_stages.py` Basic Run

```bash
python scripts/eval_stages.py --stages transit --n-episodes 5 --debug
```

**Expected**: 5 transit episodes run (no model needed), prints summary table.

### Check 2: `eval_stages.py` All Stages

```bash
python scripts/eval_stages.py --stages transit,descend,ascend --n-episodes 10
```

**Expected**: Summary table for all 3 stages. Success rate > 80% (initial calibration may need parameter tuning).

### Check 3: `eval_stress.py` Corners

```bash
python scripts/eval_stress.py --chain vertical --n-episodes 3
```

**Expected**: Tests 5 positions (4 corners + center) × 3 episodes each = 15 total. Reports per-position success rates.

### Check 4: `eval_sequence.py` Full Move

```bash
python scripts/eval_sequence.py --chain full_move --n-episodes 3 \
    --src-xy "0.88 0.2641" --dst-xy "1.00 0.40"
```

**Expected**: 3 pick-and-place episodes run. Reports per-stage success.

### Check 5: `verify_physics.py` with New Table Test

```bash
python scripts/verify_physics.py
```

**Expected**: All 6 tests pass (5 original + 1 new table geometry test).

### Check 6: Visualization Smoke Test (Optional, Requires Display)

```bash
python scripts/visualize.py --scenario transit --episodes 1 --delay 0.02
```

**Expected**: Opens MuJoCo viewer, arm performs one transit, closes cleanly.

---

## 4.10 — Parameter Calibration After Stage 4

After initial runs, calibrate these parameters in `src/chess_env/controller.py` based on observed success rates:

| Parameter | Default | Tune if... |
|-----------|---------|------------|
| `TRANSIT_MAX_STEPS` | 400 | Transit timeouts on long diagonals |
| `VERTICAL_MAX_STEPS` | 200 | Descend/ascend timeouts |
| `MAX_STEP_SIZE_M` | 0.008 | Overshoot oscillation → decrease; too slow → increase |
| `TRANSIT_TOLERANCE_M` | 0.004 | High error → decrease; timeouts → increase |
| `drift_limit` | 0.010 | Tube breach rate too high → increase temporarily |

**Do not change Z-level constants** (`SAFE_Z`, `HOVER_Z`, `GRASP_Z`) — these are calibrated to the physics.

---

## 4.11 — Expected Performance Targets

After full Stage 4 calibration, the system should achieve:

| Metric | Target |
|--------|--------|
| Transit success rate | ≥ 98% |
| Descend success rate | ≥ 95% |
| Ascend success rate | ≥ 95% |
| Grasp success rate | ≥ 90% |
| Full pick-and-place | ≥ 85% |
| All corners reachable | 100% |

These targets reflect a deterministic scripted system on a calibrated table. Failures should be almost exclusively `TIMEOUT` (parameter tuning) rather than `TUBE_BREACH` or `TABLE_HIT`.

---

## 4.12 — Summary of Changes

| Script | Action | RL dependency |
|--------|--------|--------------|
| `scripts/train.py` | Delete | Removed |
| `scripts/record_move.py` | Delete | Removed |
| `scripts/eval.py` | Delete | Removed |
| `scripts/eval_grasp.py` | Delete | Removed |
| `scripts/test_corners.py` | Delete | Removed |
| `scripts/eval_stages.py` | **New** | None |
| `scripts/eval_stress.py` | **New** | None |
| `scripts/eval_sequence.py` | Rewrite | None |
| `scripts/visualize.py` | Rewrite | None |
| `scripts/verify_physics.py` | Update | None |
| `scripts/test_grasp_physics.py` | Minor update | None |

**Stage 4 is complete when Checks 1–5 all pass and the overall success rate is > 80% across all stages.**
