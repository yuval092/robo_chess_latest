# Integration & Evaluation

## 1. `GameOrchestrator` Integration

The `GameOrchestrator` currently creates a `ScriptedController` and calls its methods. To use
the RL models, we swap in `ModelEmbeddedController` — which has the same interface.

**No changes required to `GameOrchestrator`** as long as `ModelEmbeddedController` exposes:
- `run_transit(target_xy)`
- `run_descend(tube_xy)`
- `run_ascend(tube_xy)`
- `execute_grasp()`
- `execute_place(dst_xy)`
- `transition(new_scenario, new_goal_pos, nominal_exit_pos, nominal_xy)`

All of these return `StageResult` (for movement) or the scripted dict (for grasp/place), same as before.

The swap at the orchestrator level looks like:

```python
# Before (scripted):
ctrl = ScriptedController(env, drift_limit=0.010, render_fn=render_fn)

# After (RL models):
from src.chess_env.model_controller import ModelEmbeddedController
ctrl = ModelEmbeddedController(inner, render_fn=render_fn)
ctrl.load_all(
    transit_path="checkpoints/transit_.../best_model_transit.zip",
    descend_path="checkpoints/descend_.../best_model_descend.zip",
    ascend_path="checkpoints/ascend_.../best_model_ascend.zip",
)
```

For a clean, config-driven approach, model paths can live in `configs/training.yaml`:

```yaml
# configs/training.yaml — model path registry
deployed_models:
  transit: "checkpoints/transit_20260522/best_model_transit.zip"
  descend: "checkpoints/descend_20260522/best_model_descend.zip"
  ascend:  "checkpoints/ascend_20260522/best_model_ascend.zip"
```

---

## 2. Updating `scripts/eval_stages.py`

Add a `--use-rl-models` flag. When set, create `ModelEmbeddedController` instead of `ScriptedController`.
Everything else (the evaluation loop, result collection, summary printing) stays unchanged.

```python
# In eval_stages.py main():
if args.use_rl_models:
    from src.chess_env.model_controller import ModelEmbeddedController
    from src.utils.config import load_config
    train_cfg = load_config("training")
    model_paths = train_cfg.get("deployed_models", {})
    
    ctrl = ModelEmbeddedController(
        env=env.unwrapped,
        render_fn=env.render if args.visualize else None,
        render_delay=args.delay,
    )
    ctrl.load_all(
        transit_path=args.transit_model or model_paths.get("transit"),
        descend_path=args.descend_model or model_paths.get("descend"),
        ascend_path=args.ascend_model or model_paths.get("ascend"),
    )
else:
    ctrl = ScriptedController(env, drift_limit=args.drift_limit, ...)
```

Add the CLI args:
```python
parser.add_argument("--use-rl-models", action="store_true", help="Use RL models instead of scripted controller")
parser.add_argument("--transit-model", type=str, default=None)
parser.add_argument("--descend-model", type=str, default=None)
parser.add_argument("--ascend-model", type=str, default=None)
```

**Usage:**
```bash
# Scripted (existing behaviour, unchanged):
python scripts/eval_stages.py --stages transit,descend,ascend --n-episodes 50

# RL models:
python scripts/eval_stages.py --use-rl-models --stages transit --n-episodes 50
python scripts/eval_stages.py --use-rl-models \
    --transit-model checkpoints/transit_.../best_model_transit.zip \
    --n-episodes 100
```

---

## 3. Updating `scripts/eval_sequence.py`

Same approach — add `--use-rl-models` to swap the controller. The chain logic (calling
`run_transit`, `run_descend`, `run_ascend` in sequence with `transition()` between them) is
identical for both controllers.

```bash
# RL model chain evaluation:
python scripts/eval_sequence.py --use-rl-models --chain full_move --n-episodes 20
python scripts/eval_sequence.py --use-rl-models --chain pick --n-episodes 50
```

---

## 4. A New Standalone RL Eval Script: `scripts/eval_rl_stages.py`

A purpose-built evaluation script that uses the SB3 `evaluate_policy` utility and matches
the reference project's `eval.py`. This is useful for quick standalone checks before full
integration testing:

```python
"""
eval_rl_stages.py — Standalone per-stage RL model evaluation.

Does NOT use ModelEmbeddedController or ScriptedController. Directly
rolls out episodes using model.predict() + gym loop. Useful for quick
checks that a model works at all before full integration testing.

Usage:
    python scripts/eval_rl_stages.py --stage transit --model checkpoints/.../best_model_transit.zip
    python scripts/eval_rl_stages.py --stage descend --model checkpoints/.../best_model_descend.zip
"""
import sys, os, argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC

sys.path.append(os.getcwd())
import src.chess_env
from training.envs import make_eval_env
from stable_baselines3.common.vec_env import DummyVecEnv
from src.utils.config import load_config


def evaluate(stage: str, model_path: str, n_episodes: int, visualize: bool = False, delay: float = 0.0):
    env_cfg = load_config("env")
    eval_drift = env_cfg.get("eval_drift_limit", 0.005)
    crash_penalty = env_cfg.get("crash_penalty", -500.0)

    render_mode = "human" if visualize else None
    env = gym.make(
        "ChessFetchTask-v0",
        force_scenario=stage,
        force_drift_limit=eval_drift,
        render_mode=render_mode,
    )

    # Wrap in Phase-9 observation mode
    from training.envs import WRAPPER_MAP
    env = WRAPPER_MAP[stage](env)
    model = SAC.load(model_path, env=env)

    successes, crashes, timeouts = 0, 0, 0
    total_reward = 0.0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_success = False
        ep_crash = False
        crash_reason = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if visualize:
                env.render()
                if delay > 0:
                    import time; time.sleep(delay)

            if terminated:
                if info.get("is_success"):
                    ep_success = True
                elif info.get("crash_reason"):
                    ep_crash = True
                    crash_reason = info["crash_reason"]

        if ep_success:
            successes += 1
        elif ep_crash:
            crashes += 1
        else:
            timeouts += 1
        total_reward += ep_reward

        print(
            f"Episode {ep+1:3d}/{n_episodes} | "
            f"{'SUCCESS' if ep_success else ('CRASH' if ep_crash else 'TIMEOUT'):8s} | "
            f"reward={ep_reward:8.1f}"
            + (f" | {crash_reason}" if crash_reason else "")
        )

    print(f"\n{'='*60}")
    print(f"Stage: {stage.upper()} | Model: {model_path}")
    print(f"  Success:  {successes:3d}/{n_episodes} ({successes/n_episodes:.1%})")
    print(f"  Crashes:  {crashes:3d}/{n_episodes} ({crashes/n_episodes:.1%})")
    print(f"  Timeouts: {timeouts:3d}/{n_episodes} ({timeouts/n_episodes:.1%})")
    print(f"  Avg Reward: {total_reward/n_episodes:.2f}")
    print(f"{'='*60}")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["transit", "descend", "ascend"])
    parser.add_argument("--model", required=True, help="Path to .zip model file")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()

    evaluate(args.stage, args.model, args.n_episodes, args.visualize, args.delay)


if __name__ == "__main__":
    main()
```

---

## 5. Full Implementation Checklist

Use this as a task list when executing the plan:

### Phase 1: Environment Changes (edit `src/chess_env/task.py`)
- [ ] Add `_build_phase9_observation()` method
- [ ] Add `_use_phase9_obs` flag and modify `_get_obs()` to branch on it
- [ ] Add `drift_curriculum_steps`, `force_drift_limit`, `fixed_drift` parameters to `__init__`
   - Remove the `kwargs.pop()` stubs for these
   - Add `DRIFT_LIMIT_START`, `DRIFT_CURRICULUM_STEPS` constants
   - Add `_get_current_drift_limit()` helper method
- [ ] Replace `step()` stub with full RL implementation
- [ ] Replace `compute_reward()` stub with multi-component reward
- [ ] Verify `env.yaml` has all required reward/curriculum keys (add if missing)

### Phase 2: Training Package
- [ ] Create `training/__init__.py`
- [ ] Create `training/envs/__init__.py` (factory functions)
- [ ] Create `training/envs/transit_env.py` (`TransitTrainEnv`)
- [ ] Create `training/envs/ascend_env.py` (`AscendTrainEnv`)
- [ ] Create `training/envs/descend_env.py` (`DescendTrainEnv`)
- [ ] Create `training/callbacks.py` (`DetailedLoggingCallback`, `SuccessRateEvalCallback`)
- [ ] Create `training/trainer.py` (`SACTrainer`)
- [ ] Create `configs/training.yaml`

### Phase 3: Training Script
- [ ] Create `scripts/train_rl.py` (CLI entry point)
- [ ] Test with `--stage transit --envs 1 --timesteps 5000 --fixed-drift` (smoke test)

### Phase 4: Controller
- [ ] Create `src/chess_env/model_controller.py` (`ModelEmbeddedController`)
- [ ] Verify `StageResult` is importable from `src.chess_env.controller`

### Phase 5: Eval Integration
- [ ] Update `scripts/eval_stages.py` to accept `--use-rl-models`
- [ ] Update `scripts/eval_sequence.py` to accept `--use-rl-models`
- [ ] Create `scripts/eval_rl_stages.py` (standalone RL eval)

### Phase 6: Train and Evaluate
- [ ] Train transit model: `python scripts/train_rl.py --stage transit --envs 8`
- [ ] Eval transit: `python scripts/eval_rl_stages.py --stage transit --model checkpoints/.../best_model_transit.zip --n-episodes 50`
- [ ] Train descend model, eval
- [ ] Train ascend model, eval
- [ ] Full integration eval: `python scripts/eval_stages.py --use-rl-models --stages transit,descend,ascend`
- [ ] Full chain eval: `python scripts/eval_sequence.py --use-rl-models --chain full_move`

---

## 6. Debugging Tips

| Problem | Likely Cause | Fix |
|---|---|---|
| Model predicts crazy actions | Observation mismatch | Check `_use_phase9_obs = True` before inference |
| `observation` shape is `(7,)` not `(25,)` | Phase-9 mode not enabled | Set `env._use_phase9_obs = True` |
| SubprocVecEnv hangs on Windows | Multiprocessing without guard | Run via `scripts/train_rl.py` (has `__main__` guard) |
| Drift limit always `0.010` (no curriculum) | `drift_curriculum_steps=None` not read | Verify `kwargs.pop()` was removed from `__init__` |
| No crash detected during training | `step()` returns `terminated=False` always | Verify `tube_center_xy` is set in `_reset_sim` |
| Model loads but always crashes on drift | `eval_drift_limit` too tight | Check `env.yaml` `eval_drift_limit: 0.005` |
| `SAC.load()` crashes on obs space mismatch | Wrapper obs space not set | Ensure `WRAPPER_MAP[stage](env)` is applied before `SAC.load()` |
