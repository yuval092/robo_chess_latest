# Transit Model: Diagnostic and Fixes

Date: 2026-05-23

## Summary of Failures

Two symptoms were reported after the initial integration:
1. **"Model twitches at the end"** — arm oscillates near goal, can't satisfy the stability criterion
2. **"Transit to dest fails at the start"** — the second transit (src → dst while holding piece) fails immediately

---

## Bug 1 (Critical): `PRECONDITION_FINGER` Check Fails in Grasp Mode

**File**: `src/chess_env/model_controller.py`, `_run_stage()`

**Root cause**: `_run_stage("transit", ...)` has a precondition guard that reads the current
finger joint position and compares it to `FINGER_CLOSED_JOINT = 0.0000`. The tolerance is 0.003.

When the arm is holding a chess piece (`grasp_mode=True`), the fingers are physically blocked
by the piece and sit at ~0.011–0.015 (well above the 0.003 tolerance). The check therefore
**always fails** for the second transit in any real chess move:

```
After execute_grasp():
  l_finger = 0.011574   FINGER_CLOSED_JOINT = 0.000000
  diff = 0.011574 > 0.003 → PRECONDITION_FINGER check → returns failure (0 steps)
```

This means `run_full_move(src, dst)` always fails after the pick sequence. Every chess move
requiring piece transport (i.e., every move except captures where the piece is already at
the destination) fails at the second transit with zero RL inference steps.

**Fix applied**: Skip the precondition check when `env.grasp_mode = True`. In grasp mode,
the deviation is expected and safe — the actuator is commanding closure against the piece.
Also added explicit `env.finger_target_joint` setting at the start of `_run_stage` (only when
not in grasp mode, so the gripper keeps holding the piece during transit).

```python
# BEFORE (broken):
l_finger = env._utils.get_joint_qpos(...).item()
expected_finger = env.FINGER_OPEN_JOINT if stage == "descend" else env.FINGER_CLOSED_JOINT
if abs(l_finger - expected_finger) > 0.003:
    return StageResult(success=False, ..., crash_reason="PRECONDITION_FINGER")

# AFTER (fixed):
if not env.grasp_mode:
    env.finger_target_joint = env.FINGER_OPEN_JOINT if stage == "descend" else env.FINGER_CLOSED_JOINT
if not env.grasp_mode:
    l_finger = env._utils.get_joint_qpos(...).item()
    if abs(l_finger - env.finger_target_joint) > 0.003:
        return StageResult(success=False, ..., crash_reason="PRECONDITION_FINGER")
```

---

## Bug 2 (Medium): `load_model()` Permanently Pollutes Production Env State

**File**: `src/chess_env/model_controller.py`, `load_model()`

**Root cause**: `load_model()` wraps the production env with `TransitTrainEnv` to give SAC.load()
the correct Phase-9 observation space. `TransitTrainEnv.__init__` has two permanent side-effects
on the unwrapped env:

1. `uw._use_phase9_obs = True` — env switches to Phase-9 mode
2. `uw.observation_space = phase9_space` — env's obs space changed to 3-key Phase-9 dict

After loading, the `TransitTrainEnv` wrapper is discarded, but both mutations persist. Consequences:
- `env.reset()` returns 8-key regular obs, but `observation_space` says 3-key Phase-9 → Gymnasium's
  passive env checker raises `AssertionError` on `obs.keys() != observation_space.spaces.keys()`
- `_use_phase9_obs=True` permanently leaves the env in Phase-9 mode. The `_run_stage` finally block
  restores to `previous_phase9`, but `previous_phase9 = True` (not False), so obs never reverts
  to the scripted dict. Downstream soft_reset and eval scripts that inspect obs silently get
  the wrong format.

**Fix applied**: Save and restore both attributes immediately after SAC.load():

```python
prev_phase9 = self._env._use_phase9_obs
prev_obs_space = self._env.observation_space

load_env = WRAPPER_MAP[stage](self._wrapped_env)
self._models[stage] = SAC.load(path, env=load_env)

self._env._use_phase9_obs = prev_phase9          # undo side-effect
self._env.observation_space = prev_obs_space      # undo side-effect
```

---

## Training Quality Issues

### 1. Too Few Timesteps (200K instead of 1M)

The plan specified `total_timesteps: 1_000_000`. The implementing agent reduced this to 200K
to shorten the first training run. With 4 workers this gives ~50K gradient updates after
learning starts — enough to get to ~85-90% success but not a fully converged policy.

**Evidence**: The "best" model was saved at 100K (first eval with 5/5=100%). The model at 150K
is measurably better (avg_min_dist 8.1mm → 5.6mm; 19/20 → 20/20 on standalone eval).

**Config fix**: `configs/training.yaml` updated to `total_timesteps: 1000000` for next runs.

### 2. Too Few Eval Episodes (5 instead of 20)

With `n_eval_episodes: 5`, the "best model" selection is dominated by noise. A 70% model can
pass all 5 episodes. The `best_model_transit.zip` was saved at 100K based on 5 episodes — it
does not reliably represent the best checkpoint.

**Config fix**: `configs/training.yaml` updated to `n_eval_episodes: 20`.

### 3. Deployed Model Points to `best_model` (Noisy) Instead of `latest_model`

Since `best_model` is determined by noisy 5-episode evals, `latest_model_transit.zip`
(updated at every eval) is a better proxy for the most trained weights.

**Config fix**: `configs/training.yaml` deployed_models now points to `latest_model_transit.zip`.

### 4. `drift_curriculum_steps` Inconsistency

With the new `total_timesteps: 1_000_000` and `num_envs: 4`, the per-worker steps = 250K.
The original `drift_curriculum_steps: 50000` would complete the curriculum at only 20% of
training, leaving 80% of steps at the tightest drift limit. The new value of 125K covers the
first 50% of training with curriculum, then the remaining 50% at full precision.

**Config fix**: `configs/env.yaml` updated to `drift_curriculum_steps: 125000`.

---

## "Model Twitches at the End" — Root Cause

The twitching is the model oscillating near the goal: it approaches to ~10-15mm, recedes slightly,
approaches again, never satisfying the `stability_vel_threshold: 0.02 m/s` condition.

**Cause**: The braking reward (`-0.15 * speed` when `dist < 10mm`) wasn't sufficient to fully
train deceleration in 100-150K steps. The model learned to navigate to the general area but
not to stop cleanly.

**Status**: This is a training quality issue, not a code bug. It will improve with more training
(1M steps, as configured). The latest 150K model already shows improvement (5.6mm avg vs 8.1mm).
No code change made — further training will address this.

**Short-term**: The 30% timeout rate (original best model) and ~10% timeout rate (latest model)
are acceptable for a partially-trained model. When training completes at 200K (still running),
performance should improve further.

---

## Performance After Fixes

| Test | Before Fixes | After Fixes |
|------|-------------|-------------|
| `eval_stages.py --stages transit` (20 eps) | 70% success | **90% success** |
| `eval_sequence.py --chain pick` (10 eps) | 3/3 (center only) | **10/10** |
| `eval_sequence.py --chain full_move` (5 eps) | Fails at 2nd transit | **5/5 (100%)** |
| Full test suite | 78 passed | **84 passed** |

---

## Files Changed

| File | Change |
|------|--------|
| `src/chess_env/model_controller.py` | Fix PRECONDITION_FINGER; fix obs_space/phase9 pollution in load_model |
| `configs/training.yaml` | total_timesteps 200K→1M; n_eval_episodes 5→20; deployed to latest_model |
| `configs/env.yaml` | drift_curriculum_steps 50K→125K (per-worker, consistent with 1M total/4 envs) |

---

## What Still Needs Work

1. **Transit model needs more training**: Currently at ~150-200K steps, achieving ~90% in eval.
   With 1M steps it should reach >95%. **Do not retrain until user approves.**

2. **Descend and ascend models**: Not yet trained. The plan calls for specialist models for
   both. Currently these stages fall back to `ScriptedController`.

3. **Twitching will improve with more training**: No code change needed; more gradient steps
   will teach the model to decelerate smoothly into the goal.
