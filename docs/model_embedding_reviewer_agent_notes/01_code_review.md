# Code Review: Model Embedding Commits

Reviewer agent review covering commits 3687196f through 3ed47adc (model-embedding-phase branch).

---

## Commit Overview

| Commit | Message | Key changes |
|--------|---------|-------------|
| 3687196 | Implemented training. Started transit training. | +1205 lines: task.py RL step, training package, model_controller.py, eval scripts |
| 1ad0cd9 | Starting training for all scenarios | Bug fixes to model_controller, ascend/descend checkpoints, diagnose script |
| 4856672 | Merge chess_refactor_phase | Merge only |
| b76b4b29 | Trained ascend and descend. Has bugs. | New best checkpoints, updated run_chess_ui.py, eval_sequence fix |
| 3ed47adc | Codex review notes + better selected models | Updated configs/training.yaml deployed_models, added model selection doc |

---

## TASK.PY changes (src/chess_env/task.py)

### What was added

1. **`_use_phase9_obs` flag** — gated Phase-9 observation construction behind a per-instance bool defaulting to `False`. Correct design: scripted pipeline is unaffected.

2. **`_build_phase9_observation()`** — builds the 25D dict matching FetchPickAndPlace-v4 layout. The "Holding Object" trick sets `fake_object_pos = grip_pos` so transferred weights treat every position as if holding a target. The `rel_to_goal` vector correctly uses `self.goal` (set in `_run_stage` before calling `_get_obs()`).

3. **`compute_reward()`** — replaced the no-op stub with a dense reward combining 3D distance, Z-axis error, and XY-axis error. Vectorized to support HER's batched reward queries.

4. **`step()`** — replaced the stub with a full RL training step including: curriculum drift limit, tube-breach termination, FINGER_FAULT check, braking reward, jitter penalty, floor proximity penalty, and success detection.

5. **`_get_current_drift_limit()`** — applies curriculum schedule from `drift_limit_start` (100mm) to `drift_limit_end` (8mm) over `drift_curriculum_steps` worker steps.

6. **Constructor parameters** — `drift_curriculum_steps`, `force_drift_limit`, `fixed_drift` moved from `kwargs.pop()` fallback to explicit named parameters. This is correct: it enables eval scripts to pass `force_drift_limit=0.008` without warnings.

### Code Quality Issues

**Issue 1 (Medium): `step()` hardcodes `action_copy[3] = -1.0` regardless of scenario.**

```python
# task.py step() line 1271:
action_copy[3] = -1.0
```

In `_set_action()` (simulation.py line 139-173), `action[3]` is stored as `gripper_ctrl` but is **never used in the mocap action**. The mocap action is `concat([pos_ctrl * scale, rot_ctrl_zeros])`. The finger state is set exclusively from `self.finger_target_joint`, not from `action[3]`. This means:

- The policy's 4th output dimension is trained but has no effect on the simulation.
- The 4-dim action space wastes one dimension.
- The policy may learn arbitrary values for `action[3]` without consequences.

This is benign for the current models (they work despite it) but is conceptually wrong and wastes model capacity. The action space should be 3D or `action[3]` should be plumbed into `finger_target_joint`.

**Issue 2 (Low): Velocity extraction from Phase-9 obs uses a fragile index offset.**

```python
# task.py step() lines 1277-1280:
if obs["observation"].shape[0] >= 23:
    grip_vel = obs["observation"][20:23]
else:
    grip_vel = obs["grip_vel"]
```

The Phase-9 obs vector layout is: `[grip_pos(3), fake_obj_pos(3), rel_to_goal(3), zeros(2), zeros(3), zeros(3), zeros(3), grip_velp(3), zeros(2)]`. Indices 0-8 are positions; `grip_velp` starts at index `3+3+3+2+3+3+3 = 20`. The extraction is correct **for the current layout** but would silently break if the obs layout changed. A named index constant would be safer.

**Issue 3 (Low): `stability_vel_threshold` used in `step()` is `0.02`, but model_controller uses `0.05`.**

```yaml
# configs/env.yaml:
stability_vel_threshold: 0.02
```
```python
# model_controller.py _run_stage():
"stability_vel_threshold", 0.05  # hardcoded default
```

Training declares success when speed < 20 mm/s. Inference declares success when speed < 50 mm/s. The inference threshold is 2.5x more lenient, which means the arm may be declared "arrived" while still moving at up to 50mm/s. This causes a jerk at the transit-to-descend transition: the arm is still in motion when soft_reset fires and aligns it to the nominal exit position.

The correct fix is to read `stability_vel_threshold` from `env_cfg` in `_run_stage()` instead of using the hardcoded default.

**Issue 4 (Low): `_build_phase9_observation()` uses `self.goal` with a None-guard fallback that masks missing initialization.**

```python
goal = self.goal if (hasattr(self, "goal") and self.goal is not None) else np.zeros(3)
```

If `self.goal` is not set before `_get_obs()` is called, the model receives a zero-vector desired_goal. The `_run_stage()` method does set `env.goal = target_pos` before calling the model loop, so this is protected in practice. But during training resets, `goal` is set in `_reset_sim()` via `self.goal = self.goal_pos.copy()` before the obs is returned, so it should always be available. The `hasattr` guard is defensive but the `is not None` check is the operative one.

---

## MODEL_CONTROLLER.PY (src/chess_env/model_controller.py)

### What was added

A complete `ModelEmbeddedController` class that:

1. Wraps a `ScriptedController` as fallback for stages without a loaded model.
2. Loads SAC models via `SAC.load(path, env=wrapper)`.
3. Saves/restores `_use_phase9_obs` and `observation_space` after model loading (fixes the env-pollution bug documented in `2026-05-23_diagnostic_and_fixes.md`).
4. Runs model inference in a 300-step loop, setting `env.goal`, `env.current_scenario`, and `env.tube_center_xy` before each stage.
5. Forces `action[3] = -1.0` for transit/ascend and `+1.0` for descend (but see Issue 1 above — this is harmless, not beneficial).
6. Skips the `PRECONDITION_FINGER` check when `env.grasp_mode=True` (fixes the grasp-mode bug).

### Code Quality Issues

**Issue 5 (Low): Duplicate high-level sequence code between `ModelEmbeddedController` and `ScriptedController`.**

Both `run_pick_sequence`, `run_place_sequence`, and `run_full_move` are implemented identically in both classes except that `ScriptedController` uses `waypoints.SAFE_Z`/`waypoints.HOVER_Z` constants while `ModelEmbeddedController` reads them from `self._env.SAFE_Z`/`self._env.HOVER_Z`. The logic is identical. This is ~120 lines of duplication.

The `ModelEmbeddedController` could inherit from or compose with `ScriptedController`, or both could inherit from a shared `BaseController`. This is not a bug but is a maintenance burden.

**Issue 6 (Low): `transition()` uses a fragile while-loop to find `_elapsed_steps` in wrapper chain.**

```python
curr = self._wrapped_env
while hasattr(curr, "env"):
    if hasattr(curr, "_elapsed_steps"):
        curr._elapsed_steps = 0
        break
    curr = curr.env
else:
    if hasattr(curr, "_elapsed_steps"):
        curr._elapsed_steps = 0
```

This pattern is also present in `ScriptedController.transition()`. The `else` branch of the while-loop fires when the loop **exhausts** without a `break`, not on the final iteration. This is correct Python semantics but is confusing. More importantly, if no wrapper has `_elapsed_steps`, the step counter is never reset, and the `TimeLimit` wrapper will truncate the episode early. The `ScriptedController` and `ModelEmbeddedController` have the same code, which suggests it was copied rather than refactored.

**Issue 7 (Medium): `load_model()` builds a temporary `load_env` wrapper but does not close it.**

```python
load_env = WRAPPER_MAP[stage](self._wrapped_env)
self._models[stage] = SAC.load(path, env=load_env)
```

The `WRAPPER_MAP[stage]` wrapper is a `gym.Wrapper` around `self._wrapped_env`. SAC.load() sets the model's internal env reference but does not take ownership of `load_env`. The temporary wrapper is not closed, meaning any Monitor/cleanup logic it wraps won't run. In practice this is harmless because `WRAPPER_MAP[stage]` wrappers have no cleanup side effects, but it is a resource management concern.

**Issue 8 (Low): `run_grasp()` and `run_place()` do not use `error_mm` meaningfully.**

In `run_grasp()`, `error_mm` is the displacement of the gripper during grasp (how much the arm moved). This is not a "goal distance" and is semantically incorrect as an "error" measurement. `ScriptedController.run_grasp()` has the same issue. The field exists for consistency with stage results, but the value is misleading.

---

## TRAINING PACKAGE (training/)

### What was added

- `training/trainer.py`: `SACTrainer` orchestrates fine-tuning from the pretrained FetchPickAndPlace-v4 base model.
- `training/callbacks.py`: `DetailedLoggingCallback` (rolling reward/success rate logger) and `SuccessRateEvalCallback` (saves latest+best checkpoints on eval cycles).
- `training/envs/__init__.py`: `WRAPPER_MAP`, `make_train_env()`, `make_eval_env()`.
- `training/envs/transit_env.py`, `descend_env.py`, `ascend_env.py`: Identical wrappers that enable Phase-9 obs and set the matching observation space.

### Code Quality Issues

**Issue 9 (Low): All three train wrappers (`TransitTrainEnv`, `DescendTrainEnv`, `AscendTrainEnv`) are identical.**

The only difference between them is the class name. This is 60 lines of duplication that could be one parameterized `StageTrainEnv(gym.Wrapper)` class taking a `stage` string. The current design works but will diverge over time if one wrapper gets a fix the others don't.

**Issue 10 (Medium): `SACTrainer` does not reset `ent_coef_optimizer` properly.**

```python
model.log_ent_coef = th.nn.Parameter(model.log_ent_coef, requires_grad=True)
model.ent_coef_optimizer = th.optim.Adam([model.log_ent_coef], lr=ent_coef_lr)
```

`model.log_ent_coef` is reassigned from a tensor to a `Parameter` wrapping the same tensor value, then a new Adam optimizer is created for it. The old `ent_coef_optimizer` from the checkpoint is discarded. This is intentional (entropy reset on transfer), but the comment in `training.yaml` says "Entropy reset on model load" without explaining that this also discards the optimizer state. If the learning rate from `ent_coef_lr` is not appropriate for the new task, this can cause instability in the first ~10K steps.

**Issue 11 (Low): `make_eval_env()` default `eval_drift_limit` is `0.005` m (5mm) but `configs/env.yaml` sets `eval_drift_limit: 0.008` (8mm).**

```python
def make_eval_env(stage: str, eval_drift_limit: float = 0.005, ...):
```

The trainer reads `eval_drift_limit` from `env.yaml` and passes it to `make_eval_env`, so in practice the 8mm limit is used. But the function's default argument (5mm) differs from the config default. If `make_eval_env` is called without the argument (e.g., in a test or diagnostic script), it silently uses the tighter 5mm limit.

---

## SCRIPTS changes

### eval_stages.py, eval_sequence.py

Both scripts were updated to accept `--use-rl-models`, `--transit-model`, `--descend-model`, `--ascend-model` flags and branch between `ScriptedController` and `ModelEmbeddedController`. This is clean.

**Issue 12 (Low): `eval_sequence.py` `--chain vertical` does the `ctrl.transition()` call before `run_descend`, matching production, but does not have a cube for descend.** The vertical chain was added to test descend+ascend in isolation. When run with `--use-rl-models`, the env is created with `hide_object=True` (no physical cube). The descend model was trained in this condition, so this is fine. But the comment in `run_sequence_episodes` says `hide_object = args.chain not in ("pick", "full_move")`, which means vertical=True=hide_object=True. This is correct but could be confusing.

### run_chess_ui.py

Updated to default to `ModelEmbeddedController` with `--use-scripted-controller` as opt-out. The `build_controller()` function is well structured. The `build_orchestrator()` function has `use_rl_models=True` as default, which is correct for production.

**Issue 13 (Medium): `configs/training.yaml` `deployed_models.transit` still points to an older model as of commit 3ed47adc.**

The model selection agent (commit 3ed47adc) claims to have updated `deployed_models.transit` to `checkpoints/transit_20260524_084201/latest_model_transit.zip`. However, the actual config file still shows:
```
transit: "checkpoints/transit_20260523_164127/final_transit.zip"
```

This is a critical discrepancy: the model selection agent's own note says it selected `latest_model_transit.zip` from the newer 084201 checkpoint run, but the config was not actually updated. The production UI and eval scripts that load from `configs/training.yaml` are still using the older transit model.

Verified: when running `eval_stages.py --use-rl-models`, the log shows `Loaded transit model from checkpoints/transit_20260523_164127/final_transit.zip`. The embedded eval for this old transit model shows 93.3% success, which is actually better than the newer latest model (76.7% embedded), so the discrepancy is not harmful in practice but the documentation is incorrect.

---

## DEAD / GARBAGE CODE

No significant dead code was introduced. The `logs/debug_move_*.jsonl` files removed in commit b76b4b29 were test artifacts. The old TensorBoard event files removed in b76b4b29 were superseded by new runs.

---

## TRAINING/PRODUCTION MISMATCHES SUMMARY

| Component | Training | Production / Inference | Status |
|-----------|----------|----------------------|--------|
| Observation format | Phase-9 25D dict | Same (enabled in `_run_stage`) | Match |
| Goal vector | `env.goal - grip_pos` set before reset | Same, set via `env.goal = target_pos` before loop | Match |
| Success criterion | `d_xy < 10mm AND d_z < 10mm` | Same call to `_is_success()` | Match |
| Stability threshold | 20 mm/s (`env.yaml`) | 50 mm/s (hardcoded default in model_controller) | **MISMATCH** |
| Finger action | `action[3]` ignored; `finger_target_joint` from scenario | Same; `action[3]` forced but also ignored | Match (both ignore it) |
| Episode budget | 200-step TimeLimit | 300-step inference budget | Benign |
| Drift limit | 8mm at curriculum end | 8mm (`eval_drift_limit` from env_cfg) | Match |
| Start position (transit) | Random board XY at SAFE_Z | HOME_POS at SAFE_Z | Within training distribution |
| Start position (descend) | Random board XY at SAFE_Z → arm teleported there | After transit success: arm at target_xy ± transit error | Near-match |
| Start position (ascend) | Random board XY at HOVER_Z → arm teleported there | After grasp: arm at target_xy ± grasp XY error | Near-match; grasp error up to ~11mm |

The stability threshold mismatch (50ms vs 20mm/s) is the most actionable technical discrepancy. It means the model is declared successful 2.5x sooner than it was trained to stop, which can cause the arm to still be moving at the start of the next stage's soft_reset.
