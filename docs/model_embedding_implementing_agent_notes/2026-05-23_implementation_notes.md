# Model Embedding Implementation Notes

Date: 2026-05-23

## Scope Implemented

- Read `docs/current_status/` and `docs/model_embedding_refactor_plan/new_version/`.
- Implemented Phase-9 observation support in `src/chess_env/task.py`.
- Replaced the `ChessTaskEnv.step()` and `compute_reward()` stubs with RL training behavior.
- Added drift curriculum parameters and `_get_current_drift_limit()` to `ChessTaskEnv`.
- Added the `training/` package:
  - `training/envs/*` Phase-9 wrappers and env factories.
  - `training/callbacks.py`.
  - `training/trainer.py`.
- Added `configs/training.yaml`, wired to the archived pretrained base model at
  `archive/rl_system/models/sac-FetchPickAndPlace-v4.zip`.
- Added `scripts/train_rl.py`.
- Added `src/chess_env/model_controller.py` with scripted fallback and the full controller interface.
- Updated `scripts/eval_stages.py` and `scripts/eval_sequence.py` with `--use-rl-models` and model path flags.
- Added `scripts/eval_rl_stages.py`.
- Updated `configs/env.yaml` for the requested strict 8mm drift limit:
  `drift_limit_end: 0.008` and `eval_drift_limit: 0.008`.
- Set curriculum timing for 4 envs and 200K aggregate steps:
  `drift_curriculum_steps: 50000` per worker, so the curriculum reaches the
  final 8mm limit after `50000 * 4 = 200000` total timesteps.
- Set `configs/training.yaml` to `num_envs: 4` and `total_timesteps: 200000`.

## Plan Deviations

- `training.trainer.SACTrainer` uses `DummyVecEnv` when `num_envs == 1` and `SubprocVecEnv` otherwise. The plan text showed `SubprocVecEnv` unconditionally, but also described `--envs 1` as the debugging path. This avoids subprocess overhead and makes local smoke/debug runs easier while preserving the planned multi-worker behavior.
- `SubprocVecEnv` uses `start_method="fork"`. The default forkserver mode
  failed locally with `PermissionError: [Errno 1] Operation not permitted`.
- `ChessTaskEnv.step()` tolerates the legacy 7D observation path in addition to the planned Phase-9 path. Training wrappers still enable Phase-9 as required; the fallback keeps direct Gym calls from failing during diagnostics.
- Phase-9 observation spaces and arrays use `float64`, not the planned
  `float32`, because the archived pretrained SAC model was saved with a
  `float64` dict observation space. `SAC.load(..., env=...)` rejected the
  wrapper when the dtypes were `float32`.
- `ModelEmbeddedController._run_stage()` restores the previous `_use_phase9_obs` value instead of always restoring `False`. This preserves caller state if an already wrapped Phase-9 env is passed in.
- `ModelEmbeddedController.load_model()` raises a clear `ValueError` when a configured model path is missing instead of passing `None` into `SAC.load()`.
- `ModelEmbeddedController.load_model()` now passes a matching Phase-9 wrapper
  env into `SAC.load()`. The saved checkpoint uses a HER replay buffer, and
  Stable-Baselines3 raises `AssertionError: You must pass an environment when
  using HerReplayBuffer` if the model is loaded without an env.
- `ModelEmbeddedController.load_available()` was added so transit can be
  embedded while descend/ascend remain scripted until their specialist models
  exist.
- Embedded inference now forces gripper actions the same way training does:
  closed for transit/ascend and open for descend.
- Embedded transit uses a 300-step inference budget to match
  `ScriptedController.TRANSIT_MAX_STEPS`. The Gym training/eval wrapper still
  has the registered 200-step episode cap.
- Embedded success checks now call `env._is_success()` so the integration path
  uses the same per-axis XY/Z threshold as training instead of a stricter 3D
  Euclidean norm.
- Evaluation cadence was reduced from 50 episodes every 10K timesteps to 5
  episodes every 50K timesteps. A monitored run stalled for several minutes at
  the first heavy evaluation; the lighter cadence keeps this PC-compatible
  while preserving periodic checkpointing.

## Verification

- `python -m py_compile src/chess_env/task.py src/chess_env/model_controller.py training/__init__.py training/envs/__init__.py training/envs/transit_env.py training/envs/ascend_env.py training/envs/descend_env.py training/callbacks.py training/trainer.py scripts/train_rl.py scripts/eval_stages.py scripts/eval_sequence.py scripts/eval_rl_stages.py`
- Phase-9 env smoke:
  - Created `ChessFetchTask-v0`, wrapped with `TransitTrainEnv`, reset, and stepped once.
  - Confirmed observation shapes: `(25,)`, `(3,)`, `(3,)`.
- Pretrained model compatibility:
  - `SAC.load("archive/rl_system/models/sac-FetchPickAndPlace-v4.zip", env=...)`
    succeeds after matching the archived model's `float64` observation space.
- Curriculum check:
  - `num_envs = 4`, `total_timesteps = 200000`,
    `drift_curriculum_steps = 50000`.
  - Drift limit is 100mm at worker step 0, 54mm at worker step 25K, and 8mm at
    worker step 50K.
- Controller fallback smoke:
  - Created `ModelEmbeddedController` with no loaded models.
  - Called `run_transit()` and confirmed it fell back to scripted movement successfully.
- CLI smoke:
  - `python scripts/train_rl.py --help`
  - `python scripts/eval_stages.py --help`
  - `python scripts/eval_sequence.py --help`
  - `python scripts/eval_rl_stages.py --help`
- Tests:
  - `python -m pytest tests/chess_env/test_task_chaining.py tests/chess_env/test_waypoints.py -q` passed: 12 passed.
  - `python -m pytest tests/chess_game tests/physical tests/ui -q` passed: 64 passed.
  - `python scripts/eval_stages.py --stages transit --n-episodes 1` passed with 100% success for the one scripted transit episode.
- Monitored training run:
  - Command: `MPLCONFIGDIR=/tmp/matplotlib python scripts/train_rl.py --stage transit`
  - Active checkpoint directory: `checkpoints/transit_20260523_164127/`.
  - First 50K evaluation completed and wrote `best_model_transit.zip` and
    `latest_model_transit.zip`.
  - The run continued beyond 50K, confirming that the earlier evaluation stall
    was resolved.
- Checkpoint integration pass:
  - `configs/training.yaml` now points `deployed_models.transit` at
    `checkpoints/transit_20260523_164127/best_model_transit.zip`.
  - `scripts/run_chess_ui.py` defaults to `ModelEmbeddedController` with the
    configured transit model. `--use-scripted-controller` is the opt-out.
  - UI factory smoke confirmed `ModelEmbeddedController` is constructed and its
    transit model is loaded.
  - `python scripts/eval_rl_stages.py --stage transit --model checkpoints/transit_20260523_164127/best_model_transit.zip --n-episodes 20`
    returned 13/20 success, 0 crashes, 7 timeouts.
  - `python scripts/eval_stages.py --use-rl-models --stages transit --n-episodes 20 --transit-model checkpoints/transit_20260523_164127/best_model_transit.zip`
    returned 70% success, 0% crash, 30% timeout.
  - `python scripts/eval_sequence.py --use-rl-models --chain pick --n-episodes 3 --transit-model checkpoints/transit_20260523_164127/best_model_transit.zip`
    passed 3/3 at the default center-board source.
  - `python scripts/eval_chess_game_flow.py --use-rl-models --moves e2e4 --verify-agreement --nonmoving-tolerance-mm 2.0`
    failed on the first transit with `TIMEOUT`; final grip was near the source
    square but not stable enough before timeout.
  - `python -m pytest tests/ui tests/chess_game tests/physical tests/chess_env -q`
    passed: 78 passed.

## Notes

- Importing SB3/Gymnasium-Robotics emits an upstream warning about Adroit environment reward versioning.
- Training is run one stage at a time. No concurrent training sessions were started.
- `MPLCONFIGDIR=/tmp/matplotlib` avoids the Matplotlib cache warning caused by
  `/home/user/.config/matplotlib` not being writable in this environment.
- At the time of the checkpoint integration pass, the active training process
  was still running and had progressed beyond 134K/200K timesteps. It was not
  stopped.
