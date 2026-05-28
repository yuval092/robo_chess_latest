# Training

## Goal

Training fine-tunes three SAC specialists:

- `transit`: horizontal movement at `SAFE_Z`.
- `descend`: vertical movement from `SAFE_Z` to `HOVER_Z` inside an XY tube.
- `ascend`: vertical movement from `HOVER_Z` to `SAFE_Z` inside an XY tube.

Grasp and place are not learned; they are scripted.

## CLI

```bash
robo-chess-train train --stage transit
robo-chess-train train --stage descend
robo-chess-train train --stage ascend
```

Options:

| Option | Meaning |
|---|---|
| `--stage {transit,descend,ascend}` | Required specialist |
| `--envs N` | Parallel training env count |
| `--model PATH` | Base/resume checkpoint |
| `--timesteps N` | Override total timesteps |
| `--save-dir PATH` | Output directory |
| `--fixed-drift` | Disable curriculum and use final drift limit from step 1 |
| `--debug` | Verbose env logging |

## Trainer

`SACTrainer` in `training/trainer.py`:

1. Loads training and env config.
2. Builds N train env factories with `make_train_env()`.
3. Uses `DummyVecEnv` for one env or `SubprocVecEnv(..., start_method="fork")`
   for multiple envs.
4. Builds a separate evaluation env with strict drift limit.
5. Loads a SAC checkpoint with overridden hyperparameters.
6. Resets entropy coefficient to `initial_ent_coef`.
7. Runs `model.learn()`.
8. Saves `final_{stage}.zip`.
9. Closes train and eval envs in `finally`.

Default base model:

```text
models/pretrained/sac-FetchPickAndPlace-v4.zip
```

## Training Environments

`training/envs/__init__.py` provides:

- `make_train_env(stage, drift_curriculum_steps, debug, fixed_drift)`
- `make_eval_env(stage, eval_drift_limit, debug)`

Each factory:

1. Creates `gym.make("ChessFetchTask-v0", force_scenario=stage, ...)`.
2. Wraps it in the stage-specific wrapper.
3. Wraps it with SB3 `Monitor`.

The stage wrappers enable `TRANSFER_OBS_SPACE` so the loaded policy sees the
25-D Fetch-compatible observation.

## Transfer Observation

The pretrained Fetch PickAndPlace model expects:

```text
observation shape: (25,)
achieved_goal:     (3,)
desired_goal:      (3,)
```

RoboChess preserves the shape and sets `object_pos = grip_pos`. The model
therefore treats the gripper as if it is already holding the object and should
move it to the goal.

## Drift Curriculum

Descend and ascend use a tube constraint around the target XY. The drift limit
tightens linearly:

```text
limit = drift_limit_start
      - (drift_limit_start - drift_limit_end)
        * min(total_env_steps / drift_curriculum_steps, 1.0)
```

Defaults:

| Config | Value |
|---|---:|
| `drift_limit_start` | `0.100` |
| `drift_limit_end` | `0.008` |
| `drift_curriculum_steps` | `30000` per worker |
| `eval_drift_limit` | `0.010` |

`--fixed-drift` skips the curriculum.

## Hyperparameters

Defaults from `configs/training.yaml`:

| Key | Value |
|---|---:|
| `num_envs` | `4` |
| `total_timesteps` | `600000` |
| `learning_rate` | `0.00005` |
| `batch_size` | `512` |
| `target_entropy` | `-4.0` |
| `learning_starts` | `10000` |
| `buffer_size` | `300000` |
| `initial_ent_coef` | `0.1` |
| `ent_coef_lr` | `0.001` |
| `eval_freq` | `50000` |
| `n_eval_episodes` | `20` |

## Callbacks and Artifacts

`DetailedLoggingCallback` logs rolling reward, episode length, and success rate
to:

```text
logs/training_progress.log
```

`SuccessRateEvalCallback` saves:

| File | Meaning |
|---|---|
| `latest_model_{stage}.zip` | Most recent evaluated model |
| `best_model_{stage}.zip` | Highest mean eval success rate |
| `final_{stage}.zip` | Saved by trainer after learning completes |

TensorBoard logs are written under:

```text
logs/{stage}_{timestamp}/tensorboard/
```

## Deploying a Checkpoint

Update `configs/deployed_models.yaml`:

```yaml
transit: "models/transit.zip"
descend: "models/descend.zip"
ascend: "models/ascend.zip"
```

or pass CLI overrides to `python main.py`.

Recommended validation before deployment:

```bash
pytest tests/chess_env tests/physical
RUN_EXHAUSTIVE_PHYSICAL_MOVES=1 pytest tests/integration/test_all_square_moves.py
```

