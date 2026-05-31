# Training

## Specialist Model Architecture

Three SAC (Soft Actor-Critic) models are trained independently, each specialising in one arm movement stage:

| Model | Task | Start Z | Goal Z |
|---|---|---|---|
| **transit** | Move horizontally between squares at safe height | `SAFE_Z` | `SAFE_Z` (different XY) |
| **descend** | Lower arm from `SAFE_Z` to `HOVER_Z` over a target square | `SAFE_Z` | `HOVER_Z` |
| **ascend** | Raise arm from `HOVER_Z` back to `SAFE_Z` | `HOVER_Z` | `SAFE_Z` |

All models use the 25-D pretrained observation format from `FetchPickAndPlace-v4` and are fine-tuned from a shared `FetchPickAndPlace-v4` SAC checkpoint (`models/pretrained/sac-FetchPickAndPlace-v4.zip`).

---

## Training Environment (`ChessTrainingEnv`)

### Observation

25-D pretrained observation:
- `grip_pos` (3) — gripper world XYZ
- `fake_object_pos` (3) — set equal to `grip_pos` (no real object tracking)
- `rel_to_goal` (3) — `goal - grip_pos`
- Zeros (16) — object velocity/rotation fields not used

### Reward

```
reward = - dist_weight * dist
         - z_weight * z_error
         - xy_weight * xy_error
         - braking_weight * speed  (if within braking_dist of goal)
         - jitter_weight * ||action_xyz||²
         - floor_penalty  (if near floor_limit)
         + success_bonus  (sparse, on success)
```

Stage-specific braking distances and weights allow each specialist to be tuned independently.

### Episode Termination

- **Success** — grip within `success_threshold` of goal AND speed below `stability_vel_threshold`.
- **Crash** — `FINGER_FAULT`, `FLOOR_HIT`, `TUBE_BREACH`, or `TABLE_HIT`. Returns `terminated=True` with `crash_penalty`.
- **Timeout** — `rl_max_steps_per_stage` steps reached without success or crash.

### Drift Curriculum

Descend and ascend use a cylindrical "tube" centred on the target XY. The tube radius starts at `drift_limit_start` (100 mm) and tightens linearly to `drift_limit_end` (8 mm) over `drift_curriculum_steps` environment steps. This progressively forces the arm to land precisely over the square.

---

## Training Configuration (`configs/training.yaml`)

| Key | Default | Description |
|---|---|---|
| `base_model` | `models/pretrained/sac-FetchPickAndPlace-v4.zip` | Starting checkpoint |
| `num_envs` | 4 | Parallel training environments (SubprocVecEnv if >1) |
| `total_timesteps` | 600 000 | Steps per specialist training run |
| `learning_rate` | 5e-5 | SAC actor/critic learning rate |
| `batch_size` | 512 | Replay buffer sample size |
| `target_entropy` | -4.0 | SAC entropy target |
| `learning_starts` | 10 000 | Steps before first gradient update |
| `buffer_size` | 300 000 | Replay buffer capacity |
| `eval_freq` | 50 000 | Steps between checkpoint evaluations |
| `n_eval_episodes` | 20 | Episodes per evaluation |

---

## Deployed Models

`configs/deployed_models.yaml` lists the paths to the three production-ready model files:

```yaml
transit: models/transit/model.zip
descend: models/descend/model.zip
ascend:  models/ascend/model.zip
```

These paths are resolved by `src/utils/io.resolve_model_paths()`, which accepts optional override paths (e.g. from CLI arguments or test fixtures).

---

## Evaluation

The integration test `test_all_square_to_all_square_physical_moves` (opt-in via `RUN_EXHAUSTIVE_PHYSICAL_MOVES=1`) runs all 64×63 source→destination combinations and reports per-stage success rates. This is the primary benchmark for evaluating new model checkpoints.
