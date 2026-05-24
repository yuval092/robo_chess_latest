# 2026-05-24 Model Selection And Integration Review

## Scope

Compared the latest and best checkpoints for each specialist scenario, selected the better deployed checkpoint, and re-ran integration checks through the embedded controller path used by `scripts/run_chess_ui.py`.

## Checkpoint Selection

Direct deterministic eval used `scripts/eval_rl_stages.py` with the configured 8 mm evaluation drift limit. The main comparison was three 30-episode runs per checkpoint.

| Scenario | Candidate | Runs | Result | Selection |
| --- | --- | ---: | --- | --- |
| transit | `checkpoints/transit_20260524_084201/best_model_transit.zip` | 90 episodes | 82/90 success, 8 timeouts | rejected |
| transit | `checkpoints/transit_20260524_084201/latest_model_transit.zip` | 90 episodes | 85/90 success, 5 timeouts | selected |
| descend | `checkpoints/descend_20260523_222907/best_model_descend.zip` | 90 episodes | 88/90 success, 2 tube breaches | rejected |
| descend | `checkpoints/descend_20260523_222907/latest_model_descend.zip` | 90 episodes | 90/90 success | selected |
| ascend | `checkpoints/ascend_20260523_222849/best_model_ascend.zip` | 90 episodes | 86/90 success, 4 tube breaches | selected |
| ascend | `checkpoints/ascend_20260523_222849/latest_model_ascend.zip` | 90 episodes | 69/90 success, 20 timeouts, 1 tube breach | rejected |

Additional sanity baselines:

- Old deployed transit final: `checkpoints/transit_20260523_164127/final_transit.zip` scored 90/90 direct, but only 28/30 in embedded transit eval.
- Transit latest scored 30/30 in embedded transit eval, so it is the better integrated transit choice.
- Descend final also scored 90/90 direct and is close to descend latest, but the requested latest-vs-best selection still favors latest.
- Ascend final scored 68/90 direct and is worse than ascend best.

Updated `configs/training.yaml` deployed models to:

- transit: `checkpoints/transit_20260524_084201/latest_model_transit.zip`
- descend: `checkpoints/descend_20260523_222907/latest_model_descend.zip`
- ascend: `checkpoints/ascend_20260523_222849/best_model_ascend.zip`

## Embedded Integration Results

All integration checks loaded the model paths from `configs/training.yaml`.

- `scripts/eval_stages.py --use-rl-models --stages transit,descend,ascend --n-episodes 15`
  - transit: 15/15 success, avg error 6.1 mm, p95 9.1 mm
  - descend: 15/15 success, avg error 10.3 mm, p95 10.6 mm
  - ascend: 13/15 success, 2 tube breaches at 8.2-8.3 mm
- Transit embedded follow-up:
  - selected latest: 30/30 success, avg error 5.3 mm, p95 9.7 mm
  - old final baseline: 28/30 success, 2 timeouts, avg error 5.5 mm, p95 7.4 mm
- `scripts/eval_sequence.py --use-rl-models --chain vertical --n-episodes 20`
  - 20/20 full success
- `scripts/eval_sequence.py --use-rl-models --chain full_move --n-episodes 10`
  - 10/10 full success
- `scripts/eval_chess_game_flow.py --use-rl-models --verify-agreement --nonmoving-tolerance-mm 2.0`
  - passed default four-move game flow
- `python -m pytest tests/ui tests/chess_game tests/physical tests/chess_env -q`
  - 78 passed, 11 warnings

Note: the project `.venv` does not have `pytest` installed; tests were run with the active shell Python environment.

## Main Failure Modes

The dominant remaining integration failure is not catastrophic movement. It is insufficient robustness margin around terminal conditions:

- Transit failures were timeouts: the policy reaches near the goal but does not consistently satisfy the success and stability criteria before the episode limit.
- Descend latest had one timeout in repeated direct eval; best had a strict tube breach.
- Ascend direct best is strong, but embedded per-stage eval still produced rare tube breaches at 8.2-8.3 mm against the 8.0 mm production limit.

This means the direct training wrapper and the embedded production controller are close but not identical enough. The most important mismatch is the final settling/tube-margin behavior around success, especially for ascend.

## Recommended Improvements

1. Select production models using the embedded controller path, not direct wrapper eval alone. Direct eval is useful for screening, but `eval_stages.py --use-rl-models`, sequence evals, and game-flow evals are the source of truth for `run_chess_ui.py`.
2. Add a terminal settle/hold policy behavior. The controller or training target should reward arriving with lower velocity and staying inside the target threshold for a few consecutive steps. This addresses timeouts caused by near-goal motion.
3. Train vertical stages with more safety margin than production. Production uses 8 mm; train/evaluate candidate checkpoints with a 6-7 mm effective drift margin or add a strong pre-breach penalty starting before 8 mm. This should eliminate 8.2-8.5 mm tube breaches.
4. Make reset distributions match embedded transitions. Vertical policies should see starts produced by `soft_reset()` after scripted grasp/place and after previous RL stages, not only ideal isolated starts.
5. Add a checkpoint-selection script that runs a fixed seed suite across direct stage eval, embedded stage eval, vertical sequence, full-move sequence, and game flow, then writes a simple selection report.
6. Consider a guarded inference fallback only as a runtime safety layer: if drift approaches the tube limit or timeout is imminent, switch to a short scripted correction/settle routine. This should not replace training the policy to satisfy the production constraints.

## Deviations

No intentional deviation from the requested latest-vs-best comparison. I also kept the previously known final-model behavior in mind, but the deployed registry was updated according to the latest/best selection requested here.
