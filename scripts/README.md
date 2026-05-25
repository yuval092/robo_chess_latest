# Scripts

Development tools for evaluating, validating, training, and debugging RoboChess.

All `eval_*` scripts accept `--help` and exit with code 1 on failure, making them
usable in CI pipelines. Most accept `--n-episodes`, `--visualize`, and `--debug`.

> **Application entry points** (UI, viewer, scene generation) are in `src/cli/`
> and installed as `robo-chess-ui`, `robo-chess-visualize`, `robo-chess-generate`.
> This folder contains evaluation and developer tools only.

---

## Evaluation — Movement

RL models are the default for all movement scripts. Pass `--use-scripted-only` to
use the deterministic waypoint controller without trained models.

| Script | What it tests |
|---|---|
| `eval_stages.py` | Per-stage accuracy (transit / descend / ascend), success rate + crash breakdown |
| `eval_sequence.py` | Full scenario chain (pick / full_move / vertical) |
| `eval_stress.py` | Corner and grid-position stress test across the board |
| `eval_targeted.py` | Known-difficult squares, per-destination success rate |
| `eval_all_cells_rl.py` | All-64-squares RL coverage (modes: `pawn`, `key`, `full`) |
| `eval_all_square_moves.py` | Exhaustive physical reachability sweep, all src→dst pairs |
| `eval_grasp_physics.py` | Grasp physics verification (cube hold, drop, finger closure) |
| `verify_physics.py` | MuJoCo XML asset integrity and board geometry sanity check |

```bash
# Quick per-stage check (RL default, 20 episodes each)
python scripts/eval_stages.py --n-episodes 20

# Scripted fallback — no trained models required
python scripts/eval_stages.py --use-scripted-only --stages transit --n-episodes 50

# Full pick-and-place chain, 20 episodes
python scripts/eval_sequence.py --chain full_move --n-episodes 20

# All-squares RL coverage, key positions only
python scripts/eval_all_cells_rl.py --mode key

# Full exhaustive board reachability (takes a long time)
python scripts/eval_all_square_moves.py
```

---

## Evaluation — Chess Logic

These scripts do not require MuJoCo and run fast. Good for CI.

| Script | What it tests |
|---|---|
| `eval_game_logic.py` | Complete game sequences (Scholar's mate, castling, en passant, promotion) |
| `eval_chess_game_flow.py` | Full logical + physical game flow (scripted arm) |
| `eval_chess_piece_move.py` | Single piece move through the full stack |
| `eval_chess_reachability.py` | All 64 board squares mapped to reachable XY coordinates |
| `eval_draw_conditions.py` | All draw condition types (stalemate, 50-move, repetition, …) |
| `eval_special_moves.py` | Castling, en passant, promotion command generation |

```bash
python scripts/eval_game_logic.py --verbose
python scripts/eval_draw_conditions.py --verbose
python scripts/eval_special_moves.py --verbose
python scripts/eval_chess_piece_move.py --help
```

---

## Validation

Run these before launching evaluation or training to catch missing config keys or
model files early.

| Script | What it checks |
|---|---|
| `validate_config.py` | All required YAML config keys exist; no missing sections |
| `validate_deployed_models.py` | All paths in `configs/deployed_models.yaml` point to existing files |

```bash
python scripts/validate_config.py
python scripts/validate_deployed_models.py
```

---

## Training

```bash
# Train one specialist SAC model (pick one stage)
python scripts/train_rl.py --stage transit
python scripts/train_rl.py --stage descend
python scripts/train_rl.py --stage ascend

# Resume from a checkpoint
python scripts/train_rl.py --stage transit --model checkpoints/transit_20260524/best_model.zip

# Override timesteps or save directory
python scripts/train_rl.py --stage ascend --timesteps 500000 --save-dir checkpoints/my_run
```

After training, update `configs/deployed_models.yaml` with the new checkpoint paths
and run `python scripts/validate_deployed_models.py`.

---

## Analysis

```bash
# Parse a debug_one_move JSONL log and generate a Markdown report
python scripts/analyze_debug_log.py logs/debug_move_20260524_142519.jsonl
```

`analyze_debug_log.py` reads the dense per-step log produced by
`scripts/diagnostics/debug_one_move.py` and summarises phase timing, gripper
trajectory, and failure points.

---

## Diagnostics

Scripts in `scripts/diagnostics/` are developer debug tools. They may import
training wrappers, produce verbose per-step output, and are not CI-blocking.

| Script | Purpose |
|---|---|
| `debug_one_move.py` | Run one chess move (e2→e3) with full per-step JSONL logging |
| `check_import_graph.py` | Verify `src/` has no imports from `training/` or `scripts/` |
| `diagnose_transit_full.py` | Investigate transit TIMEOUT failures step-by-step |
| `diagnose_transit_c2.py` | Transit diagnosis focused on c2-file positions |
| `diagnose_descend_afile.py` | Descend TIMEOUT diagnosis at a-file positions |
| `eval_rl_stages_direct.py` | Direct RL stage rollout with raw model output logging |

```bash
python scripts/diagnostics/check_import_graph.py
python scripts/diagnostics/debug_one_move.py --help
```
