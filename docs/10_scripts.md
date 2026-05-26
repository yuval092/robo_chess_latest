# Scripts Reference

## Overview

The `scripts/` directory contains development tools for training, evaluating, validating, and debugging RoboChess. These are separate from the installed application entry points (`robo-chess-ui`, `robo-chess-visualize`, `robo-chess-generate`) which live in `src/cli/`.

All `eval_*` scripts:
- Accept `--help` and print usage
- Exit with code `1` on failure (CI-friendly)
- Accept `--n-episodes N`, `--visualize`, `--debug` flags
- Default to RL models; use `--use-scripted-only` to skip model loading

---

## Training

### `scripts/train_rl.py`

CLI entry point for training a specialist SAC model for one movement stage.

**Usage:**
```bash
python scripts/train_rl.py --stage transit
python scripts/train_rl.py --stage descend --timesteps 1000000
python scripts/train_rl.py --stage ascend --model checkpoints/ascend_v4/best_model_ascend.zip --save-dir checkpoints/ascend_v5/
```

**Arguments:**

| Argument | Description |
|---|---|
| `--stage` | Required: `transit`, `descend`, or `ascend` |
| `--envs N` | Override number of parallel environments (default from `training.yaml`) |
| `--model PATH` | Resume from checkpoint (default: pretrained FetchPickAndPlace base) |
| `--timesteps N` | Override total training steps |
| `--save-dir PATH` | Override checkpoint output directory |
| `--debug` | Enable debug mode (DummyVecEnv, verbose logging) |
| `--fixed-drift` | Skip drift curriculum; use `drift_limit_end` from step 0 |

**What it does:**

1. Creates `SACTrainer(stage=..., num_envs=..., debug=..., fixed_drift=...)`
2. Calls `trainer.train(model_path=..., save_dir=...)`
3. Saves `best_model_{stage}.zip`, `latest_model_{stage}.zip`, `final_{stage}.zip` to the checkpoint directory

After training, validate and update `configs/deployed_models.yaml`. See [04_training.md](04_training.md) for the full training workflow.

---

## Movement Evaluation Scripts

These scripts require MuJoCo and the trained models. Use `--use-scripted-only` to run with the deterministic scripted controller instead.

### `scripts/eval_stages.py`

Per-stage accuracy evaluation. Runs N independent episodes of each stage (transit, descend, ascend) and reports success rate, crash rate, and timeout rate.

```bash
python scripts/eval_stages.py --n-episodes 20
python scripts/eval_stages.py --stages transit --n-episodes 50 --use-scripted-only
```

**Arguments:**
- `--stages STAGE[,STAGE...]`: Comma-separated list of stages (default: all three)
- `--n-episodes N`: Episodes per stage (default: 20)
- `--drift-limit F`: Tube drift limit in metres (default: from `configs/env.yaml`)
- `--use-scripted-only`: Skip RL models; use `ScriptedController`

**Output:** Per-stage table with success rate, crash count, timeout count, mean episode length.

**Key limitation:** Random goal sampling biases toward board centre. Corner squares (a1, a8, h1, h8) are underrepresented. Use `eval_all_cells_rl.py --mode key` for ground-truth production evaluation.

---

### `scripts/eval_all_cells_rl.py`

All-64-squares RL coverage evaluation. Tests representative source→destination pairs across the board and identifies problem positions.

```bash
python scripts/eval_all_cells_rl.py --mode key
python scripts/eval_all_cells_rl.py --mode full --n-reps 5
python scripts/eval_all_cells_rl.py --mode pawn --test-mode transit_only
```

**Arguments:**
- `--mode {pawn,key,full}`:
  - `pawn`: Tests typical pawn move distances (short hops)
  - `key`: Corner-heavy grid — covers corners, edges, and centre; most representative of production difficulty
  - `full`: All 64×64 source/destination combinations (very slow)
- `--n-reps N`: Repetitions per position pair (default: 3)
- `--test-mode {transit_only,...}`: Test a single stage in isolation

**Output:** Per-square success rate heatmap (text), overall success rate, list of worst-performing squares.

**Recommended usage:** Run `--mode key` after training to get the production success rate. A model should achieve >95% on key squares before deployment.

---

### `scripts/eval_sequence.py`

Full scenario chain evaluation. Tests complete pick-and-place sequences (transit → descend → grasp → ascend → transit → descend → place → ascend).

```bash
python scripts/eval_sequence.py --chain full_move --n-episodes 20
python scripts/eval_sequence.py --chain pick --src-xy "0.88 0.2641" --dst-xy "1.00 0.40"
python scripts/eval_sequence.py --chain vertical --n-episodes 10
```

**Arguments:**
- `--chain {full_move,pick,vertical}`:
  - `full_move`: Complete pick-and-place (transit + descend + grasp + ascend + transit + descend + place + ascend)
  - `pick`: Transit + descend + grasp only
  - `vertical`: Descend + ascend only (no horizontal transit)
- `--src-xy "X Y"`: Source position in metres
- `--dst-xy "X Y"`: Destination position in metres
- `--n-episodes N`: Number of episodes to run

**Output:** Per-episode result table (success/failure, which stage failed), overall success rate.

---

### `scripts/eval_stress.py`

Corner and grid-position stress test. Systematically evaluates the arm at the most difficult positions: board corners, mid-edge squares, and arbitrary diagonal pairs.

```bash
python scripts/eval_stress.py --n-episodes 10
```

Reports per-position success rates. Useful for identifying systematic failures at specific board regions (e.g., a-file descent issues, h8 corner transit).

---

### `scripts/eval_targeted.py`

Tests specific known-difficult source→destination square pairs. Used when a production bug or regression is observed at a particular square.

```bash
python scripts/eval_targeted.py --n-episodes 20
```

Provides per-destination success rates to isolate which positions are causing failures.

---

### `scripts/eval_all_square_moves.py`

Exhaustive physical reachability sweep over all source→destination pairs on the board. Tests whether the arm can physically reach every square from every other square.

```bash
python scripts/eval_all_square_moves.py
```

**Warning**: This script takes a very long time (64×64 = 4096 pairs, each requiring a full arm move). Use `eval_all_cells_rl.py --mode key` for faster production evaluation.

---

### `scripts/eval_grasp_physics.py`

Verifies the grasp physics subsystem: cube hold, drop detection, finger closure behaviour.

```bash
python scripts/eval_grasp_physics.py --n-episodes 20
```

Tests:
- Finger closure fully secures the cube
- Drop detection triggers when cube leaves grip
- Empty-grasp detection (no cube) rejects the grip correctly
- Cube remains at pick position after successful hold

---

### `scripts/verify_physics.py`

Sanity-checks MuJoCo XML assets and board geometry. Does not require RL models.

```bash
python scripts/verify_physics.py
```

Verifies:
- XML loads without errors in MuJoCo
- All 32 chess piece bodies are present
- Board geometry matches `configs/chess.yaml`
- Table extent covers the board with required margins

---

## Chess Logic Evaluation Scripts

These scripts do not require MuJoCo and run quickly. They are appropriate for CI.

### `scripts/eval_game_logic.py`

Simulates complete chess game sequences headlessly:
- Scholar's mate (4-move checkmate)
- Fool's mate (2-move checkmate)
- Full game with castling, en passant, and promotion
- Draw by stalemate, 50-move rule, and insufficient material

```bash
python scripts/eval_game_logic.py --verbose
```

Uses `GameOrchestrator.create_headless()` so no arm or MuJoCo is needed.

---

### `scripts/eval_chess_game_flow.py`

Full logical + physical game flow with the scripted arm controller. Tests that the game orchestrator correctly plans and dispatches all move types through the physical execution layer.

```bash
python scripts/eval_chess_game_flow.py
```

Unlike `eval_game_logic.py`, this exercises `PhysicalPlanExecutor`, `MovementExecutor`, and `PieceTeleporter` (using the scripted controller, not RL models). Requires MuJoCo.

---

### `scripts/eval_chess_piece_move.py`

Tests a single chess piece move through the full stack: chess validation → move planning → physical execution → occupancy update.

```bash
python scripts/eval_chess_piece_move.py --piece "white_pawn_e" --src e2 --dst e4
```

Useful for debugging a specific move path without running a full game.

---

### `scripts/eval_chess_reachability.py`

Maps all 64 board squares to world XY coordinates and verifies they fall within the expected geometry:

```bash
python scripts/eval_chess_reachability.py
```

Checks:
- All squares map to coordinates within table bounds
- File and rank mappings match `configs/chess.yaml:reachability_expected`
- `BoardMapper.nearest_square()` is the inverse of `square_to_xy()` for all squares

---

### `scripts/eval_draw_conditions.py`

Exercises all draw termination conditions:
- Stalemate
- 50-move rule
- 75-move rule
- Insufficient material (K vs K, K+B vs K, K+N vs K)
- Threefold repetition
- Fivefold repetition

```bash
python scripts/eval_draw_conditions.py --verbose
```

Uses `ChessService` with pre-configured FEN positions that trigger each condition immediately.

---

### `scripts/eval_special_moves.py`

Verifies that `MovePlanner` generates correct `PhysicalPlan` command sequences for:
- Kingside castling (e1g1)
- Queenside castling (e1c1)
- En passant capture
- Pawn promotion (with and without capture)
- Promotion + capture combination

```bash
python scripts/eval_special_moves.py --verbose
```

Does not execute moves physically; validates only the command structure of the plan.

---

## Validation Scripts

### `scripts/validate_config.py`

Checks that all required keys are present in all YAML config files and that internal geometry constraints are satisfied.

```bash
python scripts/validate_config.py
```

Validates:
- All required keys exist in `env.yaml`, `chess.yaml`, `physics.yaml`, `training.yaml`, `deployed_models.yaml`
- Board geometry: `cell_size_m == 0.08`, `board_width_m == 0.64`, margin ≥ 30mm
- Z-level ordering: `table_z < cube_z < grasp_z < hover_z < safe_z`
- Graveyard/reserve grid capacities are sufficient

Also called by `tests/test_config_schema.py::test_config_schema_is_valid`.

---

### `scripts/validate_deployed_models.py`

Checks that all paths in `configs/deployed_models.yaml` point to existing files.

```bash
python scripts/validate_deployed_models.py
```

Exits with code 1 and prints missing paths if any model file is absent. Called by `eval_stages.py` and `eval_all_cells_rl.py` before loading models.

---

## Analysis Scripts

### `scripts/analyze_debug_log.py`

Parses a per-step JSONL debug log produced by `scripts/diagnostics/debug_one_move.py` and generates a Markdown summary report.

```bash
python scripts/analyze_debug_log.py logs/debug_move_20260524_142519.jsonl
```

**Output:** Markdown report summarising:
- Phase timing (how many steps each stage took)
- Gripper trajectory (position at key waypoints)
- Crash/failure point identification
- Final position error vs. target

---

## Diagnostic Scripts (`scripts/diagnostics/`)

Developer debug tools. May produce verbose per-step output and are not CI-blocking.

### `diagnostics/debug_one_move.py`

Runs one chess move (default: e2→e3) with full per-step JSONL logging to `logs/`. Captures gripper position, velocity, action, reward, and tube drift every step.

```bash
python scripts/diagnostics/debug_one_move.py --help
python scripts/diagnostics/debug_one_move.py --src e2 --dst e3 --visualize
```

The JSONL output is consumed by `analyze_debug_log.py`. This is the primary tool for investigating step-level arm behaviour.

---

### `diagnostics/check_import_graph.py`

Verifies that `src/` modules do not import from `training/` or `scripts/`. The clean layering of `src/` vs. `training/` is a design invariant (training code must not pollute the production arm control layer).

```bash
python scripts/diagnostics/check_import_graph.py
```

Exits with code 1 and lists violation if any `src/` file imports `training` or `scripts`.

---

### `diagnostics/diagnose_transit_full.py`

Step-by-step investigation of transit `TIMEOUT` failures. Runs the transit stage and prints per-step distance-to-goal, velocity, and tube drift to identify why the arm fails to reach the target in time.

```bash
python scripts/diagnostics/diagnose_transit_full.py
```

---

### `diagnostics/diagnose_transit_c2.py`

Transit diagnosis focused on c2-file positions. The c-file is near the robot torso and can be difficult to reach. This script runs repeated transit episodes targeting c-file squares and logs failure modes.

---

### `diagnostics/diagnose_descend_afile.py`

Descent TIMEOUT diagnosis at a-file positions. The a-file (leftmost column from white's perspective) is at the extreme of the arm's reach. This script investigates why descend times out at these positions.

---

### `diagnostics/eval_rl_stages_direct.py`

Direct RL stage rollout with raw model output logging. Runs the model without the full controller wrapper, printing raw observation vectors and action outputs at each step. Used to investigate what the policy is "seeing" and "doing" in failure cases.

---

## Common Flags Across Scripts

Most evaluation scripts accept a shared set of flags via `src/utils/args.add_common_args`:

| Flag | Description |
|---|---|
| `--visualize` | Open the MuJoCo viewer window |
| `--delay F` | Per-step render delay (seconds) |
| `--debug` | Enable verbose debug logging |
| `--use-scripted-only` | Use `ScriptedController` only (no RL models) |
| `--transit-model PATH` | Override transit model path |
| `--descend-model PATH` | Override descend model path |
| `--ascend-model PATH` | Override ascend model path |
| `--n-episodes N` | Number of evaluation episodes |

---

## Recommended Evaluation Workflow

After training a new model:

```bash
# 1. Validate configs and model paths
python scripts/validate_config.py
python scripts/validate_deployed_models.py

# 2. Quick per-stage check
python scripts/eval_stages.py --n-episodes 20

# 3. Ground-truth production evaluation (corner-heavy)
python scripts/eval_all_cells_rl.py --mode key --n-reps 3

# 4. Full sequence test
python scripts/eval_sequence.py --chain full_move --n-episodes 20

# 5. Chess logic validation (no MuJoCo needed)
python scripts/eval_game_logic.py
python scripts/eval_special_moves.py
python scripts/eval_draw_conditions.py
```

Target success rates before deployment:
- `eval_stages.py`: >98% per stage
- `eval_all_cells_rl.py --mode key`: >95% overall (corners matter most)
- `eval_sequence.py --chain full_move`: >95%
