# Stage 1 — Training CLI (`training/cli/`)

## Objective

Migrate `scripts/train_rl.py` into a proper, installable CLI component that lives
inside the `training/` package. Add a second sub-command for evaluating a model
against a specific stage (delegates entirely to the eval_stage logic built in
Stage 2, so no evaluation code is duplicated here).

---

## Background

`scripts/train_rl.py` currently does exactly one thing: parse arguments and call
`training.trainer.SACTrainer`. It is 45 lines of thin glue. The `training/`
package itself (`trainer.py`, `callbacks.py`, `envs/`) is already self-contained
and has no dependency on `scripts/`. Moving the entry point into `training/cli/`
therefore requires no changes to any training logic.

---

## New Files

### `training/cli/__init__.py`

Empty package marker.

```python
"""Command-line entry points for RoboChess training."""
```

---

### `training/cli/main.py`

Single file with two sub-commands: `train` and `eval`.

#### Full argparse specification

```
robo-chess-train <sub-command> [options]

Sub-commands
------------
  train    Train a specialist SAC model for one movement stage.
  eval     Evaluate a trained model against a movement stage.
           (Delegates to robo-chess-eval-stage logic.)

robo-chess-train train
  --stage    {transit,descend,ascend}   REQUIRED.  Stage to train.
  --envs     INT          Number of parallel envs.  Default: from configs/training.yaml.
  --model    PATH         Resume from checkpoint / base model path.
  --timesteps INT         Total timesteps to train.  Default: from configs/training.yaml.
  --save-dir PATH         Where to write the model ZIP.  Default: models/.
  --fixed-drift           Skip drift curriculum; use final drift limit from step 1.
  --debug                 Verbose training logs.

robo-chess-train eval
  --stage    {transit,descend,ascend}   REQUIRED.  Stage to evaluate.
  --model    PATH         REQUIRED.  Path to the model ZIP to evaluate.
  --episodes INT          Episodes per stage.  Default: 50.
  --debug                 Verbose debug logs (forwarded to eval_stage).
```

#### Implementation sketch

```python
#!/usr/bin/env python3
"""robo-chess-train — train or evaluate a specialist SAC model."""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robo-chess-train",
        description="Train or evaluate a RoboChess specialist RL model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train a specialist SAC model.")
    train_p.add_argument(
        "--stage", required=True, choices=["transit", "descend", "ascend"]
    )
    train_p.add_argument("--envs", type=int, default=None,
                         help="Parallel training envs (default: from config).")
    train_p.add_argument("--model", type=str, default=None,
                         help="Resume from this checkpoint / base model path.")
    train_p.add_argument("--timesteps", type=int, default=None,
                         help="Total timesteps (default: from config).")
    train_p.add_argument("--save-dir", type=str, default=None,
                         help="Output directory for model ZIP (default: models/).")
    train_p.add_argument("--fixed-drift", action="store_true",
                         help="Skip drift curriculum; start at final drift limit.")
    train_p.add_argument("--debug", action="store_true",
                         help="Verbose training output.")

    # ── eval ───────────────────────────────────────────────────────────────
    eval_p = sub.add_parser(
        "eval",
        help="Evaluate a trained model on a movement stage. "
             "Delegates to robo-chess-eval-stage.",
    )
    eval_p.add_argument(
        "--stage", required=True, choices=["transit", "descend", "ascend"]
    )
    eval_p.add_argument("--model", required=True,
                        help="Path to the model ZIP to evaluate.")
    eval_p.add_argument("--episodes", type=int, default=50,
                        help="Episodes to run (default: 50).")
    eval_p.add_argument("--debug", action="store_true",
                        help="Forward --debug to robo-chess-eval-stage.")

    return parser


def _run_train(args: argparse.Namespace) -> None:
    from training.trainer import SACTrainer

    print(f"[robo-chess-train] Training stage={args.stage} ...")
    trainer = SACTrainer(
        stage=args.stage,
        num_envs=args.envs,
        debug=args.debug,
        fixed_drift=args.fixed_drift,
    )
    if args.timesteps:
        trainer.total_timesteps = args.timesteps

    trainer.train(model_path=args.model, save_dir=args.save_dir)
    print("[robo-chess-train] Training complete.")


def _run_eval(args: argparse.Namespace) -> None:
    # Import and call eval_stage directly so we share all output formatting.
    from src.cli.eval_stage import run_eval_stage

    run_eval_stage(
        stages=[args.stage],
        episodes=args.episodes,
        controller="model",
        model_overrides={args.stage: args.model},
        debug=args.debug,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        _run_eval(args) if False else _run_train(args)
    elif args.command == "eval":
        _run_eval(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

> **Note on `eval` sub-command**: it calls `src.cli.eval_stage.run_eval_stage()`
> — a callable form of the `robo-chess-eval-stage` entry point defined in Stage 2.
> This keeps evaluation logic in exactly one place.

---

## pyproject.toml addition

```toml
[project.scripts]
# ... existing entries ...
robo-chess-train = "training.cli.main:main"
```

The `training` package must also be listed in `[tool.setuptools.packages.find]`
if it is not already discovered automatically. Since `training/__init__.py` exists,
setuptools will find it; verify after `pip install -e .`.

---

## Impact on existing code

| File | Change |
|------|--------|
| `training/trainer.py` | None — API unchanged |
| `training/callbacks.py` | None |
| `training/envs/*` | None |
| `scripts/train_rl.py` | **Deleted** in Stage 6 |

---

## Validation Checklist

- [ ] `training/cli/__init__.py` exists.
- [ ] `training/cli/main.py` exists and is importable (`python -c "from training.cli.main import main"`).
- [ ] `pip install -e .` succeeds and `robo-chess-train` is on `$PATH`.
- [ ] `robo-chess-train --help` shows both sub-commands.
- [ ] `robo-chess-train train --help` shows all flags.
- [ ] `robo-chess-train eval --help` shows all flags.
- [ ] `robo-chess-train train --stage transit --timesteps 100` runs without importing from `scripts/`.
- [ ] After Stage 2 is complete: `robo-chess-train eval --stage transit --model models/transit.zip --episodes 2` produces a summary table.
