# Runtime And Training

## `python main.py`

`main.py` is the only runtime entry point.

```bash
python main.py [--host HOST] [--port PORT] [--no-visualize] [--delay SECS]
               [--debug]
               [--transit-model PATH] [--descend-model PATH] [--ascend-model PATH]
```

Default mode starts the Flask chess UI and a MuJoCo simulation. The viewer opens
unless `--no-visualize` is supplied. Model paths default to
`configs/deployed_models.yaml`.

## All-Square Sweep

```bash
RUN_EXHAUSTIVE_PHYSICAL_MOVES=1 pytest tests/integration/test_all_square_moves.py
```

This opt-in pytest case runs all 4,032 distinct source/destination pairs with
`black_rook_a` by default. It is intentionally excluded from normal runtime and
from default pytest runs because it is long.

## `robo-chess-train train`

```bash
robo-chess-train train --stage {transit,descend,ascend} [options]
```

Options:

```text
--stage STAGE    Required movement stage.
--envs N         Parallel environments; default comes from configs/training.yaml.
--model PATH     Resume from checkpoint or base model.
--timesteps N    Total training timesteps; default comes from config.
--save-dir PATH  Output directory.
--fixed-drift    Skip drift curriculum and use the final limit immediately.
--debug          Verbose training and environment logs.
```

Examples:

```bash
robo-chess-train train --stage transit
robo-chess-train train --stage descend --model checkpoints/descend_prev/model.zip
robo-chess-train train --stage ascend --timesteps 100
```

Physics, scene, and flow diagnostics now live in pytest rather than installed
runtime commands.
