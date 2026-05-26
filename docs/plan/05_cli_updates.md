# Stage 5 — Impact on Existing `src/cli/` Commands

## Objective

Audit every existing CLI entry point for overlap with the new evaluation commands
and determine whether each needs to change, and how.

---

## Existing Entry Points

| Command | Module | Status |
|---------|--------|--------|
| `robo-chess-ui` | `src.cli.run_chess_ui` | Unchanged |
| `robo-chess-generate` | `src.cli.generate_scene` | Unchanged |
| `robo-chess-visualize` | `src.cli.visualize` | Minor update (see below) |

---

## `robo-chess-ui` — Unchanged

`src/cli/run_chess_ui.py` launches the Flask web UI, creates a
`GameOrchestrator`, and drives the render loop. It has no overlap with evaluation
commands; it is a production entry point and must remain exactly as-is.

**Change required**: None.

---

## `robo-chess-generate` — Unchanged

`src/cli/generate_scene.py` regenerates MuJoCo XML assets (board, pieces, zones,
STLs). It is a build/asset tool with no evaluation logic.

**Change required**: None.

---

## `robo-chess-visualize` — Minor update

### Current behaviour

`src/cli/visualize.py` runs a human-facing render loop with `ScriptedController`.
It accepts `--scenario`, `--src-xy`, `--dst-xy`, `--episodes` (0 = infinite),
`--wait` (pause between episodes), `--setup` (teleport pieces to starting squares),
and prints a running tally of success/failure.

### Relationship to new eval commands

| Concern | `robo-chess-visualize` | `robo-chess-eval-stage` |
|---------|----------------------|------------------------|
| Purpose | Human-interactive watching | Automated metric collection |
| Controller | Scripted only | Scripted or model |
| Episodes | 0 = infinite loop | Finite, configurable |
| Output | Live tally, no table | Summary table + failure breakdown |
| Progress bar | No | Yes |
| Rendering | Always on | Optional (not added to eval) |
| Pause between eps | `--wait` | No |

The two commands serve different purposes. `visualize` is for a human to watch
the robot; `eval-stage` is for CI and offline measurement. They are **not**
redundant — keep both.

### What to update

The only useful change is to add `--controller model` support to `visualize` so
a developer can watch the RL model run interactively (mirroring what `eval-stage`
already offers). This removes the need to launch the full `robo-chess-ui` web UI
just to see the model in action.

#### New flag

```
--controller  {scripted,model}   Default: scripted.
--transit-model  PATH
--descend-model  PATH
--ascend-model   PATH
```

#### Implementation

In `src/cli/visualize.py`, after parsing `args`:

```python
if args.controller == "scripted":
    controller = ScriptedController(
        stage=args.scenario, debug=args.debug
    )
else:
    from src.chess_env.model_controller import ModelEmbeddedController
    from src.chess_env.model_registry import load_model
    overrides = {}
    if args.transit_model:
        overrides["transit"] = args.transit_model
    if args.descend_model:
        overrides["descend"] = args.descend_model
    if args.ascend_model:
        overrides["ascend"] = args.ascend_model
    models = {
        s: load_model(stage=s, override_path=overrides.get(s))
        for s in ["transit", "descend", "ascend"]
    }
    controller = ModelEmbeddedController(models=models, debug=args.debug)
```

No other logic in `visualize.py` needs to change.

### Summary of changes to `visualize.py`

1. Add `--controller`, `--transit-model`, `--descend-model`, `--ascend-model` to the parser.
2. Replace the hardcoded `ScriptedController(...)` instantiation with the
   controller-selection block above.
3. Default remains `scripted` so existing usage is fully backwards-compatible.

---

## Summary table

| Command | Change? | What changes |
|---------|---------|-------------|
| `robo-chess-ui` | No | — |
| `robo-chess-generate` | No | — |
| `robo-chess-visualize` | Minor | Add `--controller model` + model path flags |
| `robo-chess-eval-stage` | New (Stage 2) | — |
| `robo-chess-eval-physics` | New (Stage 3) | — |
| `robo-chess-eval-flow` | New (Stage 4) | — |
| `robo-chess-train` | New (Stage 1) | — |

---

## Validation Checklist

- [ ] `robo-chess-visualize --help` shows the new `--controller` flag.
- [ ] `robo-chess-visualize --scenario transit --episodes 1 --controller scripted` runs exactly as before.
- [ ] `robo-chess-visualize --scenario transit --episodes 1 --controller model` loads models and runs the RL controller with rendering on.
- [ ] `robo-chess-ui --help` unchanged.
- [ ] `robo-chess-generate --help` unchanged.
- [ ] No existing test that calls `ScriptedController` directly from `visualize.py` is broken.
