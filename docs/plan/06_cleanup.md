# Stage 6 — Cleanup: Delete `scripts/`, Update `pyproject.toml`, Update Docs

## Objective

Once Stages 1–5 are complete and validated, remove the `scripts/` folder entirely,
wire all new entry points in `pyproject.toml`, reinstall the package, and update
the relevant documentation files.

---

## 6.1 — Deletion list

The following files and directories are **deleted**. No logic in them survives
in the new CLI; either their logic was migrated (Stages 1–4) or they are
development-only tools being discarded.

### `scripts/` — entire directory

```
scripts/__init__.py
scripts/train_rl.py                    → migrated to training/cli/main.py
scripts/validate_config.py             → discarded (dev tool)
scripts/validate_deployed_models.py    → discarded (dev tool)
scripts/analyze_debug_log.py           → discarded (dev tool)
scripts/eval_stages.py                 → migrated to src/cli/eval_stage.py
scripts/eval_sequence.py               → merged into eval_stage (--chain flag)
scripts/eval_stress.py                 → merged into eval_flow (--mode complex)
scripts/eval_targeted.py               → merged into eval_flow (--mode complex)
scripts/eval_all_cells_rl.py           → merged into eval_stage (--controller model)
scripts/eval_all_square_moves.py       → migrated to eval_flow (--mode full)
scripts/eval_chess_piece_move.py       → migrated to eval_flow (--mode simple)
scripts/eval_chess_reachability.py     → merged into eval_physics (geometry phase)
scripts/eval_chess_game_flow.py        → discarded (integration test → tests/)
scripts/eval_draw_conditions.py        → discarded (unit test → tests/)
scripts/eval_game_logic.py             → discarded (unit test → tests/)
scripts/eval_special_moves.py          → discarded (unit test → tests/)
scripts/eval_grasp_physics.py          → discarded (out of scope for eval_physics)
scripts/verify_physics.py              → migrated to src/cli/eval_physics.py
scripts/diagnostics/__init__.py        → discarded
scripts/diagnostics/check_import_graph.py        → discarded
scripts/diagnostics/debug_one_move.py            → discarded
scripts/diagnostics/diagnose_descend_afile.py    → discarded
scripts/diagnostics/diagnose_transit_c2.py       → discarded
scripts/diagnostics/diagnose_transit_full.py     → discarded
scripts/diagnostics/eval_rl_stages_direct.py     → discarded
scripts/diagnostics/README.md                    → discarded
scripts/README.md                                → superseded by updated docs
```

**Shell command to execute**:

```bash
rm -rf scripts/
```

Run this only after all validation checklists in Stages 1–5 are complete.

---

## 6.2 — `pyproject.toml` — final entry points

Replace the `[project.scripts]` section with the complete final set:

```toml
[project.scripts]
# Production
robo-chess-ui           = "src.cli.run_chess_ui:main"
robo-chess-generate     = "src.cli.generate_scene:main"
robo-chess-visualize    = "src.cli.visualize:main"

# Evaluation
robo-chess-eval-stage   = "src.cli.eval_stage:main"
robo-chess-eval-physics = "src.cli.eval_physics:main"
robo-chess-eval-flow    = "src.cli.eval_flow:main"

# Training
robo-chess-train        = "training.cli.main:main"
```

After editing, reinstall:

```bash
pip install -e .
```

Verify all seven commands are on `$PATH`:

```bash
which robo-chess-ui robo-chess-generate robo-chess-visualize \
      robo-chess-eval-stage robo-chess-eval-physics robo-chess-eval-flow \
      robo-chess-train
```

---

## 6.3 — Documentation updates

### `docs/10_scripts.md` — **Replace entirely**

The current `docs/10_scripts.md` documents `scripts/` in detail (435 lines).
Replace it with a new page that:

1. States that `scripts/` was removed.
2. Documents all seven CLI entry points with their full argparse signatures.
3. Includes usage examples for each command.
4. Cross-references `docs/04_training.md` for training workflow details.

Suggested structure for the replacement:

```markdown
# CLI Reference

## Production Commands
- robo-chess-ui        — launch web UI
- robo-chess-generate  — regenerate scene assets
- robo-chess-visualize — interactive render loop

## Evaluation Commands
- robo-chess-eval-stage   — per-stage waypoint accuracy
- robo-chess-eval-physics — physics integrity check
- robo-chess-eval-flow    — full piece-move evaluation (simple/complex/full)

## Training Command
- robo-chess-train        — train or eval a specialist RL model
```

### `docs/04_training.md` — Update training invocation

The current training doc likely references `python scripts/train_rl.py`. Update
all references to use `robo-chess-train train`.

Before:
```bash
python scripts/train_rl.py --stage transit --envs 8
```

After:
```bash
robo-chess-train train --stage transit --envs 8
```

Also add a section describing `robo-chess-train eval`.

### `docs/00_overview.md` — Update architecture diagram

Add `training/cli/` to the component list alongside `src/cli/`.
Remove the `scripts/` row from the directory table.

### `docs/11_testing.md` — Update evaluation section

Replace references to running scripts directly with the new CLI commands.

---

## 6.4 — Verify no remaining imports of `scripts.*`

After deletion, grep for any surviving import of the old package:

```bash
grep -r "from scripts" src/ training/ tests/ --include="*.py"
grep -r "import scripts" src/ training/ tests/ --include="*.py"
```

Both must return empty. If any references are found, they are broken imports
that need to be redirected to the new CLI modules or removed.

---

## 6.5 — Final end-to-end test run

Run the full test suite to verify nothing broke:

```bash
pytest --tb=short -q
```

Then run the acceptance smoke-tests manually:

```bash
# Training CLI
robo-chess-train --help
robo-chess-train train --help
robo-chess-train eval --help

# Evaluation CLIs
robo-chess-eval-stage --help
robo-chess-eval-physics --help
robo-chess-eval-flow --help

# Existing CLIs
robo-chess-visualize --help
robo-chess-generate --help
robo-chess-ui --help

# Quick functional checks
robo-chess-eval-stage  --stage transit  --episodes 2  --controller scripted
robo-chess-eval-physics --settle-steps 50
robo-chess-eval-flow   --mode simple    --episodes 1  --controller scripted
robo-chess-train train --stage transit  --timesteps 100
```

---

## Validation Checklist

- [ ] `rm -rf scripts/` executed.
- [ ] `pyproject.toml` updated with all seven entry points.
- [ ] `pip install -e .` succeeds with no errors.
- [ ] All seven entry points appear in `which` output.
- [ ] `grep -r "from scripts"` returns empty.
- [ ] `grep -r "import scripts"` returns empty.
- [ ] `pytest --tb=short -q` passes.
- [ ] All seven `--help` outputs are clean.
- [ ] All four quick functional checks complete without exception.
- [ ] `docs/10_scripts.md` rewritten to document new CLI commands.
- [ ] `docs/04_training.md` updated to reference `robo-chess-train`.
- [ ] `docs/00_overview.md` updated to remove `scripts/` from architecture.
- [ ] `docs/11_testing.md` evaluation section updated.
