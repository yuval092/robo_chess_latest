# Stage 0 — Archive Cleanup and Base Model Relocation

**Objective**: The `archive/` directory contains a mix of dead code (old training scripts superseded by the top-level `training/` package) and an active asset (the base pretrained model). This stage separates them: the base model moves to a clean location, everything else is deleted.

**Must be done before Stage 1** — the base model path in `configs/training.yaml` must be valid throughout the rest of the cleanup.

---

## 0.1 Why `archive/` Cannot Be Fully Deleted

The original cleanup plan said to delete `archive/` entirely. This is now wrong.

`archive/rl_system/models/sac-FetchPickAndPlace-v4.zip` is the pretrained FetchPickAndPlace base model from Gymnasium-Robotics. It is the starting point for all three specialist SAC models (transit, descend, ascend). It is referenced in `configs/training.yaml`:

```yaml
base_model: "archive/rl_system/models/sac-FetchPickAndPlace-v4.zip"
```

Deleting `archive/` without moving this file would break all future training runs.

The dead parts of `archive/` are:
- `archive/rl_system/training/callbacks.py` — superseded by `training/callbacks.py`
- `archive/rl_system/training/trainer.py` — superseded by `training/trainer.py`
- `archive/rl_system/training.yaml` — superseded by `configs/training.yaml`

---

## 0.2 Relocate the Base Model

Move the base model to a top-level `models/` directory, which is the natural home for pretrained weights that are inputs to training rather than outputs.

```bash
mkdir -p models/pretrained
mv archive/rl_system/models/sac-FetchPickAndPlace-v4.zip models/pretrained/sac-FetchPickAndPlace-v4.zip
```

Update `configs/training.yaml`:
```yaml
# Before:
base_model: "archive/rl_system/models/sac-FetchPickAndPlace-v4.zip"

# After:
base_model: "models/pretrained/sac-FetchPickAndPlace-v4.zip"
```

Add a `models/.gitkeep` and a `models/pretrained/.gitkeep` so the directory structure is preserved in git even if the large `.zip` file is gitignored.

---

## 0.3 Inventory `archive/rl_system/models/` Before Deletion

The archive models directory contains two files:

```bash
find archive/rl_system/models -maxdepth 1 -type f -printf "%f %s\n"
# Expected output:
# sac-FetchPickAndPlace-v4.zip   <large>   ← move to models/pretrained/
# latest_model.zip               <size>    ← inspect and decide
```

`latest_model.zip` is an unnamed legacy checkpoint of unknown origin. Before deleting:
```bash
python -c "
from stable_baselines3 import SAC
m = SAC.load('archive/rl_system/models/latest_model.zip')
print('algo:', type(m).__name__)
print('obs_space:', m.observation_space)
print('act_space:', m.action_space)
"
```

If it is an intermediate FetchPickAndPlace checkpoint with no better provenance than `sac-FetchPickAndPlace-v4.zip`, delete it. If it is a specialty checkpoint worth keeping, move it to `models/legacy/` with a descriptive name.

**Required before deletion**: Create `docs/decisions/archive_latest_model.md` with the following content:

```markdown
# Decision: archive/rl_system/models/latest_model.zip

Date: YYYY-MM-DD
Obs space: <paste from inspection above>
Action space: <paste from inspection above>
Decision: DELETE  (or: KEEP AS models/legacy/<descriptive-name>.zip)
Reason: <brief justification>
```

**Do not proceed to section 0.4 until this file is written.** The Stage 0 validation checklist checks for its existence.

## 0.4 Delete the Dead Archive Code

```bash
# Only after docs/decisions/archive_latest_model.md has been written and committed:
mkdir -p docs/decisions
# (file should already exist from 0.3)
test -f docs/decisions/archive_latest_model.md || { echo "STOP: write the decision file first (section 0.3)"; exit 1; }

rm archive/rl_system/training/callbacks.py
rm archive/rl_system/training/trainer.py
rm archive/rl_system/training.yaml
rm archive/rl_system/models/latest_model.zip   # after decision recorded in 0.3
rmdir archive/rl_system/training
rmdir archive/rl_system/models   # now empty after model move and latest_model decision
rmdir archive/rl_system
rmdir archive
```

---

## 0.5 Verify No Other References to `archive/`

```bash
grep -rn "archive/" src/ scripts/ tests/ configs/ training/
```

Must return nothing after this stage.

---

## 0.6 Note: Stage 0 Validation Uses `src.utils.config`

The validation script below imports `from src.utils.config import load_config`. This works because `src/utils/config.py` already exists at this point (Stage 1 adds the shim later). No change needed — just document the sequence dependency: Stage 0 must complete before Stage 1 modifies `config.py`.

---

## Stage 0 — Validation Checklist

```bash
# 0. Decision file recorded before archive deletion
test -f docs/decisions/archive_latest_model.md && echo "OK: decision file present" || echo "FAIL: write docs/decisions/archive_latest_model.md first"

# 1. Base model exists at new location
ls -la models/pretrained/sac-FetchPickAndPlace-v4.zip

# 2. configs/training.yaml points to new path
grep "base_model" configs/training.yaml
# Expected: base_model: "models/pretrained/sac-FetchPickAndPlace-v4.zip"

# 3. No references to archive/
grep -rn "archive/" src/ scripts/ tests/ configs/ training/
# Must return nothing

# 4. archive/ directory is fully removed
test -d archive && echo "FAIL: archive still exists" || echo "OK: archive gone"

# 5. Training still launches correctly (dry run: just load config)
PYTHONPATH=. python -c "
from src.utils.config import load_config
cfg = load_config('training')
import os
assert os.path.exists(cfg['base_model']), f'Base model not found: {cfg[\"base_model\"]}'
print('OK:', cfg['base_model'])
"

# 6. Full test suite
python -m pytest tests/ -v
```
