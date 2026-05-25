# Diagnostics

Scripts in this directory are debug tools, not production evaluation commands.

`eval_rl_stages_direct.py` loads specialist checkpoints through the training
wrappers and bypasses `ModelEmbeddedController`. Use it to distinguish checkpoint
failures from production integration failures.

For production integration testing, use:

```bash
python scripts/eval_stages.py --stages transit --n-episodes 30
```
