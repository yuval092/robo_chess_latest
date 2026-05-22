# Note 2: Arm Speed Optimization

## Goal
Speed up the scripted arm without hurting grasp success rates. Current timing at 2ms/step:

| Phase | Current | Wall time |
|-------|---------|-----------|
| Transit (400 steps max, 8mm/step) | ~0.8s max | depends on distance |
| Ascend/Descend (200 steps max) | ~0.4s max | ~0.1–0.4s typical |
| Grasp close ramp | 150 steps | 0.30s |
| Grasp hold | 50 steps | 0.10s |
| Release ramp | 80 steps (hardcoded) | 0.16s |

The bottleneck is grasp/release (0.56s combined) followed by transit speed.

## Changes

### 2.1 Increase MAX_STEP_SIZE_M: 0.008 → 0.012 m/step
`src/chess_env/controller.py`: line 40
- Transit 30cm now takes ~25 steps instead of ~38 steps
- Risk: overshoot on fine approach. The final 4mm tolerance handles this.

### 2.2 Reduce grasp_close_steps: 150 → 80
`configs/env.yaml`: `grasp_close_steps: 80`
- Saves 0.14s per grasp
- Ramp rate doubles; cube has more mass now (note 4) so contact shock is absorbed

### 2.3 Reduce grasp_hold_steps: 50 → 20
`configs/env.yaml`: `grasp_hold_steps: 20`
- Saves 0.06s per grasp
- Physics settle is fast with higher damping (note 4)

### 2.4 Reduce release ramp: 80 → 40 steps (hardcoded in task.py)
`src/chess_env/task.py`: line 857 `ramp_delta = (ramp_end - ramp_start) / 80`
Change to 40 steps. Also update post-release hold.
- Saves 0.08s per release

### 2.5 Reduce TRANSIT_MAX_STEPS: 400 → 300 (safety; no perf impact on normal moves)
Longest move is board diagonal ~74cm. At 12mm/step = ~62 steps. 300 is generous.

## Validation
Run `python scripts/eval_sequence.py` — all moves should succeed.
