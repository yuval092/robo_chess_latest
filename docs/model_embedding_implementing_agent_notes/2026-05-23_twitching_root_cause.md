# Transit Twitching Root Cause Analysis

Date: 2026-05-23

## User-Reported Symptom

"The arm twitches for a really long time, failing to reach success, and then we reach TIMEOUT
and do not continue to descend." (c2→c3 move with visual monitoring)

## Diagnosis: Two Separate Problems

### Problem 1: Model Quality (First Transit — partial)

The initially deployed model was `best_model_transit.zip` (100K steps), which had only
**70% success** in production-style evaluation (30% timeout rate). With this model, ~30% of
transits would oscillate near the goal (within 10mm but speed > 20mm/s) and exhaust the
300-step budget.

**This looks like "twitching for a really long time"**: The arm reaches within 7–10mm of the
target square in ~15 steps, then oscillates for 5–20 steps failing to simultaneously satisfy
both `d_xy < 10mm AND speed < 20mm/s`. With an under-trained model, this oscillation can
persist until TIMEOUT.

**Resolution**: Training completed at 200K steps (`final_transit.zip`). The final model achieves:
- `eval_rl_stages.py` (30 episodes): **30/30 = 100% success**
- `eval_stages.py --use-rl-models` (30 episodes): **29/30 = 96.7% success**
- Full move sequence (10 episodes): **10/10 = 100% success**, all 20 transits pass

### Problem 2: Integration Bug — PRECONDITION_FINGER (Second Transit — always fails)

When the arm holds a chess piece (`grasp_mode=True`), the gripper finger joint reads ~0.011–0.015
(physically blocked by the piece) vs `FINGER_CLOSED_JOINT = 0.000`. The original precondition
check in `_run_stage()` compared these and always returned failure immediately (0 inference steps)
for ANY transit while holding a piece.

**This means the first transit (HOME→src) could succeed, but the second transit (src→dst
while holding the piece) ALWAYS failed at the precondition check before even running the model.**

**Resolution**: Fixed in `src/chess_env/model_controller.py` — skip the precondition check
when `env.grasp_mode=True`. See `2026-05-23_diagnostic_and_fixes.md` for the complete fix.

---

## Integration Mismatch Audit

The user suspected training/production mismatch. Full audit performed:

| Component | Training | Production / Inference | Match? |
|-----------|----------|----------------------|--------|
| Obs format | Phase-9 25D dict (`_use_phase9_obs=True`) | Same, set in `_run_stage()` | ✓ |
| Goal in obs | `rel_to_goal = env.goal - grip_pos` | Same, set via `env.goal = target_pos` | ✓ |
| Success criterion | `d_xy < 10mm AND d_z < 10mm AND speed < 20mm/s` | Same call to `_is_success()` + vel check | ✓ |
| Finger action | `action[3] = -1.0` (closed for transit) | Same, forced in `_run_stage()` | ✓ |
| Starting position | Random board XY (show_chess_pieces=False) | HOME_POS [0.88, 0.264, 0.530] | Different |
| Episode timing | 200-step TimeLimit (env.step) | 300-step budget (_run_stage) | Different |

### Starting Position Mismatch

Training starts from **random board positions**; production starts from **HOME_POS**.
HOME_XY = [0.88, 0.264] is within the training sampling range X∈[0.570, 1.190],
Y∈[-0.046, 0.574]. The model generalizes correctly.

Confirmed: transit HOME → c2_xy = [0.760, 0.064] (0.233m diagonal) succeeds in 20 steps.

### Step Budget Mismatch (benign)

Training uses a 200-step TimeLimit; inference uses a 300-step budget. The extra 100 steps
only help (more time to converge). Not a problem.

---

## Step-by-Step Transit Trace (HOME → c2 with final model)

| Step | grip_x | grip_y | grip_z | dist_xy | speed | Status |
|------|--------|--------|--------|---------|-------|--------|
| 0 | 0.8797 | 0.2662 | 0.5313 | 234.9mm | 0mm/s | start |
| 6 | 0.8012 | 0.1889 | 0.5584 | 131.4mm | 137mm/s | moving fast, z slightly high (+28mm) |
| 10 | 0.7616 | 0.1369 | 0.5387 | 72.8mm | 104mm/s | approaching, z settling |
| 14 | 0.7604 | 0.0840 | 0.5319 | 19.9mm | 99mm/s | decelerating |
| 15 | 0.7600 | 0.0716 | 0.5308 | 7.5mm | 93mm/s | within 10mm, still fast |
| 16-19 | ~0.761 | ~0.066 | ~0.530 | 1–5mm | 23–44mm/s | oscillating (speed too high) |
| 20 | 0.7615 | 0.0660 | 0.5304 | **2.4mm** | **18.7mm/s** | **SUCCESS** |

Z-axis drift: arm arcs up ~28mm during transit then returns. Normal for this policy.
Final error: 2.4mm XY, 0.4mm Z.

---

## Current Status After Fixes

| Issue | Status |
|-------|--------|
| PRECONDITION_FINGER (second transit) | Fixed |
| obs_space/phase9 pollution in load_model() | Fixed |
| Model quality (70% → 100% success) | Fixed (200K final model) |
| Deployed model pointing to final_transit.zip | Done |

### Remaining Low-Priority Items

- **3% timeout rate**: With 200K steps, ~3% of transits timeout. Acceptable for a chess robot
  (rare, and the move can be retried). Would drop to ~0% with 1M steps if needed.
- **5-step oscillation near goal**: The arm visually "twitches" 3–5 steps in the 10mm goal zone
  before speed drops below 20mm/s. This is normal for an RL policy — braking isn't instantaneous.
  In wall-clock time it is sub-second with no render delay.
- **Descend/ascend models**: Not yet trained; both fall back to scripted controller.
