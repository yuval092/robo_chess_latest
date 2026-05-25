# Descend HOVER_Z Offset Root Cause Analysis

Date: 2026-05-24  
Descend model: `checkpoints/descend_20260523_222907/final_descend.zip`

---

## Executive Summary

The descend model consistently stops 10–13 mm above HOVER_Z (0.460 m) instead of at it. This was previously patched with a 15 mm Z tolerance in inference (`descend_success_z_tolerance: 0.015`). This document identifies the training-level root cause and proposes specific parameter changes to fix it without patching inference code.

---

## 1. Observed Behavior

### 1.1 Quantitative (from `diagnose_descend_afile.py`)

All six tested positions (center, a1, a2, a5, a8, h1) show the same pattern:

| Position | Z at stop | Above HOVER_Z | Steps |
|----------|-----------|---------------|-------|
| center (0.88, 0.264) | 0.4721 | 12.1 mm | 7 |
| a1 (0.600, -0.016) | 0.4712 | 11.2 mm | 7 |
| a2 (0.600, 0.064) | 0.4715 | 11.5 mm | 7 |
| a5 (0.600, 0.304) | 0.4708 | 10.8 mm | 8 |
| a8 (0.600, 0.544) | 0.4714 | 11.4 mm | 8 |
| h1 (1.160, -0.016) | 0.4712 | 11.2 mm | 7 |

The equilibrium is remarkably **position-independent** — almost all positions converge to Z ≈ 0.471–0.472 (11–12 mm above HOVER_Z=0.460).

### 1.2 Velocity profile

The descent is fast initially (97–103 mm/s downward), then decelerates sharply near HOVER_Z:

```
step 0: z=0.530, speed=13mm/s  (still at SAFE_Z, model starting)
step 1: z=0.517, speed=102mm/s (model outputs [-z] action)
step 2: z=0.504, speed=97mm/s  (continuing descent)
step 3: z=0.491, speed=96mm/s  
step 4: z=0.479, speed=87mm/s  (braking begins)
step 5: z=0.474, speed=34mm/s  (sharp deceleration)
step 6: z=0.472, speed=16mm/s  (< stability threshold, NEAR+STABLE)
```

The model brakes from ~90 mm/s to 16 mm/s in 2 steps. The final position (Z=0.472) is where physics settle at the threshold of stability.

### 1.3 Current production state

With `descend_success_z_tolerance: 0.015`, all positions pass (d_z ≤ 12.1 mm < 15 mm).  
The grasp pipeline accepts ±25 mm from HOVER_Z, so a 12 mm offset is within the grasp precondition.

---

## 2. Root Cause: Training Success Criterion vs. Physical Equilibrium

### 2.1 How training terminates success

In `task.py step()`:
```python
is_near = bool(self._is_success(grip_pos, self.goal_pos))
# _is_success: d_xy < SUCCESS_THRESHOLD AND d_z < SUCCESS_THRESHOLD
# SUCCESS_THRESHOLD = 0.010 (10 mm)

is_stable = speed < stability_vel_threshold  # 20 mm/s
success = 1.0 if (is_near and is_stable) else 0.0
```

For descend, `goal_pos = [xy, HOVER_Z]`. Success requires:
- `d_z < 10 mm` AND
- `speed < 20 mm/s`

### 2.2 Why the equilibrium is at 11–12 mm

The model learned that it gets maximum reward by approaching to within a few millimeters of the 10 mm success boundary and stopping there:

1. **The braking reward** (`braking_reward_weight * speed`) activates within `braking_dist = 10 mm`. This is the exact same distance as `success_threshold`. So the model only receives velocity penalty when it's ALREADY in the success zone.

2. **The policy output at ~12–13 mm above goal**: The model outputs small Z actions (≈−0.14 to −0.19), decelerating. At 10–12 mm, the physics settle to a stable hover at the arm's natural compliance point.

3. **The key insight**: At 11 mm, the model is OUTSIDE the braking zone (braking_dist=10 mm) but very close to success. The phase-9 obs (`rel_to_goal[2] = 11 mm`) tells the model "you're close." But the reward gradient at 11 mm is nearly flat — the distance reward (`−1.5 * d_z`) changes very slowly at this distance, so the model doesn't learn to push further.

4. **The FetchPickAndPlace pretrained weights** contribute: the original policy learned to "hover above the target" since placing an object requires stopping just above the surface, not exactly at a Z target.

### 2.3 Why d_z doesn't vary much across positions

The equilibrium Z offset (11–12 mm) is consistent across board positions because:
- The arm's compliance is position-independent at this scale
- The phase-9 obs's `rel_to_goal` vector is identical when at the same offset
- The braking behavior is learned from the Z-component of the observation, not absolute position

The small variance (10.8–12.1 mm) reflects minor differences in Z-axis arm dynamics at different XY positions.

---

## 3. Reward Structure Analysis

### 3.1 Current reward function (for descend, near goal)

At d_z = 12 mm (just outside braking zone):
```
reward = -1.0 * dist_3d    # dist_reward_weight=1.0, dist_3d≈12mm
       - 1.5 * 0.012       # z_reward_weight=1.5, z_err=12mm
       - 2.0 * 0.001       # xy_reward_weight=2.0, xy_err≈1mm (small drift)
       - 0.003 * action²   # jitter penalty
= ≈ -0.012 - 0.018 - 0.002 - small
= ≈ -0.032 per step
```

At d_z = 5 mm (inside success zone):
```
reward = -1.0 * 0.005      # ≈ -0.005
       - 1.5 * 0.005       # = -0.0075
       - braking_weight * speed    # 0.15 * ~20mm/s = -0.003
       - jitter
= ≈ -0.015 per step
+ 500 (on success)
```

The reward differential between d_z=12mm and d_z=5mm is only ~0.017 per step. Over 300 steps this is ~5 reward units — vs a 500-unit success bonus. The model should learn to push through. But the issue is that at 12 mm the model is **already below the braking zone** where there's no velocity penalty for holding still. The arms is in a local minimum: staying at 12mm avoids the braking penalty while staying near the distance reward.

### 3.2 The alignment trap

`braking_dist = 0.010` and `success_threshold = 0.010` are the **same value**. This means:
- The braking reward (penalizing speed) only activates INSIDE the success zone
- But the model only needs to reach the success zone to terminate
- So there's no reward signal for "approach the success zone more slowly"

This creates a policy that: approaches fast, overshoots slightly, and settles at whatever Z physics allow near-goal.

---

## 4. Proposed Training Parameter Changes

### 4.1 Fix 1: Decouple braking distance from success threshold

**Change**: `braking_dist: 0.010` → `braking_dist: 0.025`

**Why**: Braking reward should activate 25 mm from goal, well before the 10 mm success zone. This gives the model 2–3 steps to decelerate while still receiving a velocity penalty. Currently the model reaches 12 mm above HOVER_Z (outside the 10mm braking zone) and stops — with the wider braking zone it will receive a velocity penalty at 25 mm and learn to arrive slower.

**Expected impact**: The model will approach more slowly, arriving at HOVER_Z (or close) with low speed. This should reduce the equilibrium offset from 11–12 mm to 3–6 mm.

### 4.2 Fix 2: Tighter success threshold for descend

**Change**: `success_threshold: 0.010` → `success_threshold: 0.006` (for descend only, or globally)

**Why**: A 6 mm success radius forces the model to actually get within 6 mm. The current 10 mm threshold allows the model to succeed at 9.9 mm offset — but the physics settle at 11 mm (just outside), so the model never gets the success bonus unless it pushes harder. Tightening to 6 mm forces genuine convergence.

**Risk**: May reduce success rate temporarily. Use a training curriculum: start with 10 mm, anneal to 6 mm over 200K steps.

**Implementation**: Add `descend_success_threshold: 0.006` to `env.yaml` and use it in `task.py step()` for descend scenario.

### 4.3 Fix 3: Add a final-approach bonus

**Change**: Add a `final_approach_bonus` reward when Z within 5 mm of HOVER_Z:
```python
if self.current_scenario == "descend":
    hover_z = self.env_cfg.get("hover_z", 0.460)
    z_to_hover = abs(grip_pos[2] - hover_z)
    if z_to_hover < 0.005:
        reward += self.env_cfg.get("final_approach_bonus", 5.0)
```

**Why**: Creates a strong reward gradient in the final 5 mm that pulls the model toward exact HOVER_Z. The current reward is flat in this region.

**Expected impact**: Model learns to push through 10–12 mm zone toward 5 mm zone.

### 4.4 Fix 4: Increase braking reward weight

**Change**: `braking_reward_weight: 0.15` → `braking_reward_weight: 0.30`

**Why**: With the wider braking zone (Fix 1), the velocity penalty needs to be strong enough to actually force deceleration. Doubling the weight makes slow arrival more profitable than fast arrival.

### 4.5 Summary of changes

| Parameter | Current | Proposed | Location |
|-----------|---------|----------|----------|
| `braking_dist` | 0.010 | 0.025 | `configs/env.yaml` |
| `success_threshold` | 0.010 | 0.010 (no change) | — |
| `descend_success_threshold` | (new) | 0.006 | `configs/env.yaml` + `task.py` |
| `braking_reward_weight` | 0.15 | 0.30 | `configs/env.yaml` |
| `final_approach_bonus` | (new) | 5.0 | `configs/env.yaml` + `task.py` |

**Implementation in `task.py step()`**:
```python
# Replace current braking logic:
if dist < self.env_cfg.get("braking_dist", 0.010):
    reward -= self.env_cfg.get("braking_reward_weight", 0.15) * speed

# With:
braking_d = self.env_cfg.get("braking_dist", 0.010)
if dist < braking_d:
    reward -= self.env_cfg.get("braking_reward_weight", 0.15) * speed

# Add for descend only:
if self.current_scenario == "descend":
    hover_z = self.env_cfg.get("hover_z", 0.460)
    z_to_hover = abs(grip_pos[2] - hover_z)
    if z_to_hover < 0.005:
        reward += self.env_cfg.get("final_approach_bonus", 5.0)
    # Tighter Z success criterion for descend
    descend_thresh = self.env_cfg.get("descend_success_threshold", 
                                       self.SUCCESS_THRESHOLD)
    is_near = d_xy < self.SUCCESS_THRESHOLD and d_z < descend_thresh
```

---

## 5. Why NOT to Retrain From Scratch

The current descend model is functionally correct with the 15 mm Z tolerance patch. The 12 mm offset is safe for the grasp pipeline (precondition ±25 mm). Retraining requires:
- 600K training steps
- ~2–3 hours compute time
- Risk of introducing new failure modes

**Recommendation**: Keep the inference patch for now. Apply the training parameter changes when retraining transit (which needs retaining anyway — see `07_transit_timeout_analysis.md`). Run a combined retrain with the improved reward function.

---

## 6. Existing Patch (Do Not Remove)

The current production patch in `src/chess_env/model_controller.py` (`_run_stage`):
```python
if stage == "descend":
    z_tol = env.env_cfg.get("descend_success_z_tolerance", 0.015)
    is_near = d_xy < env.SUCCESS_THRESHOLD and d_z < z_tol
```
And in `configs/env.yaml`:
```yaml
descend_success_z_tolerance: 0.015
```

This patch is working correctly (100% descend success rate) and should remain until a retrained model with proper equilibrium is deployed.
