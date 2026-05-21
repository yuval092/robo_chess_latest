# Curriculum & Learnability Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the "2mm Trap" by aligning footprint safety checks with the training curriculum and updating precision targets.

**Architecture:** We update the `env.yaml` config to use a 5mm final target, refactor the `TUBE_BREACH` logic in `task.py` to be curriculum-relative, and increase scripted transition steps for better stability.

**Tech Stack:** MuJoCo, Stable Baselines 3.

---

### Task 1: Configuration Updates

**Files:**
- Modify: `configs/env.yaml`

- [ ] **Step 1: Update drift limits in `env.yaml`**

```yaml
# configs/env.yaml updates
drift_limit_end: 0.005    # Tighten from 0.015 to 5mm center precision
eval_drift_limit: 0.005   # Synchronize evaluation with production target
```

- [ ] **Step 2: Commit configuration changes**

```bash
git add configs/env.yaml
git commit -m "config: update final drift limit to 5mm and sync eval limit"
```

---

### Task 2: Logic Refactoring (Fixing the 2mm Trap)

**Files:**
- Modify: `src/chess_env/task.py`

- [ ] **Step 1: Refactor `step()` safety check to be curriculum-relative**

```python
        # Modify TUBE_BREACH logic in step():
        elif self.current_scenario in {"descend", "ascend"}:
            drift = np.linalg.norm(gripper_pos[:2] - self.tube_center_xy)
            # Remove hard 0.035 check. The curriculum current_drift_limit 
            # now governs the center, effectively allowing exploration.
            if drift > current_drift_limit:
                crashed = True
                crash_reason = f"TUBE_BREACH (center={drift:.4f} > limit={current_drift_limit:.4f})"
```

- [ ] **Step 2: Increase scripted transition steps in `_reset_sim`**

```python
        # In _reset_sim:
        if self.current_scenario == "descend":
            # ...
            for _ in range(50): # Increased from 20
                self._set_action(dummy_action)
                self._mujoco_step(None)
        elif self.current_scenario == "ascend":
            # ... settle in open state
            for _ in range(50): # Increased from 20
                self._set_action(dummy_action)
                self._mujoco_step(None)
```

- [ ] **Step 3: Commit logic changes**

```bash
git add src/chess_env/task.py
git commit -m "feat: align footprint safety with curriculum and increase reset stability"
```

---

### Task 4: Training Restart & Verification

**Files:**
- None

- [ ] **Step 1: Stop current training and clear logs**

```bash
ps aux | grep train.py | grep -v grep | awk '{print $2}' | xargs kill -9 || true
rm -rf logs/env_debug/*
rm training_production.log
```

- [ ] **Step 2: Launch new production training**

```bash
PYTHONPATH=. python scripts/train.py --fresh > training_production.log 2>&1 &
```

- [ ] **Step 3: Verify stability after 120 seconds**
Read `training_production.log` and verify success rate and lack of instant crashes.
