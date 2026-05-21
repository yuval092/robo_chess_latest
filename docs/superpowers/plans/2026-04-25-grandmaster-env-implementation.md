# Grandmaster Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shrink the board to 56cm, implement the 25-dim masked observation space, and verify all utilities.

**Architecture:** Updates XML geometry, modifies `ChessFetchEnv` for boundaries, and re-implements `_build_phase9_observation` in `ChessFetchDenseEnv`.

**Tech Stack:** MuJoCo, Python, Gymnasium Robotics.

---

### Task 1: Geometry and Physics Overhaul

**Files:**
- Modify: `chess_env/assets/pick_and_place.xml`
- Modify: `chess_env/chess_fetch_env.py`

- [ ] **Step 1: Shrink Table in XML**
Change `table0` geom size from `0.32` to `0.28`.

```xml
		<body pos="0.88 0.2641 0.2" name="table0">
			<geom size="0.28 0.28 0.2" type="box" mass="2000" material="table_mat"></geom>
		</body>
```

- [ ] **Step 2: Update Class Boundaries**
Update `TABLE_HALF_X/Y` to `0.28` and ensure Slide X is `0.00`.

```python
    TABLE_HALF_X    = 0.28
    TABLE_HALF_Y    = 0.28
    # ... inside __init__ ...
    self.initial_qpos[0] = 0.00
```

- [ ] **Step 3: Verify XML Scene**
Run: `PYTHONPATH=. .venv/bin/python3 scripts/v8_latest/verify_scene.py`
Expected: "PASS: Table position correct", "PASS: XML Compiled Successfully".

### Task 2: Implement Masked Observation Space

**Files:**
- Modify: `chess_env/chess_fetch_dense_env.py`

- [ ] **Step 1: Implement `_build_phase9_observation`**
Apply the specific masking requested: indices 9-10 (state) and 23-24 (robot vel) to zero.

```python
    def _build_phase9_observation(self):
        (
            grip_pos,
            object_pos,
            object_rel_pos,
            gripper_state,
            object_rot,
            object_velp,
            object_velr,
            grip_velp,
            gripper_vel,
        ) = self.generate_mujoco_observations()

        scenario_map = {"transit": [1.0, 0.0, 0.0], "descend": [0.0, 1.0, 0.0], "ascend": [0.0, 0.0, 1.0]}
        scenario_id_vec = scenario_map.get(self.current_scenario, [0.0, 0.0, 0.0])
        
        target_pos_for_model = grip_pos.copy()
        relative_dist_for_model = np.zeros(3)

        obs = np.concatenate(
            [
                grip_pos,                 # 0-2
                target_pos_for_model,     # 3-5
                relative_dist_for_model,  # 6-8
                np.zeros(2),              # 9-10 (MASKED STATE)
                scenario_id_vec,          # 11-13
                np.zeros(3),              # 14-16 (MASKED OBJ VELP)
                np.zeros(3),              # 17-19 (MASKED OBJ VELR)
                grip_velp,                # 20-22
                np.zeros(2),              # 23-24 (MASKED ROBOT VEL)
            ]
        )

        return {
            "observation": obs.copy(),
            "achieved_goal": grip_pos.copy(),
            "desired_goal": self.goal.copy(),
        }
```

- [ ] **Step 2: Update Precise Heights**
Set `GRASP_Z = 0.425` and ensure `SAFE_Z = 0.550`. Ensure `SUCCESS_THRESHOLD = 0.015`.

- [ ] **Step 3: Verify Observation Shape**
Run: `PYTHONPATH=. .venv/bin/python3 -c "import gymnasium as gym; import chess_env; env=gym.make('ChessFetchDense-v0'); obs=env.reset()[0]['observation']; print(f'Shape: {obs.shape}'); assert np.all(obs[9:11] == 0); print('MASKING PASS')"`
Expected: "Shape: (25,)", "MASKING PASS".

### Task 3: Coordinate Mapping and Utilities

**Files:**
- Modify: `scripts/coordinate_mapping.py`
- Modify: `scripts/v8_latest/evaluate_production.py`

- [ ] **Step 1: Update Grid Mapping**
Update the grid center calculations for 7cm cells.

```python
    # Inside coordinate_mapping.py center calculation
    # Grid width = 0.56, cell = 0.07
    # Center = TABLE_CENTER - 0.28 + 0.035 + (index * 0.07)
```

- [ ] **Step 2: Update Evaluate Production**
Ensure the evaluation uses the tightest drift limit (3.5cm) which now matches the cell boundaries.

### Task 4: Final Validation

- [ ] **Step 1: Run Final Reachability Test**
Run: `PYTHONPATH=. .venv/bin/python3 scripts/verify_board_reach.py`
Expected: 64/64 squares reachable in vertical pose.

- [ ] **Step 2: Commit All Changes**
```bash
git add .
git commit -m "feat: implement grandmaster 56cm board with masked observations"
```
