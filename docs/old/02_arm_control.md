# Arm Control

## Overview

The arm is driven by `ModelEmbeddedController`, a hybrid controller that runs SAC specialists for `transit`, `descend`, and `ascend`, while keeping scripted transitions, grasp, and place logic. It produces `StageResult` and `SequenceResult` objects and calls the same underlying MuJoCo environment primitives.

---

## Shared Data Structures

### `StageResult`

```python
@dataclass
class StageResult:
    success:      bool
    steps:        int
    crash_reason: str | None  # e.g., "TUBE_BREACH (drift=12.4mm)"
    final_pos:    np.ndarray  # gripper XYZ at end of stage
    error_mm:     float       # distance from target in mm
```

### `SequenceResult`

```python
@dataclass
class SequenceResult:
    success:       bool
    stage_results: list[tuple[str, StageResult]]  # [("transit", sr), ...]
    failed_at:     str | None   # stage name where failure occurred
    grasp_quality: dict | None  # set after grasp
```

---

## `ModelEmbeddedController` (`src/chess_env/model_controller.py`)

### Architecture

`ModelEmbeddedController` owns the three loaded SAC specialist models. Each movement stage requires its model to be loaded; grasp, place, and scenario transitions remain scripted.

```python
def _run_stage(self, stage: str, target_pos: np.ndarray) -> StageResult:
    model = self._models[stage]

    # Model inference loop
    env._use_transfer_obs = True
    for step in range(max_steps):
        grip_pos = get_site_xpos("robot0:grip")
        grip_vel = get_site_xvelp("robot0:grip")

        if is_near(grip_pos, target_pos) and ||grip_vel|| < stability_threshold:
            success = True; break

        obs = env._get_obs()   # 25-D transfer observation
        action, _ = model.predict(obs, deterministic=True)
        action[3] = -1.0 if stage in {"transit","ascend"} else 1.0

        env._set_action(action)
        if stage == "transit":
            env.data.mocap_quat[0][:] = env.VERTICAL_QUAT  # enforce only for transit
        env._mujoco_step(action)

        crash = _check_crash(env, stage, grip_pos)
        if crash: break
    env._use_transfer_obs = False
```

### Gripper Action Override

The SAC model's gripper dimension (action[3]) is overridden deterministically:
- `transit` and `ascend`: `-1.0` → closed (fingers held; piece is not dropped)
- `descend`: `+1.0` → open (fingers must be open to receive the piece)

This prevents the model from accidentally toggling the gripper mid-stage.

### VERTICAL_QUAT Enforcement

Only applied for `transit` because the transit model was trained with `VERTICAL_QUAT` enforced in every step. Descend and ascend inference does not clamp the mocap quaternion, matching those models' training conditions.

### Crash Checks

| Stage | Crash |
|---|---|
| transit | `FLOOR_HIT`, `CUBE_DROPPED_*` (if `grasp_mode=True`) |
| descend/ascend | `TUBE_BREACH (eval_drift_limit=10mm)`, `TABLE_HIT`, `CUBE_DROPPED_*` (ascend only, `grasp_mode`) |

### Per-Stage Finger Precondition Check

Before running a model stage, the finger joint is verified:
- Descend: must be open (0.0181 m ± 3 mm)
- Transit/Ascend: must be closed (0.000 m ± 3 mm)
- Exception: `grasp_mode=True` skips this check (fingers are actuator-driven and physically blocked by the piece)

### `load_all()`

Models are loaded through `ModelEmbeddedController.load_model()`, which uses the `transfer_obs_enabled` context manager to temporarily switch the observation space before calling `SAC.load(path, env=...)`. This ensures stable-baselines3 validates the loaded model against the correct observation space.

---

## Waypoints and Transitions (`src/chess_env/waypoints.py`)

### Z-Level Map

```python
SCENARIO_EXIT_Z  = {"transit": SAFE_Z, "descend": HOVER_Z, "ascend": SAFE_Z}
SCENARIO_ENTRY_Z = {"transit": SAFE_Z, "descend": SAFE_Z,  "ascend": HOVER_Z}
```

### Valid Transitions

```python
VALID_TRANSITIONS = {
    ("transit", "transit"): True,   # same-height transit
    ("transit", "descend"): True,   # horizontal then down
    ("descend", "ascend"):  True,   # down then up (grasp/place between)
    ("ascend",  "transit"): True,   # up then horizontal
    ("ascend",  "descend"): True,   # up then down (second board square)
}
```

Any other transition (e.g., descend→transit, descend→descend) raises `ValueError`.

### Named Chains

```python
CHAIN_SHORTCUTS = {
    "full_move": ["transit","descend","ascend","transit","descend","ascend"],
    "pick":      ["transit","descend","ascend"],
    "place":     ["transit","descend","ascend"],
    "vertical":  ["descend","ascend"],
}
```

The current CLI uses its own chain shortcuts in `src/chess_env/waypoints.py`; this module remains the source of waypoint Z constants and transition validation helpers.
