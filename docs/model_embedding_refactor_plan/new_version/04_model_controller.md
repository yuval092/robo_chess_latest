# `ModelEmbeddedController` — Inference & Integration

## Overview

`ModelEmbeddedController` is a drop-in replacement for `ScriptedController`. It must expose the
**full interface** of `ScriptedController` because callers rely on the high-level methods:

- `movement_executor.py:53` calls `self.controller.run_full_move(src_xy, dst_xy)`
- `plan_executor.py:75` calls `self.controller.run_transit(home_xy)` (home return)

The RL models only replace the three movement stages (`run_transit`, `run_descend`, `run_ascend`).
Grasp, place, scenario transitions, and the high-level sequence methods (`run_full_move`,
`run_pick_sequence`, `run_place_sequence`) delegate to the internal `scripted_controller`
for scripted phases and call `self.run_transit / run_descend / run_ascend` (i.e. the RL version)
for movement phases — exactly like `ScriptedController`'s own chain does.

> **Why not inherit?** Inheriting from `ScriptedController` and overriding only the three movement
> methods would be simpler, but creates coupling to the parent's internal `_run_movement_loop`.
> The standalone approach is cleaner: `run_full_move` / `run_pick_sequence` / `run_place_sequence`
> are just thin wrappers that call `self.run_transit`, `self.run_descend`, `self.run_ascend`,
> `self.run_grasp`, `self.run_place`, and `self.transition` — all of which are defined here.

Grasp, place, and scenario transitions are **always scripted** — the RL models only handle movement.

---

## 1. `StageResult` Dataclass (shared with `ScriptedController`)

This already exists in `src/chess_env/controller.py`. We reuse it unchanged:

```python
@dataclass
class StageResult:
    success: bool
    steps: int
    crash_reason: str | None
    final_pos: np.ndarray
    error_mm: float
```

---

## 2. `src/chess_env/model_controller.py` — Full Implementation

```python
"""
ModelEmbeddedController
=======================
Drop-in replacement for ScriptedController during movement stages.

Uses three specialist SAC models for Transit, Descend, and Ascend,
falling back to the scripted controller if a model is not loaded.

Key invariant: the RL model never drives the arm during grasp/place/transition.
Those phases are always handled by the scripted pipeline.
"""
import time
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


class ModelEmbeddedController:
    """
    Wraps three specialist SAC models and provides the same interface
    as ScriptedController for the three movement stages.
    
    Args:
        env:        The (unwrapped) ChessTaskEnv instance.
        render_fn:  Optional callable called each step for visualization.
        render_delay: Seconds to sleep per step (for slow-motion viewing).
    """

    # Maximum steps per RL episode (matches max_episode_steps in registration)
    MAX_STEPS = 200

    def __init__(self, env, render_fn: Optional[Callable] = None, render_delay: float = 0.0):
        # env may be the wrapped or unwrapped env. Unwrap to the innermost ChessTaskEnv.
        self._wrapped_env = env  # Keep the original (may have TimeLimit) for step-counter reset
        inner = env
        while hasattr(inner, "env"):
            inner = inner.env
        self._env = inner  # Always the raw ChessTaskEnv
        self._render_fn = render_fn
        self._render_delay = render_delay

        # Model slots — load via load_model() or load_all()
        self._models = {
            "transit": None,
            "descend": None,
            "ascend":  None,
        }
        
        # True fallback script controller
        from src.chess_env.controller import ScriptedController
        drift_limit = env.env_cfg.get("eval_drift_limit", 0.010) if hasattr(env, "env_cfg") else 0.010
        self.scripted_controller = ScriptedController(
            env, drift_limit=drift_limit, render_fn=render_fn, render_delay=render_delay
        )

    # ── Model Loading ──────────────────────────────────────────────────────────

    def load_model(self, stage: str, path: str) -> None:
        """Load a specialist model for one stage."""
        from stable_baselines3 import SAC
        assert stage in self._models, f"Unknown stage: {stage}"
        self._models[stage] = SAC.load(path)
        print(f"[ModelEmbeddedController] Loaded {stage} model from {path}")

    def load_all(self, transit_path: str, descend_path: str, ascend_path: str) -> None:
        """Convenience method to load all three models at once."""
        self.load_model("transit", transit_path)
        self.load_model("descend", descend_path)
        self.load_model("ascend",  ascend_path)

    # ── Public Movement Interface ─────────────────────────────────────────────

    def run_transit(self, target_xy: np.ndarray):
        """Move arm horizontally to target_xy at SAFE_Z."""
        from src.chess_env.waypoints import SAFE_Z
        target_pos = np.array([target_xy[0], target_xy[1], SAFE_Z])
        return self._run_stage("transit", target_pos)

    def run_descend(self, tube_xy: np.ndarray):
        """Lower arm from SAFE_Z to HOVER_Z, staying within tube_xy's XY."""
        from src.chess_env.waypoints import HOVER_Z
        target_pos = np.array([tube_xy[0], tube_xy[1], HOVER_Z])
        return self._run_stage("descend", target_pos)

    def run_ascend(self, tube_xy: np.ndarray):
        """Raise arm from HOVER_Z to SAFE_Z, staying within tube_xy's XY."""
        from src.chess_env.waypoints import SAFE_Z
        target_pos = np.array([tube_xy[0], tube_xy[1], SAFE_Z])
        return self._run_stage("ascend", target_pos)

    # ── Passthrough to Scripted (grasp, place, transition) ────────────────────

    def transition(self, new_scenario: str, new_goal_pos, nominal_exit_pos, nominal_xy=None):
        """Delegate to the env's scripted soft_reset logic. Returns the info dict."""
        # soft_reset returns (obs, info); callers only use info
        _, info = self._env.soft_reset(
            new_scenario=new_scenario,
            new_goal_pos=new_goal_pos,
            nominal_exit_pos=nominal_exit_pos,
            nominal_xy=nominal_xy,
        )
        # Also reset TimeLimit counter so stages don't share the same episode budget
        curr = self._wrapped_env
        while hasattr(curr, "env"):
            if hasattr(curr, "_elapsed_steps"):
                curr._elapsed_steps = 0
                break
            curr = curr.env
        else:
            # curr is now the innermost env — check it too (matches ScriptedController)
            if hasattr(curr, "_elapsed_steps"):
                curr._elapsed_steps = 0
        return info

    def execute_grasp(self):
        """Delegate to the scripted grasp pipeline."""
        return self._env.execute_grasp()

    def execute_place(self, dst_xy):
        """Delegate to the scripted place pipeline."""
        return self._env.execute_place(dst_xy)

    # ── High-level sequence runners (REQUIRED by movement_executor.py) ────────
    # These mirror ScriptedController exactly, calling self.run_transit/descend/ascend
    # (the RL-augmented versions) and the scripted grasp/place/transition.

    def run_grasp(self) -> "StageResult":
        """Execute the scripted grasp pipeline. Returns StageResult."""
        from src.chess_env.controller import StageResult
        env = self._env
        grip_before = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
        result_dict = env.execute_grasp()
        success = result_dict.get("success", False)
        grip_after = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
        return StageResult(
            success=success,
            steps=result_dict.get("total_steps_used", result_dict.get("close_steps_used", 0)),
            crash_reason=None if success else result_dict.get("reason", "GRASP_FAILED"),
            final_pos=grip_after.copy(),
            error_mm=float(np.linalg.norm(grip_after - grip_before)) * 1000,
        )

    def run_place(self, dst_xy: np.ndarray) -> "StageResult":
        """Execute the scripted place pipeline. Returns StageResult."""
        from src.chess_env.controller import StageResult
        env = self._env
        grip_before = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
        result_dict = env.execute_place(dst_xy)
        success = result_dict.get("success", False)
        grip_after = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
        return StageResult(
            success=success,
            steps=result_dict.get("total_steps_used", 0),
            crash_reason=None if success else result_dict.get("reason", "PLACE_FAILED"),
            final_pos=grip_after.copy(),
            error_mm=0.0,
        )

    def run_pick_sequence(self, src_xy: np.ndarray) -> "SequenceResult":
        """Execute: transit → descend → grasp → ascend at src_xy."""
        from src.chess_env.controller import SequenceResult
        from src.chess_env.waypoints import SAFE_Z, HOVER_Z
        results = []

        result = self.run_transit(src_xy)
        results.append(("transit", result))
        if not result.success:
            return SequenceResult(False, results, "transit", None)

        nom_exit = np.array([src_xy[0], src_xy[1], SAFE_Z])
        self.transition("descend", np.array([src_xy[0], src_xy[1], HOVER_Z]), nom_exit, src_xy)

        result = self.run_descend(src_xy)
        results.append(("descend", result))
        if not result.success:
            return SequenceResult(False, results, "descend", None)

        result = self.run_grasp()
        results.append(("grasp", result))
        if not result.success:
            return SequenceResult(False, results, "grasp", None)

        nom_exit_hover = np.array([src_xy[0], src_xy[1], HOVER_Z])
        self.transition("ascend", np.array([src_xy[0], src_xy[1], SAFE_Z]), nom_exit_hover, src_xy)

        result = self.run_ascend(src_xy)
        results.append(("ascend", result))
        if not result.success:
            return SequenceResult(False, results, "ascend", None)

        return SequenceResult(True, results, None, None)

    def run_place_sequence(self, dst_xy: np.ndarray) -> "SequenceResult":
        """Execute: transit → descend → place → ascend at dst_xy."""
        from src.chess_env.controller import SequenceResult
        from src.chess_env.waypoints import SAFE_Z, HOVER_Z
        results = []

        result = self.run_transit(dst_xy)
        results.append(("transit", result))
        if not result.success:
            return SequenceResult(False, results, "transit", None)

        nom_exit = np.array([dst_xy[0], dst_xy[1], SAFE_Z])
        self.transition("descend", np.array([dst_xy[0], dst_xy[1], HOVER_Z]), nom_exit, dst_xy)

        result = self.run_descend(dst_xy)
        results.append(("descend", result))
        if not result.success:
            return SequenceResult(False, results, "descend", None)

        result = self.run_place(dst_xy)
        results.append(("place", result))
        if not result.success:
            return SequenceResult(False, results, "place", None)

        nom_exit_hover = np.array([dst_xy[0], dst_xy[1], HOVER_Z])
        self.transition("ascend", np.array([dst_xy[0], dst_xy[1], SAFE_Z]), nom_exit_hover, dst_xy)

        result = self.run_ascend(dst_xy)
        results.append(("ascend", result))
        if not result.success:
            return SequenceResult(False, results, "ascend", None)

        return SequenceResult(True, results, None, None)

    def run_full_move(self, src_xy: np.ndarray, dst_xy: np.ndarray) -> "SequenceResult":
        """Full pick-and-place: pick from src_xy, place at dst_xy. Called by MovementExecutor."""
        from src.chess_env.controller import SequenceResult
        pick = self.run_pick_sequence(src_xy)
        if not pick.success:
            return pick
        place = self.run_place_sequence(dst_xy)
        return SequenceResult(
            success=place.success,
            stage_results=pick.stage_results + place.stage_results,
            failed_at=place.failed_at,
            grasp_quality=place.grasp_quality,
        )

    # ── Internal Stage Runner ─────────────────────────────────────────────────

    def _run_stage(self, stage: str, target_pos: np.ndarray):
        """
        Core inference loop. Sets the environment goal, then steps the SAC model
        until success, crash, or timeout.
        
        The observation is built via env._get_obs() after temporarily enabling
        Phase-9 mode. This ensures the model gets exactly the observation it was
        trained on.
        """
        env = self._env
        model = self._models[stage]

        if model is None:
            # Fallback to scripted controller logic
            target_xy = target_pos[:2]
            if stage == "transit":
                return self.scripted_controller.run_transit(target_xy)
            elif stage == "descend":
                return self.scripted_controller.run_descend(target_xy)
            elif stage == "ascend":
                return self.scripted_controller.run_ascend(target_xy)

        from src.chess_env.controller import StageResult

        # Set the environment's goal for this stage (required by _build_phase9_observation)
        env.goal_pos = target_pos.copy()
        env.goal = target_pos.copy()
        env.current_scenario = stage

        # tube_center_xy MUST be set before _check_crash for descend/ascend;
        # transit leaves it None (uses FLOOR_LIMIT instead).
        if stage in {"descend", "ascend"}:
            env.tube_center_xy = target_pos[:2].copy()

        # Verify finger state is correct before handing off to RL
        l_finger = env._utils.get_joint_qpos(env.model, env.data, "robot0:l_gripper_finger_joint").item()
        expected_finger = env.FINGER_OPEN_JOINT if stage == "descend" else env.FINGER_CLOSED_JOINT
        if abs(l_finger - expected_finger) > 0.003:
            return StageResult(
                success=False, steps=0,
                crash_reason=f"PRECONDITION_FINGER (actual={l_finger:.4f}, expected={expected_finger:.4f})",
                final_pos=env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy(),
                error_mm=0.0,
            )

        # Enable Phase-9 observation for inference — restored in finally block
        env._use_phase9_obs = True

        crash_reason = None
        success = False
        step_idx = 0

        try:
            for step_idx in range(self.MAX_STEPS):
                grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
                dist = float(np.linalg.norm(target_pos - grip_pos))
                grip_vel = env._utils.get_site_xvelp(env.model, env.data, "robot0:grip").copy()
                speed = float(np.linalg.norm(grip_vel))

                # Success: close enough AND arm is stable (not just passing through)
                if dist < env.SUCCESS_THRESHOLD and speed < env.env_cfg.get("stability_vel_threshold", 0.05):
                    success = True
                    break

                # Get Phase-9 observation and predict action
                obs = env._get_obs()
                action, _ = model.predict(obs, deterministic=True)
                action = np.array(action, dtype=np.float32)
                action[3] = 0.0  # Suppress gripper — finger_target_joint controls it

                # Apply action
                env._set_action(action)
                env._mujoco_step(action)

                if self._render_fn is not None:
                    self._render_fn()
                    if self._render_delay > 0:
                        time.sleep(self._render_delay)

                # Crash check (simplified; full checks happen in step() during training)
                grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
                crash_reason = self._check_crash(env, stage, grip_pos)
                if crash_reason:
                    break
        finally:
            # Always restore — even if an exception is raised mid-inference
            env._use_phase9_obs = False

        grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
        dist = float(np.linalg.norm(target_pos - grip_pos))

        return StageResult(
            success=success,
            steps=step_idx + 1,
            crash_reason=crash_reason if not success else None,
            final_pos=grip_pos,
            error_mm=dist * 1000.0,
        )

    def _check_crash(self, env, stage: str, grip_pos: np.ndarray) -> Optional[str]:
        """Lightweight in-loop crash check for the inference path."""
        if stage == "transit":
            if grip_pos[2] < env.FLOOR_LIMIT:
                return f"FLOOR_HIT (z={grip_pos[2]:.4f})"

        elif stage in {"descend", "ascend"}:
            if env.tube_center_xy is not None:
                drift = float(np.linalg.norm(grip_pos[:2] - env.tube_center_xy))
                drift_limit = env.env_cfg.get("eval_drift_limit", 0.010)
                if drift > drift_limit:
                    return f"TUBE_BREACH (drift={drift*1000:.1f}mm)"
            if grip_pos[2] < env.TABLE_SURFACE_Z:
                return f"TABLE_HIT (z={grip_pos[2]:.4f})"

        return None
```

---

## 3. Loading Models in Practice

```python
from src.chess_env.model_controller import ModelEmbeddedController

# Create the env (same as before)
env = gym.make("ChessFetchTask-v0")
inner = env.unwrapped

# Create controller
ctrl = ModelEmbeddedController(inner, render_fn=env.render if visualize else None)

# Load all three specialist models
ctrl.load_all(
    transit_path="checkpoints/transit_20260522_143012/best_model_transit.zip",
    descend_path="checkpoints/descend_20260522_160503/best_model_descend.zip",
    ascend_path="checkpoints/ascend_20260522_174201/best_model_ascend.zip",
)

# Use exactly like ScriptedController:
obs, _ = env.reset()
result = ctrl.run_transit(target_xy=np.array([1.05, 0.35]))
```

---

## 4. Why `_use_phase9_obs` Toggle?

The same `ChessTaskEnv` instance is used for both inference (by `ModelEmbeddedController`) and
the full scripted game pipeline. During inference, the model expects the 25D Phase-9 dict.
During the game, `GameOrchestrator` and debug scripts expect the richer dict with
`grip_pos`, `grip_vel`, `goal_pos`, etc.

The flag toggle (`env._use_phase9_obs = True` before, `= False` after) ensures:
- No permanent state change to the env
- No dual observation spaces defined simultaneously
- Both paths work correctly from the same env instance

---

## 5. Fingerprint Validation Before Inference

The finger state check is already integrated into `_run_stage()` above (before the inference
loop). It verifies the arm is in the correct state before handing control to the RL model:
- **descend**: expects `FINGER_OPEN_JOINT` (arm must be open to descend and grasp)
- **transit / ascend**: expects `FINGER_CLOSED_JOINT`

If the precondition fails, `_run_stage()` returns an immediate `StageResult` with
`crash_reason="PRECONDITION_FINGER"` and `success=False`. This surfaces upstream so the
caller can log and abort rather than running an RL episode in a broken state.
