"""
ModelEmbeddedController
=======================
Drop-in replacement for ScriptedController during movement stages.

The three movement stages may be served by specialist SAC models. Grasp,
place, transitions, and high-level sequencing remain scripted.
"""
import time
from typing import Callable, Optional

import numpy as np


class ModelEmbeddedController:
    """Wraps specialist SAC models behind the ScriptedController interface."""

    MAX_STEPS = 200

    def __init__(
        self,
        env,
        render_fn: Optional[Callable] = None,
        render_delay: float = 0.0,
    ):
        self._wrapped_env = env
        inner = env
        while hasattr(inner, "env"):
            inner = inner.env
        self._env = inner
        self._render_fn = render_fn
        self._render_delay = render_delay

        self._models = {
            "transit": None,
            "descend": None,
            "ascend": None,
        }

        from src.chess_env.controller import ScriptedController

        drift_limit = self._env.env_cfg.get("eval_drift_limit", 0.010)
        self.scripted_controller = ScriptedController(
            env,
            drift_limit=drift_limit,
            render_fn=render_fn,
            render_delay=render_delay,
        )

    def load_model(self, stage: str, path: str) -> None:
        """Load a specialist model for one stage."""
        if stage not in self._models:
            raise ValueError(f"Unknown stage: {stage}")
        if not path:
            raise ValueError(f"No model path provided for {stage}")

        from stable_baselines3 import SAC

        self._models[stage] = SAC.load(path)
        print(f"[ModelEmbeddedController] Loaded {stage} model from {path}")

    def load_all(self, transit_path: str, descend_path: str, ascend_path: str) -> None:
        """Load all three specialist models."""
        self.load_model("transit", transit_path)
        self.load_model("descend", descend_path)
        self.load_model("ascend", ascend_path)

    def run_transit(self, target_xy: np.ndarray):
        """Move arm horizontally to target_xy at SAFE_Z."""
        target_pos = np.array([target_xy[0], target_xy[1], self._env.SAFE_Z])
        return self._run_stage("transit", target_pos)

    def run_descend(self, tube_xy: np.ndarray):
        """Lower arm from SAFE_Z to HOVER_Z, staying inside the target tube."""
        target_pos = np.array([tube_xy[0], tube_xy[1], self._env.HOVER_Z])
        return self._run_stage("descend", target_pos)

    def run_ascend(self, tube_xy: np.ndarray):
        """Raise arm from HOVER_Z to SAFE_Z, staying inside the target tube."""
        target_pos = np.array([tube_xy[0], tube_xy[1], self._env.SAFE_Z])
        return self._run_stage("ascend", target_pos)

    def transition(self, new_scenario: str, new_goal_pos, nominal_exit_pos, nominal_xy=None):
        """Delegate to env.soft_reset() and reset wrapper episode counters."""
        _, info = self._env.soft_reset(
            new_scenario=new_scenario,
            new_goal_pos=new_goal_pos,
            nominal_exit_pos=nominal_exit_pos,
            nominal_xy=nominal_xy,
        )
        curr = self._wrapped_env
        while hasattr(curr, "env"):
            if hasattr(curr, "_elapsed_steps"):
                curr._elapsed_steps = 0
                break
            curr = curr.env
        else:
            if hasattr(curr, "_elapsed_steps"):
                curr._elapsed_steps = 0
        return info

    def execute_grasp(self):
        """Delegate to the scripted grasp pipeline."""
        return self._env.execute_grasp()

    def execute_place(self, dst_xy):
        """Delegate to the scripted place pipeline."""
        return self._env.execute_place(dst_xy)

    def run_grasp(self):
        """Execute the scripted grasp pipeline and return StageResult."""
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
            error_mm=float(np.linalg.norm(grip_after - grip_before)) * 1000.0,
        )

    def run_place(self, dst_xy: np.ndarray):
        """Execute the scripted place pipeline and return StageResult."""
        from src.chess_env.controller import StageResult

        env = self._env
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

    def run_pick_sequence(self, src_xy: np.ndarray):
        """Execute transit -> descend -> grasp -> ascend at src_xy."""
        from src.chess_env.controller import SequenceResult

        results = []
        result = self.run_transit(src_xy)
        results.append(("transit", result))
        if not result.success:
            return SequenceResult(False, results, "transit", None)

        nom_exit = np.array([src_xy[0], src_xy[1], self._env.SAFE_Z])
        goal_descend = np.array([src_xy[0], src_xy[1], self._env.HOVER_Z])
        self.transition("descend", goal_descend, nom_exit, src_xy)

        result = self.run_descend(src_xy)
        results.append(("descend", result))
        if not result.success:
            return SequenceResult(False, results, "descend", None)

        result = self.run_grasp()
        results.append(("grasp", result))
        if not result.success:
            return SequenceResult(False, results, "grasp", None)

        nom_exit_hover = np.array([src_xy[0], src_xy[1], self._env.HOVER_Z])
        goal_ascend = np.array([src_xy[0], src_xy[1], self._env.SAFE_Z])
        self.transition("ascend", goal_ascend, nom_exit_hover, src_xy)

        result = self.run_ascend(src_xy)
        results.append(("ascend", result))
        if not result.success:
            return SequenceResult(False, results, "ascend", None)

        return SequenceResult(True, results, None, None)

    def run_place_sequence(self, dst_xy: np.ndarray):
        """Execute transit -> descend -> place -> ascend at dst_xy."""
        from src.chess_env.controller import SequenceResult

        results = []
        result = self.run_transit(dst_xy)
        results.append(("transit", result))
        if not result.success:
            return SequenceResult(False, results, "transit", None)

        nom_exit = np.array([dst_xy[0], dst_xy[1], self._env.SAFE_Z])
        goal_descend = np.array([dst_xy[0], dst_xy[1], self._env.HOVER_Z])
        self.transition("descend", goal_descend, nom_exit, dst_xy)

        result = self.run_descend(dst_xy)
        results.append(("descend", result))
        if not result.success:
            return SequenceResult(False, results, "descend", None)

        result = self.run_place(dst_xy)
        results.append(("place", result))
        if not result.success:
            return SequenceResult(False, results, "place", None)

        nom_exit_hover = np.array([dst_xy[0], dst_xy[1], self._env.HOVER_Z])
        goal_ascend = np.array([dst_xy[0], dst_xy[1], self._env.SAFE_Z])
        self.transition("ascend", goal_ascend, nom_exit_hover, dst_xy)

        result = self.run_ascend(dst_xy)
        results.append(("ascend", result))
        if not result.success:
            return SequenceResult(False, results, "ascend", None)

        return SequenceResult(True, results, None, None)

    def run_full_move(self, src_xy: np.ndarray, dst_xy: np.ndarray):
        """Full pick-and-place move. Called by MovementExecutor."""
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

    def _run_stage(self, stage: str, target_pos: np.ndarray):
        """Run a model-backed stage, or fall back to the scripted controller."""
        env = self._env
        model = self._models[stage]

        if model is None:
            target_xy = target_pos[:2]
            if stage == "transit":
                return self.scripted_controller.run_transit(target_xy)
            if stage == "descend":
                return self.scripted_controller.run_descend(target_xy)
            if stage == "ascend":
                return self.scripted_controller.run_ascend(target_xy)
            raise ValueError(f"Unknown stage: {stage}")

        from src.chess_env.controller import StageResult

        env.goal_pos = target_pos.copy()
        env.goal = target_pos.copy()
        env.current_scenario = stage
        env._debug_current_phase = stage

        if stage in {"descend", "ascend"}:
            env.tube_center_xy = target_pos[:2].copy()
        else:
            env.tube_center_xy = None

        l_finger = env._utils.get_joint_qpos(
            env.model, env.data, "robot0:l_gripper_finger_joint"
        ).item()
        expected_finger = (
            env.FINGER_OPEN_JOINT if stage == "descend" else env.FINGER_CLOSED_JOINT
        )
        if abs(l_finger - expected_finger) > 0.003:
            return StageResult(
                success=False,
                steps=0,
                crash_reason=(
                    f"PRECONDITION_FINGER (actual={l_finger:.4f}, "
                    f"expected={expected_finger:.4f})"
                ),
                final_pos=env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy(),
                error_mm=0.0,
            )

        previous_phase9 = env._use_phase9_obs
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

                if dist < env.SUCCESS_THRESHOLD and speed < env.env_cfg.get(
                    "stability_vel_threshold", 0.05
                ):
                    success = True
                    break

                obs = env._get_obs()
                action, _ = model.predict(obs, deterministic=True)
                action = np.array(action, dtype=np.float32)
                action[3] = 0.0

                env._set_action(action)
                env._mujoco_step(action)

                if self._render_fn is not None:
                    self._render_fn()
                    if self._render_delay > 0:
                        time.sleep(self._render_delay)

                grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
                crash_reason = self._check_crash(env, stage, grip_pos)
                if crash_reason:
                    break
        finally:
            env._use_phase9_obs = previous_phase9

        grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
        dist = float(np.linalg.norm(target_pos - grip_pos))
        if not success and crash_reason is None and step_idx + 1 >= self.MAX_STEPS:
            crash_reason = "TIMEOUT"

        return StageResult(
            success=success,
            steps=step_idx + 1,
            crash_reason=crash_reason if not success else None,
            final_pos=grip_pos.copy(),
            error_mm=dist * 1000.0,
        )

    def _check_crash(self, env, stage: str, grip_pos: np.ndarray) -> Optional[str]:
        """Lightweight in-loop crash check for inference."""
        if stage == "transit":
            if grip_pos[2] < env.FLOOR_LIMIT:
                return f"FLOOR_HIT (z={grip_pos[2]:.4f})"
            if env.grasp_mode:
                held, reason = env._check_cube_held(grip_pos)
                if not held:
                    return reason

        elif stage in {"descend", "ascend"}:
            if env.tube_center_xy is not None:
                drift = float(np.linalg.norm(grip_pos[:2] - env.tube_center_xy))
                drift_limit = env.env_cfg.get("eval_drift_limit", 0.010)
                if drift > drift_limit:
                    return f"TUBE_BREACH (drift={drift*1000:.1f}mm)"
            if grip_pos[2] < env.TABLE_SURFACE_Z:
                return f"TABLE_HIT (z={grip_pos[2]:.4f})"
            if stage == "ascend" and env.grasp_mode:
                held, reason = env._check_cube_held(grip_pos)
                if not held:
                    return reason

        return None
