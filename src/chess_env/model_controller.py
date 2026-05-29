"""Hybrid controller: SAC movement stages plus scripted grasp/place sequencing."""

import contextlib
import io
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from stable_baselines3 import SAC

from src.chess_env.base_env import transfer_obs_enabled
from src.chess_env.simulation import reset_elapsed_steps, unwrap_env

M_TO_MM = 1000.0
GRIPPER_CLOSED = -1.0
GRIPPER_OPEN = 1.0
ACTION_GRIPPER_IDX = 3
KNOWN_STAGES = frozenset({"transit", "descend", "ascend"})


@dataclass
class StageResult:
    """Result of a single waypoint stage."""

    success: bool
    steps: int
    crash_reason: str | None
    final_pos: np.ndarray
    error_mm: float

@dataclass
class SequenceResult:
    """Result of a full scenario chain."""

    success: bool
    stage_results: list
    failed_at: str | None


class ModelEmbeddedController:
    """Run learned transit/descend/ascend with scripted grasp/place transitions."""

    def __init__(
        self,
        env,
        render_fn: Callable | None = None,
        render_delay: float = 0.0,
    ):
        self._wrapped_env = env
        self._env = unwrap_env(env)
        self._render_fn = render_fn
        self._render_delay = render_delay
        self._max_steps = self._env.env_cfg["rl_max_steps_per_stage"]
        self._models: dict[str, SAC | None] = {stage: None for stage in KNOWN_STAGES}

    def load_model(self, stage: str, path: str) -> None:
        """Load a specialist model for one stage."""
        if stage not in KNOWN_STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        if not path:
            raise ValueError(f"No model path provided for {stage}")

        with transfer_obs_enabled(self._wrapped_env):
            with contextlib.redirect_stdout(io.StringIO()):
                self._models[stage] = SAC.load(path, env=self._wrapped_env)
        print(f"[ModelEmbeddedController] Loaded {stage} model from {path}")

    def load_all(self, transit_path: str, descend_path: str, ascend_path: str) -> None:
        """Load all three specialist models."""
        self.load_model("transit", transit_path)
        self.load_model("descend", descend_path)
        self.load_model("ascend", ascend_path)

    def run_transit(self, target_xy: np.ndarray):
        """Move arm horizontally to target_xy at SAFE_Z."""
        target_pos = np.array([target_xy[0], target_xy[1], self._env.SAFE_Z])
        return self._run_model_stage("transit", target_pos)

    def run_descend(self, tube_xy: np.ndarray):
        """Soft-reset to descend scenario, then lower arm from SAFE_Z to HOVER_Z."""
        self._prepare_stage("descend", tube_xy)
        target_pos = np.array([tube_xy[0], tube_xy[1], self._env.HOVER_Z])
        return self._run_model_stage("descend", target_pos)

    def run_ascend(self, tube_xy: np.ndarray):
        """Soft-reset to ascend scenario, then raise arm from HOVER_Z to SAFE_Z."""
        self._prepare_stage("ascend", tube_xy)
        target_pos = np.array([tube_xy[0], tube_xy[1], self._env.SAFE_Z])
        return self._run_model_stage("ascend", target_pos)

    def _prepare_stage(self, new_scenario: str, xy: np.ndarray) -> None:
        """Soft-reset the environment to new_scenario centred on xy.

        Halts the arm, aligns it to the nominal exit position of the previous stage,
        then switches the scenario and goal so the next SAC model has the correct context.
        """
        env = self._env
        goal_z =     {"descend": env.HOVER_Z, "ascend": env.SAFE_Z}[new_scenario]
        nom_exit_z = {"descend": env.SAFE_Z,  "ascend": env.HOVER_Z}[new_scenario]
        env.soft_reset(
            new_scenario=new_scenario,
            new_goal_pos=np.array([xy[0], xy[1], goal_z]),
            nominal_exit_pos=np.array([xy[0], xy[1], nom_exit_z]),
            nominal_xy=xy,
        )
        reset_elapsed_steps(self._wrapped_env)

    def run_grasp(self):
        """Execute the scripted grasp pipeline and return StageResult."""

        env = self._env
        grip_before = self._get_gripper_position()
        result_dict = env.execute_grasp()
        success = result_dict.get("success", False)
        grip_after = self._get_gripper_position()
        return StageResult(
            success=success,
            steps=result_dict.get(
                "total_steps_used", result_dict.get("close_steps_used", 0)
            ),
            crash_reason=None if success else result_dict.get("reason", "GRASP_FAILED"),
            final_pos=grip_after.copy(),
            error_mm=float(np.linalg.norm(grip_after - grip_before)) * 1000.0,
        )

    def run_place(self, dst_xy: np.ndarray):
        """Execute the scripted place pipeline and return StageResult."""

        env = self._env
        result_dict = env.execute_place(dst_xy)
        success = result_dict.get("success", False)
        grip_after = self._get_gripper_position()
        return StageResult(
            success=success,
            steps=result_dict.get("total_steps_used", 0),
            crash_reason=None if success else result_dict.get("reason", "PLACE_FAILED"),
            final_pos=grip_after.copy(),
            error_mm=0.0,
        )

    def run_pick_sequence(self, src_xy: np.ndarray) -> SequenceResult:
        """Execute transit -> descend -> grasp -> ascend at src_xy."""
        return self._run_sequence([
            ("transit", lambda: self.run_transit(src_xy)),
            ("descend", lambda: self.run_descend(src_xy)),
            ("grasp",   self.run_grasp),
            ("ascend",  lambda: self.run_ascend(src_xy)),
        ])

    def run_place_sequence(self, dst_xy: np.ndarray) -> SequenceResult:
        """Execute transit -> descend -> place -> ascend at dst_xy."""
        return self._run_sequence([
            ("transit", lambda: self.run_transit(dst_xy)),
            ("descend", lambda: self.run_descend(dst_xy)),
            ("place",   lambda: self.run_place(dst_xy)),
            ("ascend",  lambda: self.run_ascend(dst_xy)),
        ])

    def _run_sequence(self, steps: list[tuple[str, callable]]) -> SequenceResult:
        """Run a list of (stage_name, fn) steps, stopping on first failure."""
        results = []
        for stage, fn in steps:
            result = fn()
            results.append((stage, result))
            if not result.success:
                return SequenceResult(False, results, stage)
        return SequenceResult(True, results, None)

    def run_full_move(self, src_xy: np.ndarray, dst_xy: np.ndarray):
        """Full pick-and-place move. Called by MovementExecutor."""

        pick = self.run_pick_sequence(src_xy)
        if not pick.success:
            return pick
        place = self.run_place_sequence(dst_xy)
        return SequenceResult(
            success=place.success,
            stage_results=pick.stage_results + place.stage_results,
            failed_at=place.failed_at,
        )

    def _run_model_stage(self, stage: str, target_pos: np.ndarray) -> StageResult:
        """Run a model-backed movement stage."""
        model = self._get_stage_model(stage)
        self._setup_model_stage_env(stage, target_pos)

        precondition_failure = self._check_finger_precondition()
        if precondition_failure:
            return precondition_failure
        
        success, step_idx, crash_reason = self._run_inference_loop(stage, target_pos, model)
        return self._build_model_stage_result(target_pos, success, step_idx, crash_reason)

    def _get_stage_model(self, stage: str) -> SAC:
        """Helper to get the model for a stage, with error handling."""
        if stage not in KNOWN_STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        model = self._models[stage]
        if model is None:
            raise RuntimeError(
                f"No {stage} model loaded. Call load_model() before running {stage}."
            )
        return model
    
    def _setup_model_stage_env(self, stage: str, target_pos: np.ndarray) -> None:
        """Set goal, scenario, tube centre, and finger target on the environment."""
        env = self._env
        env.goal_pos = target_pos.copy()
        env.goal = target_pos.copy()
        env.current_scenario = stage
        env._debug_current_phase = stage
        env.tube_center_xy = target_pos[:2].copy() if stage in {"descend", "ascend"} else None
        # In grasp_mode the fingers are held against a piece — leave them as-is
        # so the actuator keeps gripping through the transit.
        if not env.grasp_mode:
            env.finger_target_joint = (
                env.FINGER_OPEN_JOINT if stage == "descend" else env.FINGER_CLOSED_JOINT
            )

    def _check_finger_precondition(self) -> StageResult | None:
        """Return a failed StageResult if the finger is not in the expected position, else None.

        Skipped in grasp_mode: the finger joint is physically blocked by the held piece
        and will read ~0.01 even though FINGER_CLOSED_JOINT=0.000 — this is normal and safe.
        """
        env = self._env
        if env.grasp_mode:
            return None
        # Only the left finger is checked because both fingers are always driven to the
        # same target value simultaneously, so left mirrors right exactly.
        l_finger = env._utils.get_joint_qpos(
            env.model, env.data, "robot0:l_gripper_finger_joint"
        ).item()
        expected_finger = env.finger_target_joint
        if abs(l_finger - expected_finger) > 0.003:
            return StageResult(
                success=False,
                steps=0,
                crash_reason=(
                    f"PRECONDITION_FINGER (actual={l_finger:.4f}, "
                    f"expected={expected_finger:.4f})"
                ),
                final_pos=self._get_gripper_position(),
                error_mm=0.0,
            )
        return None

    def _get_gripper_position(self) -> np.ndarray:
        """Return current gripper position."""
        env = self._env
        return env._utils.get_site_xpos(env.model, env.data, "robot0:grip").copy()
    

    def _get_gripper_speed(self) -> float:
        """Return current gripper speed."""
        env = self._env
        grip_vel = env._utils.get_site_xvelp(env.model, env.data, "robot0:grip").copy()
        return float(np.linalg.norm(grip_vel))

    def _execute_step(self, stage: str, model: SAC) -> None:
        """Predict an action, apply it to the environment, and render if needed."""
        env = self._env
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        action = np.array(action, dtype=np.float32)
        action[ACTION_GRIPPER_IDX] = GRIPPER_CLOSED if stage in {"transit", "ascend"} else GRIPPER_OPEN

        env._set_action(action)
        # Transit model was trained with VERTICAL_QUAT enforced; ascend/descend were not.
        # Only clamp wrist orientation for transit to match each model's training conditions.
        if stage == "transit":
            env.data.mocap_quat[0][:] = env.VERTICAL_QUAT
        env._mujoco_step(action)

        if self._render_fn is not None:
            self._render_fn()
            if self._render_delay > 0:
                time.sleep(self._render_delay)

    def _run_inference_loop(
        self, stage: str, target_pos: np.ndarray, model: SAC
    ) -> tuple[bool, int, str | None]:
        """Step the model until the target is reached, a crash occurs, or max steps is hit.

        Returns (success, step_idx, crash_reason).
        """
        env = self._env
        previous_transfer_obs = env._use_transfer_obs
        env._use_transfer_obs = True
        crash_reason = None
        success = False
        step_idx = 0

        try:
            for step_idx in range(self._max_steps):
                grip_pos = self._get_gripper_position()
                speed = self._get_gripper_speed()
                if bool(env._is_success(grip_pos, target_pos)) and speed < env.env_cfg["stability_vel_threshold"]:
                    success = True
                    break
                self._execute_step(stage, model)
                grip_pos = self._get_gripper_position()
                crash_reason = self._check_crash(env, stage, grip_pos)
                if crash_reason:
                    break
        finally:
            env._use_transfer_obs = previous_transfer_obs

        return success, step_idx, crash_reason

    def _build_model_stage_result(
        self,
        target_pos: np.ndarray,
        success: bool,
        step_idx: int,
        crash_reason: str | None,
    ) -> StageResult:
        """Build a StageResult from the inference loop outcome."""
        grip_pos = self._get_gripper_position()
        dist = float(np.linalg.norm(target_pos - grip_pos))
        if not success and crash_reason is None and step_idx + 1 >= self._max_steps:
            crash_reason = "TIMEOUT"
        return StageResult(
            success=success,
            steps=step_idx + 1,
            crash_reason=crash_reason if not success else None,
            final_pos=grip_pos.copy(),
            error_mm=dist * M_TO_MM,
        )

    def _check_cube_dropped(self, env, grip_pos: np.ndarray) -> str | None:
        """Return a crash reason if grasp_mode is active but the piece is no longer held."""
        if not env.grasp_mode:
            return None
        held, reason = env._check_cube_held(grip_pos)
        return None if held else reason

    def _check_tube_breach(self, env, grip_pos: np.ndarray) -> str | None:
        """Return a crash reason if the gripper has drifted outside the target tube."""
        if env.tube_center_xy is None:
            return None
        drift = float(np.linalg.norm(grip_pos[:2] - env.tube_center_xy))
        if drift > env.env_cfg["eval_drift_limit"]:
            return f"TUBE_BREACH (drift={drift * M_TO_MM:.1f}mm)"
        return None

    def _check_crash(self, env, stage: str, grip_pos: np.ndarray) -> str | None:
        """Lightweight in-loop crash check for inference."""
        if stage == "transit":
            if grip_pos[2] < env.FLOOR_LIMIT:
                return f"FLOOR_HIT (z={grip_pos[2]:.4f})"

        if stage in {"descend", "ascend"}:
            if reason := self._check_tube_breach(env, grip_pos):
                return reason
            if grip_pos[2] < env.TABLE_SURFACE_Z:
                return f"TABLE_HIT (z={grip_pos[2]:.4f})"
        
        if stage in {"transit", "ascend"}:
            if reason := self._check_cube_dropped(env, grip_pos):
                return reason

        return None
