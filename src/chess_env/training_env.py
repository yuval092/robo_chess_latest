"""RL training environment: adds drift curriculum, reward shaping, and step()."""

from __future__ import annotations

import numpy as np

from src.chess_env.base_env import PRETRAINED_OBS_SPACE, ChessBaseEnv


class ChessTrainingEnv(ChessBaseEnv):
    """Adds drift curriculum + dense RL reward on top of the shared base."""

    def __init__(
        self,
        drift_curriculum_steps=None,
        force_drift_limit=None,
        fixed_drift=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._use_pretrained_obs_format = True
        self.observation_space = PRETRAINED_OBS_SPACE

        self.drift_curriculum_steps = (
            drift_curriculum_steps or self.env_cfg["drift_curriculum_steps"]
        )
        self.force_drift_limit = force_drift_limit
        self.fixed_drift = fixed_drift

        self.DRIFT_LIMIT_START = self.env_cfg["drift_limit_start"]
        self.DRIFT_LIMIT_END = self.env_cfg["drift_limit_end"]
        self.current_drift_limit = self._get_current_drift_limit()

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Name fixed by GoalEnv interface; called by SB3 each step.
        achieved_goal = np.asarray(achieved_goal, dtype=np.float64)
        desired_goal = np.asarray(desired_goal, dtype=np.float64)
        dist = float(np.linalg.norm(achieved_goal - desired_goal))
        z_err = float(abs(achieved_goal[2] - desired_goal[2]))
        xy_err = float(np.linalg.norm(achieved_goal[:2] - desired_goal[:2]))
        return (
            -self.env_cfg["dist_reward_weight"] * dist
            - self.env_cfg["z_reward_weight"] * z_err
            - self.env_cfg["xy_reward_weight"] * xy_err
        )

    def _get_current_drift_limit(self) -> float:
        """Compute the active tube radius, applying curriculum if enabled."""
        if self.force_drift_limit is not None:
            return float(self.force_drift_limit)
        if self.fixed_drift or self.drift_curriculum_steps is None:
            return float(self.DRIFT_LIMIT_END)

        progress = min(self.total_env_steps / self.drift_curriculum_steps, 1.0)
        limit = (
            self.DRIFT_LIMIT_START
            - (self.DRIFT_LIMIT_START - self.DRIFT_LIMIT_END) * progress
        )
        return float(limit)

    def step(self, action):
        """RL training step with reward shaping and crash/success termination."""
        self.total_env_steps += 1
        self.episode_steps += 1

        if self.current_scenario == "descend":
            self.finger_target_joint = self.FINGER_OPEN_JOINT
        else:
            self.finger_target_joint = self.FINGER_CLOSED_JOINT

        action_copy = np.asarray(action, dtype=np.float32).copy()
        action_copy = np.clip(
            action_copy, self.action_space.low, self.action_space.high
        )
        self._set_action(action_copy)
        self._mujoco_step(action_copy)

        obs = self._get_obs()
        grip_pos = obs["achieved_goal"]
        grip_vel = obs["observation"][20:23]  # grip_velp in the 25D pretrained obs

        terminated, crash_reason = self._check_crash(grip_pos)
        if terminated:
            reward, success = float(self.env_cfg["crash_penalty"]), 0.0
        else:
            reward, success = self._compute_step_reward(grip_pos, grip_vel, action_copy)

        info = {
            "is_success": success,
            "scenario": self.current_scenario,
            "crash_reason": crash_reason,
        }

        if self.debug and terminated:
            outcome = f"CRASH ({crash_reason})" if crash_reason else ("SUCCESS" if success else "TIMEOUT")
            self.logger.debug(
                f"[EP {self.episode_number} END] Scenario={self.current_scenario} "
                f"Outcome={outcome} Reward={reward:.2f}"
            )

        return obs, float(reward), terminated, False, info

    def _check_crash(self, grip_pos: np.ndarray) -> tuple[bool, str | None]:
        """Check finger fault and scenario-specific boundary violations."""
        l_finger = self.get_finger_angle()
        if abs(l_finger - self.finger_target_joint) > 0.003:
            return True, f"FINGER_FAULT (actual={l_finger:.4f}, target={self.finger_target_joint:.4f})"

        current_drift_limit = self._get_current_drift_limit()
        self.current_drift_limit = current_drift_limit

        if self.current_scenario == "transit":
            if grip_pos[2] < self.FLOOR_LIMIT:
                return True, f"FLOOR_HIT (z={grip_pos[2]:.4f})"

        elif self.current_scenario in {"descend", "ascend"}:
            if self.tube_center_xy is not None:
                drift = float(np.linalg.norm(grip_pos[:2] - self.tube_center_xy))
                if drift > current_drift_limit:
                    return True, f"TUBE_BREACH (drift={drift * 1000:.1f}mm > limit={current_drift_limit * 1000:.1f}mm)"
            if grip_pos[2] < self.TABLE_SURFACE_Z:
                return True, f"TABLE_HIT (z={grip_pos[2]:.4f})"

        return False, None

    def _compute_step_reward(
        self, grip_pos: np.ndarray, grip_vel: np.ndarray, action: np.ndarray
    ) -> tuple[float, float]:
        """Compute dense reward with braking/jitter/floor penalties and success bonus."""
        reward = self.compute_reward(grip_pos, self.goal_pos, {})
        speed = float(np.linalg.norm(grip_vel))

        reward -= self._braking_penalty(grip_pos, speed)
        reward -= self._jitter_penalty(action)
        reward -= self._floor_penalty(grip_pos)

        success = self._is_episode_success(grip_pos, speed)
        if success:
            reward += float(self.env_cfg["success_bonus"])

        return reward, float(success)

    def _braking_penalty(self, grip_pos: np.ndarray, speed: float) -> float:
        """Penalise speed when close to goal to encourage deceleration."""
        dist = float(np.linalg.norm(grip_pos - self.goal_pos))
        scenario = self.current_scenario or "transit"
        braking_d = self.env_cfg.get(f"{scenario}_braking_dist", self.env_cfg["braking_dist"])
        braking_w = self.env_cfg.get(f"{scenario}_braking_reward_weight", self.env_cfg["braking_reward_weight"])
        return braking_w * speed if dist < braking_d else 0.0

    def _jitter_penalty(self, action: np.ndarray) -> float:
        """Penalise large XYZ actions to encourage smooth motion."""
        return self.env_cfg["jitter_penalty_weight"] * float(np.linalg.norm(action[:3]) ** 2)

    def _floor_penalty(self, grip_pos: np.ndarray) -> float:
        """Return floor proximity penalty (negative value from config) when near floor."""
        if grip_pos[2] < self.FLOOR_LIMIT + self.env_cfg["floor_proximity_threshold"]:
            return float(self.env_cfg["floor_penalty"])
        return 0.0

    def _is_episode_success(self, grip_pos: np.ndarray, speed: float) -> bool:
        """True when the gripper is near the goal and moving slowly enough."""
        if self.current_scenario == "descend":
            d_xy = float(np.linalg.norm(grip_pos[:2] - self.goal_pos[:2]))
            d_z = float(abs(grip_pos[2] - self.goal_pos[2]))
            is_near = d_xy < self.SUCCESS_THRESHOLD and d_z < self.env_cfg["descend_success_threshold"]
        else:
            is_near = bool(self._is_success(grip_pos, self.goal_pos))
        return is_near and speed < float(self.env_cfg["stability_vel_threshold"])
