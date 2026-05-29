"""RL training environment: adds drift curriculum, reward shaping, and step()."""

from __future__ import annotations

import numpy as np

from src.chess_env.base_env import TRANSFER_OBS_SPACE, ChessBaseEnv


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
        self._use_transfer_obs = True
        self.observation_space = TRANSFER_OBS_SPACE

        self.drift_curriculum_steps = (
            drift_curriculum_steps or self.env_cfg["drift_curriculum_steps"]
        )
        self.force_drift_limit = force_drift_limit
        self.fixed_drift = fixed_drift

        self.DRIFT_LIMIT_START = self.env_cfg["drift_limit_start"]
        self.DRIFT_LIMIT_END = self.env_cfg["drift_limit_end"]
        self.DRIFT_CURRICULUM_STEPS = self.drift_curriculum_steps
        self.current_drift_limit = self._get_current_drift_limit()

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Dense reward used by SB3 (and HER-compatible callers via vector path)."""
        achieved_goal = np.asarray(achieved_goal, dtype=np.float64)
        desired_goal = np.asarray(desired_goal, dtype=np.float64)
        scalar = achieved_goal.ndim == 1
        if scalar:
            achieved_goal = achieved_goal[None]
            desired_goal = desired_goal[None]

        dist = np.linalg.norm(achieved_goal - desired_goal, axis=1)
        reward = -self.env_cfg["dist_reward_weight"] * dist

        z_err = np.abs(achieved_goal[:, 2] - desired_goal[:, 2])
        reward -= self.env_cfg["z_reward_weight"] * z_err

        xy_err = np.linalg.norm(achieved_goal[:, :2] - desired_goal[:, :2], axis=1)
        reward -= self.env_cfg["xy_reward_weight"] * xy_err

        return float(reward[0]) if scalar else reward

    def _get_current_drift_limit(self) -> float:
        """Compute the active tube radius, applying curriculum if enabled."""
        if self.force_drift_limit is not None:
            return float(self.force_drift_limit)
        if self.fixed_drift or self.drift_curriculum_steps is None:
            return float(self.DRIFT_LIMIT_END)

        progress = min(self.total_env_steps / self.DRIFT_CURRICULUM_STEPS, 1.0)
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
        if obs["observation"].shape[0] >= 23:
            grip_vel = obs["observation"][20:23]
        else:
            grip_vel = obs["grip_vel"]

        terminated = False
        crash_reason = None

        l_finger = self._utils.get_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint"
        ).item()
        if abs(l_finger - self.finger_target_joint) > 0.003:
            terminated = True
            crash_reason = (
                f"FINGER_FAULT (actual={l_finger:.4f}, "
                f"target={self.finger_target_joint:.4f})"
            )

        if not terminated:
            current_drift_limit = self._get_current_drift_limit()
            self.current_drift_limit = current_drift_limit

            if self.current_scenario == "transit":
                if grip_pos[2] < self.FLOOR_LIMIT:
                    terminated = True
                    crash_reason = f"FLOOR_HIT (z={grip_pos[2]:.4f})"

            elif self.current_scenario in {"descend", "ascend"}:
                if self.tube_center_xy is not None:
                    drift = float(np.linalg.norm(grip_pos[:2] - self.tube_center_xy))
                    if drift > current_drift_limit:
                        terminated = True
                        crash_reason = (
                            f"TUBE_BREACH (drift={drift * 1000:.1f}mm > "
                            f"limit={current_drift_limit * 1000:.1f}mm)"
                        )
                if not terminated and grip_pos[2] < self.TABLE_SURFACE_Z:
                    terminated = True
                    crash_reason = f"TABLE_HIT (z={grip_pos[2]:.4f})"

        if terminated and crash_reason:
            reward = float(self.env_cfg["crash_penalty"])
            success = 0.0
        else:
            reward = self.compute_reward(grip_pos, self.goal_pos, {})

            dist = float(np.linalg.norm(grip_pos - self.goal_pos))
            speed = float(np.linalg.norm(grip_vel))
            scenario = self.current_scenario or "transit"
            braking_dist_key = f"{scenario}_braking_dist"
            braking_weight_key = f"{scenario}_braking_reward_weight"
            braking_d = (
                self.env_cfg[braking_dist_key]
                if braking_dist_key in self.env_cfg
                else self.env_cfg["braking_dist"]
            )
            braking_w = (
                self.env_cfg[braking_weight_key]
                if braking_weight_key in self.env_cfg
                else self.env_cfg["braking_reward_weight"]
            )
            if dist < braking_d:
                reward -= braking_w * speed

            reward -= self.env_cfg["jitter_penalty_weight"] * float(
                np.linalg.norm(action_copy[:3]) ** 2
            )

            floor_prox_thresh = self.env_cfg["floor_proximity_threshold"]
            if grip_pos[2] < self.FLOOR_LIMIT + floor_prox_thresh:
                reward += float(self.env_cfg["floor_penalty"])

            if self.current_scenario == "descend":
                d_xy = float(np.linalg.norm(grip_pos[:2] - self.goal_pos[:2]))
                d_z = float(abs(grip_pos[2] - self.goal_pos[2]))
                descend_thresh = self.env_cfg["descend_success_threshold"]
                is_near = bool(d_xy < self.SUCCESS_THRESHOLD and d_z < descend_thresh)
            else:
                is_near = bool(self._is_success(grip_pos, self.goal_pos))
            is_stable = float(np.linalg.norm(grip_vel)) < float(
                self.env_cfg["stability_vel_threshold"]
            )
            success = 1.0 if (is_near and is_stable) else 0.0

            if success:
                terminated = True
                reward += float(self.env_cfg["success_bonus"])

        info = {
            "is_success": success,
            "scenario": self.current_scenario,
            "crash_reason": crash_reason,
        }

        if self.debug and terminated:
            outcome = (
                f"CRASH ({crash_reason})"
                if crash_reason
                else ("SUCCESS" if success else "TIMEOUT")
            )
            self.logger.debug(
                f"[EP {self.episode_number} END] Scenario={self.current_scenario} "
                f"Outcome={outcome} Reward={reward:.2f}"
            )

        return obs, float(reward), terminated, False, info
