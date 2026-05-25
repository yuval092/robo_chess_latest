"""Reset, transition, reward, and step runtime for ChessTaskEnv."""

from __future__ import annotations

import logging

import mujoco
import numpy as np


class TaskRuntimeMixin:
    """Provide reset, soft-reset, reward, and step methods."""

    def _reset_sim(self):
        """
        Executes a complex reset sequence with scripted gripper transitions:
        1. Samples start/goal positions.
        2. Teleports object to a hidden or board location.
        3. Settles the arm at the starting waypoint with fingers CLOSED (stability).
        4. Executes scenario-specific scripted gripper transitions (Open/Close).
        5. Validates final finger state before handing over to RL model.
        """
        self.episode_steps = 0
        self.episode_number += 1

        # Stage 2 removed periodic debug sampling
        self.debug = self.base_debug

        if self.debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

        if self.force_scenario is not None:
            self.current_scenario = self.force_scenario
        else:
            self.current_scenario = self.np_random.choice(
                ["transit", "descend", "ascend"]
            )

        # NEW: reset grasp mode on every reset
        self.grasp_mode = False

        start_pos = self._sample_board_position()
        start_xy = start_pos[:2]

        # Define scenario-specific start/goal configurations
        if self.current_scenario == "transit":
            goal_pos = self._sample_board_position()
            while np.linalg.norm(goal_pos[:2] - start_xy) < self.MIN_GOAL_DIST:
                goal_pos = self._sample_board_position()
            self.tube_center_xy = None

            # Chess mode: always start at HOME_POS (board center); RL/eval: random or forced
            if self.force_start_pos is not None:
                arm_start_pos = self.force_start_pos.copy()
            elif self.show_chess_pieces:
                arm_start_pos = self.HOME_POS.copy()
            else:
                arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])

            self.goal_pos = np.array([goal_pos[0], goal_pos[1], self.SAFE_Z])
        elif self.current_scenario == "descend":
            self.tube_center_xy = start_xy.copy()
            arm_start_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])
            self.goal_pos = np.array([start_xy[0], start_xy[1], self.HOVER_Z])
        elif self.current_scenario == "ascend":
            self.tube_center_xy = start_xy.copy()
            arm_start_pos = np.array([start_xy[0], start_xy[1], self.HOVER_Z])
            self.goal_pos = np.array([start_xy[0], start_xy[1], self.SAFE_Z])

        # NEW: Ensure self.goal is set BEFORE any call to self.render()
        # FetchEnv's _render_callback uses self.goal, which is normally set
        # by RobotEnv.reset() AFTER _reset_sim() returns.
        if self.goal_pos is not None:
            self.goal = self.goal_pos.copy()

        # Reset parent simulation
        super()._reset_sim()

        # Place object
        obj_joint_id = self.model.joint("object0:joint").id
        qpos_start = self.model.jnt_qposadr[obj_joint_id]
        dof_start = self.model.jnt_dofadr[obj_joint_id]

        if self.hide_object:
            self.data.qpos[qpos_start : qpos_start + 3] = self.HIDDEN_OBJECT_POS
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0
        elif self.force_cube_pos is not None:
            # Support force_cube_pos override
            self.data.qpos[qpos_start : qpos_start + 3] = self.force_cube_pos[:3]
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0
        else:
            self.data.qpos[qpos_start : qpos_start + 2] = start_xy
            self.data.qpos[qpos_start + 2] = self.TABLE_Z + (self.CUBE_HEIGHT / 2.0)
            self.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
            self.data.qvel[dof_start : dof_start + 6] = 0.0

        self._reset_chess_piece_bodies()
        mujoco.mj_forward(self.model, self.data)

        # Force torso to optimal height for board reach
        torso_height = self.env_cfg["torso_height"]
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:torso_lift_joint", torso_height
        )
        mujoco.mj_forward(self.model, self.data)

        # --- PHASE 1: Settle arm CLOSED for stability ---
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT
        )
        self.finger_target_joint = self.FINGER_CLOSED_JOINT
        self._set_gripper_state()
        self._settle_arm_to_start(arm_start_pos)

        # --- PHASE 2: Scripted Transitions ---
        # Dummy action for _set_action calls during transitions
        np.zeros(4)

        if self.current_scenario == "descend":
            # Descend: closed --> opened (scripted transition before RL)
            self.finger_target_joint = self.FINGER_OPEN_JOINT
            self._set_gripper_state()
            self._move_mocap_to(
                arm_start_pos, self.VERTICAL_QUAT, max_steps=150, tolerance=0.003
            )
        elif self.current_scenario == "ascend":
            # Ascend: start opened --> closed (scripted transition before RL)
            # First open them
            self.finger_target_joint = self.FINGER_OPEN_JOINT
            self._set_gripper_state()  # Force immediately
            self._move_mocap_to(
                arm_start_pos, self.VERTICAL_QUAT, max_steps=100, tolerance=0.003
            )

            # Then close them scripted
            self.finger_target_joint = self.FINGER_CLOSED_JOINT
            self._set_gripper_state()  # Fix: Force close immediately to ensure convergence
            self._move_mocap_to(
                arm_start_pos, self.VERTICAL_QUAT, max_steps=150, tolerance=0.003
            )
        else:
            # Transit: stay closed
            self.finger_target_joint = self.FINGER_CLOSED_JOINT
            self._move_mocap_to(
                arm_start_pos, self.VERTICAL_QUAT, max_steps=150, tolerance=0.003
            )

        # --- PHASE 3: Final Validation ---
        l_pos = self._utils.get_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint"
        ).item()
        if abs(l_pos - self.finger_target_joint) > 0.0005:
            self.logger.error(
                f"Reset Failed: Finger joint at {l_pos:.6f}, target {self.finger_target_joint:.6f} (Scenario: {self.current_scenario})"
            )
            return False

        if np.linalg.norm(arm_start_pos - self.HOME_POS) < 1e-9:
            self._capture_home_posture()

        if self.debug:
            obj_pos_now = self.data.qpos[qpos_start : qpos_start + 3]
            grip_pos_now = self._grip_pos()
            arm_joints = [
                "robot0:shoulder_pan_joint",
                "robot0:shoulder_lift_joint",
                "robot0:upperarm_roll_joint",
                "robot0:elbow_flex_joint",
                "robot0:forearm_roll_joint",
                "robot0:wrist_flex_joint",
                "robot0:wrist_roll_joint",
            ]
            joint_qpos = []
            for jname in arm_joints:
                try:
                    joint_qpos.append(
                        self._utils.get_joint_qpos(self.model, self.data, jname).item()
                    )
                except Exception:
                    joint_qpos.append(float("nan"))
            jq_str = ", ".join([f"{x:.4f}" for x in joint_qpos])
            tube_str = (
                f"{self.tube_center_xy}" if self.tube_center_xy is not None else "N/A"
            )
            msg = (
                f"\n{'=' * 80}\n"
                f"[EPISODE {self.episode_number} START] Scenario: {self.current_scenario.upper()}\n"
                f"  GripPos:    [{', '.join([f'{x:.4f}' for x in grip_pos_now])}]\n"
                f"  Goal:       [{', '.join([f'{x:.4f}' for x in self.goal_pos])}]\n"
                f"  TubeCenter: {tube_str}\n"
                f"  ObjPos:     [{', '.join([f'{x:.4f}' for x in obj_pos_now])}]\n"
                f"  ArmJoints:  [{jq_str}]\n"
                f"{'=' * 80}"
            )
            self.logger.debug(msg)

        return True

    def transition_validate(self, nominal_exit_pos: np.ndarray | None = None) -> dict:
        """
        Returns arm state diagnostics at a scenario transition point.
        """
        grip_pos = self._grip_pos()
        grip_vel = self._utils.get_site_xvelp(
            self.model, self.data, "robot0:grip"
        ).copy()
        speed = float(np.linalg.norm(grip_vel))
        l_finger = self._utils.get_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint"
        ).item()

        result = {
            "grip_pos": grip_pos.tolist(),
            "grip_speed_mm_s": speed * 1000,
            "is_velocity_ok": speed < self.HALT_VEL_THRESHOLD,
            "error_from_nominal_mm": None,
            "finger_state": l_finger,
        }
        if nominal_exit_pos is not None:
            result["error_from_nominal_mm"] = float(
                np.linalg.norm(grip_pos - nominal_exit_pos) * 1000
            )
        return result

    def soft_reset(
        self,
        new_scenario: str,
        new_goal_pos: np.ndarray,
        nominal_exit_pos: np.ndarray,
        nominal_xy: np.ndarray | None = None,
    ) -> tuple[dict, dict]:
        """
        Transitions to a new scenario without teleporting the arm.
        Executes: complete halt -> waypoint alignment -> gripper transition -> state update.

        Note: The caller is responsible for resetting environment wrappers (like TimeLimit).

        Returns:
            (obs, info): obs is the new observation, info contains transition diagnostics.
        """
        # ROBOT_DOF = 15 is correct for Fetch: 3 slides + 1 torso + 2 head + 7 arm + 2 fingers
        ROBOT_DOF = 15
        HALT_HOLD_MAX_STEPS = 30
        ALIGN_TOLERANCE_M = 0.004
        ALIGN_MAX_STEPS = 80
        ALIGN_GAIN = 1.0
        ALIGN_MAX_STEP_M = 0.012

        # Phase 1: Halt (Driving arm to a dead stop)
        self._debug_current_phase = "softreset_p1_halt"
        zero_action = np.zeros(4)
        should_render = self.render_mode == "human"
        self.data.qvel[:ROBOT_DOF] = 0.0
        self.data.qacc[:ROBOT_DOF] = 0.0
        mujoco.mj_forward(self.model, self.data)
        halt_steps = 0
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        if np.linalg.norm(grip_vel) >= self.HALT_VEL_THRESHOLD:
            for _ in range(HALT_HOLD_MAX_STEPS):
                grip_vel = self._utils.get_site_xvelp(
                    self.model, self.data, "robot0:grip"
                )
                if np.linalg.norm(grip_vel) < self.HALT_VEL_THRESHOLD:
                    break
                self._set_action(zero_action)
                self._mujoco_step(None)
                if should_render:
                    self.render()
                halt_steps += 1

        # Active velocity zeroing (Robot only, preserves object physics)
        self.data.qvel[:ROBOT_DOF] = 0.0
        self.data.qacc[:ROBOT_DOF] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Confirm halt (Plan Step 1c)
        grip_vel = self._utils.get_site_xvelp(self.model, self.data, "robot0:grip")
        speed = float(np.linalg.norm(grip_vel))
        if speed >= self.HALT_VEL_THRESHOLD:
            raise RuntimeError(
                f"soft_reset HALT_FAILED: speed={speed * 1000:.3f}mm/s >= "
                f"threshold={self.HALT_VEL_THRESHOLD * 1000:.1f}mm/s"
            )

        # Phase 2: Waypoint Alignment (Physics-simulated smooth movement)
        self._debug_current_phase = "softreset_p2_align"
        converged = False
        align_steps = 0
        for loop_step in range(ALIGN_MAX_STEPS):
            grip_pos = self._grip_pos()
            error = nominal_exit_pos - grip_pos
            dist = np.linalg.norm(error)
            if dist < ALIGN_TOLERANCE_M:
                self.data.qvel[:ROBOT_DOF] = 0.0
                self.data.qacc[:ROBOT_DOF] = 0.0
                mujoco.mj_forward(self.model, self.data)
                converged = True
                break
            # Proportional step toward target
            step_vec = ALIGN_GAIN * error
            if np.linalg.norm(step_vec) > ALIGN_MAX_STEP_M:
                step_vec = step_vec / np.linalg.norm(step_vec) * ALIGN_MAX_STEP_M
            self.data.mocap_pos[0][:3] += step_vec
            self._mujoco_step(None)
            if should_render:
                self.render()
            align_steps = loop_step + 1

        if not converged:
            raise RuntimeError(
                f"soft_reset ALIGN_FAILED: did not converge to {nominal_exit_pos}"
            )

        # Phase 3: State Update
        self._debug_current_phase = "softreset_p3_state"
        prev_scenario = self.current_scenario
        self.current_scenario = new_scenario
        self.goal_pos = new_goal_pos.copy()
        self.goal = self.goal_pos.copy()  # Critical for observation desync
        self.episode_steps = 0

        if new_scenario in {"descend", "ascend"}:
            if nominal_xy is None:
                raise ValueError(f"nominal_xy required for {new_scenario}")
            self.tube_center_xy = nominal_xy.copy()
        else:
            self.tube_center_xy = None

        # Re-enforce vertical orientation
        self._utils.set_mocap_quat(
            self.model, self.data, "robot0:mocap", self.VERTICAL_QUAT
        )

        # Phase 4: Gripper Transition (Scripted finger movement)
        self._debug_current_phase = "softreset_p4_gripper"
        if not self.grasp_mode:
            needs_open = new_scenario == "descend" and prev_scenario != "descend"
            needs_close = (
                new_scenario in {"ascend", "transit"} and prev_scenario == "descend"
            )
            np.zeros(4)

            if needs_open:
                self.finger_target_joint = self.FINGER_OPEN_JOINT
                self._set_gripper_state()
                self._move_mocap_to(
                    nominal_exit_pos, self.VERTICAL_QUAT, max_steps=40, tolerance=0.004
                )
            elif needs_close:
                self.finger_target_joint = self.FINGER_CLOSED_JOINT
                self._set_gripper_state()
                self._move_mocap_to(
                    nominal_exit_pos, self.VERTICAL_QUAT, max_steps=40, tolerance=0.004
                )

            # Finger validation (same as _reset_sim Phase 3)
            l_pos = self._utils.get_joint_qpos(
                self.model, self.data, "robot0:l_gripper_finger_joint"
            ).item()
            if abs(l_pos - self.finger_target_joint) > 0.0005:
                raise RuntimeError(
                    f"soft_reset FINGER_VALIDATION_FAILED: "
                    f"actual={l_pos:.6f}, target={self.finger_target_joint:.6f}"
                )
        # else: grasp_mode=True -> fingers stay actuator-driven, no teleportation or validation needed

        info = {
            "halt_steps": halt_steps,
            "align_steps": align_steps,
        }
        return self._get_obs(), info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dense reward for RL training.

        Scripted control bypasses Gym step/reward entirely; this is used by SB3
        and remains vectorized for HER-compatible callers.
        """
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
        """Computes the active tube radius, applying curriculum if enabled."""
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
        """
        Full RL training step. ScriptedController bypasses this method and
        drives the arm through _mujoco_step() directly.
        """
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
        action_copy[3] = -1.0
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
