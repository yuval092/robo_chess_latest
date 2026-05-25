"""
Production-accurate diagnostic for transit c2 twitching.

Recreates exactly what run_chess_ui.py does:
  - show_chess_pieces=True, hide_object=True, force_scenario="transit"
  - Loads transit model from configs/training.yaml
  - Runs transit HOME -> c2, logging every step

Usage:
    python scripts/diagnose_transit_c2.py
"""

import os
import sys

import numpy as np

# Suppress matplotlib warning
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym

from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_game.board_mapper import BoardMapper
from src.utils.io import load_config


def main():
    """Parse command-line arguments and run the script."""
    env_cfg = load_config("env")
    deployed_cfg = load_config("deployed_models")
    bm = BoardMapper.from_configs()

    c2_xy = bm.square_name_to_xy("c2")
    bm.square_name_to_xy("c3")
    safe_z = env_cfg["safe_z"]
    home_xy = env_cfg["home_position_xy"]
    HOME_POS = np.array([home_xy[0], home_xy[1], safe_z])

    print(f"HOME_POS:         {HOME_POS}")
    print(f"c2_xy:            {c2_xy}")
    print(f"c2_target:        {[c2_xy[0], c2_xy[1], safe_z]}")
    print(f"Dist HOME->c2:    {np.linalg.norm(HOME_POS[:2] - c2_xy):.3f}m")
    print()

    # --- Create env exactly like run_chess_ui.py ---
    env = gym.make(
        "ChessFetchTask-v0",
        force_scenario="transit",
        show_chess_pieces=True,
        hide_object=True,
        render_mode=None,
    )

    transit_model_path = deployed_cfg.get("transit")
    if not transit_model_path:
        print("ERROR: No transit model configured in configs/training.yaml")
        sys.exit(1)

    if not os.path.exists(transit_model_path):
        print(f"ERROR: Model file not found: {transit_model_path}")
        sys.exit(1)

    print(f"Loading transit model: {transit_model_path}")
    ctrl = ModelEmbeddedController(env)
    ctrl.load_model("transit", transit_model_path)
    print("Model loaded.")
    print()

    # --- Reset env (arm goes to HOME_POS) ---
    obs, info = env.reset()
    inner = env.unwrapped

    grip = inner._utils.get_site_xpos(inner.model, inner.data, "robot0:grip").copy()
    print(f"After reset, grip_pos: {grip}")
    print(f"Expected HOME_POS:     {HOME_POS}")
    print(f"Error: {np.linalg.norm(grip - HOME_POS) * 1000:.1f}mm")
    print()

    # --- Set up for transit to c2 (mirroring _run_stage internals) ---
    target_pos = np.array([c2_xy[0], c2_xy[1], safe_z])
    inner.goal_pos = target_pos.copy()
    inner.goal = target_pos.copy()
    inner.current_scenario = "transit"
    inner._debug_current_phase = "transit"
    inner.tube_center_xy = None

    # Finger setup
    if not inner.grasp_mode:
        inner.finger_target_joint = inner.FINGER_CLOSED_JOINT

    # Finger check
    l_finger = inner._utils.get_joint_qpos(
        inner.model, inner.data, "robot0:l_gripper_finger_joint"
    ).item()
    print(
        f"Finger state: l_finger={l_finger:.5f}, target={inner.finger_target_joint:.5f}"
    )
    print(f"grasp_mode: {inner.grasp_mode}")
    print()

    # --- Run transit step by step ---
    previous_phase9 = inner._use_phase9_obs
    inner._use_phase9_obs = True

    success = False
    crash_reason = None
    max_steps = 300
    stability_vel_threshold = env_cfg["stability_vel_threshold"]
    success_threshold = env_cfg["success_threshold"]

    print(
        f"{'Step':>5} | {'grip_x':>8} {'grip_y':>8} {'grip_z':>8} | "
        f"{'dist_xy':>8} {'dist_z':>6} {'speed':>8} | "
        f"{'near':>5} {'stable':>7} | {'action':>35}"
    )
    print("-" * 115)

    last_obs = None

    for step in range(max_steps):
        grip_pos = inner._utils.get_site_xpos(
            inner.model, inner.data, "robot0:grip"
        ).copy()
        grip_vel = inner._utils.get_site_xvelp(
            inner.model, inner.data, "robot0:grip"
        ).copy()
        speed = float(np.linalg.norm(grip_vel))

        d_xy = float(np.linalg.norm(grip_pos[:2] - target_pos[:2]))
        d_z = float(abs(grip_pos[2] - target_pos[2]))
        is_near = (d_xy < success_threshold) and (d_z < success_threshold)
        is_stable = speed < stability_vel_threshold

        # Check success before stepping
        if is_near and is_stable:
            success = True
            print(
                f"{step:>5} | {grip_pos[0]:8.4f} {grip_pos[1]:8.4f} {grip_pos[2]:8.4f} | "
                f"{d_xy * 1000:8.1f}mm {d_z * 1000:6.1f}mm {speed * 1000:8.2f}mm/s | "
                f"{'YES':>5} {'YES':>7} | SUCCESS!"
            )
            break

        obs = inner._get_obs()
        last_obs = obs
        model = ctrl._models["transit"]
        action, _ = model.predict(obs, deterministic=True)
        action = np.array(action, dtype=np.float32)
        action[3] = -1.0  # transit: gripper closed

        inner._set_action(action)
        inner._mujoco_step(action)

        # Log every step, more detailed near goal
        if step < 20 or step % 20 == 0 or d_xy < 0.020 or step > 280:
            action_str = f"[{action[0]:+5.2f},{action[1]:+5.2f},{action[2]:+5.2f},{action[3]:+5.2f}]"
            near_s = "YES" if is_near else "no"
            stable_s = "YES" if is_stable else "no"
            print(
                f"{step:>5} | {grip_pos[0]:8.4f} {grip_pos[1]:8.4f} {grip_pos[2]:8.4f} | "
                f"{d_xy * 1000:8.1f}mm {d_z * 1000:6.1f}mm {speed * 1000:8.2f}mm/s | "
                f"{near_s:>5} {stable_s:>7} | {action_str}"
            )

        # Crash check
        if grip_pos[2] < inner.FLOOR_LIMIT:
            crash_reason = f"FLOOR_HIT (z={grip_pos[2]:.4f})"
            break

    inner._use_phase9_obs = previous_phase9

    if not success:
        if crash_reason is None:
            crash_reason = "TIMEOUT"
        grip_pos = inner._utils.get_site_xpos(
            inner.model, inner.data, "robot0:grip"
        ).copy()
        d_xy = float(np.linalg.norm(grip_pos[:2] - target_pos[:2]))
        d_z = float(abs(grip_pos[2] - target_pos[2]))
        grip_vel = inner._utils.get_site_xvelp(
            inner.model, inner.data, "robot0:grip"
        ).copy()
        speed = float(np.linalg.norm(grip_vel))
        print()
        print(f"FAILED: {crash_reason}")
        print(f"Final position:   {grip_pos}")
        print(f"Target:           {target_pos}")
        print(
            f"Final d_xy:       {d_xy * 1000:.1f}mm  (threshold: {success_threshold * 1000:.0f}mm)"
        )
        print(
            f"Final d_z:        {d_z * 1000:.1f}mm  (threshold: {success_threshold * 1000:.0f}mm)"
        )
        print(
            f"Final speed:      {speed * 1000:.2f}mm/s  (threshold: {stability_vel_threshold * 1000:.0f}mm/s)"
        )
    else:
        grip_pos = inner._utils.get_site_xpos(
            inner.model, inner.data, "robot0:grip"
        ).copy()
        print()
        print(f"SUCCESS! Final position: {grip_pos}, target: {target_pos}")

    print()
    print("=== OBSERVATION ANALYSIS (last obs) ===")
    if last_obs is not None:
        if isinstance(last_obs, dict):
            for k, v in last_obs.items():
                if hasattr(v, "__len__"):
                    print(f"  {k}: shape={np.array(v).shape}, vals={np.array(v)}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"  obs: {last_obs}")

    print()
    print("=== TRAINING VS PRODUCTION CONFIG CHECK ===")
    print(f"  success_threshold:      {success_threshold * 1000:.0f}mm")
    print(f"  stability_vel_thresh:   {stability_vel_threshold * 1000:.0f}mm/s")
    print("  Scripted transit tol:   4mm (no speed check)")
    print(f"  MAX_STEPS (inference):  {max_steps}")
    print("  Training env show_chess_pieces: False → arm starts at RANDOM board pos")
    print(
        f"  Production show_chess_pieces:   True  → arm starts at HOME_POS={HOME_POS}"
    )

    env.close()


if __name__ == "__main__":
    main()
