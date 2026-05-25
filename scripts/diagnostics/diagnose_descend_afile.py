"""
Diagnostic for descend TIMEOUT at a-file positions (a1, a2).

Runs descend from SAFE_Z at a1 and a2 positions, logging every step,
to understand whether the model makes progress or oscillates.

Compares against a center position (d4) for baseline.

Usage:
    PYTHONPATH=. python scripts/diagnose_descend_afile.py
"""

import os
import sys

import mujoco
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
from stable_baselines3 import SAC

from src.utils.io import load_config
from training.envs import WRAPPER_MAP


def run_descend_diagnostic(model, env, start_xy, label, max_steps=400):
    """
    Manually place the arm at start_xy, SAFE_Z, then run the descend model.
    Logs every step.
    """
    env_cfg = load_config("env")
    safe_z = env_cfg["safe_z"]
    hover_z = env_cfg["hover_z"]
    success_threshold = env_cfg["success_threshold"]
    stability_threshold = env_cfg["stability_vel_threshold"]
    eval_drift_limit = env_cfg["eval_drift_limit"]

    inner = env.unwrapped
    start_pos = np.array([start_xy[0], start_xy[1], safe_z])
    target_pos = np.array([start_xy[0], start_xy[1], hover_z])

    print(f"\n{'=' * 80}")
    print(f"DESCEND DIAGNOSTIC: {label}")
    print(f"  Start:   {start_pos}")
    print(f"  Target:  {target_pos}")
    print(f"  Descent: {(safe_z - hover_z) * 1000:.0f}mm")
    print(f"  Drift limit: {eval_drift_limit * 1000:.0f}mm")
    print(f"  Stability: {stability_threshold * 1000:.0f}mm/s")
    print(f"{'=' * 80}")

    # Reset env to descend scenario at the target XY
    inner.force_scenario = "descend"
    # We'll force the start position after reset via direct mocap manipulation
    obs, _ = env.reset()

    # Override arm position to exact start_pos
    inner._move_mocap_to(start_pos, inner.VERTICAL_QUAT, max_steps=200, tolerance=0.002)
    mujoco.mj_forward(inner.model, inner.data)

    # Set up goal and tube
    inner.goal_pos = target_pos.copy()
    inner.goal = target_pos.copy()
    inner.tube_center_xy = start_xy.copy()
    inner.current_scenario = "descend"
    inner._use_phase9_obs = True
    inner.finger_target_joint = inner.FINGER_OPEN_JOINT

    # Check actual arm position after override
    grip = inner._utils.get_site_xpos(inner.model, inner.data, "robot0:grip").copy()
    print(
        f"Arm at start: {grip} (error from target: {np.linalg.norm(grip - start_pos) * 1000:.1f}mm)"
    )
    print()
    print(
        f"{'Step':>5} | {'grip_x':>8} {'grip_y':>8} {'grip_z':>8} | "
        f"{'d_z':>7} {'drift':>6} {'speed':>8} | {'near':>5} {'stable':>7} | {'action[0:3]':>25}"
    )
    print("-" * 105)

    success = False
    crash_reason = None
    prev_z = grip[2]

    for step in range(max_steps):
        grip_pos = inner._utils.get_site_xpos(
            inner.model, inner.data, "robot0:grip"
        ).copy()
        grip_vel = inner._utils.get_site_xvelp(
            inner.model, inner.data, "robot0:grip"
        ).copy()
        speed = float(np.linalg.norm(grip_vel))
        d_z = float(abs(grip_pos[2] - target_pos[2]))
        drift = float(np.linalg.norm(grip_pos[:2] - start_xy))

        d_xy_now = float(np.linalg.norm(grip_pos[:2] - target_pos[:2]))
        d_z_now = float(abs(grip_pos[2] - target_pos[2]))
        z_tol = env_cfg["descend_success_threshold"]
        is_near = d_xy_now < success_threshold and d_z_now < z_tol
        is_stable = speed < stability_threshold

        if is_near and is_stable:
            success = True
            print(
                f"{step:>5} | {grip_pos[0]:8.4f} {grip_pos[1]:8.4f} {grip_pos[2]:8.4f} | "
                f"{d_z * 1000:7.1f}mm {drift * 1000:6.1f}mm {speed * 1000:8.2f}mm/s | "
                f"{'YES':>5} {'YES':>7} | SUCCESS!"
            )
            break

        obs = inner._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        action = np.array(action, dtype=np.float32)
        action[3] = 1.0  # descend: gripper open

        inner._set_action(action)
        inner._mujoco_step(action)

        # Crash checks
        if drift > eval_drift_limit:
            crash_reason = f"TUBE_BREACH ({drift * 1000:.1f}mm)"
            break
        if grip_pos[2] < inner.TABLE_SURFACE_Z:
            crash_reason = f"TABLE_HIT (z={grip_pos[2]:.4f})"
            break

        # Log every step for first 30, then every 20, or when near goal
        if step < 30 or step % 20 == 0 or d_z < 0.020:
            action_str = f"[{action[0]:+5.2f},{action[1]:+5.2f},{action[2]:+5.2f}]"
            near_s = "YES" if is_near else "no"
            stable_s = "YES" if is_stable else "no"
            dz_progress = (grip_pos[2] - prev_z) * 1000
            print(
                f"{step:>5} | {grip_pos[0]:8.4f} {grip_pos[1]:8.4f} {grip_pos[2]:8.4f} | "
                f"{d_z * 1000:7.1f}mm {drift * 1000:6.1f}mm {speed * 1000:8.2f}mm/s | "
                f"{near_s:>5} {stable_s:>7} | {action_str}  Δz={dz_progress:+5.1f}mm"
            )

        prev_z = grip_pos[2]

    print()
    if success:
        grip_pos = inner._utils.get_site_xpos(
            inner.model, inner.data, "robot0:grip"
        ).copy()
        print(f"SUCCESS in {step + 1} steps. Final: {grip_pos}, target: {target_pos}")
    else:
        if crash_reason is None:
            crash_reason = "TIMEOUT"
        grip_pos = inner._utils.get_site_xpos(
            inner.model, inner.data, "robot0:grip"
        ).copy()
        d = np.linalg.norm(grip_pos - target_pos)
        drift = np.linalg.norm(grip_pos[:2] - start_xy)
        speed = np.linalg.norm(
            inner._utils.get_site_xvelp(inner.model, inner.data, "robot0:grip")
        )
        print(f"FAILED: {crash_reason}")
        print(f"  Final pos:    {grip_pos}")
        print(f"  Target:       {target_pos}")
        print(f"  Dist to goal: {d * 1000:.1f}mm")
        print(f"  Drift:        {drift * 1000:.1f}mm")
        print(f"  Speed:        {speed * 1000:.2f}mm/s")

    inner._use_phase9_obs = False
    return success, crash_reason


def main():
    """Parse command-line arguments and run the script."""
    deployed_cfg = load_config("deployed_models")
    descend_path = deployed_cfg.get("descend")
    if not descend_path or not os.path.exists(descend_path):
        print(f"ERROR: descend model not found: {descend_path}")
        sys.exit(1)

    print(f"Loading descend model: {descend_path}")

    env_cfg = load_config("env")
    eval_drift = env_cfg["eval_drift_limit"]

    env = gym.make(
        "ChessFetchTask-v0",
        force_scenario="descend",
        force_drift_limit=eval_drift,
        show_chess_pieces=False,
        hide_object=True,
        render_mode=None,
    )
    env = WRAPPER_MAP["descend"](env)
    model = SAC.load(descend_path, env=env)
    print("Model loaded.\n")

    # Test positions
    positions = [
        ([0.88, 0.2641], "center (d4-ish) [BASELINE]"),
        ([0.60, -0.0159], "a1 [EXPECTED FAIL]"),
        ([0.60, 0.0641], "a2 [EXPECTED FAIL]"),
        ([0.60, 0.3041], "a5 [POSSIBLY FAIL]"),
        ([0.60, 0.5441], "a8 [EXPECTED PASS]"),
        ([1.16, -0.0159], "h1 [reference - ascend fails here]"),
    ]

    results = []
    for xy, label in positions:
        ok, reason = run_descend_diagnostic(
            model, env, np.array(xy), label, max_steps=400
        )
        results.append((label, ok, reason))

    print("\n\n=== SUMMARY ===")
    for label, ok, reason in results:
        status = "PASS" if ok else f"FAIL ({reason})"
        print(f"  {label:45s} {status}")

    env.close()


if __name__ == "__main__":
    main()
