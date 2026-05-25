"""
Full transit diagnostic script.

Records EVERY step for each episode, capturing:
- Position, velocity, distance-to-goal, action
- Near/stable conditions
- Success, timeout, or crash
- Per-position pattern analysis

Usage:
    PYTHONPATH=. python scripts/diagnose_transit_full.py [--n-episodes 30]
"""

import argparse
import os
import sys

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym

from src.chess_env.model_controller import ModelEmbeddedController
from src.chess_game.board_mapper import BoardMapper
from src.utils.io import load_config


def run_episode_detailed(ctrl, inner, target_xy, safe_z, env_cfg, verbose=False):
    """Run one transit episode, logging every step. Returns episode dict."""
    target_pos = np.array([target_xy[0], target_xy[1], safe_z])
    inner.goal_pos = target_pos.copy()
    inner.goal = target_pos.copy()
    inner.current_scenario = "transit"
    inner._debug_current_phase = "transit"
    inner.tube_center_xy = None

    if not inner.grasp_mode:
        inner.finger_target_joint = inner.FINGER_CLOSED_JOINT

    stability_vel_threshold = env_cfg["stability_vel_threshold"]
    success_threshold = env_cfg["success_threshold"]
    braking_dist = env_cfg["braking_dist"]
    max_steps = 300

    previous_phase9 = inner._use_phase9_obs
    inner._use_phase9_obs = True

    steps_log = []
    success = False
    crash_reason = None
    step_idx = 0

    model = ctrl._models["transit"]

    try:
        for step_idx in range(max_steps):
            grip_pos = inner._utils.get_site_xpos(
                inner.model, inner.data, "robot0:grip"
            ).copy()
            grip_vel = inner._utils.get_site_xvelp(
                inner.model, inner.data, "robot0:grip"
            ).copy()
            speed = float(np.linalg.norm(grip_vel))

            d_xy = float(np.linalg.norm(grip_pos[:2] - target_pos[:2]))
            d_z = float(abs(grip_pos[2] - target_pos[2]))
            dist_3d = float(np.linalg.norm(grip_pos - target_pos))
            is_near = (d_xy < success_threshold) and (d_z < success_threshold)
            is_stable = speed < stability_vel_threshold
            in_braking_zone = dist_3d < braking_dist

            if is_near and is_stable:
                success = True
                steps_log.append(
                    {
                        "step": step_idx,
                        "pos": grip_pos.copy(),
                        "vel": grip_vel.copy(),
                        "speed": speed,
                        "d_xy": d_xy,
                        "d_z": d_z,
                        "dist_3d": dist_3d,
                        "is_near": True,
                        "is_stable": True,
                        "in_braking": in_braking_zone,
                        "action": None,
                        "outcome": "SUCCESS",
                    }
                )
                break

            obs = inner._get_obs()
            action, _ = model.predict(obs, deterministic=True)
            action = np.array(action, dtype=np.float32)
            action[3] = -1.0

            steps_log.append(
                {
                    "step": step_idx,
                    "pos": grip_pos.copy(),
                    "vel": grip_vel.copy(),
                    "speed": speed,
                    "d_xy": d_xy,
                    "d_z": d_z,
                    "dist_3d": dist_3d,
                    "is_near": is_near,
                    "is_stable": is_stable,
                    "in_braking": in_braking_zone,
                    "action": action.copy(),
                    "outcome": None,
                }
            )

            inner._set_action(action)
            inner._mujoco_step(action)

            # Crash check
            grip_pos_post = inner._utils.get_site_xpos(
                inner.model, inner.data, "robot0:grip"
            ).copy()
            if grip_pos_post[2] < inner.FLOOR_LIMIT:
                crash_reason = f"FLOOR_HIT (z={grip_pos_post[2]:.4f})"
                break
    finally:
        inner._use_phase9_obs = previous_phase9

    if not success and crash_reason is None and step_idx + 1 >= max_steps:
        crash_reason = "TIMEOUT"

    # Final position
    grip_pos_final = inner._utils.get_site_xpos(
        inner.model, inner.data, "robot0:grip"
    ).copy()
    grip_vel_final = inner._utils.get_site_xvelp(
        inner.model, inner.data, "robot0:grip"
    ).copy()
    final_d_xy = float(np.linalg.norm(grip_pos_final[:2] - target_pos[:2]))
    final_d_z = float(abs(grip_pos_final[2] - target_pos[2]))
    final_speed = float(np.linalg.norm(grip_vel_final))

    return {
        "target_xy": target_xy.copy(),
        "target_pos": target_pos.copy(),
        "success": success,
        "crash_reason": crash_reason,
        "total_steps": step_idx + 1,
        "steps_log": steps_log,
        "final_pos": grip_pos_final,
        "final_d_xy": final_d_xy,
        "final_d_z": final_d_z,
        "final_speed_mms": final_speed * 1000.0,
    }


def analyze_timeout_episode(ep):
    """Analyze a timeout episode to understand what's happening near goal."""
    log = ep["steps_log"]
    if not log:
        return

    target_pos = ep["target_pos"]

    # Find steps where arm is within 20mm of goal
    near_goal_steps = [s for s in log if s["d_xy"] < 0.020]
    # Find steps where arm is within 10mm of goal (braking zone)
    [s for s in log if s["d_xy"] < 0.010]

    print("\n  --- Timeout analysis ---")
    print(f"  Target: x={target_pos[0]:.3f}, y={target_pos[1]:.3f}")
    print(
        f"  Final pos: x={ep['final_pos'][0]:.3f}, y={ep['final_pos'][1]:.3f}, z={ep['final_pos'][2]:.3f}"
    )
    print(
        f"  Final d_xy={ep['final_d_xy'] * 1000:.1f}mm, d_z={ep['final_d_z'] * 1000:.1f}mm, speed={ep['final_speed_mms']:.1f}mm/s"
    )
    print(f"  Total steps: {ep['total_steps']}")

    if near_goal_steps:
        print(f"\n  Steps within 20mm of goal ({len(near_goal_steps)} steps):")
        # Show last 20 near-goal steps
        for s in near_goal_steps[-20:]:
            if s["action"] is not None:
                a = s["action"]
                print(
                    f"    step={s['step']:3d}: pos=({s['pos'][0]:.4f},{s['pos'][1]:.4f}), "
                    f"d_xy={s['d_xy'] * 1000:5.1f}mm, speed={s['speed'] * 1000:6.1f}mm/s, "
                    f"near={s['is_near']}, stable={s['is_stable']}, "
                    f"action=[{a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f}]"
                )

        # Speed trend in last 20 near-goal steps
        speeds = [s["speed"] for s in near_goal_steps]
        if len(speeds) > 1:
            speed_trend = speeds[-1] - speeds[0]
            print(
                f"\n  Speed trend in near-goal zone: start={speeds[0] * 1000:.1f}mm/s, "
                f"end={speeds[-1] * 1000:.1f}mm/s, delta={speed_trend * 1000:+.1f}mm/s"
            )
            if speed_trend > 0:
                print("  WARNING: Speed is INCREASING near goal — oscillation likely!")
            elif speed_trend < -0.005:
                print("  Speed is decreasing — model is braking")
            else:
                print("  Speed is roughly stable — not converging")
    else:
        print("  Arm never got within 20mm of goal during episode!")

    # Check if oscillating (speed > threshold but close)
    osc_steps = [s for s in log if s["d_xy"] < 0.015 and s["speed"] > 0.020]
    if osc_steps:
        print(
            f"\n  Oscillation evidence: {len(osc_steps)} steps within 15mm but speed > 20mm/s"
        )
        # Look at X/Y velocity direction to see if it's oscillating
        if len(osc_steps) > 2:
            vx_vals = [s["vel"][0] for s in osc_steps[-10:]]
            vy_vals = [s["vel"][1] for s in osc_steps[-10:]]
            sign_changes_x = sum(
                1 for i in range(1, len(vx_vals)) if vx_vals[i] * vx_vals[i - 1] < 0
            )
            sign_changes_y = sum(
                1 for i in range(1, len(vy_vals)) if vy_vals[i] * vy_vals[i - 1] < 0
            )
            print(
                f"  Velocity sign changes (last 10 steps): X={sign_changes_x}, Y={sign_changes_y}"
            )
            if sign_changes_x + sign_changes_y >= 3:
                print("  CONFIRMED OSCILLATION: frequent velocity direction reversals")


def main():
    """Parse command-line arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Full transit diagnostic with per-step logging."
    )
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--capture-timeouts",
        type=int,
        default=5,
        help="Number of timeout episodes to analyze in detail",
    )
    args = parser.parse_args()

    env_cfg = load_config("env")
    deployed_cfg = load_config("deployed_models")
    BoardMapper.from_configs()

    transit_model_path = deployed_cfg.get("transit")
    if not transit_model_path or not os.path.exists(transit_model_path):
        print(f"ERROR: Transit model not found: {transit_model_path}")
        sys.exit(1)

    safe_z = env_cfg["safe_z"]
    success_threshold = env_cfg["success_threshold"]
    stability_vel_threshold = env_cfg["stability_vel_threshold"]

    print(f"Transit model: {transit_model_path}")
    print(f"N episodes: {args.n_episodes}")
    print(
        f"Success threshold: {success_threshold * 1000:.0f}mm, Stability: {stability_vel_threshold * 1000:.0f}mm/s"
    )
    print()

    env = gym.make(
        "ChessFetchTask-v0",
        force_scenario="transit",
        show_chess_pieces=True,
        hide_object=True,
        render_mode=None,
    )

    ctrl = ModelEmbeddedController(env)
    ctrl.load_model("transit", transit_model_path)
    inner = env.unwrapped

    successes = 0
    timeouts = 0
    crashes = 0

    timeout_episodes = []
    success_episodes = []

    print(
        f"{'EP':>4} | {'Target XY':>20} | {'Outcome':>10} | {'Steps':>6} | {'FinalDxy':>9} | {'FinalSpd':>10}"
    )
    print("-" * 80)

    for ep in range(args.n_episodes):
        obs, info = env.reset()

        # Get the target set by reset
        target_pos = inner.goal_pos.copy()
        target_xy = target_pos[:2].copy()

        ep_result = run_episode_detailed(
            ctrl, inner, target_xy, safe_z, env_cfg, verbose=args.verbose
        )

        outcome = (
            "SUCCESS"
            if ep_result["success"]
            else ep_result["crash_reason"] or "TIMEOUT"
        )
        print(
            f"{ep + 1:>4} | ({target_xy[0]:.3f},{target_xy[1]:.3f}) -> ({inner.goal_pos[0]:.3f},{inner.goal_pos[1]:.3f}) | "
            f"{outcome:>10} | {ep_result['total_steps']:>6} | "
            f"{ep_result['final_d_xy'] * 1000:>8.1f}mm | {ep_result['final_speed_mms']:>9.1f}mm/s"
        )

        if ep_result["success"]:
            successes += 1
            success_episodes.append(ep_result)
        elif "TIMEOUT" in outcome:
            timeouts += 1
            timeout_episodes.append(ep_result)
        else:
            crashes += 1

    env.close()

    print()
    print("=" * 80)
    print(
        f"Results: {successes}/{args.n_episodes} success, {timeouts} timeouts, {crashes} crashes"
    )
    print(f"Success rate: {successes / args.n_episodes:.1%}")
    print()

    # Analyze timeout episodes
    n_analyze = min(args.capture_timeouts, len(timeout_episodes))
    if n_analyze > 0:
        print(f"=== Detailed analysis of {n_analyze} timeout episodes ===")
        for i, ep in enumerate(timeout_episodes[:n_analyze]):
            print(f"\n[Timeout #{i + 1}]")
            analyze_timeout_episode(ep)

    # Position pattern analysis
    print("\n=== Position Pattern Analysis ===")
    print("\nTimeout target positions:")
    for ep in timeout_episodes:
        print(
            f"  target=({ep['target_pos'][0]:.3f},{ep['target_pos'][1]:.3f}), "
            f"final_d_xy={ep['final_d_xy'] * 1000:.1f}mm, speed={ep['final_speed_mms']:.1f}mm/s"
        )

    if success_episodes:
        print("\nSuccess target positions (sample):")
        for ep in success_episodes[:10]:
            print(
                f"  target=({ep['target_pos'][0]:.3f},{ep['target_pos'][1]:.3f}), "
                f"steps={ep['total_steps']}"
            )

    # Check if timeouts correlate with distance
    if timeout_episodes and success_episodes:
        # Use board center as reference
        home_xy = np.array(env_cfg["home_position_xy"])
        timeout_dists = [
            np.linalg.norm(np.array(ep["target_pos"][:2]) - home_xy)
            for ep in timeout_episodes
        ]
        success_dists = [
            np.linalg.norm(np.array(ep["target_pos"][:2]) - home_xy)
            for ep in success_episodes
        ]
        print("\nDistance from home (board center):")
        print(
            f"  Timeouts: avg={np.mean(timeout_dists) * 1000:.0f}mm, min={np.min(timeout_dists) * 1000:.0f}mm, max={np.max(timeout_dists) * 1000:.0f}mm"
        )
        print(
            f"  Successes: avg={np.mean(success_dists) * 1000:.0f}mm, min={np.min(success_dists) * 1000:.0f}mm, max={np.max(success_dists) * 1000:.0f}mm"
        )


if __name__ == "__main__":
    main()
