#!/usr/bin/env python3
"""
Physics verification for the grasp stage.
Run this after applying XML changes to verify the cube can be held.

Usage:
    PYTHONPATH=. python scripts/eval_grasp_physics.py --n-trials 20 --visualize --debug
"""

import argparse

import gymnasium as gym
import mujoco
import numpy as np
import scipy.spatial.transform


def assert_mandatory_preconditions(uw):
    """Call at the start of every test run."""
    print("Checking mandatory preconditions...")

    # 1. GRASP_Z check (Updated to 0.430 in Stage 3)
    assert abs(uw.GRASP_Z - 0.430) < 0.001, (
        f"GRASP_Z={uw.GRASP_Z}, expected 0.430. Restart process after env.yaml change."
    )

    # 2. Cube mass check
    cube_body_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_BODY, "object0")
    mass = uw.model.body_mass[cube_body_id]
    print(f"  - Cube mass: {mass:.3f}kg")
    assert abs(mass - 0.05) < 0.001, (
        f"Cube mass={mass:.3f}, expected 0.05. XML not updated."
    )

    # 3. Finger threshold check (Updated to 0.016 in Stage 4)
    assert abs(uw.GRASP_VERIFY_FINGER_THRESHOLD - 0.016) < 0.001, (
        f"GRASP_VERIFY_FINGER_THRESHOLD={uw.GRASP_VERIFY_FINGER_THRESHOLD}, expected 0.016"
    )

    # 3.5. Actuator Kp check (Updated to 20000 in Stage 3)
    l_act_id = mujoco.mj_name2id(
        uw.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_joint"
    )
    kp = uw.model.actuator_gainprm[l_act_id, 0]
    print(f"  - Actuator Kp: {kp}")
    assert abs(kp - 20000) < 1.0, f"Expected 20000, got {kp}"

    # 4. Geom parameters
    cube_geom_id = mujoco.mj_name2id(uw.model, mujoco.mjtObj.mjOBJ_GEOM, "object0")
    solref = uw.model.geom_solref[cube_geom_id]
    solimp = uw.model.geom_solimp[cube_geom_id]
    condim = uw.model.geom_condim[cube_geom_id]
    conaff = uw.model.geom_conaffinity[cube_geom_id]
    contype = uw.model.geom_contype[cube_geom_id]
    print(f"  - Cube solref: {solref}")
    print(f"  - Cube solimp: {solimp}")
    print(f"  - Cube condim: {condim} | conaff: {conaff} | contype: {contype}")

    l_finger_id = mujoco.mj_name2id(
        uw.model, mujoco.mjtObj.mjOBJ_GEOM, "robot0:l_gripper_finger_link"
    )
    f_solref = uw.model.geom_solref[l_finger_id]
    f_condim = uw.model.geom_condim[l_finger_id]
    f_conaff = uw.model.geom_conaffinity[l_finger_id]
    f_contype = uw.model.geom_contype[l_finger_id]
    print(f"  - Finger solref: {f_solref}")
    print(f"  - Finger condim: {f_condim} | conaff: {f_conaff} | contype: {f_contype}")

    print("[OK] Mandatory preconditions verified")


def _valid_board_position(uw):
    """Samples a board position using config dimensions."""
    cx, cy = uw.env_cfg["table_center_xy"]
    hx, hy = uw.env_cfg["table_half_x"], uw.env_cfg["table_half_y"]
    margin = uw.env_cfg["edge_margin"]

    min_x, max_x = cx - hx + margin, cx + hx - margin
    min_y, max_y = cy - hy + margin, cy + hy - margin

    x = uw.np_random.uniform(min_x, max_x)
    y = uw.np_random.uniform(min_y, max_y)
    return np.array([x, y])


def move_arm_to_target(uw, target, tolerance=0.001, max_steps=200):
    """Moves arm directly using mocap for test setup."""
    return uw._move_mocap_to(
        target, uw.VERTICAL_QUAT, max_steps=max_steps, tolerance=tolerance
    )


def test_static_grasp(env, debug=False) -> dict:
    """Test 1: Place arm at HOVER_Z, then execute grasp, check stability."""
    uw = env.unwrapped
    uw.force_scenario = "descend"
    uw.hide_object = False
    env.reset()

    src_xy = _valid_board_position(uw)
    cube_z = uw.TABLE_SURFACE_Z + uw.CUBE_HEIGHT / 2.0

    # Teleport cube to valid position
    obj_joint_id = uw.model.joint("object0:joint").id
    qpos_start = uw.model.jnt_qposadr[obj_joint_id]
    uw.data.qpos[qpos_start : qpos_start + 2] = src_xy
    uw.data.qpos[qpos_start + 2] = cube_z
    uw.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
    uw.data.qvel[:] = 0.0
    mujoco.mj_forward(uw.model, uw.data)

    cube_pos = uw.get_cube_position()
    target_hover = np.array([cube_pos[0], cube_pos[1], uw.HOVER_Z])

    # Open fingers first for setup
    uw.finger_target_joint = uw.FINGER_OPEN_JOINT
    uw._set_gripper_state()

    move_arm_to_target(uw, target_hover)

    grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
    arm_alignment_error = np.linalg.norm(grip_pos - target_hover) * 1000  # mm

    # Execute grasp (includes Plunge, Grasp, Retract)
    result = uw.execute_grasp()

    cube_pos_after = uw.get_cube_position()
    cube_z_displacement = abs(cube_pos_after[2] - cube_z) * 1000
    cube_xy_displacement = np.linalg.norm(cube_pos_after[:2] - src_xy) * 1000

    if debug:
        print(
            f"  [Static] Alignment err: {arm_alignment_error:.1f}mm | Success: {result['success']} ({result.get('reason')})"
        )

    return {
        "success": result["success"],
        "reason": result.get("reason"),
        "arm_alignment_error_mm": arm_alignment_error,
        "cube_z_displacement_mm": cube_z_displacement,
        "cube_xy_displacement_mm": cube_xy_displacement,
        "close_steps_used": result.get("close_steps_used"),
        "final_finger_pos": result.get("final_finger_pos"),
    }


def test_lift(env, debug=False) -> dict:
    """Test 2: Grasp cube and lift to SAFE_Z. Verify no drop."""
    uw = env.unwrapped
    uw.force_scenario = "descend"
    uw.hide_object = False
    env.reset()

    src_xy = _valid_board_position(uw)
    cube_z = uw.TABLE_SURFACE_Z + uw.CUBE_HEIGHT / 2.0

    obj_joint_id = uw.model.joint("object0:joint").id
    qpos_start = uw.model.jnt_qposadr[obj_joint_id]
    uw.data.qpos[qpos_start : qpos_start + 2] = src_xy
    uw.data.qpos[qpos_start + 2] = cube_z
    uw.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
    uw.data.qvel[:] = 0.0
    mujoco.mj_forward(uw.model, uw.data)

    # Setup
    uw.finger_target_joint = uw.FINGER_OPEN_JOINT
    uw._set_gripper_state()
    cube_pos = uw.get_cube_position()
    move_arm_to_target(uw, np.array([cube_pos[0], cube_pos[1], uw.HOVER_Z]))

    grasp_result = uw.execute_grasp()
    if not grasp_result["success"]:
        return {
            "success": False,
            "reason": f"GRASP_FAILED: {grasp_result.get('reason')}",
        }

    # Lift to SAFE_Z manually (not using Controller to test raw physics)
    target_z = uw.SAFE_Z
    max_z_deviation = 0.0
    cube_dropped = False
    drop_reason = None

    # 3mm per step for smooth lift
    for step in range(300):
        grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
        if grip_pos[2] >= target_z - 0.001:
            break

        uw.data.mocap_pos[0][2] += 0.003
        uw._mujoco_step(None)

        # Check cube
        cube_pos = uw.get_cube_position()
        grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
        z_offset = (
            grip_pos[2] - cube_pos[2]
        )  # expected ~0.015m if at top, but actually site-to-COM
        # Corrected: site is at finger tips. Cube is 30mm tall.
        # COM is at 15mm from top.
        # site_z should be near cube_top_z. So site_z - cube_COM_z should be ~15mm.
        deviation = abs(z_offset - 0.015)
        max_z_deviation = max(max_z_deviation, deviation)

        xy_error = np.linalg.norm(cube_pos[:2] - grip_pos[:2]) * 1000
        if xy_error > 30.0:
            cube_dropped = True
            drop_reason = f"CUBE_DROPPED_XY ({xy_error:.1f}mm)"
            break
        if deviation > 0.040:
            cube_dropped = True
            drop_reason = f"CUBE_DROPPED_Z (deviation={deviation * 1000:.1f}mm)"
            break

        if env.unwrapped.render_mode == "human":
            env.render()

    if debug:
        print(
            f"  [Lift] Success: {not cube_dropped} | Max Z-dev: {max_z_deviation * 1000:.1f}mm"
        )

    return {
        "success": not cube_dropped,
        "reason": drop_reason,
        "max_z_deviation_mm": max_z_deviation * 1000,
    }


def test_transit_held(env, debug=False) -> dict:
    """Test 3: Transit horizontally while holding cube."""
    uw = env.unwrapped
    uw.force_scenario = "descend"
    uw.hide_object = False
    env.reset()

    src_xy = _valid_board_position(uw)
    cube_z = uw.TABLE_SURFACE_Z + uw.CUBE_HEIGHT / 2.0

    obj_joint_id = uw.model.joint("object0:joint").id
    qpos_start = uw.model.jnt_qposadr[obj_joint_id]
    uw.data.qpos[qpos_start : qpos_start + 2] = src_xy
    uw.data.qpos[qpos_start + 2] = cube_z
    uw.data.qpos[qpos_start + 3 : qpos_start + 7] = [1, 0, 0, 0]
    uw.data.qvel[:] = 0.0
    mujoco.mj_forward(uw.model, uw.data)

    # Setup
    uw.finger_target_joint = uw.FINGER_OPEN_JOINT
    uw._set_gripper_state()
    cube_pos = uw.get_cube_position()
    move_arm_to_target(uw, np.array([cube_pos[0], cube_pos[1], uw.HOVER_Z]))

    grasp_result = uw.execute_grasp()
    if not grasp_result["success"]:
        return {
            "success": False,
            "reason": f"GRASP_FAILED: {grasp_result.get('reason')}",
        }

    # Lift to SAFE_Z first
    cube_pos = uw.get_cube_position()
    move_arm_to_target(uw, np.array([cube_pos[0], cube_pos[1], uw.SAFE_Z]))

    # Horizontal transit 100mm
    start_grip = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip").copy()
    target_xy = start_grip[:2] + np.array([0.10, 0.0])
    target = np.array([target_xy[0], target_xy[1], uw.SAFE_Z])

    cube_dropped = False
    drop_reason = None
    max_yaw_deg = 0.0

    for _ in range(200):
        grip_pos = uw._utils.get_site_xpos(uw.model, uw.data, "robot0:grip")
        error = target - grip_pos
        if np.linalg.norm(error) < 0.001:
            break

        step_vec = 0.5 * error
        if np.linalg.norm(step_vec) > 0.005:
            step_vec = step_vec / np.linalg.norm(step_vec) * 0.005
        uw.data.mocap_pos[0][:3] += step_vec
        uw._mujoco_step(None)

        # Check drop
        cube_pos = uw.get_cube_position()
        xy_err = np.linalg.norm(cube_pos[:2] - grip_pos[:2]) * 1000
        z_dev = abs((grip_pos[2] - cube_pos[2]) - 0.015) * 1000

        if xy_err > 30.0 or z_dev > 40.0:
            cube_dropped = True
            drop_reason = f"CUBE_DROPPED (XY={xy_err:.1f}mm, Z_dev={z_dev:.1f}mm)"
            break

        # Check rotation
        quat = uw.get_cube_quat()
        r = scipy.spatial.transform.Rotation.from_quat(
            [quat[1], quat[2], quat[3], quat[0]]
        )
        euler = r.as_euler("xyz", degrees=True)
        max_yaw_deg = max(max_yaw_deg, abs(euler[2]))

        if env.unwrapped.render_mode == "human":
            env.render()

    if debug:
        print(
            f"  [Transit] Success: {not cube_dropped} | Max Yaw: {max_yaw_deg:.1f} deg"
        )

    return {
        "success": not cube_dropped,
        "reason": drop_reason,
        "max_yaw_deg": max_yaw_deg,
    }


def main():
    """Parse command-line arguments and run the script."""
    p = argparse.ArgumentParser()
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    render_mode = "human" if args.visualize else None
    env = gym.make(
        "ChessFetchTask-v0",
        render_mode=render_mode,
        hide_object=False,
        debug=args.debug,
    )
    uw = env.unwrapped

    assert_mandatory_preconditions(uw)

    stats = {
        "static": {"success": 0, "fail_reasons": {}},
        "lift": {"success": 0, "fail_reasons": {}},
        "transit": {"success": 0, "fail_reasons": {}},
    }

    for i in range(args.n_trials):
        print(f"\nTrial {i + 1}/{args.n_trials}...")

        # Test 1
        res = test_static_grasp(env, args.debug)
        if res["success"]:
            stats["static"]["success"] += 1
        else:
            reason = res.get("reason", "UNKNOWN")
            stats["static"]["fail_reasons"][reason] = (
                stats["static"]["fail_reasons"].get(reason, 0) + 1
            )

        # Test 2
        res = test_lift(env, args.debug)
        if res["success"]:
            stats["lift"]["success"] += 1
        else:
            reason = res.get("reason", "UNKNOWN")
            stats["lift"]["fail_reasons"][reason] = (
                stats["lift"]["fail_reasons"].get(reason, 0) + 1
            )

        # Test 3
        res = test_transit_held(env, args.debug)
        if res["success"]:
            stats["transit"]["success"] += 1
        else:
            reason = res.get("reason", "UNKNOWN")
            stats["transit"]["fail_reasons"][reason] = (
                stats["transit"]["fail_reasons"].get(reason, 0) + 1
            )

    print("\n" + "=" * 40)
    print("PHYSICS TEST SUMMARY")
    print("=" * 40)
    for test in ["static", "lift", "transit"]:
        s = stats[test]
        rate = s["success"] / args.n_trials * 100 if args.n_trials > 0 else 0
        print(f"{test.capitalize():<10}: {s['success']}/{args.n_trials} ({rate:.1f}%)")
        if s["fail_reasons"]:
            print(f"  Failures: {s['fail_reasons']}")
    print("=" * 40)

    env.close()


if __name__ == "__main__":
    main()
