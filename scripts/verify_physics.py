import sys
import os
import argparse
import numpy as np
import gymnasium as gym
import mujoco

# Ensure src is importable
sys.path.append(os.getcwd())

from src.chess_env.task import ChessTaskEnv
from src.utils.config import load_config

def test_xml_integrity(debug=False):
    print("Testing XML Integrity...")
    try:
        env = ChessTaskEnv(debug=debug)
        print("  - Model loaded successfully.")
        # Check for key geoms/sites
        expected_sites = ["robot0:grip"]
        for site in expected_sites:
            try:
                env.model.site(site)
                print(f"  - Site '{site}' found.")
            except Exception:
                print(f"  - ERROR: Site '{site}' NOT found.")
                return False
        env.close()
        return True
    except Exception as e:
        print(f"  - ERROR: Failed to load environment: {e}")
        return False

def test_table_geometry(debug=False):
    print("Testing Table Geometry (70x70cm, 4 legs)...")
    try:
        cfg = load_config("env")
        if cfg["table_half_x"] != 0.35 or cfg["table_half_y"] != 0.35:
            print(f"  - ERROR: table_half_x/y expected 0.35, got {cfg['table_half_x']}/{cfg['table_half_y']}")
            return False
            
        env = ChessTaskEnv(debug=debug)
        model = env.model
        # Verify 4 leg geoms exist
        for leg_name in ["table0_leg_far_plus", "table0_leg_far_minus",
                         "table0_leg_near_plus", "table0_leg_near_minus"]:
            try:
                model.geom(leg_name)
                print(f"  - Leg '{leg_name}' found.")
            except Exception:
                print(f"  - ERROR: Leg '{leg_name}' NOT found.")
                env.close()
                return False
        
        # Verify surface top Z = 0.400
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table0_surface")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table0")
        surface_top = model.body_pos[body_id][2] + model.geom_pos[geom_id][2] + model.geom_size[geom_id][2]
        if abs(surface_top - 0.400) > 0.001:
            print(f"  - ERROR: Surface Z = {surface_top:.4f}, expected 0.400")
            env.close()
            return False
        print(f"  - Surface top Z = {surface_top:.4f} (correct)")
        env.close()
        return True
    except Exception as e:
        print(f"  - ERROR during Table Geometry test: {e}")
        return False

def test_static_stability(debug=False):
    print("Testing Static Stability (no-action drift check)...")
    try:
        env = ChessTaskEnv(force_scenario="transit", hide_object=False, debug=debug) # Keep object visible to check drift
        env.reset()
        
        obj_joint_id = env.model.joint("object0:joint").id
        qpos_start = env.model.jnt_qposadr[obj_joint_id]
        
        start_pos = env.data.qpos[qpos_start:qpos_start+3].copy()
        
        for _ in range(100):
            env.step(np.zeros(4)) # No action
            
        end_pos = env.data.qpos[qpos_start:qpos_start+3].copy()
        print(f"  - Start Pos: {start_pos}")
        print(f"  - End Pos:   {end_pos}")
        drift = np.linalg.norm(start_pos - end_pos)
        print(f"  - Object drift over 100 steps: {drift:.6f}m")
        
        env.close()
        return drift < 0.001
    except Exception as e:
        print(f"  - ERROR during Static Stability test: {e}")
        return False

def test_kinematic_reachability(debug=False):
    print("Testing Kinematic Reachability (arm at new position)...")
    try:
        env = ChessTaskEnv(debug=debug)
        env.reset() # Critical: ensures torso is raised
        cfg = load_config("env")
        safe_z = cfg["safe_z"]
        grasp_z = cfg["grasp_z"]
        cx, cy = cfg["table_center_xy"]
        hx, hy = cfg["table_half_x"], cfg["table_half_y"]
        margin = cfg.get("edge_margin", 0.02)
        
        # Test corners and centers
        test_positions = [
            np.array([cx - hx + margin, cy]),               # near edge center
            np.array([cx + hx - margin, cy]),               # far edge center
            np.array([cx, cy - hy + margin]),               # right edge center
            np.array([cx, cy + hy - margin]),               # left edge center
            np.array([cx - hx + margin, cy - hy + margin]), # near-right corner
            np.array([cx + hx - margin, cy + hy - margin]), # far-left corner
        ]
        
        for pt_xy in test_positions:
            # Check safe_z
            pt_safe = np.array([pt_xy[0], pt_xy[1], safe_z])
            env._settle_arm_to_start(pt_safe)
            grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip")
            err = np.linalg.norm(pt_safe - grip_pos)
            print(f"  - Reach safe_z at {pt_xy}: err={err:.6f}m")
            if err > 0.005: 
                print(f"    ERROR: High error reaching safe_z")
                env.close()
                return False
            
            # Check grasp_z
            pt_grasp = np.array([pt_xy[0], pt_xy[1], grasp_z])
            env._settle_arm_to_start(pt_grasp)
            grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip")
            err = np.linalg.norm(pt_grasp - grip_pos)
            print(f"  - Reach grasp_z at {pt_xy}: err={err:.6f}m")
            if err > 0.005:
                print(f"    ERROR: High error reaching grasp_z")
                env.close()
                return False
            
        env.close()
        return True
    except Exception as e:
        print(f"  - ERROR during Kinematic Reachability test: {e}")
        return False

def test_teleport_verification(debug=False):
    print("Testing Teleport Verification (object-hiding logic)...")
    try:
        env = ChessTaskEnv(hide_object=True, debug=debug)
        env.reset()
        
        # Check immediately after reset
        obj_joint_id = env.model.joint("object0:joint").id
        qpos_start = env.model.jnt_qposadr[obj_joint_id]
        obj_pos = env.data.qpos[qpos_start:qpos_start+3].copy()
        
        hidden_pos = np.array(load_config("env")["hidden_object_pos"])
        dist = np.linalg.norm(obj_pos - hidden_pos)
        
        print(f"  - Object position: {obj_pos}, Hidden target: {hidden_pos}")
        print(f"  - Distance to hidden target: {dist:.6f}m")
        
        env.close()
        # Stable position on floor should be very accurate
        return dist < 0.001
    except Exception as e:
        print(f"  - ERROR during Teleport Verification test: {e}")
        return False

def verify_grasp_xml_changes(debug=False):
    print("Verifying Grasp-Stage XML changes...")
    try:
        env = ChessTaskEnv(debug=debug)
        model = env.model
        uw = env.unwrapped
        
        # 1. Cube mass
        cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object0")
        mass = model.body_mass[cube_body_id]
        print(f"  - Cube mass: {mass:.3f}kg")
        assert abs(mass - 0.05) < 0.001, f"Expected 0.05, got {mass:.3f}"
        
        # 2. Actuator Kp (Updated to 20000 in Stage 3)
        l_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_joint")
        kp = model.actuator_gainprm[l_act_id, 0]
        print(f"  - Actuator Kp: {kp}")
        assert abs(kp - 20000) < 1.0, f"Expected 20000, got {kp}"
        
        # 3. Actuator ctrlrange
        ctrl_max = model.actuator_ctrlrange[l_act_id, 1]
        print(f"  - Actuator ctrlrange max: {ctrl_max:.3f}")
        assert abs(ctrl_max - 0.05) < 0.001, f"Expected 0.05, got {ctrl_max:.3f}"
        
        # 4. GRASP_Z (Updated to 0.430 in Stage 3)
        print(f"  - GRASP_Z: {uw.GRASP_Z:.3f}m")
        assert abs(uw.GRASP_Z - 0.430) < 0.001, f"Expected 0.430, got {uw.GRASP_Z:.3f}"
        
        env.close()
        return True
    except Exception as e:
        print(f"  - ERROR during Grasp XML verification: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Automated system health and stability check.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging in the environment.")
    args = parser.parse_args()

    print("--- RoboChess Physics Verification ---")
    results = {
        "XML Integrity": test_xml_integrity(debug=args.debug),
        "Table Geometry": test_table_geometry(debug=args.debug),
        "Grasp XML Verification": verify_grasp_xml_changes(debug=args.debug),
        "Static Stability": test_static_stability(debug=args.debug),
        "Kinematic Reachability": test_kinematic_reachability(debug=args.debug),
        "Teleport Verification": test_teleport_verification(debug=args.debug)
    }
    
    print("\nSummary:")
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{name:<25}: {status}")
        if not passed: all_passed = False
        
    if all_passed:
        print("\nSYSTEM HEALTHY")
    else:
        print("\nSYSTEM UNHEALTHY - check failures above")
        sys.exit(1)

if __name__ == "__main__":
    main()
