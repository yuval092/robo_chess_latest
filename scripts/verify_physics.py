"""Verify MuJoCo physics assets and chess scene geometry."""
import argparse
import sys

import chess
import mujoco
import numpy as np

# Ensure src is importable
from src.chess_env.task import ChessTaskEnv
from src.chess_game.board_mapper import BoardMapper
from src.physical.piece_registry import PieceRegistry, reserve_piece_ids
from src.utils.io import load_config


def test_xml_integrity(debug=False):
    """Run the xml integrity check."""
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
        target_site_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_SITE, "target0"
        )
        if target_site_id != -1:
            print(
                "  - ERROR: Fetch target0 red-dot site is still present in loaded chess scene."
            )
            env.close()
            return False
        print("  - Fetch target0 red-dot site absent.")
        env.close()
        return True
    except Exception as e:
        print(f"  - ERROR: Failed to load environment: {e}")
        return False


def test_table_geometry(debug=False):
    """Run the table geometry check."""
    print("Testing Table Geometry (70x70cm, 4 legs)...")
    try:
        cfg = load_config("env")
        if cfg["table_half_x"] != 0.35 or cfg["table_half_y"] != 0.35:
            print(
                f"  - ERROR: table_half_x/y expected 0.35, got {cfg['table_half_x']}/{cfg['table_half_y']}"
            )
            return False

        env = ChessTaskEnv(debug=debug)
        model = env.model
        # Verify 4 leg geoms exist
        for leg_name in [
            "table0_leg_far_plus",
            "table0_leg_far_minus",
            "table0_leg_near_plus",
            "table0_leg_near_minus",
        ]:
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
        surface_top = (
            model.body_pos[body_id][2]
            + model.geom_pos[geom_id][2]
            + model.geom_size[geom_id][2]
        )
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


def test_board_visual_geometry(debug=False):
    """Run the board visual geometry check."""
    print("Testing Board Visual Geometry (64 non-colliding exact-8cm squares)...")
    try:
        env = ChessTaskEnv(debug=debug)
        model = env.model
        mapper = BoardMapper.from_configs()
        env_cfg = load_config("env")
        table_cx, table_cy = env_cfg["table_center_xy"]

        table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table0")
        surface_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "table0_surface"
        )
        if surface_id == -1:
            print("  - ERROR: table0_surface geom missing.")
            env.close()
            return False
        if model.geom_bodyid[surface_id] != table_body_id:
            print("  - ERROR: table0_surface is not attached to table0.")
            env.close()
            return False

        max_center_error = 0.0
        for square in chess.SQUARES:
            square_name = chess.square_name(square)
            geom_name = f"board_{square_name}_visual"
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id == -1:
                print(f"  - ERROR: Missing board square geom {geom_name}.")
                env.close()
                return False
            if model.geom_bodyid[geom_id] != table_body_id:
                print(f"  - ERROR: {geom_name} is not attached to table0.")
                env.close()
                return False
            if model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0:
                print(
                    f"  - ERROR: {geom_name} must be visual-only contype=0 conaffinity=0."
                )
                env.close()
                return False

            expected_xy = mapper.square_to_xy(square)
            expected_local = np.array(
                [expected_xy[0] - table_cx, expected_xy[1] - table_cy]
            )
            actual_local = model.geom_pos[geom_id][:2]
            center_error = float(np.linalg.norm(actual_local - expected_local))
            max_center_error = max(max_center_error, center_error)
            if center_error > 0.0002:
                print(
                    f"  - ERROR: {geom_name} local XY {actual_local} does not match "
                    f"BoardMapper {expected_local}."
                )
                env.close()
                return False
            if not np.allclose(
                model.geom_size[geom_id], [0.04, 0.04, 0.0005], atol=1e-6
            ):
                print(
                    f"  - ERROR: {geom_name} has unexpected size {model.geom_size[geom_id]}."
                )
                env.close()
                return False

        print("  - Found 64 board square visual geoms.")
        print(f"  - Max square center mismatch: {max_center_error * 1000:.3f}mm")
        print(
            "  - Board square geoms are visual-only and table0_surface remains the support geom."
        )
        env.close()
        return True
    except Exception as e:
        print(f"  - ERROR during Board Visual Geometry test: {e}")
        return False


def test_zone_visual_geometry(debug=False):
    """Run the zone visual geometry check."""
    print("Testing Off-Board Zone Visual Geometry...")
    try:
        env = ChessTaskEnv(debug=debug)
        model = env.model
        cfg = load_config("chess")
        zones = [
            (
                "zone_white_graveyard",
                "graveyards",
                "white",
                cfg["reserves"]["graveyard_slot_spacing_m"],
            ),
            (
                "zone_black_graveyard",
                "graveyards",
                "black",
                cfg["reserves"]["graveyard_slot_spacing_m"],
            ),
            (
                "zone_white_reserve",
                "promotion_reserve",
                "white",
                cfg["reserves"]["promotion_slot_spacing_m"],
            ),
            (
                "zone_black_reserve",
                "promotion_reserve",
                "black",
                cfg["reserves"]["promotion_slot_spacing_m"],
            ),
        ]
        for geom_name, section, color, spacing in zones:
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id == -1:
                print(f"  - ERROR: Missing zone geom {geom_name}.")
                env.close()
                return False
            if model.geom_bodyid[geom_id] != 0:
                print(
                    f"  - ERROR: {geom_name} must be in world coordinates, not inside a body."
                )
                env.close()
                return False
            if model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0:
                print(
                    f"  - ERROR: {geom_name} must be visual-only contype=0 conaffinity=0."
                )
                env.close()
                return False
            zone_cfg = cfg[section][color]
            rows = zone_cfg["rows"]
            cols = zone_cfg["cols"]
            origin = zone_cfg["origin_xyz"]
            expected_pos = np.array(
                [
                    origin[0] + (rows - 1) * spacing / 2.0,
                    origin[1] + (cols - 1) * spacing / 2.0,
                    0.002,
                ]
            )
            expected_size = np.array(
                [
                    (rows * spacing) / 2.0 + 0.005,
                    (cols * spacing) / 2.0 + 0.005,
                    0.002,
                ]
            )
            if not np.allclose(model.geom_pos[geom_id], expected_pos, atol=1e-6):
                print(
                    f"  - ERROR: {geom_name} pos {model.geom_pos[geom_id]} != {expected_pos}."
                )
                env.close()
                return False
            if not np.allclose(model.geom_size[geom_id], expected_size, atol=1e-6):
                print(
                    f"  - ERROR: {geom_name} size {model.geom_size[geom_id]} != {expected_size}."
                )
                env.close()
                return False
        print(
            "  - Found 4 world-space visual zone markers aligned with configured slots."
        )
        env.close()
        return True
    except Exception as e:
        print(f"  - ERROR during Zone Visual Geometry test: {e}")
        return False


def test_chess_piece_modeling(debug=False):
    """Run the chess piece modeling check."""
    print("Testing Chess Piece Modeling (32 active + 64 reserve bodies)...")
    try:
        env = ChessTaskEnv(debug=debug, show_chess_pieces=True, hide_object=True)
        env.reset()
        model = env.model
        registry = PieceRegistry()
        mapper = BoardMapper.from_configs()
        damping = load_config("chess")["pieces"]["freejoint_damping"]

        max_start_error = 0.0
        for piece in registry.all_pieces():
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, piece.body_name
            )
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, piece.joint_name
            )
            cube_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, piece.cube_geom_name
            )
            visual_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, piece.visual_geom_name
            )
            if min(body_id, joint_id, cube_id, visual_id) == -1:
                print(f"  - ERROR: Missing model object for {piece.piece_id}.")
                env.close()
                return False
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                print(f"  - ERROR: {piece.joint_name} is not a freejoint.")
                env.close()
                return False
            dof_start = model.jnt_dofadr[joint_id]
            if not np.allclose(model.dof_damping[dof_start : dof_start + 6], damping):
                print(f"  - ERROR: {piece.joint_name} damping is not {damping}.")
                env.close()
                return False
            if not np.allclose(model.geom_size[cube_id], [0.015, 0.015, 0.015]):
                print(f"  - ERROR: {piece.cube_geom_name} has wrong cube size.")
                env.close()
                return False
            if (
                model.geom_contype[visual_id] != 0
                or model.geom_conaffinity[visual_id] != 0
            ):
                print(f"  - ERROR: {piece.visual_geom_name} must be visual-only.")
                env.close()
                return False

            qpos_start = model.jnt_qposadr[joint_id]
            actual = env.data.qpos[qpos_start : qpos_start + 3]
            expected = mapper.square_to_piece_xyz(
                chess.parse_square(piece.initial_square)
            )
            max_start_error = max(
                max_start_error, float(np.linalg.norm(actual - expected))
            )

        for piece_id, _, _ in reserve_piece_ids():
            body_name = f"piece_{piece_id}"
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) == -1:
                print(f"  - ERROR: Missing reserve body {body_name}.")
                env.close()
                return False

        if max_start_error > 0.002:
            print(
                f"  - ERROR: Active piece start error {max_start_error * 1000:.1f}mm exceeds 2mm."
            )
            env.close()
            return False

        print("  - Found all 32 active pieces and 64 reserve pieces.")
        print(f"  - Max active start error: {max_start_error * 1000:.2f}mm")
        env.close()
        return True
    except Exception as e:
        print(f"  - ERROR during Chess Piece Modeling test: {e}")
        return False


def test_chess_piece_idle_stability(debug=False):
    """Run the chess piece idle stability check."""
    print("Testing Chess Piece Idle Stability (5s visible-board drift check)...")
    try:
        env = ChessTaskEnv(debug=debug, show_chess_pieces=True, hide_object=True)
        env.reset()
        registry = PieceRegistry()

        starts = {}
        for piece in registry.all_pieces():
            joint_id = env.model.joint(piece.joint_name).id
            qpos_start = env.model.jnt_qposadr[joint_id]
            starts[piece.piece_id] = env.data.qpos[qpos_start : qpos_start + 3].copy()

        for _ in range(2500):
            env._mujoco_step(None)

        max_drift = 0.0
        worst_piece = None
        for piece in registry.all_pieces():
            joint_id = env.model.joint(piece.joint_name).id
            qpos_start = env.model.jnt_qposadr[joint_id]
            current = env.data.qpos[qpos_start : qpos_start + 3].copy()
            drift = float(np.linalg.norm(current - starts[piece.piece_id]))
            if drift > max_drift:
                max_drift = drift
                worst_piece = piece.piece_id

        print(
            f"  - Max active piece drift over 5s: {max_drift * 1000:.3f}mm ({worst_piece})"
        )
        env.close()
        return max_drift < 0.001
    except Exception as e:
        print(f"  - ERROR during Chess Piece Idle Stability test: {e}")
        return False


def test_static_stability(debug=False):
    """Run the static stability check."""
    print("Testing Static Stability (no-action drift check)...")
    try:
        env = ChessTaskEnv(
            force_scenario="transit", hide_object=False, debug=debug
        )  # Keep object visible to check drift
        env.reset()

        obj_joint_id = env.model.joint("object0:joint").id
        qpos_start = env.model.jnt_qposadr[obj_joint_id]

        start_pos = env.data.qpos[qpos_start : qpos_start + 3].copy()

        for _ in range(100):
            env.step(np.zeros(4))  # No action

        end_pos = env.data.qpos[qpos_start : qpos_start + 3].copy()
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
    """Run the kinematic reachability check."""
    print("Testing Kinematic Reachability (all 64 exact-8cm chess squares)...")
    try:
        env = ChessTaskEnv(debug=debug)
        env.reset()  # Critical: ensures torso is raised
        cfg = load_config("env")
        safe_z = cfg["safe_z"]
        grasp_z = cfg["grasp_z"]
        mapper = BoardMapper.from_configs()
        squares = mapper.all_square_centers()

        max_err = 0.0
        worst = None
        THRESHOLD = 0.005  # 5mm

        for square_name, xy in squares.items():
            for z, z_name in [(safe_z, "safe_z"), (grasp_z, "grasp_z")]:
                target = np.array([xy[0], xy[1], z])
                env._settle_arm_to_start(target)
                grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip")
                err = np.linalg.norm(target - grip_pos)
                if err > max_err:
                    max_err = err
                    worst = (square_name, z_name, xy[0], xy[1], err)

        print(f"  - Max error across 64 squares × 2 heights: {max_err * 1000:.1f}mm")
        if worst:
            square_name, zn, sx, sy, e = worst
            print(
                f"  - Worst: square={square_name} ({zn}) pos=({sx:.3f},{sy:.3f}) err={e * 1000:.1f}mm"
            )

        if max_err > THRESHOLD:
            print(f"  ERROR: max_err={max_err * 1000:.1f}mm exceeds 5mm threshold.")
            env.close()
            return False

        print(
            f"  - All 64 squares reachable within {THRESHOLD * 1000:.0f}mm threshold."
        )
        env.close()
        return True
    except Exception as e:
        print(f"  - ERROR during Kinematic Reachability test: {e}")
        return False


def test_teleport_verification(debug=False):
    """Run the teleport verification check."""
    print("Testing Teleport Verification (object-hiding logic)...")
    try:
        env = ChessTaskEnv(hide_object=True, debug=debug)
        env.reset()

        # Check immediately after reset
        obj_joint_id = env.model.joint("object0:joint").id
        qpos_start = env.model.jnt_qposadr[obj_joint_id]
        obj_pos = env.data.qpos[qpos_start : qpos_start + 3].copy()

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
    """Return verify grasp xml changes."""
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
        l_act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_joint"
        )
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
    """Parse command-line arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Automated system health and stability check."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging in the environment."
    )
    args = parser.parse_args()

    print("--- RoboChess Physics Verification ---")
    results = {
        "XML Integrity": test_xml_integrity(debug=args.debug),
        "Table Geometry": test_table_geometry(debug=args.debug),
        "Board Visual Geometry": test_board_visual_geometry(debug=args.debug),
        "Zone Visual Geometry": test_zone_visual_geometry(debug=args.debug),
        "Chess Piece Modeling": test_chess_piece_modeling(debug=args.debug),
        "Chess Piece Idle Stability": test_chess_piece_idle_stability(debug=args.debug),
        "Grasp XML Verification": verify_grasp_xml_changes(debug=args.debug),
        "Static Stability": test_static_stability(debug=args.debug),
        "Kinematic Reachability": test_kinematic_reachability(debug=args.debug),
        "Teleport Verification": test_teleport_verification(debug=args.debug),
    }

    print("\nSummary:")
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{name:<25}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nSYSTEM HEALTHY")
    else:
        print("\nSYSTEM UNHEALTHY - check failures above")
        sys.exit(1)


if __name__ == "__main__":
    main()
