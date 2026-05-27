"""robo-chess-eval-physics — MuJoCo physics and geometry integrity verification."""

from __future__ import annotations

import argparse
import sys

import chess
import mujoco
import numpy as np

from src.chess_env.task import ChessTaskEnv
from src.chess_game.board_mapper import BoardMapper
from src.physical.piece_registry import PieceRegistry, reserve_piece_ids
from src.utils.io import load_config


# ── Individual test functions ──────────────────────────────────────────────
# Each returns True on pass, False on failure.


def test_xml_integrity(debug: bool = False, render_mode: str | None = None) -> bool:
    """Load the MuJoCo model; verify required sites exist and legacy sites are absent."""
    print("Testing XML Integrity...")
    try:
        env = ChessTaskEnv(debug=debug, render_mode=render_mode)
        for site in ["robot0:grip"]:
            try:
                env.model.site(site)
                print(f"  - Site '{site}' found.")
            except Exception:
                print(f"  - ERROR: Site '{site}' NOT found.")
                env.close()
                return False
        target_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "target0")
        if target_id != -1:
            print("  - ERROR: Fetch target0 red-dot site is still present.")
            env.close()
            return False
        print("  - Fetch target0 red-dot site absent.")
        env.close()
        return True
    except Exception as exc:
        print(f"  - ERROR: Failed to load environment: {exc}")
        return False


def test_table_geometry(debug: bool = False, render_mode: str | None = None) -> bool:
    """Verify table is 70×70 cm with 4 legs and surface Z = 0.400 m."""
    print("Testing Table Geometry (70x70cm, 4 legs)...")
    try:
        cfg = load_config("env")
        if cfg["table_half_x"] != 0.35 or cfg["table_half_y"] != 0.35:
            print(
                f"  - ERROR: table_half_x/y expected 0.35, "
                f"got {cfg['table_half_x']}/{cfg['table_half_y']}"
            )
            return False

        env = ChessTaskEnv(debug=debug, render_mode=render_mode)
        model = env.model
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
    except Exception as exc:
        print(f"  - ERROR during Table Geometry test: {exc}")
        return False


def test_board_visual_geometry(debug: bool = False, render_mode: str | None = None) -> bool:
    """Verify 64 board square geoms exist, are spaced exactly 8 cm apart, and match BoardMapper."""
    print("Testing Board Visual Geometry (64 non-colliding exact-8cm squares)...")
    try:
        env = ChessTaskEnv(debug=debug, render_mode=render_mode)
        model = env.model
        mapper = BoardMapper.from_configs()
        env_cfg = load_config("env")
        table_cx, table_cy = env_cfg["table_center_xy"]

        table_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table0")
        surface_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table0_surface")
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
            expected_xy = mapper.square_name_to_xy(square_name)
            geom_name = f"board_sq_{square_name}"
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id == -1:
                print(f"  - ERROR: Missing board square geom '{geom_name}'.")
                env.close()
                return False
            geom_pos = model.geom_pos[geom_id]
            center_xy = np.array([geom_pos[0] + table_cx, geom_pos[1] + table_cy])
            err = float(np.linalg.norm(center_xy - expected_xy))
            max_center_error = max(max_center_error, err)

        print(f"  - Max board square center error: {max_center_error * 1000:.3f}mm")
        if max_center_error > 0.001:
            print(f"  - ERROR: Max center error {max_center_error * 1000:.3f}mm exceeds 1mm.")
            env.close()
            return False

        print("  - All 64 squares present and correctly positioned.")
        env.close()
        return True
    except Exception as exc:
        print(f"  - ERROR during Board Visual Geometry test: {exc}")
        return False


def test_zone_visual_geometry(debug: bool = False, render_mode: str | None = None) -> bool:
    """Verify off-board graveyard and reserve zone marker geoms are world-space and visual-only."""
    print("Testing Off-Board Zone Visual Geometry...")
    try:
        env = ChessTaskEnv(debug=debug, render_mode=render_mode)
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
                print(f"  - ERROR: {geom_name} must be world-space (body 0).")
                env.close()
                return False
            if model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0:
                print(f"  - ERROR: {geom_name} must be visual-only (contype=0 conaffinity=0).")
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
        print("  - Found 4 world-space visual zone markers aligned with configured slots.")
        env.close()
        return True
    except Exception as exc:
        print(f"  - ERROR during Zone Visual Geometry test: {exc}")
        return False


def test_chess_piece_modeling(debug: bool = False, render_mode: str | None = None) -> bool:
    """Verify all 32 active pieces and 64 reserve pieces exist with correct bodies and geometry."""
    print("Testing Chess Piece Modeling (32 active + 64 reserve bodies)...")
    try:
        env = ChessTaskEnv(debug=debug, render_mode=render_mode, show_chess_pieces=True, hide_object=True)
        env.reset()
        model = env.model
        registry = PieceRegistry()
        mapper = BoardMapper.from_configs()
        damping = load_config("chess")["pieces"]["freejoint_damping"]

        max_start_error = 0.0
        for piece in registry.all_pieces():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, piece.body_name)
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, piece.joint_name)
            cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, piece.cube_geom_name)
            visual_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, piece.visual_geom_name)
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
            if model.geom_contype[visual_id] != 0 or model.geom_conaffinity[visual_id] != 0:
                print(f"  - ERROR: {piece.visual_geom_name} must be visual-only.")
                env.close()
                return False
            qpos_start = model.jnt_qposadr[joint_id]
            actual = env.data.qpos[qpos_start : qpos_start + 3]
            expected = mapper.square_to_piece_xyz(chess.parse_square(piece.initial_square))
            max_start_error = max(max_start_error, float(np.linalg.norm(actual - expected)))

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
    except Exception as exc:
        print(f"  - ERROR during Chess Piece Modeling test: {exc}")
        return False


def test_chess_piece_idle_stability(settle_steps: int = 2500, debug: bool = False, render_mode: str | None = None) -> bool:
    """
    Step the simulation `settle_steps` times with no arm actions and verify
    all chess pieces drift less than 1 mm from their initial positions.
    """
    print(
        f"Testing Chess Piece Idle Stability ({settle_steps} steps / no-action drift check)..."
    )
    try:
        env = ChessTaskEnv(debug=debug, render_mode=render_mode, show_chess_pieces=True, hide_object=True)
        env.reset()
        registry = PieceRegistry()

        starts: dict[str, np.ndarray] = {}
        for piece in registry.all_pieces():
            joint_id = env.model.joint(piece.joint_name).id
            qpos_start = env.model.jnt_qposadr[joint_id]
            starts[piece.piece_id] = env.data.qpos[qpos_start : qpos_start + 3].copy()

        for _ in range(settle_steps):
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
            f"  - Max active piece drift over {settle_steps} steps: "
            f"{max_drift * 1000:.3f}mm ({worst_piece})"
        )
        env.close()
        if max_drift >= 0.001:
            print("  - ERROR: Piece drift exceeds 1mm threshold.")
            return False
        return True
    except Exception as exc:
        print(f"  - ERROR during Chess Piece Idle Stability test: {exc}")
        return False


def test_static_stability(debug: bool = False, render_mode: str | None = None) -> bool:
    """Step 100 times with zero action; object must not drift more than 1 mm."""
    print("Testing Static Stability (object zero-action drift check)...")
    try:
        env = ChessTaskEnv(force_scenario="transit", hide_object=False, debug=debug, render_mode=render_mode)
        env.reset()

        obj_joint_id = env.model.joint("object0:joint").id
        qpos_start = env.model.jnt_qposadr[obj_joint_id]
        start_pos = env.data.qpos[qpos_start : qpos_start + 3].copy()

        for _ in range(100):
            env.step(np.zeros(4))

        end_pos = env.data.qpos[qpos_start : qpos_start + 3].copy()
        drift = float(np.linalg.norm(start_pos - end_pos))
        print(f"  - Object drift over 100 steps: {drift * 1000:.3f}mm")
        env.close()
        if drift >= 0.001:
            print("  - ERROR: Object drift exceeds 1mm threshold.")
            return False
        return True
    except Exception as exc:
        print(f"  - ERROR during Static Stability test: {exc}")
        return False


def test_kinematic_reachability(debug: bool = False, render_mode: str | None = None) -> bool:
    """
    Settle the arm to every board square at SAFE_Z and GRASP_Z.
    All 64 × 2 positions must be reachable within 5 mm.
    """
    print("Testing Kinematic Reachability (all 64 exact-8cm chess squares)...")
    try:
        env = ChessTaskEnv(debug=debug, render_mode=render_mode)
        env.reset()
        cfg = load_config("env")
        safe_z = cfg["safe_z"]
        grasp_z = cfg["grasp_z"]
        mapper = BoardMapper.from_configs()
        squares = mapper.all_square_centers()

        THRESHOLD = 0.005
        max_err = 0.0
        worst: tuple | None = None

        for square_name, xy in squares.items():
            for z, z_name in [(safe_z, "safe_z"), (grasp_z, "grasp_z")]:
                target = np.array([xy[0], xy[1], z])
                env._settle_arm_to_start(target)
                grip_pos = env._utils.get_site_xpos(env.model, env.data, "robot0:grip")
                err = float(np.linalg.norm(target - grip_pos))
                if err > max_err:
                    max_err = err
                    worst = (square_name, z_name, xy[0], xy[1], err)

        print(f"  - Max error across 64 squares × 2 heights: {max_err * 1000:.1f}mm")
        if worst:
            sn, zn, sx, sy, e = worst
            print(
                f"  - Worst: square={sn} ({zn}) pos=({sx:.3f},{sy:.3f}) "
                f"err={e * 1000:.1f}mm"
            )
        env.close()
        if max_err > THRESHOLD:
            print(f"  - ERROR: max_err {max_err * 1000:.1f}mm exceeds 5mm threshold.")
            return False
        print(f"  - All 64 squares reachable within {THRESHOLD * 1000:.0f}mm.")
        return True
    except Exception as exc:
        print(f"  - ERROR during Kinematic Reachability test: {exc}")
        return False


def test_grasp_xml(debug: bool = False, render_mode: str | None = None) -> bool:
    """Verify grasp-stage XML parameters: cube mass, actuator Kp, ctrlrange, GRASP_Z."""
    print("Verifying Grasp-Stage XML parameters...")
    try:
        env = ChessTaskEnv(debug=debug, render_mode=render_mode)
        model = env.model
        uw = env.unwrapped

        cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object0")
        mass = model.body_mass[cube_body_id]
        print(f"  - Cube mass: {mass:.3f}kg")
        if abs(mass - 0.05) >= 0.001:
            print(f"  - ERROR: Expected cube mass 0.05, got {mass:.3f}")
            env.close()
            return False

        l_act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robot0:l_gripper_finger_joint"
        )
        kp = model.actuator_gainprm[l_act_id, 0]
        print(f"  - Actuator Kp: {kp}")
        if abs(kp - 20000) >= 1.0:
            print(f"  - ERROR: Expected actuator Kp 20000, got {kp}")
            env.close()
            return False

        ctrl_max = model.actuator_ctrlrange[l_act_id, 1]
        print(f"  - Actuator ctrlrange max: {ctrl_max:.3f}")
        if abs(ctrl_max - 0.05) >= 0.001:
            print(f"  - ERROR: Expected ctrlrange max 0.05, got {ctrl_max:.3f}")
            env.close()
            return False

        print(f"  - GRASP_Z: {uw.GRASP_Z:.3f}m")
        if abs(uw.GRASP_Z - 0.430) >= 0.001:
            print(f"  - ERROR: Expected GRASP_Z 0.430, got {uw.GRASP_Z:.3f}")
            env.close()
            return False

        env.close()
        return True
    except Exception as exc:
        print(f"  - ERROR during Grasp XML verification: {exc}")
        return False


def test_teleport_verification(debug: bool = False, render_mode: str | None = None) -> bool:
    """After reset with hide_object=True, cube must be within 1 mm of the hidden position."""
    print("Testing Teleport Verification (object-hiding logic)...")
    try:
        env = ChessTaskEnv(hide_object=True, debug=debug, render_mode=render_mode)
        env.reset()

        obj_joint_id = env.model.joint("object0:joint").id
        qpos_start = env.model.jnt_qposadr[obj_joint_id]
        obj_pos = env.data.qpos[qpos_start : qpos_start + 3].copy()

        hidden_pos = np.array(load_config("env")["hidden_object_pos"])
        dist = float(np.linalg.norm(obj_pos - hidden_pos))
        print(f"  - Distance to hidden target: {dist * 1000:.3f}mm")
        env.close()
        if dist >= 0.001:
            print("  - ERROR: Object is not hidden correctly.")
            return False
        return True
    except Exception as exc:
        print(f"  - ERROR during Teleport Verification test: {exc}")
        return False


def _run_check(results: dict[str, bool], name: str, fn, **kwargs) -> None:
    """Run one physics check and record a boolean result."""
    try:
        results[name] = bool(fn(**kwargs))
    except Exception as exc:
        print(f"  - ERROR during {name}: {exc}")
        results[name] = False


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for robo-chess-eval-physics."""
    p = argparse.ArgumentParser(
        prog="robo-chess-eval-physics",
        description=(
            "Verify MuJoCo physics assets, scene geometry, and simulation stability. "
            "Exits 0 if all checks pass, 1 if any fail."
        ),
    )
    p.add_argument(
        "--settle-steps",
        type=int,
        default=2500,
        help=(
            "Simulation steps for the piece-idle-stability settle loop "
            "(default: 2500 ≈ 5 simulated seconds)."
        ),
    )
    p.add_argument(
        "--skip-geometry",
        action="store_true",
        help="Skip all static geometry checks (XML, table, board squares, zones, pieces).",
    )
    p.add_argument(
        "--skip-stability",
        action="store_true",
        help="Skip piece idle stability and static stability settle loops.",
    )
    p.add_argument(
        "--skip-reachability",
        action="store_true",
        help="Skip kinematic reachability check (fastest to skip for quick runs).",
    )
    p.add_argument(
        "--visualize",
        action="store_true",
        help="Open the MuJoCo viewer window during each check.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose environment debug logging.",
    )
    args = p.parse_args()

    render_mode = "human" if args.visualize else None

    print("--- RoboChess Physics Verification ---")

    results: dict[str, bool] = {}

    if not args.skip_geometry:
        for name, fn in [
            ("XML Integrity", test_xml_integrity),
            ("Table Geometry", test_table_geometry),
            ("Board Visual Geometry", test_board_visual_geometry),
            ("Zone Visual Geometry", test_zone_visual_geometry),
            ("Chess Piece Modeling", test_chess_piece_modeling),
            ("Grasp XML Parameters", test_grasp_xml),
            ("Teleport Verification", test_teleport_verification),
        ]:
            _run_check(results, name, fn, debug=args.debug, render_mode=render_mode)

    if not args.skip_stability:
        _run_check(
            results,
            "Chess Piece Idle Stability",
            test_chess_piece_idle_stability,
            settle_steps=args.settle_steps,
            debug=args.debug,
            render_mode=render_mode,
        )
        _run_check(
            results,
            "Static Stability",
            test_static_stability,
            debug=args.debug,
            render_mode=render_mode,
        )

    if not args.skip_reachability:
        _run_check(
            results,
            "Kinematic Reachability",
            test_kinematic_reachability,
            debug=args.debug,
            render_mode=render_mode,
        )

    if not results:
        print("\nNo checks ran — all categories were skipped.")
        sys.exit(0)

    print("\nSummary:")
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name:<30s}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nRESULT: SYSTEM HEALTHY")
        sys.exit(0)
    else:
        print("\nRESULT: SYSTEM UNHEALTHY — check failures above")
        sys.exit(1)


if __name__ == "__main__":
    main()
