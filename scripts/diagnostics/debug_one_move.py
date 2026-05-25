"""
Run one chess move (e2→e3) with dense per-step debug logging.

Logs every physics step during the full pick-and-place sequence to:
  logs/debug_move_TIMESTAMP.jsonl

Each line is a JSON object with:
  step, wall_time_s, phase,
  grip_pos [x,y,z], grip_vel [vx,vy,vz],
  l_finger_pos, r_finger_pos,
  finger_target, mocap_pos [x,y,z], mocap_quat [w,x,y,z],
  arm_joints {name: qpos, ...},
  src_piece_pos [x,y,z], src_piece_vel [vx,vy,vz],
  dst_square_xy [x,y],
  all_pieces_pos {piece_id: [x,y,z], ...}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium

import src.chess_env  # noqa: F401 — registers ChessFetchTask-v0
from src.chess_env.controller import ScriptedController
from src.chess_game.board_mapper import BoardMapper
from src.physical.piece_registry import PieceRegistry


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run one chess move with dense per-step debug logging."
    )
    parser.add_argument(
        "--piece",
        default="white_pawn_e",
        help="Piece ID to move (default: white_pawn_e)",
    )
    parser.add_argument(
        "--src", default="e2", help="Source square in algebraic notation (default: e2)"
    )
    parser.add_argument(
        "--dst",
        default="e3",
        help="Destination square in algebraic notation (default: e3)",
    )
    return parser.parse_args()


_args = _parse_args()
SRC_SQUARE = _args.src
DST_SQUARE = _args.dst
SRC_PIECE_ID = _args.piece

ARM_JOINTS = [
    "robot0:shoulder_pan_joint",
    "robot0:shoulder_lift_joint",
    "robot0:upperarm_roll_joint",
    "robot0:elbow_flex_joint",
    "robot0:forearm_roll_joint",
    "robot0:wrist_flex_joint",
    "robot0:wrist_roll_joint",
]

FINGER_JOINTS = [
    "robot0:l_gripper_finger_joint",
    "robot0:r_gripper_finger_joint",
]


def _piece_pos(env, joint_name: str) -> list[float]:
    """Return piece pos."""
    try:
        joint_id = env.model.joint(joint_name).id
        start = env.model.jnt_qposadr[joint_id]
        return env.data.qpos[start : start + 3].tolist()
    except Exception:
        return [0.0, 0.0, 0.0]


def _piece_vel(env, joint_name: str) -> list[float]:
    """Return piece vel."""
    try:
        joint_id = env.model.joint(joint_name).id
        start = env.model.jnt_dofadr[joint_id]
        return env.data.qvel[start : start + 3].tolist()
    except Exception:
        return [0.0, 0.0, 0.0]


def main() -> None:
    """Parse command-line arguments and run the script."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"debug_move_{ts}.jsonl"

    env = gymnasium.make(
        "ChessFetchTask-v0",
        render_mode=None,  # headless — faster, no display needed
        force_scenario="transit",
        show_chess_pieces=True,
        hide_object=False,
    )
    raw_env = env.unwrapped

    # Build piece/board helpers
    registry = PieceRegistry()
    mapper = BoardMapper.from_configs()

    src_piece = registry.by_id(SRC_PIECE_ID)
    src_joint = src_piece.joint_name
    dst_xy = mapper.square_name_to_xy(DST_SQUARE).tolist()

    all_joint_names = {p.piece_id: p.joint_name for p in registry.all_pieces()}

    # Reset
    env.reset()
    raw_env.set_active_piece(SRC_PIECE_ID)

    step_counter = [0]
    t0 = time.perf_counter()

    records: list[dict] = []

    def log_step(phase: str) -> None:
        """Return log step."""
        env_inner = raw_env
        model, data = env_inner.model, env_inner.data

        grip_pos = env_inner._utils.get_site_xpos(model, data, "robot0:grip").tolist()
        grip_vel = env_inner._utils.get_site_xvelp(model, data, "robot0:grip").tolist()

        l_finger = float(
            env_inner._utils.get_joint_qpos(
                model, data, "robot0:l_gripper_finger_joint"
            ).item()
        )
        r_finger = float(
            env_inner._utils.get_joint_qpos(
                model, data, "robot0:r_gripper_finger_joint"
            ).item()
        )

        arm_joints = {}
        for jname in ARM_JOINTS:
            try:
                arm_joints[jname] = float(
                    env_inner._utils.get_joint_qpos(model, data, jname).item()
                )
            except Exception:
                arm_joints[jname] = 0.0

        src_pos = _piece_pos(env_inner, src_joint)
        src_vel = _piece_vel(env_inner, src_joint)

        all_pos = {
            pid: _piece_pos(env_inner, jn) for pid, jn in all_joint_names.items()
        }

        records.append(
            {
                "step": step_counter[0],
                "wall_time_s": round(time.perf_counter() - t0, 5),
                "phase": phase,
                "grip_pos": grip_pos,
                "grip_vel": grip_vel,
                "l_finger_pos": l_finger,
                "r_finger_pos": r_finger,
                "finger_target": float(getattr(env_inner, "finger_target_joint", 0.0)),
                "mocap_pos": data.mocap_pos[0][:3].tolist(),
                "mocap_quat": data.mocap_quat[0].tolist(),
                "arm_joints": arm_joints,
                "src_piece_pos": src_pos,
                "src_piece_vel": src_vel,
                "dst_square_xy": dst_xy,
                "all_pieces_pos": all_pos,
            }
        )
        step_counter[0] += 1

    raw_env._debug_step_callback = log_step

    # Run the move
    controller = ScriptedController(env)
    src_xy = mapper.square_name_to_xy(SRC_SQUARE)
    dst_xy_np = mapper.square_name_to_xy(DST_SQUARE)

    print(f"Running {SRC_PIECE_ID}: {SRC_SQUARE} → {DST_SQUARE}")
    t_move_start = time.perf_counter()
    result = controller.run_full_move(src_xy, dst_xy_np)
    elapsed = time.perf_counter() - t_move_start

    raw_env._debug_step_callback = None
    env.close()

    # Write log
    with log_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Move result: success={result.success}, failed_at={result.failed_at}")
    for stage_name, stage_res in result.stage_results:
        print(
            f"  {stage_name}: steps={stage_res.steps} success={stage_res.success} "
            f"err={stage_res.error_mm:.1f}mm reason={stage_res.crash_reason}"
        )
    print(f"Total steps logged: {step_counter[0]}  Wall time: {elapsed:.2f}s")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
