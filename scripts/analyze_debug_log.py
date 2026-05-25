"""
Analyze a debug_one_move JSONL log and produce a Markdown report.

Usage:
  python scripts/analyze_debug_log.py logs/debug_move_TIMESTAMP.jsonl

Output:
  logs/debug_analysis_TIMESTAMP.md
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PHYSICS_DT = 0.002  # seconds per step

# Expected upper-bound step counts per phase (for stall detection)
PHASE_EXPECTED_STEPS = {
    "transit": 300,
    "descend": 100,
    "grasp_p0_halt": 30,
    "grasp_p12_align": 100,
    "grasp_p3_plunge": 50,
    "grasp_p4_close": 24,
    "grasp_p5_hold": 10,
    "grasp_p6_retract": 80,
    "softreset_p1_halt": 30,
    "softreset_p2_align": 80,
    "softreset_p3_state": 5,
    "softreset_p4_gripper": 50,
    "place_p0_halt": 20,
    "place_p12_align": 100,
    "place_p3_plunge": 50,
    "place_p4_release": 8,
    "place_p5_verify": 5,
    "place_p6_retract": 80,
    "ascend": 100,
}

STALL_VEL_THRESHOLD = 0.002  # m/s — grip speed below this is a stall
STALL_MIN_STEPS = 10  # consecutive below-threshold steps = stall
PIECE_VIBRATION_THRESHOLD = 0.003  # m/s — piece velocity while the arm is stationary
STALL_PHASES = {
    "transit",
    "descend",
    "ascend",
    "grasp_p12_align",
    "grasp_p3_plunge",
    "grasp_p6_retract",
    "softreset_p2_align",
    "place_p12_align",
    "place_p3_plunge",
    "place_p6_retract",
}


def load_records(path: Path) -> list[dict]:
    """Load records."""
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def phase_segments(records: list[dict]) -> list[tuple[str, list[dict]]]:
    """Split records into contiguous (phase_name, [records]) segments."""
    segments: list[tuple[str, list[dict]]] = []
    for rec in records:
        phase = rec["phase"]
        if segments and segments[-1][0] == phase:
            segments[-1][1].append(rec)
        else:
            segments.append((phase, [rec]))
    return segments


def grip_speed(rec: dict) -> float:
    """Return grip speed."""
    v = rec["grip_vel"]
    return float(np.linalg.norm(v))


def detect_stalls(recs: list[dict]) -> list[dict]:
    """Return list of stall events: {start_step, end_step, phase, duration_s}."""
    stalls = []
    in_stall = False
    stall_start = None
    stall_phase = None
    for rec in recs:
        if rec["phase"] not in STALL_PHASES:
            slow = False
        else:
            slow = grip_speed(rec) < STALL_VEL_THRESHOLD
        if in_stall and rec["phase"] != stall_phase:
            span = rec["step"] - stall_start
            if span >= STALL_MIN_STEPS:
                stalls.append(
                    {
                        "start_step": stall_start,
                        "end_step": rec["step"],
                        "phase": stall_phase,
                        "duration_s": round(span * PHYSICS_DT, 4),
                        "steps": span,
                    }
                )
            in_stall = False
            stall_start = None
            stall_phase = None
        if slow and not in_stall:
            in_stall = True
            stall_start = rec["step"]
            stall_phase = rec["phase"]
        elif not slow and in_stall:
            span = rec["step"] - stall_start
            if span >= STALL_MIN_STEPS:
                stalls.append(
                    {
                        "start_step": stall_start,
                        "end_step": rec["step"],
                        "phase": stall_phase,
                        "duration_s": round(span * PHYSICS_DT, 4),
                        "steps": span,
                    }
                )
            in_stall = False
            stall_start = None
            stall_phase = None
    # Handle stall that runs to end
    if in_stall:
        span = recs[-1]["step"] - stall_start
        if span >= STALL_MIN_STEPS:
            stalls.append(
                {
                    "start_step": stall_start,
                    "end_step": recs[-1]["step"],
                    "phase": stall_phase,
                    "duration_s": round(span * PHYSICS_DT, 4),
                    "steps": span,
                }
            )
    return stalls


def piece_vibration(recs: list[dict], piece_key: str = "src_piece_pos") -> list[dict]:
    """Detect piece motion while the grip is stationary.

    This avoids treating normal piece transport from source to destination as vibration.
    """
    events = []
    for rec in recs:
        piece_vel = rec.get("src_piece_vel")
        if piece_vel is None or grip_speed(rec) >= STALL_VEL_THRESHOLD:
            continue
        speed = float(np.linalg.norm(piece_vel[:2]))
        if speed > PIECE_VIBRATION_THRESHOLD:
            events.append(
                {
                    "step": rec["step"],
                    "phase": rec["phase"],
                    "speed_mm_s": round(speed * 1000, 2),
                    "pos": [round(x * 1000, 1) for x in rec[piece_key]],
                }
            )
    return events


def finger_convergence(recs: list[dict]) -> list[dict]:
    """Find steps where actual l_finger diverges from target by > 0.5mm."""
    events = []
    for rec in recs:
        err = abs(rec["l_finger_pos"] - rec["finger_target"])
        if err > 0.0005:
            events.append(
                {
                    "step": rec["step"],
                    "phase": rec["phase"],
                    "finger_actual": round(rec["l_finger_pos"], 5),
                    "finger_target": round(rec["finger_target"], 5),
                    "error_m": round(err, 5),
                }
            )
    return events


def fmt_xyz(xyz) -> str:
    """Return fmt xyz."""
    return f"({xyz[0] * 1000:.1f}, {xyz[1] * 1000:.1f}, {xyz[2] * 1000:.1f}) mm"


def write_report(records: list[dict], report_path: Path, log_path: Path) -> None:
    """Return write report."""
    segs = phase_segments(records)
    total_steps = len(records)
    total_time_s = total_steps * PHYSICS_DT

    stalls = detect_stalls(records)
    vibrations = piece_vibration(records)
    finger_errs = finger_convergence(records)

    # Per-phase summary
    phase_stats: dict[str, dict] = defaultdict(
        lambda: {"steps": 0, "sim_time_s": 0.0, "over_budget": False}
    )
    for phase_name, recs in segs:
        s = len(recs)
        phase_stats[phase_name]["steps"] += s
        phase_stats[phase_name]["sim_time_s"] += round(s * PHYSICS_DT, 4)
        budget = PHASE_EXPECTED_STEPS.get(phase_name, None)
        if budget and phase_stats[phase_name]["steps"] > budget:
            phase_stats[phase_name]["over_budget"] = True

    lines = []
    lines += [
        "# Debug Move Analysis",
        "",
        f"**Log file:** `{log_path.name}`  ",
        f"**Total physics steps:** {total_steps}  ",
        f"**Total simulated time:** {total_time_s:.3f}s  ",
        f"**Wall time at last step:** {records[-1]['wall_time_s']:.2f}s  ",
        "",
        "---",
        "",
        "## Phase Breakdown",
        "",
        "| Phase | Steps | Sim time (s) | Budget | Status |",
        "|-------|-------|-------------|--------|--------|",
    ]

    for phase_name, stats in phase_stats.items():
        budget = PHASE_EXPECTED_STEPS.get(phase_name, "—")
        status = "⚠️ OVER BUDGET" if stats["over_budget"] else "✓"
        lines.append(
            f"| `{phase_name}` | {stats['steps']} | {stats['sim_time_s']:.3f} | {budget} | {status} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Stall Detection",
        "",
        f"Grip speed < {STALL_VEL_THRESHOLD * 1000:.0f} mm/s for ≥ {STALL_MIN_STEPS} consecutive steps.",
        "",
    ]

    if stalls:
        lines.append(f"**{len(stalls)} stall(s) detected:**")
        lines.append("")
        lines.append("| Start step | End step | Phase | Duration (s) | Steps |")
        lines.append("|-----------|---------|-------|-------------|-------|")
        for s in stalls:
            lines.append(
                f"| {s['start_step']} | {s['end_step']} | `{s['phase']}` | {s['duration_s']:.3f} | {s['steps']} |"
            )
    else:
        lines.append("No significant stalls detected.")

    lines += [
        "",
        "---",
        "",
        "## Piece Vibration",
        "",
        f"Steps where source-piece XY speed is > {PIECE_VIBRATION_THRESHOLD * 1000:.0f} mm/s while the grip is stationary.",
        "",
    ]

    if vibrations:
        sample = vibrations[:20]
        lines.append(
            f"**{len(vibrations)} vibration step(s)** (showing first {len(sample)}):"
        )
        lines.append("")
        lines.append("| Step | Phase | Speed (mm/s) | Piece pos (mm) |")
        lines.append("|------|-------|-----------|----------------|")
        for v in sample:
            pos_str = fmt_xyz([x / 1000 for x in v["pos"]])
            lines.append(
                f"| {v['step']} | `{v['phase']}` | {v['speed_mm_s']} | {pos_str} |"
            )
        if len(vibrations) > 20:
            lines.append(f"| … | | ({len(vibrations) - 20} more) | |")
    else:
        lines.append("No vibration events detected.")

    lines += [
        "",
        "---",
        "",
        "## Finger Convergence",
        "",
        "Steps where |l_finger_actual − target| > 0.5 mm.",
        "",
    ]

    if finger_errs:
        sample = finger_errs[:20]
        lines.append(
            f"**{len(finger_errs)} step(s) with finger lag** (showing first {len(sample)}):"
        )
        lines.append("")
        lines.append("| Step | Phase | Actual | Target | Error (m) |")
        lines.append("|------|-------|--------|--------|-----------|")
        for fe in sample:
            lines.append(
                f"| {fe['step']} | `{fe['phase']}` | {fe['finger_actual']:.5f} | {fe['finger_target']:.5f} | {fe['error_m']:.5f} |"
            )
        if len(finger_errs) > 20:
            lines.append(f"| … | | | | ({len(finger_errs) - 20} more) |")
    else:
        lines.append("Finger tracking within tolerance throughout.")

    # Grip position trace per phase
    lines += [
        "",
        "---",
        "",
        "## Grip Height Trace (Per Phase Entry/Exit)",
        "",
        "| Phase | Entry grip Z (mm) | Exit grip Z (mm) | ΔZ (mm) |",
        "|-------|------------------|-----------------|---------|",
    ]

    for phase_name, recs in segs:
        if not recs:
            continue
        z_in = recs[0]["grip_pos"][2] * 1000
        z_out = recs[-1]["grip_pos"][2] * 1000
        dz = z_out - z_in
        lines.append(f"| `{phase_name}` | {z_in:.1f} | {z_out:.1f} | {dz:+.1f} |")

    # Issues summary
    issues = []
    for phase_name, stats in phase_stats.items():
        if stats["over_budget"]:
            budget = PHASE_EXPECTED_STEPS[phase_name]
            issues.append(
                f"- **{phase_name}** used {stats['steps']} steps vs budget {budget} (+{stats['steps'] - budget})"
            )
    for s in stalls:
        issues.append(
            f"- **Stall** in `{s['phase']}` steps {s['start_step']}–{s['end_step']} ({s['duration_s']:.3f}s)"
        )
    if len(vibrations) > 50:
        issues.append(
            f"- **Excessive vibration**: {len(vibrations)} steps with piece drift > {PIECE_VIBRATION_THRESHOLD * 1000:.0f}mm"
        )
    if len(finger_errs) > 100:
        issues.append(
            f"- **Finger lag**: {len(finger_errs)} steps with finger not at target"
        )

    lines += [
        "",
        "---",
        "",
        "## Detected Issues Summary",
        "",
    ]

    if issues:
        for issue in issues:
            lines.append(issue)
    else:
        lines.append("No issues detected — move looks clean.")

    report_path.write_text("\n".join(lines) + "\n")
    print(f"Report saved to: {report_path}")


def main() -> None:
    """Parse command-line arguments and run the script."""
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/analyze_debug_log.py logs/debug_move_TIMESTAMP.jsonl"
        )
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        print(f"File not found: {log_path}")
        sys.exit(1)

    records = load_records(log_path)
    print(f"Loaded {len(records)} step records from {log_path.name}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = log_path.parent / f"debug_analysis_{ts}.md"
    write_report(records, report_path, log_path)

    # Quick console summary
    phases = defaultdict(int)
    for rec in records:
        phases[rec["phase"]] += 1
    print("\nPhase step counts:")
    for phase, count in phases.items():
        print(f"  {phase}: {count}")


if __name__ == "__main__":
    main()
