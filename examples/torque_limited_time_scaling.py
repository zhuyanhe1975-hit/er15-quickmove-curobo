from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from er15_quickmove import (
    ER15QuickMovePlanner,
    audit_torque_limits,
    find_torque_limited_time_scale,
    quickmove_profile,
)

START = [0.0, -0.9, 1.25, 0.0, 0.55, 0.0]
GOAL = [1.1, -0.35, 1.75, 1.2, 0.25, 2.4]


def main() -> int:
    parser = argparse.ArgumentParser(description="Torque-limited retiming for an ER15 QuickMove trajectory.")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-scale", type=float, default=0.25)
    parser.add_argument("--max-scale", type=float, default=1024.0)
    parser.add_argument("--smoothstep", action="store_true")
    args = parser.parse_args()

    planner = ER15QuickMovePlanner(quickmove_profile())
    planned = planner.plan_cspace(START, GOAL, warmup=not args.no_warmup)
    if planned.report is None:
        raise RuntimeError("QuickMove planning failed")
    traj = planned.result.get_interpolated_plan()
    positions = traj.position.detach().cpu().reshape(-1, traj.position.shape[-1]).numpy()

    original = audit_torque_limits(positions, planner.profile.interpolation_dt, planner.joint_control_limits)
    retimed = find_torque_limited_time_scale(
        positions,
        planner.profile.interpolation_dt,
        planner.joint_control_limits,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        smoothstep=args.smoothstep,
    )
    payload = {
        "quickmove_report": asdict(planned.report),
        "torque_audit_original": asdict(original),
        "torque_limited_retimed": asdict(retimed),
        "saved_vs_quickmove_s": planned.report.duration_s - retimed.duration_s,
        "saved_vs_quickmove_percent": 100.0 * (planned.report.duration_s - retimed.duration_s) / planned.report.duration_s,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"quickmove_duration_s={planned.report.duration_s:.3f}")
        print(f"original_peak_torque_ratio={original.peak_torque_ratio:.3f}")
        print(f"original_peak_velocity_ratio={original.peak_velocity_ratio:.3f}")
        print(f"torque_limited_duration_s={retimed.duration_s:.3f}")
        print(f"torque_limited_time_scale={retimed.time_scale:.3f}")
        print(f"retimed_peak_torque_ratio={retimed.peak_torque_ratio:.3f}")
        print(f"retimed_peak_velocity_ratio={retimed.peak_velocity_ratio:.3f}")
        print(f"limiting_joint={retimed.limiting_joint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
