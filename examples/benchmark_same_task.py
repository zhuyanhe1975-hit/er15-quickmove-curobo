from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from er15_quickmove import ER15QuickMovePlanner, quickmove_profile, run_same_task_benchmark

START = [0.0, -0.9, 1.25, 0.0, 0.55, 0.0]
GOAL = [1.1, -0.35, 1.75, 1.2, 0.25, 2.4]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark ER15 cycle time on the same joint-space task."
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    planner = ER15QuickMovePlanner(quickmove_profile())
    planned = planner.plan_cspace(START, GOAL, warmup=not args.no_warmup)
    if planned.report is None:
        raise RuntimeError("QuickMove planning failed")
    traj = planned.result.get_interpolated_plan()
    positions = traj.position.detach().cpu().reshape(-1, traj.position.shape[-1]).numpy()

    results = run_same_task_benchmark(
        START,
        GOAL,
        positions,
        planner.profile.interpolation_dt,
        planner.joint_control_limits,
    )
    payload = {
        "task": {"start": START, "goal": GOAL},
        "quickmove_profile_report": asdict(planned.report),
        "results": [asdict(result) for result in results],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("ER15 same-task cycle-time benchmark")
        for result in results:
            if result.duration_s is None:
                print(f"{result.name}: {result.status} ({result.details.get('reason')})")
            else:
                print(
                    f"{result.name}: duration={result.duration_s:.3f}s "
                    f"torque={result.peak_torque_ratio:.3f} "
                    f"velocity={result.peak_velocity_ratio:.3f} "
                    f"limit={result.limiting_joint} status={result.status}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
