from __future__ import annotations

from er15_quickmove import ER15QuickMovePlanner, QuickMoveProfile


def main() -> int:
    planner = ER15QuickMovePlanner(QuickMoveProfile())

    start = [0.0, -0.9, 1.25, 0.0, 0.55, 0.0]
    goal = [1.1, -0.35, 1.75, 1.2, 0.25, 2.4]
    planned = planner.plan_cspace(start, goal, warmup=True)

    if planned.report is None:
        print("Planning failed")
        return 1

    print("Planning succeeded")
    print(f"duration_s={planned.report.duration_s:.3f}")
    print(f"waypoints={planned.report.waypoints}")
    print(f"peak_velocity_ratio={planned.report.peak_velocity_ratio:.3f}")
    print(f"peak_acceleration_ratio={planned.report.peak_acceleration_ratio:.3f}")
    print(f"peak_jerk_ratio={planned.report.peak_jerk_ratio:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

