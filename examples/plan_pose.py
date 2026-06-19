from __future__ import annotations

from er15_quickmove import ER15QuickMovePlanner, QuickMoveProfile


def main() -> int:
    planner = ER15QuickMovePlanner(QuickMoveProfile())

    start = [0.0, -0.95, 1.35, 0.0, 0.65, 0.0]
    goal_position_m = [0.78, -0.18, 0.72]
    planned = planner.plan_pose(start, goal_position_m, warmup=True)

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

