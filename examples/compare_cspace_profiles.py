from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from er15_quickmove import (
    CycleTimeComparison,
    ER15QuickMovePlanner,
    baseline_profile,
    quickmove_profile,
)


DEFAULT_START = [0.0, -0.9, 1.25, 0.0, 0.55, 0.0]
DEFAULT_GOAL = [1.1, -0.35, 1.75, 1.2, 0.25, 2.4]


def _run_profile(profile, start, goal, warmup: bool):
    planner = ER15QuickMovePlanner(profile)
    planned = planner.plan_cspace(start, goal, warmup=warmup)
    if planned.report is None:
        raise RuntimeError(f"{profile.name} planning failed")
    return planned.report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare conservative and QuickMove-like ER15-1400 profiles."
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip cuRobo warmup for faster debugging.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text.",
    )
    args = parser.parse_args()

    warmup = not args.no_warmup
    baseline = _run_profile(baseline_profile(), DEFAULT_START, DEFAULT_GOAL, warmup)
    quickmove = _run_profile(quickmove_profile(), DEFAULT_START, DEFAULT_GOAL, warmup)
    comparison = CycleTimeComparison(baseline=baseline, candidate=quickmove)

    if args.json:
        print(
            json.dumps(
                {
                    "baseline": asdict(baseline),
                    "quickmove": asdict(quickmove),
                    "saved_time_s": comparison.saved_time_s,
                    "saved_percent": comparison.saved_percent,
                },
                indent=2,
            )
        )
        return 0

    print("ER15-1400 cycle-time comparison")
    print(f"baseline_duration_s={baseline.duration_s:.3f}")
    print(f"quickmove_duration_s={quickmove.duration_s:.3f}")
    print(f"saved_time_s={comparison.saved_time_s:.3f}")
    print(f"saved_percent={comparison.saved_percent:.1f}")
    print(f"quickmove_peak_velocity_ratio={quickmove.peak_velocity_ratio:.3f}")
    print(f"quickmove_peak_acceleration_ratio={quickmove.peak_acceleration_ratio:.3f}")
    print(f"quickmove_peak_jerk_ratio={quickmove.peak_jerk_ratio:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
