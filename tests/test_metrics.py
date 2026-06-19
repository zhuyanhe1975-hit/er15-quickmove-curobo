from __future__ import annotations

import torch

from er15_quickmove.metrics import CycleTimeComparison, summarize_joint_trajectory


def test_summarize_joint_trajectory_computes_duration_and_ratios():
    position = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.4],
        ],
        dtype=torch.float32,
    )
    report = summarize_joint_trajectory(
        position=position,
        dt=0.1,
        velocity_limits=[2.0, 4.0],
        acceleration_limits=[10.0, 10.0],
        jerk_limits=[100.0, 100.0],
        profile_name="test",
    )

    assert report.profile_name == "test"
    assert report.duration_s == 0.2
    assert report.waypoints == 3
    assert report.peak_velocity_ratio == 0.5
    assert report.is_within_limits


def test_cycle_time_comparison_reports_saved_percent():
    baseline = summarize_joint_trajectory(
        position=torch.zeros((11, 2)),
        dt=0.1,
        velocity_limits=[1.0, 1.0],
        acceleration_limits=[1.0, 1.0],
        jerk_limits=[1.0, 1.0],
        profile_name="baseline",
    )
    candidate = summarize_joint_trajectory(
        position=torch.zeros((6, 2)),
        dt=0.1,
        velocity_limits=[1.0, 1.0],
        acceleration_limits=[1.0, 1.0],
        jerk_limits=[1.0, 1.0],
        profile_name="quickmove",
    )

    comparison = CycleTimeComparison(baseline=baseline, candidate=candidate)

    assert comparison.saved_time_s == 0.5
    assert comparison.saved_percent == 50.0
