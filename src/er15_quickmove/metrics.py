"""Trajectory metrics for QuickMove-like cycle-time evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class TrajectoryLimitReport:
    profile_name: str
    duration_s: float
    waypoints: int
    peak_velocity_ratio: float
    peak_acceleration_ratio: float
    peak_jerk_ratio: float

    @property
    def is_within_limits(self) -> bool:
        return (
            self.peak_velocity_ratio <= 1.0
            and self.peak_acceleration_ratio <= 1.0
            and self.peak_jerk_ratio <= 1.0
        )


@dataclass(frozen=True)
class CycleTimeComparison:
    baseline: TrajectoryLimitReport
    candidate: TrajectoryLimitReport

    @property
    def saved_time_s(self) -> float:
        return self.baseline.duration_s - self.candidate.duration_s

    @property
    def saved_percent(self) -> float:
        if self.baseline.duration_s <= 0.0:
            return 0.0
        return 100.0 * self.saved_time_s / self.baseline.duration_s


def _peak_ratio(values: torch.Tensor | None, limits: Sequence[float] | None) -> float:
    if values is None or limits is None:
        return 0.0
    limit_tensor = torch.as_tensor(limits, device=values.device, dtype=values.dtype)
    ratio = torch.abs(values) / torch.clamp(limit_tensor.view(1, -1), min=1e-9)
    return float(torch.max(ratio).detach().cpu())


def summarize_joint_trajectory(
    position: torch.Tensor,
    dt: float,
    velocity_limits: Sequence[float],
    acceleration_limits: Sequence[float] | None,
    jerk_limits: Sequence[float] | None,
    profile_name: str = "trajectory",
    velocity: torch.Tensor | None = None,
    acceleration: torch.Tensor | None = None,
    jerk: torch.Tensor | None = None,
) -> TrajectoryLimitReport:
    """Summarize limit utilization for a dense joint trajectory.

    Pass ``None`` for acceleration or jerk limits when they are not direct
    constraints. In that case the corresponding utilization ratio is reported
    as 0.0 and does not affect ``is_within_limits``.
    """

    if position.ndim == 3:
        position = position.squeeze(0)
    if position.ndim != 2:
        raise ValueError(f"position must have shape [T, DOF], got {tuple(position.shape)}")

    if velocity is None and position.shape[0] > 1:
        velocity = torch.diff(position, dim=0) / dt
    if acceleration is None and velocity is not None and velocity.shape[0] > 1:
        acceleration = torch.diff(velocity, dim=0) / dt
    if jerk is None and acceleration is not None and acceleration.shape[0] > 1:
        jerk = torch.diff(acceleration, dim=0) / dt

    duration = max(position.shape[0] - 1, 0) * dt
    return TrajectoryLimitReport(
        profile_name=profile_name,
        duration_s=float(duration),
        waypoints=int(position.shape[0]),
        peak_velocity_ratio=_peak_ratio(velocity, velocity_limits),
        peak_acceleration_ratio=_peak_ratio(acceleration, acceleration_limits),
        peak_jerk_ratio=_peak_ratio(jerk, jerk_limits),
    )
