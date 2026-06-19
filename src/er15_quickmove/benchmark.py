"""Same-task cycle-time benchmarks for ER15 trajectory methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Any

import numpy as np

from er15_quickmove.control import JointControlLimits, build_control_limits
from er15_quickmove.torque import (
    MujocoTorqueModel,
    TorqueLimitReport,
    audit_torque_limits,
    find_torque_limited_time_scale,
    smoothstep_resample_path,
)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    status: str
    duration_s: float | None
    feasible: bool
    peak_torque_ratio: float | None
    peak_velocity_ratio: float | None
    limiting_joint: str | None
    details: dict[str, Any]


def _result_from_report(name: str, report: TorqueLimitReport, details: dict[str, Any] | None = None) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        status="ok" if report.feasible else "infeasible",
        duration_s=report.duration_s,
        feasible=report.feasible,
        peak_torque_ratio=report.peak_torque_ratio,
        peak_velocity_ratio=report.peak_velocity_ratio,
        limiting_joint=report.limiting_joint,
        details={**(details or {}), "torque_report": asdict(report)},
    )


def _skipped(name: str, reason: str) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        status="skipped",
        duration_s=None,
        feasible=False,
        peak_torque_ratio=None,
        peak_velocity_ratio=None,
        limiting_joint=None,
        details={"reason": reason},
    )


def quintic_joint_trajectory(start: list[float], goal: list[float], duration: float, dt: float) -> tuple[np.ndarray, float]:
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    sample_count = max(3, int(np.ceil(duration / dt)) + 1)
    actual_dt = duration / (sample_count - 1)
    r = np.linspace(0.0, 1.0, sample_count)
    s = 10.0 * r**3 - 15.0 * r**4 + 6.0 * r**5
    start_arr = np.asarray(start, dtype=float)
    delta = np.asarray(goal, dtype=float) - start_arr
    return start_arr[None, :] + s[:, None] * delta[None, :], float(actual_dt)


def trapezoid_like_trajectory(start: list[float], goal: list[float], duration: float, dt: float) -> tuple[np.ndarray, float]:
    sample_count = max(3, int(np.ceil(duration / dt)) + 1)
    actual_dt = duration / (sample_count - 1)
    r = np.linspace(0.0, 1.0, sample_count)
    s = np.where(r < 0.5, 2.0 * r * r, 1.0 - 2.0 * (1.0 - r) * (1.0 - r))
    start_arr = np.asarray(start, dtype=float)
    delta = np.asarray(goal, dtype=float) - start_arr
    return start_arr[None, :] + s[:, None] * delta[None, :], float(actual_dt)


def _fastest_generated_trajectory(
    name: str,
    generator,
    start: list[float],
    goal: list[float],
    dt: float,
    control_limits: JointControlLimits,
    torque_model: MujocoTorqueModel,
    min_duration: float = 0.05,
    max_duration: float = 20.0,
    tolerance: float = 1e-3,
) -> BenchmarkResult:
    def evaluate(duration: float) -> TorqueLimitReport:
        positions, sample_dt = generator(start, goal, duration, dt)
        return audit_torque_limits(positions, sample_dt, control_limits, torque_model)

    upper = max_duration
    report = evaluate(upper)
    while not report.feasible and upper < 300.0:
        upper *= 2.0
        report = evaluate(upper)
    if not report.feasible:
        return _result_from_report(name, report, {"generator": generator.__name__})

    lower = min_duration
    lower_report = evaluate(lower)
    if lower_report.feasible:
        return _result_from_report(name, lower_report, {"generator": generator.__name__})

    best = report
    iterations = 0
    for iterations in range(1, 40):
        mid = 0.5 * (lower + upper)
        candidate = evaluate(mid)
        if candidate.feasible:
            best = candidate
            upper = mid
        else:
            lower = mid
        if upper - lower <= tolerance:
            break
    best = TorqueLimitReport(**{**best.__dict__, "iterations": iterations})
    return _result_from_report(name, best, {"generator": generator.__name__})


def ruckig_like_baseline(
    start: list[float],
    goal: list[float],
    dt: float,
    control_limits: JointControlLimits,
    torque_model: MujocoTorqueModel,
) -> BenchmarkResult:
    return _fastest_generated_trajectory(
        "ruckig_like_quintic",
        quintic_joint_trajectory,
        start,
        goal,
        dt,
        control_limits,
        torque_model,
    )


def toppra_like_baseline(
    start: list[float],
    goal: list[float],
    dt: float,
    control_limits: JointControlLimits,
    torque_model: MujocoTorqueModel,
) -> BenchmarkResult:
    # TOPP-RA is path retiming. Use a dense straight joint-space path and the
    # same torque/velocity boundary search as the project method.
    path = np.linspace(np.asarray(start, dtype=float), np.asarray(goal, dtype=float), 101)
    nominal_dt = dt
    report = find_torque_limited_time_scale(
        path, nominal_dt, control_limits, torque_model, min_scale=0.01, max_scale=100.0
    )
    return _result_from_report("toppra_like_path_retiming", report, {"path_samples": 101})


def moveit_like_baseline(
    start: list[float],
    goal: list[float],
    dt: float,
    control_limits: JointControlLimits,
    torque_model: MujocoTorqueModel,
) -> BenchmarkResult:
    return _fastest_generated_trajectory(
        "moveit_like_iterative_parabolic",
        trapezoid_like_trajectory,
        start,
        goal,
        dt,
        control_limits,
        torque_model,
    )


def optional_dependency_results() -> list[BenchmarkResult]:
    results = []
    for package, label in [
        ("ruckig", "ruckig_python"),
        ("toppra", "toppra_python"),
        ("moveit_commander", "moveit_commander"),
    ]:
        if find_spec(package) is None:
            results.append(_skipped(label, f"missing optional dependency: {package}"))
        else:
            results.append(_skipped(label, "installed but direct adapter is not implemented yet"))
    return results


def run_same_task_benchmark(
    start: list[float],
    goal: list[float],
    quickmove_positions: np.ndarray,
    quickmove_dt: float,
    control_limits: JointControlLimits | None = None,
    torque_model: MujocoTorqueModel | None = None,
) -> list[BenchmarkResult]:
    control_limits = control_limits or build_control_limits()
    torque_model = torque_model or MujocoTorqueModel()
    quickmove_report = find_torque_limited_time_scale(
        quickmove_positions, quickmove_dt, control_limits, torque_model, min_scale=0.05
    )
    results = [
        _result_from_report(
            "curobo_quickmove_torque_limited",
            quickmove_report,
            {"source": "cuRobo path + MuJoCo inverse-dynamics retiming"},
        ),
        ruckig_like_baseline(start, goal, quickmove_dt, control_limits, torque_model),
        toppra_like_baseline(start, goal, quickmove_dt, control_limits, torque_model),
        moveit_like_baseline(start, goal, quickmove_dt, control_limits, torque_model),
    ]
    results.extend(optional_dependency_results())
    return results
