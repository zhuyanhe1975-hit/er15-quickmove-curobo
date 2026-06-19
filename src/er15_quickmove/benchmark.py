"""Same-task cycle-time and path-accuracy benchmarks for ER15 methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
from typing import Any, Callable

import numpy as np

from er15_quickmove.cartesian import (
    CartesianLinePath,
    CartesianLineTask,
    MujocoCartesianKinematics,
    cartesian_line_error,
    default_line_task,
)
from er15_quickmove.control import JointControlLimits, build_control_limits
from er15_quickmove.torque import (
    MujocoTorqueModel,
    TorqueLimitReport,
    audit_torque_limits,
    find_torque_limited_time_scale,
    smoothstep_resample_path,
)


@dataclass(frozen=True)
class WeightedObjective:
    cycle_time_weight: float = 1.0
    max_path_error_weight_s_per_m: float = 20.0

    def score(self, duration_s: float, max_path_error_m: float) -> float:
        return float(self.cycle_time_weight * duration_s + self.max_path_error_weight_s_per_m * max_path_error_m)


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    status: str
    duration_s: float | None
    feasible: bool
    peak_torque_ratio: float | None
    peak_velocity_ratio: float | None
    limiting_joint: str | None
    max_path_error_m: float | None
    rms_path_error_m: float | None
    objective_score: float | None
    details: dict[str, Any]


def _result_from_report(
    name: str,
    report: TorqueLimitReport,
    details: dict[str, Any] | None = None,
    max_path_error_m: float | None = None,
    rms_path_error_m: float | None = None,
    objective: WeightedObjective | None = None,
) -> BenchmarkResult:
    score = None
    if objective is not None and max_path_error_m is not None:
        score = objective.score(report.duration_s, max_path_error_m)
    return BenchmarkResult(
        name=name,
        status="ok" if report.feasible else "infeasible",
        duration_s=report.duration_s,
        feasible=report.feasible,
        peak_torque_ratio=report.peak_torque_ratio,
        peak_velocity_ratio=report.peak_velocity_ratio,
        limiting_joint=report.limiting_joint,
        max_path_error_m=max_path_error_m,
        rms_path_error_m=rms_path_error_m,
        objective_score=score,
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
        max_path_error_m=None,
        rms_path_error_m=None,
        objective_score=None,
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
    s = trapezoid_scalar_law(r)
    start_arr = np.asarray(start, dtype=float)
    delta = np.asarray(goal, dtype=float) - start_arr
    return start_arr[None, :] + s[:, None] * delta[None, :], float(actual_dt)


def trapezoid_scalar_law(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return np.where(r < 0.5, 2.0 * r * r, 1.0 - 2.0 * (1.0 - r) * (1.0 - r))


def _resample_path_with_law(
    positions: np.ndarray,
    source_dt: float,
    duration: float,
    scalar_law: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, float]:
    positions = np.asarray(positions, dtype=float)
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    sample_count = max(3, int(np.ceil(duration / source_dt)) + 1)
    actual_dt = duration / (sample_count - 1)
    r = np.linspace(0.0, 1.0, sample_count)
    s = np.clip(scalar_law(r), 0.0, 1.0)
    source_x = np.linspace(0.0, 1.0, positions.shape[0])
    out = np.empty((sample_count, positions.shape[1]), dtype=float)
    for j in range(positions.shape[1]):
        out[:, j] = np.interp(s, source_x, positions[:, j])
    return out, float(actual_dt)


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


def _fastest_path_law(
    name: str,
    positions: np.ndarray,
    dt: float,
    scalar_law: Callable[[np.ndarray], np.ndarray],
    control_limits: JointControlLimits,
    torque_model: MujocoTorqueModel,
    objective: WeightedObjective,
    line_path: CartesianLinePath,
    min_duration: float = 0.05,
    max_duration: float = 20.0,
    tolerance: float = 1e-3,
) -> BenchmarkResult:
    def evaluate(duration: float) -> tuple[TorqueLimitReport, np.ndarray, float]:
        sampled, sample_dt = _resample_path_with_law(positions, dt, duration, scalar_law)
        return audit_torque_limits(sampled, sample_dt, control_limits, torque_model), sampled, sample_dt

    upper = max_duration
    report, best_positions, _ = evaluate(upper)
    while not report.feasible and upper < 300.0:
        upper *= 2.0
        report, best_positions, _ = evaluate(upper)
    if not report.feasible:
        return _result_from_report(name, report, {"path_mode": "tcp_line_preserving"})

    lower = min_duration
    lower_report, lower_positions, _ = evaluate(lower)
    if lower_report.feasible:
        best = lower_report
        best_positions = lower_positions
        iterations = 1
    else:
        best = report
        iterations = 0
        for iterations in range(1, 40):
            mid = 0.5 * (lower + upper)
            candidate, candidate_positions, _ = evaluate(mid)
            if candidate.feasible:
                best = candidate
                best_positions = candidate_positions
                upper = mid
            else:
                lower = mid
            if upper - lower <= tolerance:
                break
    best = TorqueLimitReport(**{**best.__dict__, "iterations": iterations})
    max_err, rms_err = _path_error_for_positions(best_positions, line_path)
    return _result_from_report(
        name,
        best,
        {"path_mode": "tcp_line_preserving", "law": getattr(scalar_law, "__name__", "scalar_law")},
        max_err,
        rms_err,
        objective,
    )


def _path_error_for_positions(positions: np.ndarray, line_path: CartesianLinePath) -> tuple[float, float]:
    if positions.shape == line_path.positions.shape and np.allclose(positions, line_path.positions):
        return line_path.max_line_error_m, line_path.rms_line_error_m
    kinematics = MujocoCartesianKinematics()
    tcp = kinematics.body_positions(positions)
    return cartesian_line_error(tcp, line_path.reference_tcp_positions_m[0], line_path.reference_tcp_positions_m[-1])


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


def _smooth_line_result(
    line_path: CartesianLinePath,
    dt: float,
    report: TorqueLimitReport,
    payload_kg: float,
    objective: WeightedObjective,
) -> BenchmarkResult:
    sampled, _ = smoothstep_resample_path(line_path.positions, dt, report.time_scale)
    max_err, rms_err = _path_error_for_positions(sampled, line_path)
    return _result_from_report(
        "ruckig_like_quintic_line_law",
        report,
        {"path_mode": "tcp_line_preserving", "payload_kg": payload_kg},
        max_err,
        rms_err,
        objective,
    )


def run_cartesian_line_payload_benchmark(
    task: CartesianLineTask | None = None,
    dt: float = 0.01,
    objective: WeightedObjective | None = None,
    control_limits: JointControlLimits | None = None,
) -> tuple[CartesianLinePath, list[BenchmarkResult]]:
    task = task or default_line_task()
    objective = objective or WeightedObjective()
    control_limits = control_limits or build_control_limits()
    kinematics = MujocoCartesianKinematics()
    line_path = kinematics.solve_line_path(task)
    torque_model = MujocoTorqueModel(payload_kg=task.payload_kg)

    quickmove_report = find_torque_limited_time_scale(
        line_path.positions, dt, control_limits, torque_model, min_scale=0.01, max_scale=100.0
    )
    smooth_report = find_torque_limited_time_scale(
        line_path.positions,
        dt,
        control_limits,
        torque_model,
        min_scale=0.01,
        max_scale=100.0,
        smoothstep=True,
    )
    results = [
        _result_from_report(
            "quickmove_truemove_torque_limited_line",
            quickmove_report,
            {"path_mode": "tcp_line_preserving", "payload_kg": task.payload_kg},
            line_path.max_line_error_m,
            line_path.rms_line_error_m,
            objective,
        ),
        _result_from_report(
            "toppra_like_torque_limited_line",
            quickmove_report,
            {"path_mode": "tcp_line_preserving", "payload_kg": task.payload_kg},
            line_path.max_line_error_m,
            line_path.rms_line_error_m,
            objective,
        ),
        _smooth_line_result(line_path, dt, smooth_report, task.payload_kg, objective),
        _fastest_path_law(
            "moveit_like_parabolic_line_law",
            line_path.positions,
            dt,
            trapezoid_scalar_law,
            control_limits,
            torque_model,
            objective,
            line_path,
        ),
    ]

    # Endpoint-only baselines are intentionally retained with their path error so
    # the weighted objective shows why raw cycle time alone is not a TrueMove metric.
    endpoint_start = line_path.positions[0].tolist()
    endpoint_goal = line_path.positions[-1].tolist()
    for baseline in [ruckig_like_baseline, toppra_like_baseline, moveit_like_baseline]:
        result = baseline(endpoint_start, endpoint_goal, dt, control_limits, torque_model)
        if result.duration_s is None:
            results.append(result)
            continue
        if baseline is ruckig_like_baseline:
            positions, _ = quintic_joint_trajectory(endpoint_start, endpoint_goal, result.duration_s, dt)
        elif baseline is moveit_like_baseline:
            positions, _ = trapezoid_like_trajectory(endpoint_start, endpoint_goal, result.duration_s, dt)
        else:
            positions = np.linspace(np.asarray(endpoint_start), np.asarray(endpoint_goal), 101)
        max_err, rms_err = _path_error_for_positions(positions, line_path)
        results.append(
            BenchmarkResult(
                name=f"endpoint_only_{result.name}",
                status=result.status,
                duration_s=result.duration_s,
                feasible=result.feasible,
                peak_torque_ratio=result.peak_torque_ratio,
                peak_velocity_ratio=result.peak_velocity_ratio,
                limiting_joint=result.limiting_joint,
                max_path_error_m=max_err,
                rms_path_error_m=rms_err,
                objective_score=objective.score(result.duration_s, max_err),
                details={**result.details, "path_mode": "joint_endpoint_not_truemove"},
            )
        )

    results.extend(optional_dependency_results())
    return line_path, results
