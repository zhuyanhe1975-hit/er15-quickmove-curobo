"""Torque-limited time scaling for ER15 trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from er15_quickmove.config import ProjectPaths
from er15_quickmove.control import JointControlLimits, build_control_limits


@dataclass(frozen=True)
class TorqueLimitReport:
    duration_s: float
    dt_s: float
    time_scale: float
    feasible: bool
    peak_torque_nm: list[float]
    peak_torque_ratio: float
    peak_torque_ratio_by_joint: list[float]
    peak_velocity_ratio: float
    peak_velocity_ratio_by_joint: list[float]
    limiting_joint: str
    iterations: int


def finite_difference_state(positions: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if positions.ndim != 2:
        raise ValueError(f"positions must have shape [T, DOF], got {positions.shape}")
    if positions.shape[0] < 3:
        raise ValueError("at least 3 waypoints are required for torque audit")
    velocities = np.gradient(positions, dt, axis=0, edge_order=2)
    accelerations = np.gradient(velocities, dt, axis=0, edge_order=2)
    return velocities, accelerations


def smoothstep_resample_path(
    positions: np.ndarray,
    source_dt: float,
    time_scale: float,
    max_samples: int = 2001,
) -> tuple[np.ndarray, float]:
    """Retiming helper that preserves path geometry and smooths endpoint timing.

    The source trajectory is treated as a geometric path parameterized by sample
    index. A quintic smoothstep law produces zero endpoint velocity and
    acceleration in the scalar path coordinate.
    """

    positions = np.asarray(positions, dtype=float)
    source_count = positions.shape[0]
    source_duration = max(source_count - 1, 1) * source_dt
    target_duration = source_duration * time_scale
    sample_count = min(max(source_count, int(np.ceil(target_duration / source_dt)) + 1), max_samples)
    if sample_count < 3:
        sample_count = 3
    new_dt = target_duration / (sample_count - 1)
    r = np.linspace(0.0, 1.0, sample_count)
    s = 10.0 * r**3 - 15.0 * r**4 + 6.0 * r**5
    source_x = np.linspace(0.0, 1.0, source_count)
    out = np.empty((sample_count, positions.shape[1]), dtype=float)
    for j in range(positions.shape[1]):
        out[:, j] = np.interp(s, source_x, positions[:, j])
    return out, float(new_dt)


class MujocoTorqueModel:
    """MuJoCo inverse dynamics wrapper for the repository ER15 MJCF model."""

    def __init__(self, paths: ProjectPaths | None = None):
        self.paths = paths or ProjectPaths()
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(self.paths.robot_mjcf))
        self.data = mujoco.MjData(self.model)

    def inverse_dynamics(self, positions: np.ndarray, dt: float) -> np.ndarray:
        positions = np.asarray(positions, dtype=float)
        velocities, accelerations = finite_difference_state(positions, dt)
        torques = np.zeros_like(positions)
        mass_matrix = np.zeros((self.model.nv, self.model.nv), dtype=float)
        for i, (q, qd, qdd) in enumerate(zip(positions, velocities, accelerations, strict=True)):
            self.data.qpos[:] = q
            self.data.qvel[:] = qd
            self.data.qacc[:] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
            self.mujoco.mj_fullM(self.model, mass_matrix, self.data.qM)
            torques[i, :] = (mass_matrix @ qdd + self.data.qfrc_bias)[: positions.shape[1]]
        return torques


def audit_torque_limits(
    positions: np.ndarray,
    dt: float,
    control_limits: JointControlLimits | None = None,
    torque_model: MujocoTorqueModel | None = None,
    time_scale: float = 1.0,
) -> TorqueLimitReport:
    control_limits = control_limits or build_control_limits()
    torque_model = torque_model or MujocoTorqueModel()
    scaled_dt = dt * time_scale
    positions = np.asarray(positions, dtype=float)
    velocities, _ = finite_difference_state(positions, scaled_dt)
    torques = torque_model.inverse_dynamics(positions, scaled_dt)

    torque_limits = np.asarray(control_limits.actuator_torque_upper_nm, dtype=float)
    velocity_limits = np.asarray(control_limits.velocity_upper_rad_s, dtype=float)
    torque_ratio_by_joint = np.max(np.abs(torques), axis=0) / np.maximum(torque_limits, 1e-9)
    velocity_ratio_by_joint = np.max(np.abs(velocities), axis=0) / np.maximum(velocity_limits, 1e-9)
    combined = np.maximum(torque_ratio_by_joint, velocity_ratio_by_joint)
    limiting_idx = int(np.argmax(combined))
    return TorqueLimitReport(
        duration_s=float(max(positions.shape[0] - 1, 0) * scaled_dt),
        dt_s=float(scaled_dt),
        time_scale=float(time_scale),
        feasible=bool(np.max(combined) <= 1.0),
        peak_torque_nm=np.max(np.abs(torques), axis=0).tolist(),
        peak_torque_ratio=float(np.max(torque_ratio_by_joint)),
        peak_torque_ratio_by_joint=torque_ratio_by_joint.tolist(),
        peak_velocity_ratio=float(np.max(velocity_ratio_by_joint)),
        peak_velocity_ratio_by_joint=velocity_ratio_by_joint.tolist(),
        limiting_joint=control_limits.joint_names[limiting_idx],
        iterations=0,
    )


def find_torque_limited_time_scale(
    positions: np.ndarray,
    dt: float,
    control_limits: JointControlLimits | None = None,
    torque_model: MujocoTorqueModel | None = None,
    min_scale: float = 0.25,
    max_scale: float = 1024.0,
    tolerance: float = 1e-3,
    max_iterations: int = 32,
    max_samples: int = 2001,
    smoothstep: bool = False,
) -> TorqueLimitReport:
    """Find the fastest retiming that respects torque and velocity limits.

    ``time_scale < 1`` speeds up the original trajectory, ``time_scale > 1``
    slows it down. By default the cuRobo trajectory samples are uniformly
    retimed, preserving the optimizer's time law while searching for the
    torque/velocity boundary. Set ``smoothstep=True`` to replace the original
    time law with a conservative quintic path-coordinate timing.
    """

    if min_scale <= 0 or max_scale <= min_scale:
        raise ValueError("expected 0 < min_scale < max_scale")
    control_limits = control_limits or build_control_limits()
    torque_model = torque_model or MujocoTorqueModel()

    def evaluate(scale: float) -> TorqueLimitReport:
        if smoothstep:
            resampled, resampled_dt = smoothstep_resample_path(
                positions, dt, scale, max_samples=max_samples
            )
            report = audit_torque_limits(
                resampled, resampled_dt, control_limits, torque_model, time_scale=1.0
            )
            return TorqueLimitReport(**{**report.__dict__, "time_scale": scale})
        return audit_torque_limits(positions, dt, control_limits, torque_model, time_scale=scale)

    original = evaluate(1.0)
    original = TorqueLimitReport(**{**original.__dict__, "iterations": 0})

    # If the lower bound is feasible, it is the fastest allowed by the search box.
    lower = min_scale
    lower_report = evaluate(lower)
    if lower_report.feasible:
        return TorqueLimitReport(**{**lower_report.__dict__, "iterations": 1})

    # Find a feasible upper bound. If the original trajectory is feasible this
    # is normally 1.0; otherwise expand upward until dynamics become feasible.
    upper = 1.0
    upper_report = original
    while not upper_report.feasible and upper < max_scale:
        upper = min(max_scale, upper * 1.5)
        upper_report = evaluate(upper)
    if not upper_report.feasible:
        return TorqueLimitReport(**{**upper_report.__dict__, "iterations": 0})

    best = upper_report
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        mid = 0.5 * (lower + upper)
        candidate = evaluate(mid)
        if candidate.feasible:
            best = candidate
            upper = mid
        else:
            lower = mid
        if upper - lower <= tolerance:
            break
    return TorqueLimitReport(**{**best.__dict__, "iterations": iterations})

