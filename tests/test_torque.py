from __future__ import annotations

import numpy as np
import pytest

from er15_quickmove.control import JointControlLimits
from er15_quickmove.torque import audit_torque_limits, find_torque_limited_time_scale, smoothstep_resample_path


class FakeTorqueModel:
    def inverse_dynamics(self, positions, dt):
        # Torque grows with the retimed path acceleration.
        velocity = np.gradient(positions, dt, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
        return 100.0 * np.abs(acceleration)


def _limits() -> JointControlLimits:
    return JointControlLimits(
        joint_names=["joint_1", "joint_2"],
        position_lower_rad=[-1.0, -1.0],
        position_upper_rad=[1.0, 1.0],
        velocity_upper_rad_s=[10.0, 10.0],
        actuator_torque_upper_nm=[100.0, 100.0],
        wrist_load_torque_upper_nm=[None, None],
        wrist_load_inertia_upper_kgm2=[None, None],
        source={},
    )


def test_torque_audit_reports_peak_ratio():
    positions = np.linspace([0.0, 0.0], [0.2, 0.1], 5)

    report = audit_torque_limits(positions, 0.2, _limits(), FakeTorqueModel())

    assert report.peak_torque_ratio > 0.0
    assert report.peak_torque_ratio < 1.0
    assert report.feasible


def test_torque_limited_time_scale_speeds_up_until_boundary():
    positions = np.linspace([0.0, 0.0], [0.2, 0.1], 5)

    report = find_torque_limited_time_scale(
        positions,
        0.1,
        _limits(),
        FakeTorqueModel(),
        min_scale=0.25,
        tolerance=1e-2,
    )

    assert report.feasible
    assert report.time_scale < 1.0
    assert max(report.peak_torque_ratio, report.peak_velocity_ratio) <= 1.0


def test_smoothstep_resample_preserves_endpoints_and_duration():
    positions = np.linspace([0.0, 0.0], [1.0, 0.5], 5)

    retimed, dt = smoothstep_resample_path(positions, 0.1, 2.0)

    assert np.allclose(retimed[0], positions[0])
    assert np.allclose(retimed[-1], positions[-1])
    assert dt * (retimed.shape[0] - 1) == pytest.approx(0.8)


def test_torque_limited_time_scale_slows_down_when_original_infeasible():
    positions = np.linspace([0.0, 0.0], [2.0, 1.0], 5)
    limits = JointControlLimits(
        joint_names=["joint_1", "joint_2"],
        position_lower_rad=[-10.0, -10.0],
        position_upper_rad=[10.0, 10.0],
        velocity_upper_rad_s=[0.5, 0.5],
        actuator_torque_upper_nm=[100.0, 100.0],
        wrist_load_torque_upper_nm=[None, None],
        wrist_load_inertia_upper_kgm2=[None, None],
        source={},
    )

    report = find_torque_limited_time_scale(
        positions,
        0.1,
        limits,
        FakeTorqueModel(),
        min_scale=0.25,
        tolerance=1e-2,
    )

    assert report.feasible
    assert report.time_scale > 1.0
    assert report.peak_velocity_ratio <= 1.0
