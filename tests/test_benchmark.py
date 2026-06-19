from __future__ import annotations

import numpy as np

from er15_quickmove.benchmark import (
    moveit_like_baseline,
    optional_dependency_results,
    ruckig_like_baseline,
    toppra_like_baseline,
)
from er15_quickmove.control import JointControlLimits


class FakeTorqueModel:
    def inverse_dynamics(self, positions, dt):
        velocity = np.gradient(positions, dt, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, dt, axis=0, edge_order=2)
        return 10.0 * np.abs(acceleration)


def _limits() -> JointControlLimits:
    return JointControlLimits(
        joint_names=["joint_1", "joint_2"],
        position_lower_rad=[-10.0, -10.0],
        position_upper_rad=[10.0, 10.0],
        velocity_upper_rad_s=[5.0, 5.0],
        actuator_torque_upper_nm=[100.0, 100.0],
        wrist_load_torque_upper_nm=[None, None],
        wrist_load_inertia_upper_kgm2=[None, None],
        source={},
    )


def test_like_baselines_return_feasible_results():
    start = [0.0, 0.0]
    goal = [0.5, 0.25]
    limits = _limits()
    model = FakeTorqueModel()

    for fn in [ruckig_like_baseline, toppra_like_baseline, moveit_like_baseline]:
        result = fn(start, goal, 0.01, limits, model)
        assert result.status == "ok"
        assert result.duration_s is not None
        assert result.duration_s > 0.0
        assert result.peak_torque_ratio <= 1.0
        assert result.peak_velocity_ratio <= 1.0


def test_optional_dependency_results_include_requested_tools():
    names = {result.name for result in optional_dependency_results()}

    assert {"ruckig_python", "toppra_python", "moveit_commander"}.issubset(names)
