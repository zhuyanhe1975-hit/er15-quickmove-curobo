from __future__ import annotations

import numpy as np

from er15_quickmove.benchmark import (
    WeightedObjective,
    moveit_like_baseline,
    optional_dependency_results,
    ruckig_like_baseline,
    toppra_like_baseline,
)
from er15_quickmove.control import JointControlLimits
from er15_quickmove.cartesian import cartesian_line_error


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


def test_weighted_objective_combines_cycle_time_and_path_error():
    objective = WeightedObjective(cycle_time_weight=1.0, max_path_error_weight_s_per_m=20.0)

    assert objective.score(0.5, 0.002) == 0.54


def test_cartesian_line_error_reports_deviation_from_tcp_line():
    tcp = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.01, 0.0],
        [1.0, 0.0, 0.0],
    ])

    max_error, rms_error = cartesian_line_error(tcp, np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))

    assert np.isclose(max_error, 0.01)
    assert rms_error > 0.0
