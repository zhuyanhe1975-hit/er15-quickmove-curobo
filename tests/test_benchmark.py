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
from er15_quickmove.cartesian import cartesian_line_error, cartesian_path_error, rounded_door_reference_path
from er15_quickmove.demo_trajectory import rounded_door_metadata


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


def test_rounded_door_reference_path_has_expected_extent():
    path = rounded_door_reference_path(
        np.array([1.0, 0.0, 1.0]),
        width_y_m=0.12,
        height_z_m=0.08,
        corner_radius_m=0.02,
        samples=51,
    )

    assert path.shape == (51, 3)
    assert np.isclose(path[0, 0], 1.0)
    assert np.isclose(path[-1, 1], -0.12)
    assert np.isclose(path[-1, 0], 1.0)
    assert np.isclose(path[:, 2].max(), 1.08, atol=1e-4)


def test_cartesian_path_error_uses_full_reference_polyline():
    reference = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ])
    tcp = np.array([[0.5, 0.0, 0.5]])

    max_error, _ = cartesian_path_error(tcp, reference)

    assert np.isclose(max_error, 0.5)


def test_rounded_door_metadata_marks_shared_visualization_source():
    class FakeTask:
        start_q = [0.0] * 6
        width_y_m = 0.12
        height_z_m = 0.08
        corner_radius_m = 0.02
        payload_kg = 15.0
        target_rotation = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        body_name = "link_6"
        samples = 3

    class FakePath:
        start_tcp_m = [1.0, 0.0, 0.8]
        goal_tcp_m = [1.0, -0.12, 0.8]
        max_target_error_m = 1e-5
        max_path_error_m = 2e-5
        rms_path_error_m = 1e-5
        max_orientation_error_rad = 3e-5
        rms_orientation_error_rad = 2e-5

    metadata = rounded_door_metadata(FakeTask(), FakePath())

    assert metadata["path_shape"] == "rounded_door"
    assert metadata["width_y_m"] == 0.12
    assert metadata["ik_max_orientation_error_rad"] == 3e-5


def test_default_rounded_door_task_is_large_and_central():
    from er15_quickmove.cartesian import default_path_task

    task = default_path_task()

    assert task.width_y_m == 0.65
    assert task.height_z_m == 0.40
    assert task.corner_radius_m == 0.10
    assert task.target_rotation == [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
