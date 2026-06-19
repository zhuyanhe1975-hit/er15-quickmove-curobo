"""Shared demo trajectory generation for benchmark and visualization scripts."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from er15_quickmove.benchmark import WeightedObjective, run_cartesian_line_payload_benchmark
from er15_quickmove.cartesian import CartesianPath, CartesianRoundedDoorTask, default_path_task


def rounded_door_payload_trajectory(
    dt: float = 0.01,
    payload_kg: float = 15.0,
    path_weight: float = 20.0,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    base_task = default_path_task()
    task = CartesianRoundedDoorTask(
        start_q=base_task.start_q,
        width_y_m=base_task.width_y_m,
        height_z_m=base_task.height_z_m,
        corner_radius_m=base_task.corner_radius_m,
        samples=base_task.samples,
        body_name=base_task.body_name,
        payload_kg=payload_kg,
        ik_tolerance_m=base_task.ik_tolerance_m,
        ik_orientation_tolerance_rad=base_task.ik_orientation_tolerance_rad,
        ik_max_iterations=base_task.ik_max_iterations,
    )
    objective = WeightedObjective(max_path_error_weight_s_per_m=path_weight)
    reference_path, results = run_cartesian_line_payload_benchmark(task=task, dt=dt, objective=objective)
    best = next(result for result in results if result.name == "quickmove_truemove_torque_limited_path")
    metadata = rounded_door_metadata(task, reference_path)
    metadata.update(
        {
            "objective": asdict(objective),
            "selected_result": asdict(best),
            "benchmark_results": [asdict(result) for result in results],
        }
    )
    return reference_path.positions, dt * best.details["torque_report"]["time_scale"], metadata


def rounded_door_metadata(task: CartesianRoundedDoorTask, path: CartesianPath) -> dict[str, Any]:
    return {
        "path_shape": "rounded_door",
        "start_q": task.start_q,
        "width_y_m": task.width_y_m,
        "height_z_m": task.height_z_m,
        "corner_radius_m": task.corner_radius_m,
        "payload_kg": task.payload_kg,
        "body_name": task.body_name,
        "samples": task.samples,
        "tcp_start_m": path.start_tcp_m,
        "tcp_goal_m": path.goal_tcp_m,
        "ik_max_target_error_m": path.max_target_error_m,
        "ik_max_path_error_m": path.max_path_error_m,
        "ik_rms_path_error_m": path.rms_path_error_m,
        "ik_max_orientation_error_rad": path.max_orientation_error_rad,
        "ik_rms_orientation_error_rad": path.rms_orientation_error_rad,
    }
