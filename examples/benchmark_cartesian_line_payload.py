from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from er15_quickmove.benchmark import WeightedObjective, run_cartesian_line_payload_benchmark
from er15_quickmove.cartesian import CartesianLineTask, default_line_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ER15 QuickMove+TrueMove on a loaded TCP straight line")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--payload-kg", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--path-weight", type=float, default=20.0)
    args = parser.parse_args()

    base_task = default_line_task()
    task = CartesianLineTask(
        start_q=base_task.start_q,
        delta_xyz_m=base_task.delta_xyz_m,
        samples=base_task.samples,
        body_name=base_task.body_name,
        payload_kg=args.payload_kg,
        ik_tolerance_m=base_task.ik_tolerance_m,
        ik_max_iterations=base_task.ik_max_iterations,
    )
    objective = WeightedObjective(max_path_error_weight_s_per_m=args.path_weight)
    line_path, results = run_cartesian_line_payload_benchmark(task=task, dt=args.dt, objective=objective)

    payload = {
        "task": {
            "start_q": task.start_q,
            "delta_xyz_m": task.delta_xyz_m,
            "payload_kg": task.payload_kg,
            "body_name": task.body_name,
            "samples": task.samples,
            "tcp_start_m": line_path.start_tcp_m,
            "tcp_goal_m": line_path.goal_tcp_m,
            "ik_max_target_error_m": line_path.max_target_error_m,
            "ik_max_line_error_m": line_path.max_line_error_m,
            "ik_rms_line_error_m": line_path.rms_line_error_m,
        },
        "objective": asdict(objective),
        "results": [asdict(result) for result in results],
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    print("Cartesian line payload benchmark")
    print(f"payload_kg={task.payload_kg:.1f} line_error={line_path.max_line_error_m * 1000.0:.3f} mm")
    for result in sorted(results, key=lambda item: float("inf") if item.objective_score is None else item.objective_score):
        if result.duration_s is None:
            print(f"{result.name}: {result.status} ({result.details.get('reason', 'n/a')})")
            continue
        print(
            f"{result.name}: objective={result.objective_score:.4f} duration={result.duration_s:.4f}s "
            f"max_path_error={result.max_path_error_m * 1000.0:.3f}mm "
            f"torque={result.peak_torque_ratio:.3f} velocity={result.peak_velocity_ratio:.3f} "
            f"limit={result.limiting_joint}"
        )


if __name__ == "__main__":
    main()
