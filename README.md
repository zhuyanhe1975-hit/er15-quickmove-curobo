# ER15 QuickMove cuRobo

QuickMove-like motion generation for the EFORT ER15-1400 industrial arm using
cuRobo trajectory optimization.

This project intentionally does not claim to reproduce ABB RobotWare internals.
It implements the public behavior target: minimize cycle time by aggressively
using robot velocity and dynamics limits while preserving a feasible trajectory.
Acceleration and jerk are audited as trajectory properties, not direct hardware
limits.

## Status

- Target robot: EFORT ER15-1400, 6-axis, 15 kg payload, 1420 mm reach.
- Planning backend: cuRobo `MotionPlanner` / `TrajOptSolver`.
- Planning model: real ER15-1400 URDF/STL generated from
  `assets/er15_1400/ER15-1400-fulldyn-local.urdf`
  into `assets/er15_1400/ER15-1400-fulldyn-curobo.urdf`; only the zero joint
  velocity placeholders are filled with public ER15 speed limits for cuRobo.
- Dynamics/display model: real ER15-1400 MJCF/STL from
  `assets/er15_1400/er15-1400.mjcf.xml`.
- The project no longer carries or falls back to the old simplified cylinder
  arm model; planning, dynamics, and visualization all point at the real ER15
  assets.

Public ER15-1400 data used here comes from the EFORT ER15-1400 product
page and product leaflet PDF published by EFORT in 2025:

- Payload: 15 kg
- Reach: 1420 mm
- Repeatability: +/-0.03 mm
- Joint speed limits: J1 260 deg/s, J2 255 deg/s, J3 210 deg/s,
  J4 450 deg/s, J5 450 deg/s, J6 600 deg/s
- Joint ranges: J1 +/-170 deg, J2 +90/-160 deg, J3 +175/-85 deg,
  J4 +/-190 deg, J5 +/-130 deg, J6 +/-360 deg
- Published wrist payload torque limits: J4 42 N*m, J5 42 N*m, J6 20 N*m
- Published wrist payload inertia limits: J4 2 kg*m^2, J5 2 kg*m^2,
  J6 0.7 kg*m^2
- Engineering-default actuator torque limits for method validation:
  J1 1800 N*m, J2 1200 N*m, J3 700 N*m, J4 180 N*m, J5 140 N*m, J6 90 N*m

The public leaflet does not publish actuator/drive torque limits for J1-J6.
The actuator torque limits above are therefore deliberately marked as
engineering defaults, not vendor data. They are synchronized into the cuRobo
URDF `effort` fields and the control API so planning/control experiments have
reasonable saturation values. Acceleration and jerk are not treated as direct
robot limits; usable acceleration should come from torque-limited retiming.

Sources:

- EFORT product page: `https://efort.com.cn/index.php/product/product.html`
- ER15-1400 product leaflet PDF: `https://download.efort.com.cn:20250/pdf/web/upload/2025/05/06/17465086985666xiybb.pdf`

## Layout

```text
assets/er15_1400/              real ER15 URDF/MJCF/STL assets
configs/er15_1400_curobo.yml   cuRobo config pointing to real ER15 URDF/STL
examples/plan_cspace.py        minimal joint-space QuickMove-like demo
examples/compare_cspace_profiles.py
                                baseline vs QuickMove-like cycle-time report
examples/plan_pose.py          Cartesian pose planning demo
examples/benchmark_cartesian_line_payload.py
                                QuickMove+TrueMove rounded-door/payload benchmark
src/er15_quickmove/            Python package
tests/                         lightweight tests
```

## Install

Use the Python environment where cuRobo is already available:

```bash
cd /home/yhzhu/IndustrialRobotBase/er15-quickmove-curobo
python -m pip install -e .
```

If cuRobo is only available from the local source checkout:

```bash
export PYTHONPATH=/home/yhzhu/AI/NVlabs/curobo:$PYTHONPATH
```

Preferred environment for this workspace:

```bash
cd /home/yhzhu/IndustrialRobotBase/er15-quickmove-curobo
export PYTHONPATH=$PWD/src:/home/yhzhu/AI/NVlabs/curobo:$PYTHONPATH
TERM=xterm /home/yhzhu/isaaclab/isaaclab.sh -p -m pip install -e .
TERM=xterm /home/yhzhu/isaaclab/isaaclab.sh -p scripts/check_env.py
```

The `TERM=xterm` prefix avoids a non-interactive terminal issue where
`isaaclab.sh` tries to reset tabs while `TERM=dumb`.

## Run

Full comparison + visualization demo from the project root:

```bash
./run_quickmove_demo.sh
```

Joint-space cycle-time compression:

```bash
TERM=xterm /home/yhzhu/isaaclab/isaaclab.sh -p examples/plan_cspace.py
```

Baseline vs QuickMove-like comparison:

```bash
TERM=xterm /home/yhzhu/isaaclab/isaaclab.sh -p examples/compare_cspace_profiles.py
```

Torque-limited retiming audit:

```bash
TERM=xterm /home/yhzhu/isaaclab/isaaclab.sh -p examples/torque_limited_time_scaling.py --no-warmup
```


QuickMove+TrueMove fair benchmark on a rounded-door TCP path with rated payload:

```bash
TERM=xterm /home/yhzhu/isaaclab/isaaclab.sh -p examples/benchmark_cartesian_line_payload.py
```

This is now the preferred comparison for method validation. It fixes the task as
a reachable 500 mm x 300 mm rounded-door path with 75 mm top corner radius,
keeps the TCP orientation fixed over the whole path, injects the rated 15 kg
payload into the MuJoCo dynamics model at runtime, and ranks methods by a
weighted objective:

```text
objective = cycle_time_weight * duration_s
          + max_path_error_weight_s_per_m * max_tcp_path_error_m
          + max_orientation_error_weight_s_per_rad * max_tcp_orientation_error_rad
```

The default weight is `cycle_time + 20 s/m * max TCP path error`. This makes the
QuickMove part prefer shorter cycle time while the TrueMove part penalizes
leaving the commanded door path.

Legacy same-start/same-goal benchmark against Ruckig/TOPP-RA/MoveIt-style baselines:

```bash
TERM=xterm /home/yhzhu/isaaclab/isaaclab.sh -p examples/benchmark_same_task.py --no-warmup
```

The IsaacLab environment used here does not currently include the real Python
packages `ruckig`, `toppra`, or ROS MoveIt. The benchmark therefore reports
those adapters as skipped and includes local like-for-like baselines:
`ruckig_like_quintic`, `toppra_like_path_retiming`, and
`moveit_like_iterative_parabolic`. These are useful for method comparison on the
same ER15 start/goal task, but should be replaced by direct adapters once the
real packages are installed.

This legacy benchmark is retained for manual reference, but the Cartesian
line/payload benchmark above is the preferred health metric because it includes
the TrueMove path-accuracy term.

This torque-limited pass removes acceleration/jerk as direct hardware limits.
It audits the cuRobo path with MuJoCo inverse dynamics and searches for the
fastest uniform retiming that respects joint velocity and engineering-default
torque limits. A `--smoothstep` option is available for conservative path-law
experiments, but the default preserves the cuRobo time law and pushes it toward
the velocity/torque boundary.

MuJoCo video render with the real ER15-1400 MJCF/STL model. By default this
plays the rounded-door QuickMove+TrueMove trajectory used by the fair benchmark:

```bash
./run_mujoco_visualization.sh video --no-warmup
```

MuJoCo interactive viewer, requires a desktop display:

```bash
./run_mujoco_visualization.sh viewer --no-warmup
```

Pass `--trajectory cspace` only when you explicitly want to inspect the legacy
joint start/goal demo. MuJoCo is the recommended fast visualization path for
trajectory playback. Isaac Sim is available in the environment and is better
suited for later high-fidelity scene integration, sensor rendering, and
USD-based workflows.

Cartesian pose planning:

```bash
TERM=xterm /home/yhzhu/isaaclab/isaaclab.sh -p examples/plan_pose.py
```

Both examples print duration and limit-utilization metrics. The planner uses
CUDA when available because cuRobo is GPU-first.

Current smoke-test result in the IsaacLab environment:

```text
Planning succeeded
duration_s=2.000
waypoints=201
peak_velocity_ratio=0.214
peak_acceleration_ratio=0.894
peak_jerk_ratio=0.324
```

Current comparison result after removing direct acceleration/jerk limits:

```text
baseline_duration_s=1.200
quickmove_duration_s=0.600
saved_time_s=0.600
saved_percent=50.0
```

Current QuickMove+TrueMove fair benchmark result on a rounded-door TCP path with
15 kg payload and fixed TCP orientation:

```text
quickmove_truemove_torque_limited_path: objective=0.5766 duration=0.5762 s path_error=0.017 mm orientation_error=0.042 mrad
moveit_like_parabolic_path_law:         objective=0.7822 duration=0.7818 s path_error=0.016 mm orientation_error=0.042 mrad
ruckig_like_quintic_path_law:           objective=0.9502 duration=0.9498 s path_error=0.017 mm orientation_error=0.042 mrad
endpoint_only_toppra_like_path_retiming objective=5.1654 duration=0.1328 s path_error=250.465 mm orientation_error=23.341 mrad
```

The endpoint-only result is intentionally shown: it can look fast on raw cycle
time, but it does not satisfy the same rounded-door TrueMove task and is
penalized by the path-error term.

Legacy same-start/same-goal joint-space benchmark result:

```text
curobo_quickmove_torque_limited: 0.481 s
ruckig_like_quintic:             0.478 s
toppra_like_path_retiming:       0.256 s
moveit_like_iterative_parabolic: 0.501 s
```

The legacy TOPP-RA-like baseline retimes a straight joint-space path, so it is a
same-start/same-goal task comparison rather than an identical TCP-path
comparison. The direct `ruckig`, `toppra`, and MoveIt adapters are reported as
skipped when those optional dependencies are not installed.

Current torque-limited retiming result on the cuRobo quickmove path:

```text
quickmove_duration_s=0.600
torque_limited_duration_s=0.480
time_scale=0.801
peak_torque_ratio=0.9997
peak_velocity_ratio=0.8746
limiting_joint=joint_2
```

The J1 static-gravity sanity check is also important: because J1 is the vertical
base yaw axis, its static gravity torque should be near zero. The torque audit
uses `tau = M(q) qdd + qfrc_bias(q, qd)` after `mj_forward`, which gives J1 = 0
for static poses and places the gravity load primarily on J2/J3.

## How It Maps To QuickMove

ABB QuickMove is treated here as a behavior target:

1. Solve a feasible path with IK, graph seeding, collision constraints, and
   B-spline trajectory optimization.
2. Re-run time-optimal finetune passes that shrink trajectory `dt`.
3. Validate the interpolated plan against velocity, torque/dynamics, collision,
   and convergence constraints. Acceleration and jerk are not direct limits.
4. Rank solutions by feasibility, motion time, and TCP path accuracy when TrueMove mode is enabled.

For production use, generate cuRobo collision spheres from the real STL meshes
and tune `QuickMoveProfile` plus dynamics limits against measured ER15-1400
controller traces.
