# ER15 QuickMove cuRobo

QuickMove-like motion generation for the EFORT ER15-1400 industrial arm using
cuRobo trajectory optimization.

This project intentionally does not claim to reproduce ABB RobotWare internals.
It implements the public behavior target: minimize cycle time by aggressively
using the robot velocity, acceleration, jerk, collision, and optional dynamics
limits while preserving a feasible trajectory.

## Status

- Target robot: EFORT ER15-1400, 6-axis, 15 kg payload, 1420 mm reach.
- Planning backend: cuRobo `MotionPlanner` / `TrajOptSolver`.
- Planning model: real ER15-1400 URDF/STL generated from
  `assets/er15_1400/ER15-1400-fulldyn-local.urdf`
  into `assets/er15_1400/ER15-1400-fulldyn-curobo.urdf`; only the zero joint
  velocity placeholders are filled with public ER15 speed limits for cuRobo.
- Dynamics/display model: real ER15-1400 MJCF/STL from
  `assets/er15_1400/er15-1400.mjcf.xml`.
- The old project-local cylinder URDF remains only as a fallback/reference asset;
  default planning and visualization paths no longer point to it.

Public ER15-1400 data used here:

- Payload: 15 kg
- Reach: 1420 mm
- Repeatability: about +/-0.03 mm
- Joint speed limits: J1 260 deg/s, J2 255 deg/s, J3 210 deg/s,
  J4 450 deg/s, J5 450 deg/s, J6 600 deg/s
- Joint ranges: J1 +/-170 deg, J2 +90/-160 deg, J3 +175/-85 deg,
  J4 +/-190 deg, J5 +/-130 deg, J6 +/-360 deg

The acceleration, jerk, link geometry, mass, inertia, and collision spheres are
engineering defaults until official data is supplied.

## Layout

```text
assets/er15_1400/              legacy approximate URDF fallback/reference
configs/er15_1400_curobo.yml   cuRobo config pointing to real ER15 URDF/STL
examples/plan_cspace.py        minimal joint-space QuickMove-like demo
examples/compare_cspace_profiles.py
                                baseline vs QuickMove-like cycle-time report
examples/plan_pose.py          Cartesian pose planning demo
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


MuJoCo video render with the real ER15-1400 MJCF/STL model:

```bash
./run_mujoco_visualization.sh video --no-warmup
```

MuJoCo interactive viewer, requires a desktop display:

```bash
./run_mujoco_visualization.sh viewer --no-warmup
```

MuJoCo is the recommended fast visualization path for trajectory playback. Isaac Sim is available in the environment and is better suited for later high-fidelity scene integration, sensor rendering, and USD-based workflows.

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

Current comparison result:

```text
baseline_duration_s=2.800
quickmove_duration_s=2.000
saved_time_s=0.800
saved_percent=28.6
```

## How It Maps To QuickMove

ABB QuickMove is treated here as a behavior target:

1. Solve a feasible path with IK, graph seeding, collision constraints, and
   B-spline trajectory optimization.
2. Re-run time-optimal finetune passes that shrink trajectory `dt`.
3. Validate the interpolated plan against velocity, acceleration, jerk,
   collision, and convergence constraints.
4. Rank solutions by feasibility and motion time.

For production use, generate cuRobo collision spheres from the real STL meshes
and tune `QuickMoveProfile` plus dynamics limits against measured ER15-1400
controller traces. The old hand-authored cylinder-style collision spheres are
not enabled by default.
