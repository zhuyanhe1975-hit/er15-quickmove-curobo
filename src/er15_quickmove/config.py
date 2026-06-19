"""Project configuration and ER15-1400 public limit data."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # Isaac Sim kit Python may not ship PyYAML.
    yaml = None


def deg_to_rad(value: float) -> float:
    return value * pi / 180.0


REAL_ER15_MODEL_ROOT = Path(__file__).resolve().parents[2] / "assets" / "er15_1400"
REAL_ER15_SOURCE_URDF = REAL_ER15_MODEL_ROOT / "ER15-1400-fulldyn-local.urdf"
REAL_ER15_CUROBO_URDF = REAL_ER15_MODEL_ROOT / "ER15-1400-fulldyn-curobo.urdf"
REAL_ER15_MJCF = REAL_ER15_MODEL_ROOT / "er15-1400.mjcf.xml"


ER15_LIMIT_SOURCE = {
    "vendor": "EFORT ER15-1400 robot product leaflet",
    "product_page": "https://efort.com.cn/index.php/product/product.html",
    "pdf": "https://download.efort.com.cn:20250/pdf/web/upload/2025/05/06/17465086985666xiybb.pdf",
    "pdf_creation_date": "2025-04-28",
}


ER15_PUBLIC_LIMITS = {
    "joint_names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    "position_lower_rad": [
        deg_to_rad(-170),
        deg_to_rad(-160),
        deg_to_rad(-85),
        deg_to_rad(-190),
        deg_to_rad(-130),
        deg_to_rad(-360),
    ],
    "position_upper_rad": [
        deg_to_rad(170),
        deg_to_rad(90),
        deg_to_rad(175),
        deg_to_rad(190),
        deg_to_rad(130),
        deg_to_rad(360),
    ],
    "velocity_upper_deg_s": [260, 255, 210, 450, 450, 600],
    "velocity_upper_rad_s": [
        deg_to_rad(260),
        deg_to_rad(255),
        deg_to_rad(210),
        deg_to_rad(450),
        deg_to_rad(450),
        deg_to_rad(600),
    ],
    # The public ER15-1400 leaflet publishes wrist payload torque, not actuator
    # torque. These actuator torque values are engineering defaults for method
    # validation, sized for method validation on a 15 kg / 1.42 m class industrial arm.
    "actuator_torque_upper_nm": [1800.0, 1200.0, 700.0, 180.0, 140.0, 90.0],
    "actuator_torque_source": "engineering_default_not_vendor_published",
    "wrist_load_torque_upper_nm": [None, None, None, 42.0, 42.0, 20.0],
    "wrist_load_inertia_upper_kgm2": [None, None, None, 2.0, 2.0, 0.7],
    "source": ER15_LIMIT_SOURCE,
}


@dataclass(frozen=True)
class ProjectPaths:
    root: Path = Path(__file__).resolve().parents[2]

    @property
    def robot_config(self) -> Path:
        return self.root / "configs" / "er15_1400_curobo.yml"

    @property
    def robot_urdf(self) -> Path:
        return REAL_ER15_CUROBO_URDF

    @property
    def robot_source_urdf(self) -> Path:
        return REAL_ER15_SOURCE_URDF

    @property
    def robot_mjcf(self) -> Path:
        return REAL_ER15_MJCF

    @property
    def robot_asset_root(self) -> Path:
        return REAL_ER15_MODEL_ROOT


@dataclass(frozen=True)
class QuickMoveProfile:
    """Aggressive time-optimal settings layered over cuRobo TrajOpt."""

    name: str = "quickmove"
    minimum_trajectory_dt: float = 0.004
    maximum_trajectory_dt: float = 0.18
    interpolation_dt: float = 0.01
    num_trajopt_seeds: int = 8
    num_ik_seeds: int = 48
    max_attempts: int = 4
    finetune_attempts: int = 5
    finetune_dt_scale: float = 0.72
    initial_iters: int | None = None
    time_optimal_iters: int | None = 125
    finetune_iters: int | None = 50
    use_cuda_graph: bool = True

    # Public speed data exists; public acceleration/jerk data is not always in
    # flyers. These scales let integration users tune conservatism without
    # editing robot YAML.
    velocity_scale: float = 0.95
    # Acceleration and jerk are not modeled as direct robot limits. They are
    # left effectively unbounded in cuRobo; torque-limited retiming determines
    # usable acceleration from inverse dynamics.
    acceleration_scale: float = 1.0
    jerk_scale: float = 1.0
    velocity_limit_source: str = "efort_public_leaflet"
    torque_scale: float = 0.80


def baseline_profile() -> QuickMoveProfile:
    """Conservative reference profile for cycle-time comparisons."""

    return QuickMoveProfile(
        name="baseline",
        minimum_trajectory_dt=0.012,
        maximum_trajectory_dt=0.24,
        interpolation_dt=0.01,
        num_trajopt_seeds=4,
        num_ik_seeds=24,
        finetune_attempts=1,
        finetune_dt_scale=0.90,
        time_optimal_iters=100,
        finetune_iters=50,
        velocity_scale=0.55,
        torque_scale=0.50,
    )


def quickmove_profile() -> QuickMoveProfile:
    """Default QuickMove-like profile for ER15-1400 cycle-time compression."""

    return QuickMoveProfile()


def load_er15_robot_config(paths: ProjectPaths | None = None) -> dict[str, Any]:
    """Load the cuRobo robot config and patch project-local absolute paths."""

    paths = paths or ProjectPaths()
    if yaml is None:
        data = _fallback_er15_robot_config()
    else:
        with paths.robot_config.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)

    kinematics = data["robot_cfg"]["kinematics"]
    kinematics["urdf_path"] = str(paths.robot_urdf)
    kinematics["asset_root_path"] = str(paths.robot_asset_root)
    return data


def _fallback_er15_robot_config() -> dict[str, Any]:
    """Minimal config used when PyYAML is unavailable.

    The YAML file carries richer collision spheres. This fallback keeps the
    planner usable in Isaac Sim kit Python environments that already provide
    torch and cuRobo but omit PyYAML.
    """

    return {
        "robot_cfg": {
            "kinematics": {
                "format_version": 2.0,
                "asset_root_path": "",
                "urdf_path": "",
                "base_link": "base_link",
                "tool_frames": ["link_6"],
                "mesh_link_names": [
                    "base_link",
                    "link_1",
                    "link_2",
                    "link_3",
                    "link_4",
                    "link_5",
                    "link_6",
                ],
                "collision_link_names": [],
                "collision_spheres": None,
                "collision_sphere_buffer": 0.0,
                "self_collision_buffer": None,
                "self_collision_ignore": None,
                "lock_joints": None,
                "cspace": {
                    "joint_names": ER15_PUBLIC_LIMITS["joint_names"],
                    "cspace_distance_weight": [1.0, 1.0, 1.0, 0.6, 0.6, 0.4],
                    "null_space_weight": [1.0, 1.0, 1.0, 0.5, 0.5, 0.4],
                    "max_acceleration": 1000000.0,
                    "max_jerk": 1000000.0,
                    "velocity_scale": [0.95] * 6,
                    "acceleration_scale": [1.0] * 6,
                    "jerk_scale": [1.0] * 6,
                    "position_limit_clip": 0.05,
                    "default_joint_position": [0.0, -1.0, 1.35, 0.0, 0.65, 0.0],
                },
            }
        }
    }
