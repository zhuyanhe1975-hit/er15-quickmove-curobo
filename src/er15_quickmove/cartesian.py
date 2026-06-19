"""Cartesian reference-path helpers for QuickMove/TrueMove benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from er15_quickmove.config import ProjectPaths


@dataclass(frozen=True)
class CartesianLineTask:
    start_q: list[float]
    delta_xyz_m: list[float]
    samples: int = 101
    body_name: str = "link_6"
    payload_kg: float = 15.0
    ik_tolerance_m: float = 5e-4
    ik_orientation_tolerance_rad: float = 5e-4
    ik_max_iterations: int = 120


@dataclass(frozen=True)
class CartesianRoundedDoorTask:
    start_q: list[float]
    width_y_m: float = 0.12
    height_z_m: float = 0.08
    corner_radius_m: float = 0.02
    samples: int = 161
    body_name: str = "link_6"
    payload_kg: float = 15.0
    ik_tolerance_m: float = 5e-4
    ik_orientation_tolerance_rad: float = 5e-4
    ik_max_iterations: int = 160


@dataclass(frozen=True)
class CartesianPath:
    positions: np.ndarray
    tcp_positions_m: np.ndarray
    reference_tcp_positions_m: np.ndarray
    tcp_rotations: np.ndarray
    reference_rotation: np.ndarray
    max_line_error_m: float
    rms_line_error_m: float
    max_target_error_m: float
    max_orientation_error_rad: float
    rms_orientation_error_rad: float

    @property
    def start_tcp_m(self) -> list[float]:
        return self.reference_tcp_positions_m[0].tolist()

    @property
    def goal_tcp_m(self) -> list[float]:
        return self.reference_tcp_positions_m[-1].tolist()

    @property
    def max_path_error_m(self) -> float:
        return self.max_line_error_m

    @property
    def rms_path_error_m(self) -> float:
        return self.rms_line_error_m


CartesianLinePath = CartesianPath


class MujocoCartesianKinematics:
    """MuJoCo FK/Jacobian wrapper using the repository ER15 MJCF model."""

    def __init__(self, paths: ProjectPaths | None = None):
        self.paths = paths or ProjectPaths()
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(self.paths.robot_mjcf))
        self.data = mujoco.MjData(self.model)

    def body_id(self, body_name: str) -> int:
        body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"unknown MuJoCo body: {body_name}")
        return int(body_id)

    def fk_body_position(self, q: np.ndarray, body_name: str = "link_6") -> np.ndarray:
        body_id = self.body_id(body_name)
        self.data.qpos[: len(q)] = q
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return np.array(self.data.xpos[body_id], dtype=float)

    def body_positions(self, positions: np.ndarray, body_name: str = "link_6") -> np.ndarray:
        return np.vstack([self.fk_body_position(np.asarray(q, dtype=float), body_name) for q in positions])

    def body_rotations(self, positions: np.ndarray, body_name: str = "link_6") -> np.ndarray:
        return np.stack([self.fk_body_pose(np.asarray(q, dtype=float), body_name)[1] for q in positions])

    def solve_line_path(self, task: CartesianLineTask) -> CartesianPath:
        start_tcp, start_rotation = self.fk_body_pose(np.asarray(task.start_q, dtype=float), task.body_name)
        delta = np.asarray(task.delta_xyz_m, dtype=float)
        references = start_tcp[None, :] + np.linspace(0.0, 1.0, task.samples)[:, None] * delta[None, :]
        return self.solve_reference_path(
            references,
            start_q=task.start_q,
            body_name=task.body_name,
            ik_tolerance_m=task.ik_tolerance_m,
            ik_orientation_tolerance_rad=task.ik_orientation_tolerance_rad,
            ik_max_iterations=task.ik_max_iterations,
            target_rotation=start_rotation,
        )

    def solve_rounded_door_path(self, task: CartesianRoundedDoorTask) -> CartesianPath:
        start_tcp, start_rotation = self.fk_body_pose(np.asarray(task.start_q, dtype=float), task.body_name)
        references = rounded_door_reference_path(
            start_tcp,
            width_y_m=task.width_y_m,
            height_z_m=task.height_z_m,
            corner_radius_m=task.corner_radius_m,
            samples=task.samples,
        )
        return self.solve_reference_path(
            references,
            start_q=task.start_q,
            body_name=task.body_name,
            ik_tolerance_m=task.ik_tolerance_m,
            ik_orientation_tolerance_rad=task.ik_orientation_tolerance_rad,
            ik_max_iterations=task.ik_max_iterations,
            target_rotation=start_rotation,
        )

    def solve_reference_path(
        self,
        references: np.ndarray,
        start_q: list[float],
        body_name: str = "link_6",
        ik_tolerance_m: float = 5e-4,
        ik_orientation_tolerance_rad: float = 5e-4,
        ik_max_iterations: int = 120,
        target_rotation: np.ndarray | None = None,
    ) -> CartesianPath:
        if references.shape[0] < 3:
            raise ValueError("Cartesian reference path needs at least 3 samples")
        q = np.asarray(start_q, dtype=float).copy()
        body_id = self.body_id(body_name)

        positions = np.empty((references.shape[0], len(q)), dtype=float)
        achieved = np.empty_like(references)
        rotations = np.empty((references.shape[0], 3, 3), dtype=float)
        target_errors = np.empty(references.shape[0], dtype=float)
        orientation_errors = np.empty(references.shape[0], dtype=float)
        q_lower = self.model.jnt_range[: len(q), 0]
        q_upper = self.model.jnt_range[: len(q), 1]
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        damping = 2e-4
        orientation_weight_m_per_rad = 0.25
        if target_rotation is None:
            _, target_rotation = self.fk_body_pose(q, body_name)

        for i, target in enumerate(references):
            for _ in range(ik_max_iterations):
                self.data.qpos[: len(q)] = q
                self.data.qvel[:] = 0.0
                self.data.qacc[:] = 0.0
                self.mujoco.mj_forward(self.model, self.data)
                current = np.array(self.data.xpos[body_id], dtype=float)
                current_rotation = np.array(self.data.xmat[body_id], dtype=float).reshape(3, 3)
                position_error = target - current
                rotation_error = rotation_vector_error(current_rotation, target_rotation)
                if (
                    float(np.linalg.norm(position_error)) <= ik_tolerance_m
                    and float(np.linalg.norm(rotation_error)) <= ik_orientation_tolerance_rad
                ):
                    break
                self.mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
                j = np.vstack([jacp[:, : len(q)], orientation_weight_m_per_rad * jacr[:, : len(q)]])
                error = np.concatenate([position_error, orientation_weight_m_per_rad * rotation_error])
                dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(6), error)
                max_step = float(np.max(np.abs(dq)))
                if max_step > 0.03:
                    dq *= 0.03 / max_step
                q = np.clip(q + dq, q_lower, q_upper)

            self.data.qpos[: len(q)] = q
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
            achieved[i] = np.array(self.data.xpos[body_id], dtype=float)
            rotations[i] = np.array(self.data.xmat[body_id], dtype=float).reshape(3, 3)
            target_errors[i] = float(np.linalg.norm(references[i] - achieved[i]))
            orientation_errors[i] = float(np.linalg.norm(rotation_vector_error(rotations[i], target_rotation)))
            positions[i] = q

        max_path_error, rms_path_error = cartesian_path_error(achieved, references)
        return CartesianPath(
            positions=positions,
            tcp_positions_m=achieved,
            reference_tcp_positions_m=references,
            tcp_rotations=rotations,
            reference_rotation=target_rotation,
            max_line_error_m=max_path_error,
            rms_line_error_m=rms_path_error,
            max_target_error_m=float(np.max(target_errors)),
            max_orientation_error_rad=float(np.max(orientation_errors)),
            rms_orientation_error_rad=float(np.sqrt(np.mean(orientation_errors**2))),
        )

    def fk_body_pose(self, q: np.ndarray, body_name: str = "link_6") -> tuple[np.ndarray, np.ndarray]:
        body_id = self.body_id(body_name)
        self.data.qpos[: len(q)] = q
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return (
            np.array(self.data.xpos[body_id], dtype=float),
            np.array(self.data.xmat[body_id], dtype=float).reshape(3, 3),
        )


def rounded_door_reference_path(
    start_tcp_m: np.ndarray,
    width_y_m: float,
    height_z_m: float,
    corner_radius_m: float,
    samples: int,
) -> np.ndarray:
    """Build a rounded-corner door path in the TCP Y-Z plane.

    The path starts at the lower-right post, moves upward, blends through a
    top-right quarter circle, moves laterally in the TCP Y direction, blends
    through a top-left quarter circle, and finishes downward at the lower-left
    post. Placing the door in the Y-Z plane keeps the ER15 away from the
    outer X-reach boundary while still exercising line and circular blending.
    """

    if samples < 5:
        raise ValueError("rounded door path needs at least 5 samples")
    if width_y_m <= 0.0 or height_z_m <= 0.0:
        raise ValueError("door width and height must be positive")
    radius = float(corner_radius_m)
    if radius <= 0.0 or radius >= min(width_y_m, height_z_m) / 2.0:
        raise ValueError("corner radius must be positive and smaller than half the door width/height")

    dense: list[np.ndarray] = []
    start = np.asarray(start_tcp_m, dtype=float)

    def point(y: float, z: float) -> np.ndarray:
        return start + np.array([0.0, y, z], dtype=float)

    dense.extend(point(0.0, z) for z in np.linspace(0.0, height_z_m - radius, 40, endpoint=False))
    center_right = np.array([0.0, -radius, height_z_m - radius], dtype=float)
    for theta in np.linspace(0.0, np.pi / 2.0, 32, endpoint=False):
        dense.append(start + center_right + np.array([0.0, radius * np.cos(theta), radius * np.sin(theta)]))
    dense.extend(point(y, height_z_m) for y in np.linspace(-radius, -(width_y_m - radius), 60, endpoint=False))
    center_left = np.array([0.0, -(width_y_m - radius), height_z_m - radius], dtype=float)
    for theta in np.linspace(np.pi / 2.0, np.pi, 32, endpoint=False):
        dense.append(start + center_left + np.array([0.0, radius * np.cos(theta), radius * np.sin(theta)]))
    dense.extend(point(-width_y_m, z) for z in np.linspace(height_z_m - radius, 0.0, 40))

    return _resample_polyline(np.vstack(dense), samples)


def _resample_polyline(points: np.ndarray, samples: int) -> np.ndarray:
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(deltas)])
    if s[-1] <= 1e-12:
        raise ValueError("reference path length is zero")
    target = np.linspace(0.0, s[-1], samples)
    out = np.empty((samples, points.shape[1]), dtype=float)
    for axis in range(points.shape[1]):
        out[:, axis] = np.interp(target, s, points[:, axis])
    return out


def cartesian_path_error(tcp_positions_m: np.ndarray, reference_tcp_positions_m: np.ndarray) -> tuple[float, float]:
    tcp = np.asarray(tcp_positions_m, dtype=float)
    reference = np.asarray(reference_tcp_positions_m, dtype=float)
    if reference.shape[0] < 2:
        distances = np.linalg.norm(tcp - reference[0][None, :], axis=1)
    else:
        distances = np.array([_distance_to_polyline(point, reference) for point in tcp], dtype=float)
    return float(np.max(distances)), float(np.sqrt(np.mean(distances**2)))


def rotation_vector_error(current_rotation: np.ndarray, target_rotation: np.ndarray) -> np.ndarray:
    """Small-angle orientation error that drives current frame toward target."""

    current = np.asarray(current_rotation, dtype=float)
    target = np.asarray(target_rotation, dtype=float)
    return 0.5 * (
        np.cross(current[:, 0], target[:, 0])
        + np.cross(current[:, 1], target[:, 1])
        + np.cross(current[:, 2], target[:, 2])
    )


def _distance_to_polyline(point: np.ndarray, polyline: np.ndarray) -> float:
    starts = polyline[:-1]
    ends = polyline[1:]
    segments = ends - starts
    denom = np.einsum("ij,ij->i", segments, segments)
    safe = np.maximum(denom, 1e-12)
    alpha = np.clip(np.einsum("ij,ij->i", point[None, :] - starts, segments) / safe, 0.0, 1.0)
    closest = starts + alpha[:, None] * segments
    return float(np.min(np.linalg.norm(point[None, :] - closest, axis=1)))


def cartesian_line_error(tcp_positions_m: np.ndarray, line_start_m: np.ndarray, line_goal_m: np.ndarray) -> tuple[float, float]:
    return cartesian_path_error(tcp_positions_m, np.vstack([line_start_m, line_goal_m]))


def default_line_task() -> CartesianLineTask:
    return CartesianLineTask(
        start_q=[0.0, -1.0, 1.5, 0.0, 0.4, 0.0],
        delta_xyz_m=[-0.08, -0.04, 0.02],
        samples=101,
        payload_kg=15.0,
    )


def default_path_task() -> CartesianRoundedDoorTask:
    return CartesianRoundedDoorTask(
        start_q=[0.023, 0.075, -0.479, 0.669, -0.767, 0.0],
        width_y_m=0.50,
        height_z_m=0.30,
        corner_radius_m=0.075,
        samples=241,
        payload_kg=15.0,
    )
