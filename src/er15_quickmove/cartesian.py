"""Cartesian straight-line task helpers for QuickMove/TrueMove benchmarks."""

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
    ik_max_iterations: int = 120


@dataclass(frozen=True)
class CartesianLinePath:
    positions: np.ndarray
    tcp_positions_m: np.ndarray
    reference_tcp_positions_m: np.ndarray
    max_line_error_m: float
    rms_line_error_m: float
    max_target_error_m: float

    @property
    def start_tcp_m(self) -> list[float]:
        return self.reference_tcp_positions_m[0].tolist()

    @property
    def goal_tcp_m(self) -> list[float]:
        return self.reference_tcp_positions_m[-1].tolist()


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

    def solve_line_path(self, task: CartesianLineTask) -> CartesianLinePath:
        if task.samples < 3:
            raise ValueError("Cartesian line task needs at least 3 samples")
        q = np.asarray(task.start_q, dtype=float).copy()
        body_id = self.body_id(task.body_name)
        start_tcp = self.fk_body_position(q, task.body_name)
        delta = np.asarray(task.delta_xyz_m, dtype=float)
        references = start_tcp[None, :] + np.linspace(0.0, 1.0, task.samples)[:, None] * delta[None, :]

        positions = np.empty((task.samples, len(q)), dtype=float)
        achieved = np.empty_like(references)
        target_errors = np.empty(task.samples, dtype=float)
        q_lower = self.model.jnt_range[: len(q), 0]
        q_upper = self.model.jnt_range[: len(q), 1]
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        damping = 2e-4

        for i, target in enumerate(references):
            for _ in range(task.ik_max_iterations):
                self.data.qpos[: len(q)] = q
                self.data.qvel[:] = 0.0
                self.data.qacc[:] = 0.0
                self.mujoco.mj_forward(self.model, self.data)
                current = np.array(self.data.xpos[body_id], dtype=float)
                error = target - current
                if float(np.linalg.norm(error)) <= task.ik_tolerance_m:
                    break
                self.mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
                j = jacp[:, : len(q)]
                dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(3), error)
                max_step = float(np.max(np.abs(dq)))
                if max_step > 0.04:
                    dq *= 0.04 / max_step
                q = np.clip(q + dq, q_lower, q_upper)

            self.data.qpos[: len(q)] = q
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            self.mujoco.mj_forward(self.model, self.data)
            achieved[i] = np.array(self.data.xpos[body_id], dtype=float)
            target_errors[i] = float(np.linalg.norm(references[i] - achieved[i]))
            positions[i] = q

        max_line_error, rms_line_error = cartesian_line_error(achieved, references[0], references[-1])
        return CartesianLinePath(
            positions=positions,
            tcp_positions_m=achieved,
            reference_tcp_positions_m=references,
            max_line_error_m=max_line_error,
            rms_line_error_m=rms_line_error,
            max_target_error_m=float(np.max(target_errors)),
        )


def cartesian_line_error(tcp_positions_m: np.ndarray, line_start_m: np.ndarray, line_goal_m: np.ndarray) -> tuple[float, float]:
    tcp = np.asarray(tcp_positions_m, dtype=float)
    start = np.asarray(line_start_m, dtype=float)
    goal = np.asarray(line_goal_m, dtype=float)
    line = goal - start
    denom = float(np.dot(line, line))
    if denom <= 1e-12:
        distances = np.linalg.norm(tcp - start[None, :], axis=1)
    else:
        alpha = np.clip(((tcp - start[None, :]) @ line) / denom, 0.0, 1.0)
        closest = start[None, :] + alpha[:, None] * line[None, :]
        distances = np.linalg.norm(tcp - closest, axis=1)
    return float(np.max(distances)), float(np.sqrt(np.mean(distances**2)))


def default_line_task() -> CartesianLineTask:
    return CartesianLineTask(
        start_q=[0.0, -1.0, 1.5, 0.0, 0.4, 0.0],
        delta_xyz_m=[-0.08, -0.04, 0.02],
        samples=101,
        payload_kg=15.0,
    )
