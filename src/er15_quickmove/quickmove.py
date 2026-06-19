"""cuRobo-backed QuickMove-like planner for ER15-1400."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from er15_quickmove.config import (
    ER15_PUBLIC_LIMITS,
    ProjectPaths,
    QuickMoveProfile,
    load_er15_robot_config,
)
from er15_quickmove.control import JointControlLimits, build_control_limits
from er15_quickmove.metrics import TrajectoryLimitReport, summarize_joint_trajectory


@dataclass
class QuickMoveResult:
    result: Any
    report: TrajectoryLimitReport | None


class ER15QuickMovePlanner:
    """Small facade that turns cuRobo TrajOpt into a QuickMove-like mode."""

    def __init__(
        self,
        profile: QuickMoveProfile | None = None,
        paths: ProjectPaths | None = None,
        scene_model: str | dict[str, Any] | None = None,
        collision_cache: dict[str, int] | None = None,
        self_collision_check: bool = False,
    ):
        self.profile = profile or QuickMoveProfile()
        self.paths = paths or ProjectPaths()
        self.robot_config = load_er15_robot_config(self.paths)
        self.control_limits = build_control_limits(self.profile)
        self._apply_profile_to_robot_limits()

        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_cfg = DeviceCfg(device=device, dtype=torch.float32)
        cfg = MotionPlannerCfg.create(
            robot=self.robot_config,
            scene_model=scene_model,
            collision_cache=collision_cache,
            self_collision_check=self_collision_check,
            device_cfg=device_cfg,
            num_ik_seeds=self.profile.num_ik_seeds,
            num_trajopt_seeds=self.profile.num_trajopt_seeds,
            use_cuda_graph=self.profile.use_cuda_graph and torch.cuda.is_available(),
        )
        cfg.trajopt_solver_config.minimum_trajectory_dt = self.profile.minimum_trajectory_dt
        cfg.trajopt_solver_config.maximum_trajectory_dt = self.profile.maximum_trajectory_dt
        cfg.trajopt_solver_config.interpolation_dt = self.profile.interpolation_dt
        self.planner = MotionPlanner(cfg)

    @property
    def joint_names(self) -> list[str]:
        return list(self.planner.joint_names)

    @property
    def tool_frames(self) -> list[str]:
        return list(self.planner.tool_frames)

    @property
    def joint_control_limits(self) -> JointControlLimits:
        return self.control_limits

    def warmup(self, iterations: int = 3) -> None:
        self.planner.warmup(
            enable_graph=self.profile.use_cuda_graph and torch.cuda.is_available(),
            num_warmup_iterations=iterations,
        )

    def make_joint_state(self, position: list[float] | torch.Tensor):
        from curobo.types import JointState

        if not torch.is_tensor(position):
            position = torch.tensor(position, device=self.planner.device_cfg.device, dtype=torch.float32)
        if position.ndim == 1:
            position = position.unsqueeze(0)
        return JointState.from_position(position, joint_names=self.joint_names)

    def make_goal_pose(self, position: list[float], quaternion_wxyz: list[float] | None = None):
        from curobo.types import GoalToolPose

        q = quaternion_wxyz or [1.0, 0.0, 0.0, 0.0]
        device = self.planner.device_cfg.device
        pos = torch.tensor([[[[[position]]]]], device=device, dtype=torch.float32)
        quat = torch.tensor([[[[[q]]]]], device=device, dtype=torch.float32)
        return GoalToolPose(tool_frames=self.tool_frames, position=pos, quaternion=quat)

    def plan_pose(
        self,
        start_position: list[float] | torch.Tensor,
        goal_position: list[float],
        goal_quaternion_wxyz: list[float] | None = None,
        warmup: bool = False,
    ) -> QuickMoveResult:
        """Plan to a Cartesian tool pose with aggressive time finetuning."""

        if warmup:
            self.warmup()

        current_state = self.make_joint_state(start_position)
        goal_pose = self.make_goal_pose(goal_position, goal_quaternion_wxyz)

        ik_result = self.planner.ik_solver.solve_pose(
            goal_pose,
            return_seeds=self.profile.num_trajopt_seeds,
            current_state=current_state,
        )
        if torch.count_nonzero(ik_result.success) == 0:
            return QuickMoveResult(result=ik_result, report=None)

        seed_config = ik_result.solution
        if torch.count_nonzero(ik_result.success) < self.profile.num_trajopt_seeds:
            good = seed_config[ik_result.success][0:1, :].clone()
            seed_config[~ik_result.success][:, :] = good

        result = self.planner.trajopt_solver.solve_pose(
            goal_pose,
            current_state,
            seed_config=seed_config,
            use_implicit_goal=True,
            finetune_attempts=self.profile.finetune_attempts,
            finetune_dt_scale=self.profile.finetune_dt_scale,
            initial_iters=self.profile.initial_iters,
            time_optimal_iters=self.profile.time_optimal_iters,
            finetune_iters=self.profile.finetune_iters,
        )
        return QuickMoveResult(result=result, report=self._report(result))

    def plan_cspace(
        self,
        start_position: list[float] | torch.Tensor,
        goal_position: list[float] | torch.Tensor,
        warmup: bool = False,
    ) -> QuickMoveResult:
        """Plan directly in joint space with aggressive time finetuning."""

        if warmup:
            self.warmup()

        start = self.make_joint_state(start_position)
        goal = self.make_joint_state(goal_position)
        result = self.planner.trajopt_solver.solve_cspace(
            goal,
            start,
            finetune_attempts=self.profile.finetune_attempts,
            finetune_dt_scale=self.profile.finetune_dt_scale,
            initial_iters=self.profile.initial_iters,
            time_optimal_iters=self.profile.time_optimal_iters,
            finetune_iters=self.profile.finetune_iters,
        )
        return QuickMoveResult(result=result, report=self._report(result))

    def _apply_profile_to_robot_limits(self) -> None:
        cspace = self.robot_config["robot_cfg"]["kinematics"]["cspace"]
        cspace["velocity_scale"] = [self.profile.velocity_scale] * 6
        cspace["acceleration_scale"] = [1.0] * 6
        cspace["jerk_scale"] = [1.0] * 6

    def _report(self, result: Any) -> TrajectoryLimitReport | None:
        if result is None or not hasattr(result, "success") or not torch.any(result.success):
            return None

        plan = result.get_interpolated_plan()
        position = plan.position.squeeze(0)
        velocity = plan.velocity.squeeze(0) if plan.velocity is not None else None
        acceleration = plan.acceleration.squeeze(0) if plan.acceleration is not None else None
        jerk = plan.jerk.squeeze(0) if plan.jerk is not None else None

        velocity_limits = self.control_limits.velocity_upper_rad_s
        return summarize_joint_trajectory(
            position=position,
            dt=self.profile.interpolation_dt,
            velocity_limits=velocity_limits,
            acceleration_limits=None,
            jerk_limits=None,
            profile_name=self.profile.name,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
        )
