"""Control-side limits derived from public ER15-1400 data."""

from __future__ import annotations

from dataclasses import dataclass

from er15_quickmove.config import ER15_PUBLIC_LIMITS, QuickMoveProfile


@dataclass(frozen=True)
class JointControlLimits:
    joint_names: list[str]
    position_lower_rad: list[float]
    position_upper_rad: list[float]
    velocity_upper_rad_s: list[float]
    actuator_torque_upper_nm: list[float | None]
    wrist_load_torque_upper_nm: list[float | None]
    wrist_load_inertia_upper_kgm2: list[float | None]
    source: dict[str, str]

    def clamp_velocity(self, command_rad_s: list[float]) -> list[float]:
        if len(command_rad_s) != len(self.velocity_upper_rad_s):
            raise ValueError("velocity command must have one value per ER15 joint")
        return [
            max(-limit, min(limit, value))
            for value, limit in zip(command_rad_s, self.velocity_upper_rad_s, strict=True)
        ]

    def clamp_torque(self, command_nm: list[float]) -> list[float]:
        if len(command_nm) != len(self.actuator_torque_upper_nm):
            raise ValueError("torque command must have one value per ER15 joint")
        clamped: list[float] = []
        for value, limit in zip(command_nm, self.actuator_torque_upper_nm, strict=True):
            if limit is None:
                clamped.append(value)
            else:
                clamped.append(max(-limit, min(limit, value)))
        return clamped

    def validate_wrist_payload(
        self,
        torque_nm: dict[str, float] | None = None,
        inertia_kgm2: dict[str, float] | None = None,
    ) -> None:
        torque_nm = torque_nm or {}
        inertia_kgm2 = inertia_kgm2 or {}
        for joint, limit in zip(self.joint_names, self.wrist_load_torque_upper_nm, strict=True):
            if limit is not None and joint in torque_nm and abs(torque_nm[joint]) > limit:
                raise ValueError(f"{joint} wrist payload torque {torque_nm[joint]} exceeds {limit} N*m")
        for joint, limit in zip(self.joint_names, self.wrist_load_inertia_upper_kgm2, strict=True):
            if limit is not None and joint in inertia_kgm2 and inertia_kgm2[joint] > limit:
                raise ValueError(f"{joint} wrist payload inertia {inertia_kgm2[joint]} exceeds {limit} kg*m^2")


def build_control_limits(profile: QuickMoveProfile | None = None) -> JointControlLimits:
    profile = profile or QuickMoveProfile()
    return JointControlLimits(
        joint_names=list(ER15_PUBLIC_LIMITS["joint_names"]),
        position_lower_rad=list(ER15_PUBLIC_LIMITS["position_lower_rad"]),
        position_upper_rad=list(ER15_PUBLIC_LIMITS["position_upper_rad"]),
        velocity_upper_rad_s=[
            value * profile.velocity_scale for value in ER15_PUBLIC_LIMITS["velocity_upper_rad_s"]
        ],
        actuator_torque_upper_nm=[
            None if value is None else value * profile.torque_scale
            for value in ER15_PUBLIC_LIMITS["actuator_torque_upper_nm"]
        ],
        wrist_load_torque_upper_nm=list(ER15_PUBLIC_LIMITS["wrist_load_torque_upper_nm"]),
        wrist_load_inertia_upper_kgm2=list(ER15_PUBLIC_LIMITS["wrist_load_inertia_upper_kgm2"]),
        source=dict(ER15_PUBLIC_LIMITS["source"]),
    )
