from __future__ import annotations

from math import isclose, pi
from xml.etree import ElementTree as ET

import pytest

from er15_quickmove import ER15_PUBLIC_LIMITS, build_control_limits, quickmove_profile
from er15_quickmove.config import ProjectPaths


def test_public_velocity_limits_match_efort_leaflet():
    assert ER15_PUBLIC_LIMITS["velocity_upper_deg_s"] == [260, 255, 210, 450, 450, 600]
    assert isclose(ER15_PUBLIC_LIMITS["velocity_upper_rad_s"][0], 260 * pi / 180.0)


def test_control_limits_expose_engineering_actuator_and_wrist_payload_torque():
    limits = build_control_limits(quickmove_profile())

    assert limits.actuator_torque_upper_nm == [1440.0, 960.0, 560.0, 144.0, 112.0, 72.0]
    assert limits.clamp_torque([9999.0, -999.0, 1.0, 999.0, -999.0, 999.0]) == [
        1440.0,
        -960.0,
        1.0,
        144.0,
        -112.0,
        72.0,
    ]
    assert limits.wrist_load_torque_upper_nm == [None, None, None, 42.0, 42.0, 20.0]
    limits.validate_wrist_payload({"joint_4": 42.0, "joint_6": 20.0})
    with pytest.raises(ValueError):
        limits.validate_wrist_payload({"joint_6": 21.0})


def test_control_velocity_clamp_uses_profile_scaled_public_limits():
    limits = build_control_limits(quickmove_profile())
    command = [99.0, -99.0, 0.1, 99.0, -99.0, 99.0]
    clamped = limits.clamp_velocity(command)

    assert clamped[0] == limits.velocity_upper_rad_s[0]
    assert clamped[1] == -limits.velocity_upper_rad_s[1]
    assert clamped[2] == 0.1


def test_curobo_urdf_velocity_matches_public_limits_and_effort_engineering_defaults():
    root = ET.parse(ProjectPaths().robot_urdf).getroot()
    expected = dict(
        zip(
            ER15_PUBLIC_LIMITS["joint_names"],
            ER15_PUBLIC_LIMITS["velocity_upper_rad_s"],
            strict=True,
        )
    )
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        limit = joint.find("limit")
        if name in expected:
            assert limit is not None
            assert isclose(float(limit.attrib["velocity"]), expected[name], rel_tol=1e-7)
            assert float(limit.attrib["effort"]) == ER15_PUBLIC_LIMITS["actuator_torque_upper_nm"][
                ER15_PUBLIC_LIMITS["joint_names"].index(name)
            ]


def test_project_paths_use_only_real_er15_models():
    paths = ProjectPaths()

    assert paths.robot_urdf.name == "ER15-1400-fulldyn-curobo.urdf"
    assert paths.robot_source_urdf.name == "ER15-1400-fulldyn-local.urdf"
    assert paths.robot_mjcf.name == "er15-1400.mjcf.xml"
    assert not (paths.robot_asset_root / "er15_1400_approx.urdf").exists()
