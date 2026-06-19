from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from er15_quickmove import ER15QuickMovePlanner, quickmove_profile

START = [0.0, -0.9, 1.25, 0.0, 0.55, 0.0]
GOAL = [1.1, -0.35, 1.75, 1.2, 0.25, 2.4]

REAL_ER15_MODEL = Path(__file__).resolve().parents[1] / "assets" / "er15_1400" / "er15-1400.mjcf.xml"



def plan_quickmove_positions(no_warmup: bool) -> tuple[np.ndarray, float, dict]:
    planner = ER15QuickMovePlanner(quickmove_profile())
    planned = planner.plan_cspace(START, GOAL, warmup=not no_warmup)
    if planned.report is None:
        raise RuntimeError("QuickMove planning failed; cannot visualize in MuJoCo")
    traj = planned.result.get_interpolated_plan()
    positions = traj.position.detach().cpu().reshape(-1, traj.position.shape[-1]).numpy()
    return positions, planner.profile.interpolation_dt, planned.report.__dict__


def _prepare_model_xml(model_path: Path, width: int, height: int) -> str:
    xml = model_path.read_text(encoding="utf-8")
    if "offwidth=" not in xml:
        old_global = '<global azimuth="120" elevation="-18"/>'
        new_global = f'<global azimuth="120" elevation="-18" offwidth="{width}" offheight="{height}"/>'
        xml = xml.replace(old_global, new_global)
    return xml


def load_mujoco_model(model_path: Path, width: int = 1280, height: int = 720):
    if not os.environ.get("DISPLAY") and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    import mujoco

    xml = _prepare_model_xml(model_path, width, height)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=model_path.parent, delete=False) as handle:
        handle.write(xml)
        xml_path = handle.name
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    finally:
        Path(xml_path).unlink(missing_ok=True)
    data = mujoco.MjData(model)
    return mujoco, model, data


def play_viewer(positions: np.ndarray, dt: float, model_path: Path) -> None:
    mujoco, model, data = load_mujoco_model(model_path)
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for q in positions:
                if not viewer.is_running():
                    break
                data.qpos[:] = q
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt)


def render_video(positions: np.ndarray, output_path: Path, fps: int, width: int, height: int, model_path: Path) -> None:
    mujoco, model, data = load_mujoco_model(model_path, width=width, height=height)
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    stride = max(1, round((1.0 / fps) / 0.01))
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.azimuth = 135
    camera.elevation = -22
    camera.distance = 2.7
    camera.lookat[:] = np.array([0.2, 0.0, 0.85])
    for q in positions[::stride]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())
    renderer.close()
    imageio.mimsave(output_path, frames, fps=fps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize ER15 QuickMove trajectory in MuJoCo.")
    parser.add_argument("--mode", choices=["video", "viewer"], default="video")
    parser.add_argument("--output", type=Path, default=Path("outputs/quickmove_demo/quickmove_mujoco.mp4"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--model-path", type=Path, default=REAL_ER15_MODEL)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    positions, dt, report = plan_quickmove_positions(no_warmup=args.no_warmup)
    if args.mode == "viewer":
        play_viewer(positions, dt, args.model_path)
    else:
        render_video(positions, args.output, args.fps, args.width, args.height, args.model_path)
        metrics_path = args.output.with_suffix(".json")
        metrics_path.write_text(
            __import__("json").dumps({"trajectory": report, "video": str(args.output), "model": str(args.model_path)}, indent=2),
            encoding="utf-8",
        )
        print(f"video={args.output}")
        print(f"metrics={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
