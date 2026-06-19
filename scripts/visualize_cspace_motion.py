from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from er15_quickmove import ER15QuickMovePlanner, quickmove_profile

START = [0.0, -0.9, 1.25, 0.0, 0.55, 0.0]
GOAL = [1.1, -0.35, 1.75, 1.2, 0.25, 2.4]


def _rotx(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack([
        torch.stack([torch.ones_like(c), torch.zeros_like(c), torch.zeros_like(c)]),
        torch.stack([torch.zeros_like(c), c, -s]),
        torch.stack([torch.zeros_like(c), s, c]),
    ])


def _roty(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack([
        torch.stack([c, torch.zeros_like(c), s]),
        torch.stack([torch.zeros_like(c), torch.ones_like(c), torch.zeros_like(c)]),
        torch.stack([-s, torch.zeros_like(c), c]),
    ])


def _rotz(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack([
        torch.stack([c, -s, torch.zeros_like(c)]),
        torch.stack([s, c, torch.zeros_like(c)]),
        torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)]),
    ])


def _fk_points(q: torch.Tensor) -> torch.Tensor:
    p = torch.zeros(3, dtype=q.dtype, device=q.device)
    r = torch.eye(3, dtype=q.dtype, device=q.device)
    points = [p.clone()]

    def move(local_xyz: list[float]) -> None:
        nonlocal p
        p = p + r @ torch.tensor(local_xyz, dtype=q.dtype, device=q.device)
        points.append(p.clone())

    move([0.0, 0.0, 0.24])
    r = r @ _rotz(q[0])
    move([0.0, 0.0, 0.32])
    r = r @ _roty(q[1])
    move([0.64, 0.0, 0.0])
    r = r @ _roty(q[2])
    move([0.54, 0.0, 0.0])
    r = r @ _rotx(q[3])
    move([0.18, 0.0, 0.0])
    r = r @ _roty(q[4])
    move([0.0, 0.0, 0.16])
    r = r @ _rotz(q[5])
    move([0.0, 0.0, 0.12])
    return torch.stack(points)


def _subsample_indices(n: int, max_frames: int) -> list[int]:
    if n <= max_frames:
        return list(range(n))
    return torch.linspace(0, n - 1, max_frames).round().to(torch.int64).tolist()


def _write_plot(path: Path, positions: torch.Tensor, dt: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = torch.arange(positions.shape[0], dtype=torch.float32).numpy() * dt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = [f"j{i}" for i in range(1, positions.shape[1] + 1)]
    velocity = torch.diff(positions, dim=0) / dt
    acceleration = torch.diff(velocity, dim=0) / dt
    series = [positions, velocity, acceleration]
    titles = ["Position (rad)", "Velocity (rad/s)", "Acceleration (rad/s^2)"]

    for ax, values, title in zip(axes, series, titles):
        local_t = t[: values.shape[0]]
        for joint_idx, label in enumerate(labels):
            ax.plot(local_t, values[:, joint_idx].cpu().numpy(), label=label, linewidth=1.2)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_html(path: Path, positions: torch.Tensor, dt: float, max_frames: int) -> None:
    frame_idxs = _subsample_indices(positions.shape[0], max_frames)
    frames = []
    for idx in frame_idxs:
        pts = _fk_points(positions[idx]).cpu().tolist()
        frames.append({"t": round(idx * dt, 4), "points": pts})

    payload = json.dumps({"frames": frames}, separators=(",", ":"))
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>ER15 QuickMove Motion</title>
<style>
  body {{ margin: 0; font-family: Arial, sans-serif; background: #111; color: #eee; }}
  #hud {{ position: fixed; left: 16px; top: 12px; z-index: 2; font-size: 14px; }}
  canvas {{ display: block; width: 100vw; height: 100vh; }}
</style>
</head>
<body>
<div id="hud">ER15-1400 QuickMove-like motion<br><span id="time"></span></div>
<canvas id="view"></canvas>
<script>
const DATA = {payload};
const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const timeLabel = document.getElementById('time');
let frame = 0;
function resize() {{ canvas.width = window.innerWidth * devicePixelRatio; canvas.height = window.innerHeight * devicePixelRatio; }}
window.addEventListener('resize', resize); resize();
function project(p) {{
  const yaw = -0.7, pitch = 0.55;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  let x = p[0] * cy - p[1] * sy;
  let y = p[0] * sy + p[1] * cy;
  let z = p[2];
  let zz = y * sp + z * cp;
  const scale = Math.min(canvas.width, canvas.height) * 0.34;
  return [canvas.width * 0.52 + x * scale, canvas.height * 0.66 - zz * scale];
}}
function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 1 * devicePixelRatio; ctx.strokeStyle = '#333';
  for (let i = 0; i < 8; i++) {{ const y = canvas.height * (0.18 + i * 0.09); ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke(); }}
  const current = DATA.frames[frame % DATA.frames.length];
  const pts = current.points.map(project);
  ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#42d392'; ctx.lineWidth = 8 * devicePixelRatio;
  ctx.beginPath(); pts.forEach((p, i) => i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.stroke();
  ctx.fillStyle = '#f5c542';
  for (const p of pts) {{ ctx.beginPath(); ctx.arc(p[0], p[1], 6 * devicePixelRatio, 0, Math.PI * 2); ctx.fill(); }}
  timeLabel.textContent = `t=${{current.t.toFixed(3)}}s  frame=${{frame % DATA.frames.length}}/${{DATA.frames.length}}`;
  frame += 1; requestAnimationFrame(draw);
}}
draw();
</script>
</body>
</html>"""
    path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize the ER15 QuickMove-like trajectory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/quickmove_demo"))
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    planner = ER15QuickMovePlanner(quickmove_profile())
    planned = planner.plan_cspace(START, GOAL, warmup=not args.no_warmup)
    if planned.report is None:
        raise RuntimeError("QuickMove planning failed; no trajectory to visualize")

    traj = planned.result.get_interpolated_plan()
    positions = traj.position.detach().cpu().reshape(-1, traj.position.shape[-1])
    dt = planner.profile.interpolation_dt
    metrics_path = args.output_dir / "quickmove_metrics.json"
    plot_path = args.output_dir / "quickmove_joint_trajectory.png"
    html_path = args.output_dir / "quickmove_motion.html"

    metrics_path.write_text(json.dumps(asdict(planned.report), indent=2), encoding="utf-8")
    _write_plot(plot_path, positions, dt)
    _write_html(html_path, positions, dt, args.max_frames)
    print(f"metrics={metrics_path}")
    print(f"joint_plot={plot_path}")
    print(f"motion_html={html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
