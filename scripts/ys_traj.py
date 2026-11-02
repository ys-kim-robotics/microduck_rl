"""Top-down trajectories under one fixed command, one panel per policy (ys personal tool).

Positions are drawn in each robot's start frame (start at origin, initial heading = +x), so a
perfect straight walk is a horizontal line. Reuses ys_eval's env/policy setup.

  uv run scripts/ys_traj.py Mjlab-Velocity-Flat-MicroDuck a.onnx b.pt --cmd 0.3 0 0 --out traj.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ys_eval import make_env, make_policy  # noqa: E402


def rollout(task, path, cmd, seconds, n):
  env, venv = make_env(task, cmd, n)
  policy = make_policy(task, path, venv)
  data = env.scene["robot"].data
  obs = venv.get_observations()
  xy = []
  with torch.inference_mode():
    for _ in range(int(seconds / env.step_dt)):
      obs, *_ = venv.step(policy(obs))
      xy.append(data.root_link_pos_w[:, :2].clone())
      if len(xy) == 1:
        yaw0 = data.heading_w.clone()
  env.close()
  p = torch.stack(xy) - xy[0]  # (T, n, 2), start at origin
  c, s = torch.cos(-yaw0), torch.sin(-yaw0)  # rotate into the start heading frame
  return torch.stack([c * p[..., 0] - s * p[..., 1], s * p[..., 0] + c * p[..., 1]], -1).cpu().numpy()


def main():
  p = argparse.ArgumentParser()
  p.add_argument("task")
  p.add_argument("policies", nargs="+")
  p.add_argument("--cmd", type=float, nargs=3, default=(0.3, 0.0, 0.0))
  p.add_argument("--seconds", type=float, default=15.0)
  p.add_argument("--envs", type=int, default=16)
  p.add_argument("--labels", nargs="+")
  p.add_argument("--out", default="traj.png")
  a = p.parse_args()
  labels = a.labels or [Path(x).parent.name + "/" + Path(x).name for x in a.policies]
  fig, axes = plt.subplots(len(a.policies), 1, figsize=(7, 2.4 * len(a.policies)), sharex=True, sharey=True)
  for ax, path, label in zip(axes, a.policies, labels):
    t = rollout(a.task, path, tuple(a.cmd), a.seconds, a.envs)
    for i in range(t.shape[1]):
      ax.plot(t[:, i, 0], t[:, i, 1], lw=0.8, alpha=0.6)
    ax.plot([0, a.cmd[0] * a.seconds], [0, 0], "k--", lw=1, label="ideal")
    ax.set_title(f"{label}   (end: mean x={t[-1, :, 0].mean():.2f} m, |y|={abs(t[-1, :, 1]).mean():.2f} m)", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
  axes[-1].set_xlabel("x forward (m)")
  fig.suptitle(f"cmd vx,vy,wz={tuple(a.cmd)}  {a.seconds:.0f}s  {a.envs} robots each (start frame)")
  fig.tight_layout()
  fig.savefig(a.out, dpi=130)
  print("saved", a.out)


if __name__ == "__main__":
  main()
