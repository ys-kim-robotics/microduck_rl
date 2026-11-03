"""Stage 5b eval: how big a mouth_tip circle does a policy draw in the FEET frame? (ys personal tool)

Runs Mjlab-YsBodyCircle-Flat-MicroDuck (or any head-circle task) with the circle command's radius
fixed per env (R list x reps), tip load off, and measures over the last WINDOW seconds:
  Ay / Az (mm)   first-harmonic tip amplitude, feet-frame lateral / vertical (target = R)
  tipE (mm)      mean distance to the feet-frame target circle (phase-locked)
  score          phase-free circle score the 5b-3 reward pays (radius x turning speed), 0-1
  trunk (mm)     first-harmonic trunk lateral / vertical amplitude: how much the BODY moves
  off %          share of steps with a foot off the ground,  fall %

  uv run scripts/ys_body_circle.py Mjlab-YsBodyCircle-Flat-MicroDuck model.pt --radii 0.035 0.06 0.08 0.1
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from ys_eval import make_env, make_policy  # noqa: E402
from ys_head_circle import harmonic  # noqa: E402
from mjlab_microduck.tasks.ys_head_circle_env_cfg import TIP_M_PER_RAD, circle_score, feet_frame, tip_error_feet, tip_feet  # noqa: E402


def run(task, path, radii, reps, warmup, window, freq=None):
  n = len(radii) * reps
  env, venv = make_env(task, (0.0, 0.0, 0.0), n)
  policy = make_policy(task, path, venv)
  cmd = env.command_manager.get_term("head_pose")
  cmd.cfg.freeze_radius = True
  if freq:
    cmd.cfg.freq_hz = freq
  cmd.freq[:] = cmd.cfg.freq_hz
  R = torch.tensor([r for r in radii for _ in range(reps)], device=env.device)
  robot = env.scene["robot"]
  sid = robot.find_sites("mouth_tip")[0][0]
  feet = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)[0]
  contact = env.scene["feet_ground_contact"]
  dt, f = env.step_dt, cmd.cfg.freq_hz
  ts, tip, trunk, err, off, score = [], [], [], [], [], []
  fell = torch.zeros(n, dtype=torch.bool, device=env.device)
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(int((warmup + window) / dt)):
      if k == 0:
        cmd.center[:] = tip_feet(robot, sid, feet)[0][:, 1:]
      cmd.amp[:] = R / TIP_M_PER_RAD
      cmd.freq[:] = f
      cmd.tip_force.zero_()
      cmd.phase.copy_(torch.full_like(cmd.phase, 2 * math.pi * f * k * dt))
      cmd._write()
      obs, _, dones, extras = venv.step(policy(obs))
      fell |= dones.bool() & ~extras["time_outs"].bool()
      if k * dt >= warmup:
        o, rot = feet_frame(robot, feet)
        rt = rot.transpose(1, 2)
        tip.append((rt @ (robot.data.site_pos_w[:, sid] - o).unsqueeze(-1)).squeeze(-1))
        trunk.append((rt @ (robot.data.root_link_pos_w - o).unsqueeze(-1)).squeeze(-1))
        err.append(tip_error_feet(robot, sid, feet, cmd.amp, cmd.phase))
        score.append(circle_score(robot, sid, feet, cmd.amp, cmd.freq, cmd.center))
        off.append((contact.data.found == 0).any(1).float())
        ts.append(k * dt)
  env.close()
  t = torch.tensor(ts, device=env.device, dtype=torch.float64)
  fr = torch.full((n,), f, device=env.device, dtype=torch.float64)
  tip, trunk = torch.stack(tip).double(), torch.stack(trunk).double()
  amp = lambda x: 1000 * harmonic(x, t, fr).abs()  # noqa: E731
  m = {"Ay": amp(tip[..., 1]), "Az": amp(tip[..., 2]), "tipE": 1000 * torch.stack(err).double().mean(0),
       "trY": amp(trunk[..., 1]), "trZ": amp(trunk[..., 2]), "off%": 100 * torch.stack(off).double().mean(0),
       "score": torch.stack(score).double().mean(0)}
  rows = []
  for i, r in enumerate(radii):
    sl = slice(i * reps, (i + 1) * reps)
    ok = ~fell[sl]
    row = {"R": 1000 * r, "fall%": 100.0 * fell[sl].float().mean().item()}
    for k, v in m.items():
      row[k] = v[sl][ok].mean().item() if ok.any() else float("nan")
    rows.append(row)
  return rows


def main():
  p = argparse.ArgumentParser()
  p.add_argument("task")
  p.add_argument("policies", nargs="+")
  p.add_argument("--radii", type=float, nargs="+", default=[0.035, 0.06, 0.08, 0.10])
  p.add_argument("--reps", type=int, default=8)
  p.add_argument("--freq", type=float, help="override the task's circle frequency (Hz)")
  p.add_argument("--warmup", type=float, default=2.0)
  p.add_argument("--window", type=float, default=8.0)
  a = p.parse_args()
  keys = ["R", "Ay", "Az", "tipE", "score", "trY", "trZ", "off%", "fall%"]
  out = [f"{'policy':34s} " + " ".join(f"{k:>6s}" for k in keys)]
  for path in a.policies:
    name = "/".join(Path(path).parts[-2:])[-34:]
    for r in run(a.task, path, a.radii, a.reps, a.warmup, a.window, a.freq):
      out.append(f"{name:34s} " + " ".join(f"{r[k]:6.2f}" if k == "score" else f"{r[k]:6.1f}" for k in keys))
  print("\n" + "\n".join(out))


if __name__ == "__main__":
  main()
