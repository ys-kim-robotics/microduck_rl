"""Stage 5 rope-turner checks from a resting rope (ys personal tool): spin-up time, whether a frequency
sustains the turn, and how far the rope drags the duck. Per frequency schedule, N envs from reset:
  spin(s)   first time the 0.8 s mean of rope_turn exceeds 0.8 (nan: never), mean over envs that spun
  spun%     share of envs that spun up;  turn  mean rope_turn over the last WINDOW s (1 = with the handle)
  dxy(mm)   trunk horizontal displacement from its start;  dyaw(deg)  heading change;  fall%
A schedule "1.25>1.0@8" turns at 1.25 Hz, then 1.0 Hz from t = 8 s (phase continuous).

  uv run scripts/ys_rope_check.py Mjlab-YsRopeBob-Flat-MicroDuck model.pt --sched 1.25 1.0 "1.25>1.0@8"
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from ys_eval import make_env, make_policy  # noqa: E402
from mjlab_microduck.tasks.ys_head_circle_env_cfg import TIP_M_PER_RAD  # noqa: E402
from mjlab_microduck.tasks.ys_rope_env_cfg import rope_turn  # noqa: E402


def parse(s):
  """'1.25' -> (1.25, 1.25, inf); '1.25>1.0@8' -> (1.25, 1.0, 8.0)"""
  if ">" not in s:
    return float(s), float(s), math.inf
  a, rest = s.split(">")
  b, t = rest.split("@")
  return float(a), float(b), float(t)


def yaw(q):
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run(task, path, sched, n, radius, seconds, window):
  f0, f1, t_sw = parse(sched)
  env, venv = make_env(task, (0.0, 0.0, 0.0), n)
  policy = make_policy(task, path, venv)
  cmd = env.command_manager.get_term("head_pose")
  cmd.cfg.freeze_radius = True
  robot = env.scene["robot"]
  dt = env.step_dt
  p0, y0 = robot.data.root_link_pos_w[:, :2].clone(), yaw(robot.data.root_link_quat_w).clone()
  hist, turns = [], []
  spin = torch.full((n,), math.nan, device=env.device)
  fell = torch.zeros(n, dtype=torch.bool, device=env.device)
  win = int(0.8 / dt)
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(int(seconds / dt)):
      t = k * dt
      cmd.amp[:] = radius / TIP_M_PER_RAD
      cmd.freq[:] = f0 if t < t_sw else f1
      cmd.tip_force.zero_()
      cmd._write()
      obs, _, dones, extras = venv.step(policy(obs))
      fell |= dones.bool() & ~extras["time_outs"].bool()
      r = rope_turn(env)
      hist.append(r)
      if len(hist) >= win:
        m = torch.stack(hist[-win:]).mean(0)
        spin = torch.where(torch.isnan(spin) & (m > 0.8), torch.full_like(spin, t), spin)
      if t >= seconds - window:
        turns.append(r)
    dxy = 1000 * (robot.data.root_link_pos_w[:, :2] - p0).norm(dim=-1)
    dyaw = torch.rad2deg((yaw(robot.data.root_link_quat_w) - y0 + math.pi) % (2 * math.pi) - math.pi).abs()
  env.close()
  ok = ~fell
  spun = ~torch.isnan(spin)
  return {"sched": sched, "spin": spin[spun & ok].mean().item() if (spun & ok).any() else math.nan,
          "spun%": 100 * spun[ok].float().mean().item(), "turn": torch.stack(turns).mean(0)[ok].mean().item(),
          "dxy": dxy[ok].mean().item(), "dyaw": dyaw[ok].mean().item(), "fall%": 100 * fell.float().mean().item()}


def main():
  p = argparse.ArgumentParser()
  p.add_argument("task")
  p.add_argument("policy")
  p.add_argument("--sched", nargs="+", default=["1.0", "1.1", "1.25", "1.5", "1.25>1.0@8"])
  p.add_argument("--n", type=int, default=16)
  p.add_argument("--radius", type=float, default=0.12)
  p.add_argument("--seconds", type=float, default=19.0, help="< episode length (20 s): no timeout reset")
  p.add_argument("--window", type=float, default=6.0)
  a = p.parse_args()
  keys = ["spin", "spun%", "turn", "dxy", "dyaw", "fall%"]
  out = [f"{'sched':>12s} " + " ".join(f"{k:>6s}" for k in keys)]
  for s in a.sched:
    r = run(a.task, a.policy, s, a.n, a.radius, a.seconds, a.window)
    out.append(f"{s:>12s} " + " ".join(f"{r[k]:6.2f}" if k in ("spin", "turn") else f"{r[k]:6.1f}" for k in keys))
    print(out[-1], flush=True)
  print("\n" + "\n".join(out))


if __name__ == "__main__":
  main()
