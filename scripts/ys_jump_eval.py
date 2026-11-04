"""Stage 7-1: score the timed hop of Mjlab-YsJump (ys personal tool, eval only).

Clock at 1.25 Hz, play cfg (target 4 cm). Over the last WINDOW seconds, per hop (one clock cycle):
  peak      max foot clearance (min of both foot sites) inside the reference window, mm
  clear@Xmm share of hops whose clearance stays >= X mm through the CORE window (±CORE_MS around the
            reference apex) = the ~60 ms the rope spends under the feet (STAGES-ys.md 7)
  apex dt   time of the max clearance minus the reference apex, ms (+ = late)
  off-beat  share of time both feet are up outside the reference window (should be ~0)
  impact    p95 of the per-hop peak net foot force, N

  uv run scripts/ys_jump_eval.py logs/rsl_rl/ys_jump/<run>/model_N.pt [--still] [--video out.mp4]
"""

import argparse
import math
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import mjlab_microduck.tasks  # noqa: F401,E402
from ys_eval import make_env, make_policy  # noqa: E402
import mjlab_microduck.tasks.ys_jump_env_cfg as J  # noqa: E402

TASK = "Mjlab-YsJump-Flat-MicroDuck"
CORE_MS = 30


def main():
  p = argparse.ArgumentParser()
  p.add_argument("policy")
  p.add_argument("--n", type=int, default=64)
  p.add_argument("--seconds", type=float, default=15.0)
  p.add_argument("--window", type=float, default=10.0)
  p.add_argument("--still", action="store_true", help="clock off (amp 0): must stand still")
  p.add_argument("--video", help="render env 0 to this mp4")
  p.add_argument("--lead-ms", type=float, default=0.0, help="the policy's clock runs this far ahead of the rope (7-3 calibration)")
  a = p.parse_args()

  env, venv = make_env(TASK, (0.0, 0.0, 0.0), 1 if a.video else a.n, render=bool(a.video))
  pol = make_policy(TASK, a.policy, venv)
  cmd = env.command_manager.get_term("head_pose")
  robot, sensor = env.scene["robot"], env.scene["feet_ground_contact"]
  n, dt = env.num_envs, env.step_dt
  core = 2 * math.pi * J.JUMP_FREQ_HZ * CORE_MS / 1000
  p0 = robot.data.root_link_pos_w[:, :2].clone()
  fell = torch.zeros(n, dtype=torch.bool, device=env.device)
  rows, taus, zs = [], [], []  # per step: d, clearance, force / actuator torques / root z
  writer = None
  if a.video:
    writer = imageio.get_writer(a.video, fps=round(1 / dt), codec="libx264", quality=7, macro_block_size=8)
    r = env._offline_renderer
    r._cam.azimuth, r._cam.elevation = 90, -10
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(int(a.seconds / dt)):
      cmd.amp[:] = 0.0 if a.still else 1.0
      cmd.freq[:] = J.JUMP_FREQ_HZ
      cmd._write()
      obs, *_ = venv.step(pol(obs))
      fell |= robot.data.projected_gravity_b[:, 2] > -0.5
      if k * dt >= a.seconds - a.window:
        f = sensor.data.force.norm(dim=-1).sum(-1) if sensor.data.force.dim() == 3 else sensor.data.force.norm(dim=-1)
        taus.append(robot.data.qfrc_actuator.abs())
        zs.append(robot.data.root_link_pos_w[:, 2])
        lead = 2 * math.pi * J.JUMP_FREQ_HZ * a.lead_ms / 1000
        rows.append(torch.stack([torch.remainder(cmd.window() - lead + math.pi, 2 * math.pi) - math.pi, J.jump_clearance(robot, cmd._sole), f.reshape(n, -1).sum(-1)], 1))
      if writer:
        r.update(env.sim.data)
        writer.append_data(r.render().copy())
  if writer:
    writer.close()
    print("saved", a.video)

  R = torch.stack(rows).cpu().numpy()  # (T, n, 3)
  d, c, F = R[..., 0], R[..., 1], R[..., 2]
  ok = ~fell.cpu().numpy()
  drift = 1000 * (robot.data.root_link_pos_w[:, :2] - p0).norm(dim=-1).cpu().numpy()
  print(f"{'still' if a.still else 'clock'}  envs {n}  fell {100 * (~ok).mean():.1f}%  drift {drift[ok].mean():.0f} mm (p90 {np.percentile(drift[ok], 90):.0f})")
  tau = torch.stack(taus)[:, torch.from_numpy(ok).to(env.device)].flatten(0, 1).cpu().numpy()
  z = 1000 * torch.stack(zs).cpu().numpy()[:, ok]
  print(f"  realism: torque p99/stall(0.96 Nm) max joint {np.percentile(tau, 99, axis=0).max() / 0.96:.2f}   trunk z mm p1 {np.percentile(z, 1):.0f} p99 {np.percentile(z, 99):.0f} (stand {1000 * J.TRUNK_Z0:.0f})")
  inw = np.abs(d) < J.JUMP_HALF_W
  print(f"  off-beat airborne (both feet > 5 mm outside the window): {100 * ((c > 0.005) & ~inw)[:, ok].mean():.1f}% of time")
  if a.still:
    print(f"  airborne (both feet > 5 mm): {100 * (c > 0.005)[:, ok].mean():.1f}% of time")
    return
  # split into hops: a hop = a contiguous in-window run per env
  peaks, mins, dts, imp = [], {20: [], 26: [], 30: []}, [], []
  for e in np.flatnonzero(ok):
    t = 0
    while t < len(d):
      if not inw[t, e]:
        t += 1
        continue
      s = t
      while t < len(d) and inw[t, e]:
        t += 1
      if s == 0 or t == len(d):
        continue  # partial window
      seg_c, seg_d = c[s:t, e], d[s:t, e]
      peaks.append(seg_c.max())
      dts.append(seg_d[seg_c.argmax()] / (2 * math.pi * J.JUMP_FREQ_HZ) * 1000)
      cmin = seg_c[np.abs(seg_d) < core].min() if (np.abs(seg_d) < core).any() else 0.0
      for x in mins:
        mins[x].append(cmin >= x / 1000)
      imp.append(F[t : min(t + 10, len(d)), e].max())  # landing: 200 ms after the window
  peaks, dts = 1000 * np.array(peaks), np.array(dts)
  print(f"  hops {len(peaks)}  peak clearance mm: p10 {np.percentile(peaks, 10):.0f}  p50 {np.percentile(peaks, 50):.0f}  p90 {np.percentile(peaks, 90):.0f}")
  print(f"  rope-ready (clearance through ±{CORE_MS} ms core): >=20 mm {100 * np.mean(mins[20]):.0f}%   >=26 mm {100 * np.mean(mins[26]):.0f}%   >=30 mm {100 * np.mean(mins[30]):.0f}%")
  print(f"  apex dt ms: mean {dts.mean():+.0f}  sd {dts.std():.0f}    landing force N: p50 {np.percentile(imp, 50):.0f}  p95 {np.percentile(imp, 95):.0f}  (weight {0.742 * 9.81:.1f} N)")


if __name__ == "__main__":
  main()
