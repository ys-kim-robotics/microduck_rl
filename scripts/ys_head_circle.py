"""Stage 4a: can an existing policy draw a circle with its head, untrained? (ys personal tool)

Holds a zero twist (standing) and drives the head_pose command with
  head_yaw = A·sin(2πft),  head_pitch = A·cos(2πft)
— a circle in JOINT space. Each env gets one (A, f) combo, so one rollout sweeps the grid.
Measured on the first harmonic (Fourier coefficient at f) over a whole number of periods:
  joint gain / lag   actual head joint vs command
  tip Ay / Az (mm)   mouth_tip lateral / vertical amplitude in the robot's start frame
  round              min(Ay, Az) / max(Ay, Az), 1 = circle
  tipE (mm)          mean y-z distance of mouth_tip (trunk frame) to the 4b-3 tip target
                     (centre TIP_CENTER_B, radius 0.05·A, y=+R·sin, z=−R·cos; same phase as the command)
  sway (mm)          trunk horizontal std,  fall %

  uv run scripts/ys_head_circle.py Mjlab-Velocity-Flat-MicroDuck ../policies/velstand.onnx model.pt
"""

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ys_eval import make_env, make_policy  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402
from mjlab_microduck.tasks.ys_head_circle_env_cfg import TIP_CENTER_B, TIP_M_PER_RAD, apply_tip_force  # noqa: E402

HEAD_PITCH, HEAD_YAW = 1, 2  # head_pose command order: neck_pitch, head_pitch, head_yaw, head_roll


def set_head_circle(env, t, amp, freq, lead=0.0):
  """Write the circle command for time t (s). amp/freq: floats or per-env tensors.
  lead (rad): phase advance, for feedforward compensation of a known lag."""
  cmd = env.command_manager.get_term("head_pose").command
  w = 2 * math.pi * freq * t + lead
  cmd.zero_()
  cmd[:, HEAD_YAW] = amp * torch.sin(torch.as_tensor(w, device=cmd.device))
  cmd[:, HEAD_PITCH] = amp * torch.cos(torch.as_tensor(w, device=cmd.device))


def harmonic(x, t, f):
  """Complex first-harmonic coefficient of x(t) (T, n) at per-env frequency f (n,)."""
  return 2 * (x * torch.exp(-1j * 2 * math.pi * f * t[:, None])).mean(0)


def run(task, path, combos, reps, warmup, window, direct=False, boost=1.0, lead_deg=0.0, tip_force=None):
  n = len(combos) * reps
  env, venv = make_env(task, (0.0, 0.0, 0.0), n)
  policy = make_policy(task, path, venv)
  robot = env.scene["robot"]
  dev = env.device
  amp = torch.tensor([c[0] for c in combos for _ in range(reps)], device=dev)
  freq = torch.tensor([c[1] for c in combos for _ in range(reps)], device=dev)
  jid = [robot.joint_names.index(j) for j in ("head_yaw", "head_pitch")]
  act_names = env.action_manager.get_term("joint_pos")._target_names
  aid = [act_names.index(j) for j in ("head_yaw", "head_pitch")]  # action = target - HOME (scale 1)
  sid = robot.find_sites("mouth_tip")[0][0]
  bid = robot.find_bodies("jaw_soft")[0][0]
  dt = env.step_dt
  ts, yaw, pitch, tip, trunk, tipe, tipb = [], [], [], [], [], [], []
  fell = torch.zeros(n, dtype=torch.bool, device=dev)
  obs = venv.get_observations()
  steps = int((warmup + window) / dt)
  with torch.inference_mode():
    for k in range(steps):
      t = (k + 1) * dt
      act = policy(obs)
      set_head_circle(env, t, amp * boost, freq, math.radians(lead_deg))  # the obs returned by this step carries cmd(t)
      if direct:  # bypass the policy for the two head joints: servo-only bandwidth
        w = 2 * math.pi * freq * t
        act[:, aid[0]], act[:, aid[1]] = amp * torch.sin(w), amp * torch.cos(w)
      if tip_force:
        apply_tip_force(robot, bid, sid, *tip_force)
      obs, _, dones, extras = venv.step(act)
      fell |= dones.bool() & ~extras["time_outs"].bool()
      if k == 0:
        yaw0 = robot.data.heading_w.clone()
        p0 = robot.data.root_link_pos_w.clone()
      if t > warmup:
        q = robot.data.joint_pos - robot.data.default_joint_pos
        ts.append(t)
        yaw.append(q[:, jid[0]].clone())
        pitch.append(q[:, jid[1]].clone())
        tip.append(robot.data.site_pos_w[:, sid].clone() - p0)
        trunk.append(robot.data.root_link_pos_w[:, :2].clone() - p0[:, :2])
        d = robot.data
        tb = quat_apply_inverse(d.root_link_quat_w, d.site_pos_w[:, sid] - d.root_link_pos_w)
        w = 2 * math.pi * freq * t
        ey = tb[:, 1] - (TIP_CENTER_B[1] + TIP_M_PER_RAD * amp * torch.sin(w))
        ez = tb[:, 2] - (TIP_CENTER_B[2] - TIP_M_PER_RAD * amp * torch.cos(w))
        tipe.append(torch.sqrt(ey**2 + ez**2))
        tipb.append(tb[:, 1:].clone())
  env.close()
  t = torch.tensor(ts, device=dev, dtype=torch.float64)
  f = freq.double()
  c, s = torch.cos(-yaw0), torch.sin(-yaw0)
  tip = torch.stack(tip).double()
  tip_y = s * tip[..., 0] + c * tip[..., 1]  # lateral, start frame
  tip_z = tip[..., 2]
  ref_yaw = harmonic(amp.double() * torch.sin(2 * math.pi * f * t[:, None]), t, f)
  ref_pitch = harmonic(amp.double() * torch.cos(2 * math.pi * f * t[:, None]), t, f)
  h_yaw, h_pitch = harmonic(torch.stack(yaw).double(), t, f), harmonic(torch.stack(pitch).double(), t, f)
  ay, az = 1000 * harmonic(tip_y, t, f).abs(), 1000 * harmonic(tip_z, t, f).abs()
  sway = 1000 * torch.stack(trunk).double().std(0).norm(dim=-1)
  m = {
    "gain_yaw": (h_yaw.abs() / ref_yaw.abs()),
    "lag_yaw": torch.rad2deg(torch.angle(ref_yaw / h_yaw)),
    "gain_pitch": (h_pitch.abs() / ref_pitch.abs()),
    "lag_pitch": torch.rad2deg(torch.angle(ref_pitch / h_pitch)),
    "tipE": 1000 * torch.stack(tipe).double().mean(0),
    "Ay": ay, "Az": az, "round": torch.minimum(ay, az) / torch.maximum(ay, az), "sway": sway,
  }
  rows = []
  tipb = 1000 * torch.stack(tipb).cpu().numpy()  # (T, n, 2) trunk-frame y, z in mm
  paths = [(tipb[:, i * reps, 0], tipb[:, i * reps, 1]) for i in range(len(combos))]
  for i, (a, fr) in enumerate(combos):
    sl = slice(i * reps, (i + 1) * reps)
    ok = ~fell[sl]
    row = {"A": a, "f": fr, "fall%": 100.0 * fell[sl].float().mean().item()}
    for k, v in m.items():
      row[k] = v[sl][ok].mean().item() if ok.any() else float("nan")
    row["path"] = paths[i]
    rows.append(row)
  return rows


def main():
  p = argparse.ArgumentParser()
  p.add_argument("task")
  p.add_argument("policies", nargs="+")
  p.add_argument("--amps", type=float, nargs="+", default=[0.2, 0.4, 0.7])
  p.add_argument("--freqs", type=float, nargs="+", default=[0.25, 0.5, 1.0])
  p.add_argument("--reps", type=int, default=8)
  p.add_argument("--warmup", type=float, default=2.0)
  p.add_argument("--window", type=float, default=8.0, help="whole number of periods for every freq")
  p.add_argument("--plot", help="save mouth_tip y-z paths (first rep of each combo) to this png")
  p.add_argument("--boost", type=float, default=1.0, help="feedforward: command amplitude x this")
  p.add_argument("--lead", type=float, default=0.0, help="feedforward: command phase advance (deg)")
  p.add_argument("--tip-force", type=float, nargs=3, metavar=("FX", "FR", "FZ"),
                 help="rope-tension stand-in at mouth_tip, N (see apply_tip_force)")
  p.add_argument("--direct", action="store_true", help="write the head joint targets straight to the servos")
  a = p.parse_args()
  combos = list(itertools.product(a.amps, a.freqs))
  hdr = f"{'policy':34s} {'A':>4s} {'f':>5s} {'gYaw':>5s} {'lagY':>5s} {'gPit':>5s} {'lagP':>5s} {'Ay':>6s} {'Az':>6s} {'round':>5s} {'tipE':>5s} {'sway':>5s} {'fall%':>5s}"
  out = [hdr]
  allrows = []
  for path in a.policies:
    name = (("DIRECT " if a.direct else "") + "/".join(Path(path).parts[-2:]))[-34:]
    for r in run(a.task, path, combos, a.reps, a.warmup, a.window, a.direct, a.boost, a.lead, a.tip_force):
      allrows.append((name, r))
      out.append(
        f"{name:34s} {r['A']:4.2f} {r['f']:5.2f} {r['gain_yaw']:5.2f} {r['lag_yaw']:5.0f} {r['gain_pitch']:5.2f} "
        f"{r['lag_pitch']:5.0f} {r['Ay']:6.1f} {r['Az']:6.1f} {r['round']:5.2f} {r['tipE']:5.1f} {r['sway']:5.1f} {r['fall%']:5.0f}"
      )
  print("\n" + "\n".join(out))
  if a.plot:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(a.policies), len(combos), figsize=(2.6 * len(combos), 2.8 * len(a.policies)), squeeze=False)
    for k, (name, r) in enumerate(allrows):
      ax = axes[k // len(combos)][k % len(combos)]
      y, z = r["path"]
      R = 1000 * TIP_M_PER_RAD * r["A"]
      th = np.linspace(0, 2 * np.pi, 100)
      cy, cz = 1000 * TIP_CENTER_B[1], 1000 * TIP_CENTER_B[2]
      ax.plot(cy + R * np.sin(th), cz - R * np.cos(th), "k--", lw=0.8)
      ax.plot(y, z, lw=0.8)
      ax.set_title(f"{name[-24:]}\nA={r['A']} f={r['f']}Hz round={r['round']:.2f} tipE={r['tipE']:.1f}mm", fontsize=7)
      ax.set_aspect("equal")
      ax.grid(alpha=0.3)
    fig.supxlabel("mouth_tip y, trunk frame (mm)   - - target circle (4b-3)")
    fig.supylabel("mouth_tip z, trunk frame (mm)")
    fig.tight_layout()
    fig.savefig(a.plot, dpi=120)
    print("saved", a.plot)


if __name__ == "__main__":
  main()
