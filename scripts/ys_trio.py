"""Stage 7-3 smoke (ys personal tool, eval only, no training): two turners + a jumper in the middle.

ys_rope_duo's scene (robot ══ rope ══ partner, one turner policy, partner mirrored) plus a third duck
"jumper" at the rope's midpoint, facing +x (the side stance), running a jump policy (side waddle by
default). The jumper's command slots are written here: twist = walk back to its spot (as
JumpClockCommand._center), head = the jump clock at amp 1 with phase = turner phase + delta.
delta is swept across envs (calibration: where must the jumper's clock sit relative to the rope);
then rerun with --delta-deg at the best value.

Rope hit: at the jumper's x section the rope point (y, z) lies inside a foot's y band (jumper frame)
with the rope top above that sole's lowest corner, below 0.17 m (above it the rope goes over the head).
Only the feet collide with the rope, so a hit is also where a real rope would catch.

  uv run scripts/ys_trio.py <turner.pt> <jumper.pt> [--span 0.60] [--n 64] [--delta-deg D] [--video out.mp4]
"""

import argparse
import copy
import math
import sys
from dataclasses import replace
from pathlib import Path

import imageio
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import mjlab_microduck.tasks  # noqa: F401,E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers import ObservationTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from ys_eval import make_env, make_policy  # noqa: E402
import ys_rope_duo as D  # noqa: E402
import mjlab_microduck.tasks.ys_jump_env_cfg as J  # noqa: E402
from mjlab_microduck.tasks.ys_rope import ROPE_RADIUS  # noqa: E402
from mjlab_microduck.tasks.ys_rope_env_cfg import STEM_END_NOM  # noqa: E402

JTASK = "Mjlab-YsSideJump-Flat-MicroDuck"
BANDS = {"right": (-0.065, -0.018), "left": (0.020, 0.064)}  # foot y bands, jumper frame (side.py)
HEAD_TOP = 0.17


def trio_cfg(n, span, jtask):
  cfg = D.duo_cfg(n, span=span)
  jc = load_env_cfg(jtask, play=True)
  mid = STEM_END_NOM[0] + span / 2
  jr = jc.scene.entities["robot"]
  jumper = replace(jr, init_state=replace(jr.init_state, pos=(mid, 0.0, jr.init_state.pos[2]), rot=(1.0, 0.0, 0.0, 0.0)))
  e = cfg.scene.entities
  cfg.scene.entities = {"robot": e["robot"], "partner": e["partner"], "jumper": jumper, "rope": e["rope"]}  # ducks before rope
  for k, ev in list(cfg.events.items()):  # jumper copies of the robot events (reset pose, joints, DR)
    a = (ev.params or {}).get("asset_cfg")
    if k.endswith("_partner") or not ((a is not None and a.name == "robot") or k == "reset_base"):
      continue
    e2 = copy.deepcopy(ev)
    e2.params = dict(e2.params or {})
    e2.params["asset_cfg"] = replace(a, name="jumper") if a is not None else SceneEntityCfg("jumper")
    if k == "reset_base":
      e2.params["pose_range"] = jc.events["reset_base"].params["pose_range"]
    cfg.events[f"{k}_jumper"] = e2
  cfg.actions["jumper_joint_pos"] = replace(cfg.actions["joint_pos"], entity_name="jumper")
  jo = copy.deepcopy(jc.observations["actor"])
  for t in jo.terms.values():
    if t.params and "asset_cfg" in t.params:
      t.params = {**t.params, "asset_cfg": replace(t.params["asset_cfg"], name="jumper")}
  for t in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
    jo.terms[t].params = {**(jo.terms[t].params or {}), "asset_cfg": SceneEntityCfg("jumper")}
  jo.terms["actions"] = ObservationTermCfg(func=D.own_action(28, 42))
  cfg.observations["jumper"] = jo
  return cfg, mid


def yaw(q):
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def jumper_slots(jumper, spot, phase):
  """(n, 13) command block: centering twist, jump clock (amp 1), body pose 0 (side-jump obs, probe.py)."""
  ya = yaw(jumper.data.root_link_quat_w)
  dx, dy = (spot - jumper.data.root_link_pos_w[:, :2]).unbind(-1)
  c, s = ya.cos(), ya.sin()
  out = torch.zeros(len(phase), 13, device=phase.device)
  out[:, 0] = (J.CENTER_GAIN * (c * dx + s * dy)).clamp(-0.3, 0.3)
  out[:, 1] = (J.CENTER_GAIN * (-s * dx + c * dy)).clamp(-0.3, 0.3)
  out[:, 2] = (-J.CENTER_GAIN * ya).clamp(-0.5, 0.5)
  out[:, 4], out[:, 5] = torch.cos(phase), torch.sin(phase)  # head_pitch, head_yaw
  return out


def rope_section(pts, xq):
  """rope polyline (n, L, 3) where it crosses x = xq (n,) -> (n, 2) y, z (nan if it does not)."""
  x = pts[..., 0] - xq[:, None]
  out = torch.full((pts.shape[0], 2), math.nan, device=pts.device)
  for i in range(pts.shape[1] - 1):
    a, b = x[:, i], x[:, i + 1]
    s = -a / torch.where(b != a, b - a, torch.full_like(a, 1e-9))
    hit = (s >= 0) & (s <= 1) & out[:, 0].isnan()
    yz = pts[:, i, 1:] + s[:, None] * (pts[:, i + 1, 1:] - pts[:, i, 1:])
    out[hit] = yz[hit]
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument("turner")
  p.add_argument("jumper")
  p.add_argument("--jtask", default=JTASK)
  p.add_argument("--span", type=float, default=0.60)
  p.add_argument("--n", type=int, default=64)
  p.add_argument("--seconds", type=float, default=19.0)
  p.add_argument("--window", type=float, default=12.0)
  p.add_argument("--delta-deg", type=float, help="jumper clock = turner phase + delta (default: sweep 0..360 over envs)")
  p.add_argument("--y", default="0", help="jumper spot y (m), comma list: envs split over (y, delta) (the rope circle sits ~-66 mm off the mid line)")
  p.add_argument("--video", help="render env 0 (2x2 views)")
  p.add_argument("--rope-len", type=float, help="rope length (m; default ys_rope.ROPE_LEN 0.85), mass unchanged")
  p.add_argument("--dump", help="save per-step rope y (jumper frame), rope z, sole z (L, R) to this npz")
  a = p.parse_args()

  e1, v1 = make_env(D.TASK, (0.0, 0.0, 0.0), 1)
  tpol = make_policy(D.TASK, a.turner, v1)
  e2, v2 = make_env(a.jtask, (0.0, 0.0, 0.0), 1)
  jpol = make_policy(a.jtask, a.jumper, v2)

  if a.rope_len:
    import mjlab_microduck.tasks.ys_rope_env_cfg as RE
    RE.ROPE_LEN = a.rope_len  # rope_spec reads it at scene build (and the reset keyframe)
  ys = [float(v) for v in a.y.split(",")]
  n = 1 if a.video else a.n * len(ys)
  cfg, mid = trio_cfg(n, a.span, a.jtask)
  if a.video:
    cfg.viewer.height, cfg.viewer.width, cfg.viewer.distance = 360, 480, 1.4
    cfg.viewer.origin_type, cfg.viewer.lookat = cfg.viewer.OriginType.WORLD, (mid, 0.0, 0.12)
    cfg.viewer.max_extra_envs = 0
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if a.video else None)
  venv = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(D.TASK).clip_actions)
  robot, partner, jumper, rope = (env.scene[k] for k in ("robot", "partner", "jumper", "rope"))
  cmd = env.command_manager.get_term("head_pose")
  dev, dt = env.device, env.step_dt
  nd = n // len(ys)
  delta = torch.full((n,), math.radians(a.delta_deg), device=dev) if a.delta_deg is not None else (torch.arange(n, device=dev) % nd) * 2 * math.pi / nd
  yspot = torch.tensor(ys, device=dev).repeat_interleave(nd)
  spot = env.scene.env_origins[:, :2] + torch.stack([torch.full_like(yspot, mid), yspot], 1)
  sole = jumper.find_geoms(["left_foot_collision", "right_foot_collision"], preserve_order=True)[0]
  fell = torch.zeros(n, 3, dtype=torch.bool, device=dev)
  hits, ang, dump = [], [], []
  sid_r, sid_p = robot.find_sites("stem_end")[0][0], partner.find_sites("stem_end")[0][0]
  writer = None
  if a.video:
    writer = imageio.get_writer(a.video, fps=round(1 / dt), codec="libx264", quality=7, macro_block_size=8)
    r = env._offline_renderer
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(int(a.seconds / dt)):
      cmd.amp[:] = 0.12 / D.hc.TIP_M_PER_RAD
      cmd.freq[:] = 1.25
      cmd._write()
      po = obs.clone()
      po["actor"] = D.mirror_obs(obs["partner"], obs["actor"])
      jo = obs.clone()
      jo["actor"] = obs["jumper"].clone()
      jo["actor"][:, 48:61] = jumper_slots(jumper, spot, cmd.phase + delta)
      obs, *_ = venv.step(torch.cat([tpol(obs), D.mirror_act(tpol(po)), jpol(jo)], 1))
      for i, d in enumerate((robot, partner, jumper)):
        fell[:, i] |= d.data.projected_gravity_b[:, 2] > -0.5
      if k * dt >= a.seconds - a.window:
        pts = rope.data.body_link_pos_w  # (n, L, 3) link origins = the polyline (side.py)
        sec = rope_section(pts, jumper.data.root_link_pos_w[:, 0])
        ya = yaw(jumper.data.root_link_quat_w)
        dy = sec[:, 0] - jumper.data.root_link_pos_w[:, 1]
        dx_ = torch.zeros_like(dy)
        yj = -ya.sin() * dx_ + ya.cos() * dy  # rope y in the jumper frame (x offset 0 at the section)
        soles = J.foot_clearance(jumper, sole) + J.FOOT_Z0  # (n, 2) lowest sole corner z (left, right)
        h = torch.zeros(n, dtype=torch.bool, device=dev)
        for j, nm in ((0, "left"), (1, "right")):
          lo, hi = BANDS[nm]
          h |= (yj > lo) & (yj < hi) & (sec[:, 1] + ROPE_RADIUS > soles[:, j]) & (sec[:, 1] - ROPE_RADIUS < HEAD_TOP)
        hits.append(h)
        dump.append(torch.cat([yj[:, None], sec[:, 1:], soles], 1))
        axis = 0.5 * (robot.data.site_pos_w[:, sid_r, 1:] + partner.data.site_pos_w[:, sid_p, 1:])
        m = rope.data.body_link_pos_w[:, rope.data.body_link_pos_w.shape[1] // 2, 1:] - axis
        ang.append(torch.atan2(m[:, 0], -m[:, 1]))
      if writer:
        tiles = []
        for az, el in ((0, -10), (90, -10), (135, -25), (90, -89)):
          r._cam.azimuth, r._cam.elevation = az, el
          r.update(env.sim.data)
          tiles.append(r.render().copy())
        writer.append_data(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
  if writer:
    writer.close()
    print("saved", a.video)

  if a.dump:
    np.savez(a.dump, x=torch.stack(dump).cpu().numpy(), dt=dt)
  H = torch.stack(hits).cpu().numpy()  # (T, n)
  per = round(0.8 / dt)
  cyc = H[: (len(H) // per) * per].reshape(-1, per, n).any(1)  # (cycles, n) any hit in the cycle
  ang = torch.stack(ang)
  turn = ((ang.diff(dim=0) + math.pi) % (2 * math.pi) - math.pi).sum(0) / (2 * math.pi) / (1.25 * a.window)
  ok = ~fell.any(1).cpu().numpy()
  rate = cyc.mean(0)
  print(f"trio  envs {n}  span {a.span}  fell robot/partner/jumper {[round(100 * x, 1) for x in fell.float().mean(0).tolist()]}%  rope turn {turn[torch.from_numpy(ok).to(dev)].mean():.2f}")
  if a.delta_deg is None:  # rows y, columns delta: % of cycles with a hit (X = jumper fell)
    dd = np.degrees(delta.cpu().numpy()[:nd])
    print("  y mm \\ delta deg " + " ".join(f"{x:4.0f}" for x in dd))
    for i, yv in enumerate(ys):
      print(f"  {1000 * yv:6.0f}            " + " ".join(f"{100 * rate[i * nd + j]:4.0f}" if ok[i * nd + j] else "   X" for j in range(nd)))
  else:
    for i, yv in enumerate(ys):
      s_ = slice(i * nd, (i + 1) * nd)
      o = ok[s_]
      print(f"  y {1000 * yv:.0f} mm  delta {a.delta_deg} deg: hit {100 * cyc[:, s_][:, o].mean():.1f}% of cycles (clean envs {100 * (rate[s_][o] == 0).mean():.0f}%)  fell {100 * (~o).mean():.0f}%  cycles {cyc.shape[0]}")


if __name__ == "__main__":
  main()
