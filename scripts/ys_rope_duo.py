"""Stage 5d: two turner ducks, one rope, one policy (ys personal tool, eval only, no training).

The single-duck rope env (Mjlab-YsRopeBoth) with its mirrored mocap far handle replaced by a second
duck ("partner") facing the first across the rope, both running the same checkpoint. Facing the other way,
the same world rope rotation is the REVERSED circle for the partner, which the policy never learned
(a reverse flag in a command slot failed: the slot was always 0 in training, so the obs normalizer
blew it up 10x+, rope6). Instead the partner runs the policy through left-right mirroring
(symmetry.py tables): its obs is mirrored, the robot's forward command written in, and the action
mirrored back. A mirrored forward circle is exactly the reversed one, and the robot is symmetric. Measured over the last WINDOW seconds:
  turn      rope mid-link revolutions per commanded revolution about the stem-to-stem axis
  apex/low  rope's highest centre point / lowest surface point (mm), per 0.8 s window, mean
  dxy/dyaw  each duck's trunk drift (mm) and heading change (deg), over envs where neither ever tilted
            past 60 deg;  'ever' = tilted past 60 deg at some step, 'down at end' = still tilted at the end

  uv run scripts/ys_rope_duo.py logs/rsl_rl/ys_rope/<run>/model_N.pt --freq 1.25 --video out.mp4
"""

import argparse
import copy
import math
import sys
from dataclasses import replace
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import mjlab_microduck.tasks  # noqa: F401,E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers import ObservationTermCfg, TerminationTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402
from ys_eval import make_env, make_policy  # noqa: E402
import mjlab_microduck.tasks.ys_head_circle_env_cfg as hc  # noqa: E402
from mjlab_microduck.tasks.symmetry import _get_tensors  # noqa: E402
from mjlab_microduck.tasks.ys_rope import ROPE_LINKS, ROPE_RADIUS  # noqa: E402
from mjlab_microduck.tasks.ys_rope_env_cfg import FAR_NOM, STEM_END_NOM, make_ys_rope_bob_env_cfg  # noqa: E402

TASK = "Mjlab-YsRopeBoth-Flat-MicroDuck"
PARTNER_X = FAR_NOM[0] + STEM_END_NOM[0]  # partner root: its stem end lands on the far handle spot


def connect_duo(spec):
  for a, b in (("rope/rope_start", "partner/stem_end"), ("rope/rope_end", "robot/stem_end")):
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE, name1=a, name2=b)
    eq.solref = [0.05, 1.0]  # as connect_rope (ys_rope_env_cfg)


def mirror_obs(x, robot_obs):
  """Partner obs -> its mirror image, with the robot's (forward) circle command in the head slots."""
  perm, sign, _, _ = _get_tensors(x.device)
  m = x[:, perm] * sign
  m[:, 51:55] = robot_obs[:, 51:55]
  return m


def mirror_act(a):
  _, _, perm, sign = _get_tensors(a.device)
  return a[:, perm] * sign


def own_action(lo, hi):
  return lambda env: env.action_manager.action[:, lo:hi]


def duo_cfg(n, connect=True):
  cfg = make_ys_rope_bob_env_cfg(play=True, stay=True, reverse=True)
  cfg.scene.num_envs, cfg.seed = n, 0
  ents = cfg.scene.entities
  robot = ents["robot"]
  partner = replace(robot, init_state=replace(robot.init_state, pos=(PARTNER_X, 0.0, robot.init_state.pos[2]), rot=(0.0, 0.0, 0.0, 1.0)))
  cfg.scene.entities = {"robot": robot, "partner": partner, "rope": ents["rope"]}  # partner before rope (default_joint_pos slicing)
  cfg.scene.spec_fn = connect_duo if connect else None
  cfg.scene.sim_nconmax = None if not hasattr(cfg.scene, "sim_nconmax") else cfg.scene.sim_nconmax
  cfg.sim.nconmax = 96
  cfg.commands["head_pose"] = hc.HeadCircleCommandCfg(**{k: getattr(cfg.commands["head_pose"], k) for k in hc.HeadCircleCommandCfg.__dataclass_fields__})
  cfg.commands["head_pose"].freeze_radius = True
  cfg.rewards, cfg.curriculum = {}, {}
  cfg.terminations = {"time_out": cfg.terminations["time_out"]}
  cfg.events.pop("reset_far_handle")
  for k, e in list(cfg.events.items()):  # partner copies of every robot event (reset pose, joints, DR)
    a = (e.params or {}).get("asset_cfg")
    if (a is not None and a.name == "robot") or k == "reset_base":
      e2 = copy.deepcopy(e)
      e2.params = dict(e2.params or {})
      e2.params["asset_cfg"] = replace(a, name="partner") if a is not None else SceneEntityCfg("partner")
      cfg.events[f"{k}_partner"] = e2
  act = cfg.actions["joint_pos"]
  cfg.actions["partner_joint_pos"] = replace(act, entity_name="partner")
  actor = cfg.observations["actor"]
  actor.terms["actions"] = ObservationTermCfg(func=own_action(0, 14))
  part = copy.deepcopy(actor)
  for t in part.terms.values():
    if t.params and "asset_cfg" in t.params:
      t.params = {**t.params, "asset_cfg": replace(t.params["asset_cfg"], name="partner")}
  part.terms["actions"] = ObservationTermCfg(func=own_action(14, 28))
  for t in ("base_ang_vel", "projected_gravity"):  # their asset defaults to robot: name it
    part.terms[t].params = {**(part.terms[t].params or {}), "asset_cfg": SceneEntityCfg("partner")}
  part.terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg("partner")
  part.terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg("partner")
  cfg.observations = {"actor": actor, "critic": cfg.observations["critic"], "partner": part}
  return cfg


def yaw(q):
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
  p = argparse.ArgumentParser()
  p.add_argument("policy")
  p.add_argument("--freq", type=float, default=1.25, help="circle frequency (Hz), > 0")
  p.add_argument("--radius", type=float, default=0.12)
  p.add_argument("--n", type=int, default=16)
  p.add_argument("--seconds", type=float, default=19.0)
  p.add_argument("--window", type=float, default=6.0)
  p.add_argument("--partner", help="partner checkpoint (default: the same policy)")
  p.add_argument("--no-connect", action="store_true", help="rope not attached: isolates the scene/wiring from rope forces")
  p.add_argument("--video", help="render env 0 (2x2 views) to this mp4")
  a = p.parse_args()

  env1, venv1 = make_env(TASK, (0.0, 0.0, 0.0), 1)  # only to build the policy (28-D duo actions would not load)
  policy = make_policy(TASK, a.policy, venv1)
  ppolicy = make_policy(TASK, a.partner, venv1) if a.partner else policy

  cfg = duo_cfg(a.n if not a.video else 1, connect=not a.no_connect)
  if a.video:
    cfg.viewer.height, cfg.viewer.width, cfg.viewer.distance = 360, 480, 1.7
    cfg.viewer.origin_type, cfg.viewer.lookat = cfg.viewer.OriginType.WORLD, ((PARTNER_X) / 2, 0.0, 0.15)
    cfg.viewer.max_extra_envs = 0
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if a.video else None)
  venv = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
  robot, partner, rope = env.scene["robot"], env.scene["partner"], env.scene["rope"]
  assert torch.allclose(partner.data.default_joint_pos, robot.data.default_joint_pos), "partner default_joint_pos mis-sliced"
  cmd = env.command_manager.get_term("head_pose")
  n, dt = env.num_envs, env.step_dt
  sid_r, sid_p = robot.find_sites("stem_end")[0][0], partner.find_sites("stem_end")[0][0]
  ducks = (robot, partner)
  p0 = [d.data.root_link_pos_w[:, :2].clone() for d in ducks]
  y0 = [yaw(d.data.root_link_quat_w).clone() for d in ducks]
  fell = torch.zeros(n, 2, dtype=torch.bool, device=env.device)
  t_fall = torch.full((n,), math.nan, device=env.device)
  t_duck = torch.full((n, 2), math.nan, device=env.device)  # first tilt per duck (robot, partner)
  t_blow = torch.full((n,), math.nan, device=env.device)  # rope link > 15 m/s or non-finite (rope_blew_up)
  ang, low, apex, frames = [], [], [], []
  writer = None
  if a.video:
    writer = imageio.get_writer(a.video, fps=round(1 / dt), codec="libx264", quality=7, macro_block_size=8)
    r = env._offline_renderer
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(int(a.seconds / dt)):
      cmd.amp[:] = a.radius / hc.TIP_M_PER_RAD
      cmd.freq[:] = a.freq
      cmd._write()
      act_r = policy(obs)
      po = obs.clone()
      po["actor"] = mirror_obs(obs["partner"], obs["actor"])
      act_p = mirror_act(ppolicy(po))
      obs, *_ = venv.step(torch.cat([act_r, act_p], dim=1))
      for i, d in enumerate(ducks):
        fell[:, i] |= d.data.projected_gravity_b[:, 2] > -0.5  # tilted > 60 deg
      t = k * dt
      t_fall = torch.where(torch.isnan(t_fall) & fell.any(1), torch.full_like(t_fall, t), t_fall)
      t_duck = torch.where(torch.isnan(t_duck) & fell, torch.full_like(t_duck, t), t_duck)
      v = rope.data.body_link_lin_vel_w.norm(dim=-1).max(1).values
      blow = (v > 15) | ~torch.isfinite(v)
      t_blow = torch.where(torch.isnan(t_blow) & blow, torch.full_like(t_blow, t), t_blow)
      if k * dt >= a.seconds - a.window:
        axis = 0.5 * (robot.data.site_pos_w[:, sid_r, 1:] + partner.data.site_pos_w[:, sid_p, 1:])
        m = rope.data.body_link_pos_w[:, ROPE_LINKS // 2, 1:] - axis
        ang.append(torch.atan2(m[:, 0], -m[:, 1]))
        z = rope.data.body_link_pos_w[..., 2]
        low.append(z.min(1).values - ROPE_RADIUS)
        apex.append(z.max(1).values)
      if writer:
        tiles = []
        for az, el in ((0, -15), (90, -15), (135, -25), (90, -89)):
          r._cam.azimuth, r._cam.elevation = az, el
          r.update(env.sim.data)
          tiles.append(r.render().copy())
        writer.append_data(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
  if writer:
    writer.close()
    print("saved", a.video)
  ang = torch.stack(ang)
  turn = (ang.diff(dim=0) + math.pi) % (2 * math.pi) - math.pi
  turn = turn.sum(0) / (2 * math.pi) / (a.freq * a.window)
  T = int(0.8 / dt)
  w = lambda x: x[: (x.shape[0] // T) * T].reshape(-1, T, n)  # noqa: E731
  low_w, apex_w = 1000 * w(torch.stack(low)).min(1).values, 1000 * w(torch.stack(apex)).max(1).values
  ok = ~fell.any(1)
  fb = ~torch.isnan(t_fall)
  bb = ~torch.isnan(t_blow)
  first_blow = bb & (~fb | (t_blow <= t_fall))
  print(f"  rope blew up {100 * bb.float().mean():.0f}% of envs; blow-up came first (before any tilt) in {first_blow.sum().item()} of {fb.sum().item()} fallen envs")
  tr, tp = t_duck.nan_to_num(1e9).unbind(1)
  print(f"  who tilts first (of {int(fb.sum())} fallen envs): robot {int(((tr < tp) & fb).sum())}  partner(mirrored) {int(((tp < tr) & fb).sum())}  same step {int(((tp == tr) & fb).sum())}")
  print("  first tilt t(s):", [round(x, 2) for x in t_fall[fb].tolist()], " first blow t(s):", [round(x, 2) for x in t_blow[bb].tolist()])
  print(f"freq {a.freq} Hz  radius {a.radius} m  envs {n}")
  print(f"  rope turn {turn[ok].mean():.2f}   apex {apex_w[:, ok].mean():.0f} mm   lowest {low_w[:, ok].mean():.1f} mm   fall% {100 * fell.any(1).float().mean():.1f}")
  for i, (name, d) in enumerate(zip(("robot", "partner"), ducks)):
    dxy = 1000 * (d.data.root_link_pos_w[:, :2] - p0[i]).norm(dim=-1)
    dyaw = torch.rad2deg((yaw(d.data.root_link_quat_w) - y0[i] + math.pi) % (2 * math.pi) - math.pi).abs()
    down = 100 * (d.data.projected_gravity_b[:, 2] > -0.5).float().mean()
    print(f"  {name:8s} dxy {dxy[ok].mean():5.1f} mm  dyaw {dyaw[ok].mean():5.1f} deg  tilted>60deg ever {100 * fell[:, i].float().mean():.0f}%  down at end {down:.0f}%")


if __name__ == "__main__":
  main()
