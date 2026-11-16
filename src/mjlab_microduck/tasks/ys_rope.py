"""Stage 5 (ys): a jump rope as a capsule chain (MJWarp has no `cable` plugin).

The rope hangs from a fixed wall anchor (ball joint) and its far end is pinned to a handle by an
equality CONNECT (a free swivel, like a real jump-rope handle's bearing), so rope twist never
torques the handle. Built into an MjSpec so the same rope serves the CPU physics probe (5a,
handle = mocap body) and later the training scene (handle = the duck's jaw).
"""

import math
import os

import mujoco
import numpy as np
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

ROPE_LEN = 0.85  # m; span 0.7 + a ~0.2 m arc (STAGES-ys.md stage 5 sizing)
ROPE_MASS = float(os.environ.get("YS_ROPE_MASS", "0.02"))  # kg (7-13 check 9: YS_ROPE_MASS to vary it)
ROPE_LINKS = 12
ROPE_RADIUS = 0.003  # m
ROPE_DAMPING = float(os.environ.get("YS_ROPE_DAMPING", "5e-4"))  # N·m·s/rad per ball joint (7-13 check 9: YS_ROPE_DAMPING): 1e-4 held on CPU (5a) but blew up 3/16 in the robot scene (5c-2); 5e-4 held 16/16
ROPE_ARMATURE = 1e-6  # kg·m²: ~0.4x a link's own inertia; 0 blew up, 1e-5 visibly stiffened the rope (5a)


def add_rope(spec, anchor, end, name="rope", length=ROPE_LEN, mass=ROPE_MASS, links=ROPE_LINKS, free=False):
  """Chain from `anchor` toward `end`, initialised as a V sagging below the straight line so its
  length fits. The first link hangs from a world-fixed ball joint at `anchor`, or with free=True
  has a free joint (connect its start site to a moving handle). Never parent the rope to a mocap
  body: moving a mocap teleports its subtree, so the chain is dragged rigidly with no inertia (5a-2).
  Returns the names of the (start, end) sites."""
  anchor, end = np.asarray(anchor, float), np.asarray(end, float)
  seg = length / links
  span = np.linalg.norm(end - anchor)
  sag = np.sqrt(max((length / 2) ** 2 - (span / 2) ** 2, 0.0))
  mid = (anchor + end) / 2 - np.array([0, 0, sag])
  pts = [anchor + (mid - anchor) * t for t in np.linspace(0, 1, links // 2 + 1)]
  pts += [mid + (end - mid) * t for t in np.linspace(0, 1, links - links // 2 + 1)[1:]]
  parent = spec.worldbody
  prev = np.zeros(3)  # parent frame origin in world (worldbody: origin)
  for i in range(links):
    d = pts[i + 1] - pts[i]
    d /= np.linalg.norm(d)
    body = parent.add_body(name=f"{name}_{i}", pos=pts[i] - prev)
    prev = pts[i]
    if i == 0 and free:
      body.add_freejoint(name=f"{name}_free")
      body.add_site(name=f"{name}_start", size=[0.004, 0, 0])
    else:
      body.add_joint(name=f"{name}_j{i}", type=mujoco.mjtJoint.mjJNT_BALL, damping=ROPE_DAMPING, armature=ROPE_ARMATURE)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, 0, 0, *(d * seg)], size=[ROPE_RADIUS, 0, 0],
                  mass=mass / links, rgba=[0.9, 0.2, 0.2, 1], contype=2, conaffinity=1, condim=3)
    parent = body
  parent.add_site(name=f"{name}_end", pos=d * seg, size=[0.004, 0, 0])
  return f"{name}_start", f"{name}_end"


ROPE_DR = tuple(float(v) for v in os.environ["YS_ROPE_DR"].split(",")) if os.environ.get("YS_ROPE_DR") else None  # 7-14: "lo,hi" kg


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def rope_mass_scale(env, env_ids, asset_cfg, mass_range=(0.01, 0.04)):
  """7-14 startup event (user 2026-11-09: L / F and J should cope with whatever rope comes): the whole rope's mass, one draw
  per env, log-uniform in mass_range (kg; median at the geometric mean), every link scaled alike (mjlab's pseudo_inertia
  draws per body, which averages the rope's total out). Mass and inertia scale together (density), CoM unchanged."""
  import torch
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device)
  bi = torch.as_tensor(env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids], device=env.device).long(); ids = ids.long()
  m0, i0 = env.sim.get_default_field("body_mass")[bi], env.sim.get_default_field("body_inertia")[bi]
  lo, hi = (math.log(m / float(m0.sum())) for m in mass_range)
  s = torch.exp(torch.empty(len(ids), device=env.device).uniform_(lo, hi))
  g = torch.meshgrid(ids, bi, indexing="ij")
  env.sim.model.body_mass[g] = m0[None] * s[:, None]
  env.sim.model.body_inertia[g] = i0[None] * s[:, None, None]


# 3-a (LF3, STAGES 7-39): the rope's mass in the L·F input, in the twist vx slot (48). rope14 reads 0 there while it turns (L·F walk out
# on the standing ONNX), and vx mirrors to itself (the self-play partner reads the same). 20 g -> 0, so LF3 starts as rope14 at 20 g;
# +-5 g -> +-0.087 = rope14's normalizer std in that slot (left from its walking ancestors, ~frozen after 9e8 samples): about +-1 normalized.
LF_MASS_IN = os.environ.get("YS_LF_MASS_IN") == "1"  # trio / eval scripts: feed the turners the rope's mass (an LF3 checkpoint)
MASS_IN_REF, MASS_IN_GAIN, MASS_IN_SLOT = 0.020, 0.087 / 0.005, 48


def rope_mass_enc(env):
  """(N,) the env's rope mass as the L·F input value (0 at 20 g)."""
  import torch
  bi = env.__dict__.get("_ys_rope_bi")
  if bi is None:
    bi = env._ys_rope_bi = torch.as_tensor(env.scene["rope"].indexing.body_ids, device=env.device).long()
  m = env.sim.model.body_mass[..., bi].sum(-1)  # (N,) with a per-env mass draw, else (1,) or ()
  return (m - MASS_IN_REF) * MASS_IN_GAIN * torch.ones(env.num_envs, device=env.device)


def twist_with_rope_mass(env, command_name: str = "twist"):
  """Observation term: the twist command block, its vx slot replaced by rope_mass_enc."""
  c = env.command_manager.get_command(command_name).clone()
  c[:, 0] = rope_mass_enc(env)
  return c


def add_mass_input(cfg) -> None:
  """3-a: every obs group with the twist command term (actor, critic, the duo's partner) reads the rope's mass in its vx slot."""
  for g in cfg.observations.values():
    t = g.terms.get("command")
    if t is not None and (t.params or {}).get("command_name") == "twist":
      t.func = twist_with_rope_mass


def demo():
  spec = mujoco.MjSpec()
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
  _, site = add_rope(spec, (0, 0, 0.2), (0.7, 0, 0.2))
  m = spec.compile()
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  assert m.nbody == ROPE_LINKS + 1 and abs(m.body_subtreemass[1] - ROPE_MASS) < 1e-9
  assert np.allclose(d.site(site).xpos, (0.7, 0, 0.2), atol=1e-6), d.site(site).xpos
  print("ok", d.site(site).xpos)


if __name__ == "__main__":
  demo()
