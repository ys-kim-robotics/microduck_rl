"""Stage 5 (ys): a jump rope as a capsule chain (MJWarp has no `cable` plugin).

The rope hangs from a fixed wall anchor (ball joint) and its far end is pinned to a handle by an
equality CONNECT (a free swivel, like a real jump-rope handle's bearing), so rope twist never
torques the handle. Built into an MjSpec so the same rope serves the CPU physics probe (5a,
handle = mocap body) and later the training scene (handle = the duck's jaw).
"""

import mujoco
import numpy as np

ROPE_LEN = 0.85  # m; span 0.7 + a ~0.2 m arc (STAGES-ys.md stage 5 sizing)
ROPE_MASS = 0.02  # kg
ROPE_LINKS = 12
ROPE_RADIUS = 0.003  # m
ROPE_DAMPING = 1e-4  # N·m·s/rad per ball joint: <1e-4 blew up at dt 5 ms + implicitfast (5a grid)
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
