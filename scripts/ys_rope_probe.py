"""Stage 5a: what handle circle turns a jump rope? Rope physics only, no robot (ys personal tool).

The rope hangs between a far handle at (0, 0, H) and the duck-side handle at (SPAN, 0, H) (free swivels).
The duck-side handle circles in the y-z plane at radius r, frequency f, from a hanging start; the far
handle circles in phase with it (a mirroring partner), or with --fixed-anchor stays put (a wall, 5a).
Measured on the rope's middle link over the last WINDOW seconds:
  rev      mid revolutions per handle revolution (1 = the rope turns with the handle), mean of trials
  ok%      share of trials with rev > 0.8;  blow = trials dropped for solver blow-up (Fmax > 20 N)
  zlo/zhi  lowest / highest mid height (mm): a jumper needs zlo ≈ 0 and zhi above its head
  F        handle force (N): mean, 95th percentile, max of the connect constraint
CPU MuJoCo, same timestep/integrator as training (5 ms, implicitfast).

  uv run scripts/ys_rope_probe.py --radii 0.02 0.05 0.1 --freqs 1.0 1.5
  uv run scripts/ys_rope_probe.py --radii 0.05 --freqs 1.5 --video out.mp4
"""

import argparse
import itertools
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from mjlab_microduck.tasks.ys_rope import ROPE_LINKS, ROPE_MASS, add_rope  # noqa: E402

H, SPAN = 0.2, 0.7  # m: anchor/handle-centre height (≈ duck mouth), anchor-to-handle distance
SETTLE, RUN, WINDOW = 1.0, 16.0, 8.0  # s
KICK_S = 5.0  # s


def build(length, mass=ROPE_MASS, both_ends=True):
  spec = mujoco.MjSpec()
  spec.option.timestep = 0.005
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.worldbody.add_light(pos=[0.35, 0, 1.5], dir=[0, 0, -1])
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1], rgba=[0.8, 0.8, 0.8, 1], friction=[0.5, 0.005, 0.0001])
  spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[-0.02, 0, 0.25], size=[0.02, 0.3, 0.25], rgba=[0.6, 0.6, 0.7, 1],
                          contype=0, conaffinity=0)
  start, site = add_rope(spec, (0, 0, H), (SPAN, 0, H), length=length, mass=mass, free=both_ends)
  if both_ends:  # far turner: a second mocap handle, the rope's start connected to it like the duck side
    far = spec.worldbody.add_body(name="far_handle", mocap=True, pos=[0, 0, H])
    far.add_site(name="far_site")
    spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE, name1=start, name2="far_site")
  hand = spec.worldbody.add_body(name="handle", mocap=True, pos=[SPAN, 0, H])
  hand.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[0, -0.03, 0, 0, 0.03, 0], size=[0.006, 0, 0],
                rgba=[0.2, 0.4, 0.9, 1], contype=0, conaffinity=0)
  eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE, name1=site, name2="handle_site")
  hand.add_site(name="handle_site")
  eq.name = "grip"
  return spec.compile()


def run(length, r, f, video=None, kick=None, mass=ROPE_MASS, zscale=1.0, phase0=0.0, both_ends=True, f_sustain=None):
  """kick: radius held for the first KICK_S seconds (spin-up), then r -> does r SUSTAIN the turn?
  f_sustain: after KICK_S the frequency drops from f to this (phase continuous); rev is then
  measured against f_sustain. both_ends: the far end circles in phase instead of a fixed anchor."""
  m = build(length, mass, both_ends)
  d = mujoco.MjData(m)
  mid = m.body(f"rope_{ROPE_LINKS // 2}").id
  n_eq = 3
  frames, renderer, cams = [], None, []
  if video:
    renderer = mujoco.Renderer(m, 360, 480)
    for az, el in ((180, -5), (135, -20)):  # looking down the rope from behind the handle; 3/4
      c = mujoco.MjvCamera()
      c.lookat[:] = (SPAN / 2, 0, H * 0.8)
      c.distance, c.azimuth, c.elevation = 1.3, az, el
      cams.append(c)
  ang, zs, forces = [], [], []
  steps = int((SETTLE + RUN) / m.opt.timestep)
  w = phase0
  f_end = f_sustain or f
  for k in range(steps):
    t = k * m.opt.timestep - SETTLE
    if t > 0:
      w += 2 * np.pi * (f_sustain if f_sustain and t >= KICK_S else f) * m.opt.timestep
    r_now = kick if kick and t < KICK_S else r
    rr = r_now * np.clip(t, 0.0, 1.0)  # 1 s radius ramp: no position jump at t=0
    off = (rr * np.sin(w), -zscale * rr * np.cos(w))
    d.mocap_pos[m.body("handle").mocapid[0]] = (SPAN, off[0], H + off[1])
    if both_ends:  # the far turner mirrors the duck: same circle, same phase, same sense
      d.mocap_pos[m.body("far_handle").mocapid[0]] = (0.0, off[0], H + off[1])
    mujoco.mj_step(m, d)
    if t > RUN - WINDOW:
      p = d.xpos[mid]
      ang.append(np.arctan2(p[1], -(p[2] - H)))  # 0 = straight down, same sense as the handle
      zs.append(p[2])
      eq_rows = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
      forces.append(np.linalg.norm(d.efc_force[eq_rows][:n_eq]))
    if renderer and k % 4 == 0:
      tiles = []
      for c in cams:
        renderer.update_scene(d, c)
        tiles.append(renderer.render())
      frames.append(np.hstack(tiles))
  if renderer:
    import imageio

    imageio.mimsave(video, frames, fps=50, macro_block_size=8)
  rev = (np.unwrap(ang)[-1] - np.unwrap(ang)[0]) / (2 * np.pi) / (f_end * WINDOW)
  forces = np.array(forces)
  return dict(rev=rev, zlo=1000 * min(zs), zhi=1000 * max(zs), F=forces.mean(), F95=np.percentile(forces, 95), Fmax=forces.max())


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--radii", type=float, nargs="+", default=[0.02, 0.035, 0.05, 0.08, 0.12])
  p.add_argument("--freqs", type=float, nargs="+", default=[1.0, 1.25, 1.5])
  p.add_argument("--lengths", type=float, nargs="+", default=[0.85])
  p.add_argument("--masses", type=float, nargs="+", default=[ROPE_MASS])
  p.add_argument("--zscale", type=float, default=1.0, help="vertical radius = this x r (ellipse)")
  p.add_argument("--fixed-anchor", action="store_true", help="far end fixed to the wall (5a) instead of circling in phase")
  p.add_argument("--sustain-freqs", type=float, nargs="+", help="after KICK_S drop the frequency to each of these")
  p.add_argument("--trials", type=int, default=6, help="start phases per combo (rope motion is chaotic)")
  p.add_argument("--video")
  p.add_argument("--kick", type=float, help="spin up at this radius for KICK_S s, then drop to --radii")
  a = p.parse_args()
  print(f"{'M':>5s} {'L':>5s} {'r':>5s} {'f':>5s} {'rev':>5s} {'ok%':>4s} {'zlo':>5s} {'zhi':>5s} {'F':>5s} {'F95':>5s} {'Fmax':>5s} {'blow':>4s}")
  for M, L, r, f, fs in itertools.product(a.masses, a.lengths, a.radii, a.freqs, a.sustain_freqs or [None]):
    os = [run(L, r, f, a.video if i == 0 else None, a.kick, M, a.zscale, 2 * np.pi * i / a.trials, not a.fixed_anchor, fs)
          for i in range(a.trials)]
    good = [o for o in os if o["Fmax"] < 20]  # blown-up runs (solver divergence) excluded
    if not good:
      print(f"{M:5.3f} {L:5.2f} {r:5.3f} {f:5.2f}  all {len(os)} trials blew up", flush=True)
      continue
    m = {k: np.mean([o[k] for o in good]) for k in good[0]}
    ok = 100 * np.mean([o["rev"] > 0.8 for o in good])
    print(f"{M:5.3f} {L:5.2f} {r:5.3f} {f:5.2f}{'>' + format(fs, '.2f') if fs else ''} {m['rev']:5.2f} {ok:4.0f} {m['zlo']:5.0f} {m['zhi']:5.0f} {m['F']:5.2f} {m['F95']:5.2f} {m['Fmax']:5.2f} {len(os) - len(good):4d}", flush=True)


if __name__ == "__main__":
  main()
