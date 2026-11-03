"""Record a trained checkpoint as a synchronized 2x2 multi-view MP4 (ys personal tool).

One rollout, four cameras per frame (front / side / back-3/4 / top), so the views
stay in sync. Mirrors mjlab's play.py env/policy setup.

  uv run scripts/ys_multiview.py Mjlab-Velocity-Flat-MicroDuck path/to/model_499.pt --seconds 20
"""

import argparse
import math
import sys
from pathlib import Path

import imageio
import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

sys.path.insert(0, str(Path(__file__).parent))
from ys_eval import fix_command, make_policy  # noqa: E402
from ys_head_circle import set_head_circle  # noqa: E402
from mjlab_microduck.tasks.ys_head_circle_env_cfg import apply_tip_force  # noqa: E402

# (label, azimuth, elevation). MuJoCo azimuth 0 = camera placed at -x looking +x (sees the face).
VIEWS = [("front", 0, -15), ("side", 90, -15), ("back-3/4", 135, -25), ("top", 90, -89)]


def main():
  p = argparse.ArgumentParser()
  p.add_argument("task")
  p.add_argument("checkpoint")
  p.add_argument("--seconds", type=float, default=20.0)
  p.add_argument("--size", type=int, default=480, help="height of each view (px)")
  p.add_argument("--distance", type=float, default=0.8, help="camera distance (m)")
  p.add_argument("--out", default=None)
  p.add_argument("--lookat", type=float, nargs=3, help="fixed world point to look at (default: follow the robot)")
  p.add_argument("--cmd", type=float, nargs=3, help="hold this vx vy wz (default: the task's random commands)")
  p.add_argument("--head-circle", type=float, nargs="+", metavar="A F [BOOST LEAD_DEG]",
                 help="drive head_yaw/pitch on a circle (needs --cmd)")
  p.add_argument("--tip-force", type=float, nargs=3, metavar=("FX", "FR", "FZ"), help="rope-tension stand-in at mouth_tip (N)")
  p.add_argument("--body-circle", type=float, metavar="R", help="5b task: hold the feet-frame circle radius R (m), no load")
  p.add_argument("--body-freq", type=float, help="5b task: circle frequency (Hz)")
  p.add_argument("--views", nargs="+", help="subset of view labels (1, 2 or 4)")
  a = p.parse_args()
  views = [v for v in VIEWS if not a.views or v[0] in a.views]

  env_cfg = load_env_cfg(a.task, play=True)
  agent_cfg = load_rl_cfg(a.task)
  env_cfg.scene.num_envs = 1
  env_cfg.seed = 0  # same command sequence for every checkpoint -> videos are comparable
  torch.manual_seed(0)
  if a.cmd:
    fix_command(env_cfg, tuple(a.cmd))
  env_cfg.viewer.height, env_cfg.viewer.width = a.size, a.size * 4 // 3
  env_cfg.viewer.distance = a.distance
  env_cfg.viewer.max_extra_envs = 0
  if a.lookat:  # e.g. the rope's middle, so the far end stays in frame
    env_cfg.viewer.origin_type, env_cfg.viewer.lookat = env_cfg.viewer.OriginType.WORLD, tuple(a.lookat)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode="rgb_array")
  venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  policy = make_policy(a.task, a.checkpoint, venv)  # .pt or .onnx

  r = env._offline_renderer
  fps = round(1.0 / env.step_dt)
  out = Path(a.out or Path(a.checkpoint).with_name(f"{Path(a.checkpoint).stem}_multiview.mp4"))
  writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=7, macro_block_size=8)

  if a.body_circle is not None:
    from mjlab_microduck.tasks.ys_head_circle_env_cfg import TIP_M_PER_RAD
    hc = env.command_manager.get_term("head_pose")
    hc.cfg.freeze_radius = True
  obs = venv.get_observations()
  with torch.inference_mode():
    for k in range(int(a.seconds * fps)):
      act = policy(obs)
      if a.head_circle:
        A, F, boost, lead = (list(a.head_circle) + [1.0, 0.0])[:4]
        set_head_circle(env, (k + 1) * env.step_dt, A * boost, F, math.radians(lead))
      if a.body_circle is not None:
        hc.amp[:] = a.body_circle / TIP_M_PER_RAD
        if a.body_freq:
          hc.freq[:] = a.body_freq
        hc.tip_force.zero_()
        hc._write()
      if a.tip_force:
        rb = env.scene["robot"]
        apply_tip_force(rb, rb.find_bodies("jaw_soft")[0][0], rb.find_sites("mouth_tip")[0][0], *a.tip_force)
      obs, *_ = venv.step(act)
      tiles = []
      for label, az, el in views:
        r._cam.azimuth, r._cam.elevation = az, el
        r.update(env.sim.data)
        tiles.append(r.render().copy())
      grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]) if len(tiles) == 4 else np.hstack(tiles)
      writer.append_data(grid)
  writer.close()
  print(f"saved {out}  ({a.seconds:.0f}s @ {fps}fps, views: {', '.join(v[0] for v in views)})")


if __name__ == "__main__":
  main()
