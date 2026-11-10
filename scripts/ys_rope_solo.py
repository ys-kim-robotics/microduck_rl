"""7-15 (user: are L and F out of sync?): one turner duck (robot = L) and no partner duck. The rope's far end follows L's own
stem end about its running centre (same circle, same phase: RopeCircleCommand, the rope5-era partner stand-in), so the two
ends are in sync by construction. --handle rigid: the far end is that mocap point; soft: a 0.1 kg gravity-compensated body
on a ~2 Hz spring to it (rope9's compliant partner: it yields to the rope as a duck would). --track saves the same layout as
ys_rope_duo --track, with the far end in the partner's slot. Eval only.

  YS_ROPE_MASS=0.010 uv run scripts/ys_rope_solo.py logs/rsl_rl/ys_rope/<run>/model_N.pt --handle soft --track out.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import mjlab_microduck.tasks  # noqa: F401,E402
from mjlab.entity import EntityCfg  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs.mdp import events as mjlab_events  # noqa: E402
from mjlab.managers import EventTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402
from ys_eval import make_policy  # noqa: E402
import mjlab_microduck.tasks.ys_head_circle_env_cfg as hc  # noqa: E402
import mjlab_microduck.tasks.ys_rope_env_cfg as RC  # noqa: E402

TASK = "Mjlab-YsRopeBoth-Flat-MicroDuck"


def main():
  p = argparse.ArgumentParser()
  p.add_argument("policy")
  p.add_argument("--handle", choices=("rigid", "soft"), default="soft")
  p.add_argument("--freq", type=float, default=1.25)
  p.add_argument("--radius", type=float, default=0.12)
  p.add_argument("--n", type=int, default=64)
  p.add_argument("--seconds", type=float, default=19.0)
  p.add_argument("--span", type=float, default=0.60)
  p.add_argument("--track", required=True)
  a = p.parse_args()

  cfg = RC.make_ys_rope_bob_env_cfg(play=True, stay=True, reverse=True)
  cfg.scene.num_envs, cfg.seed = a.n, 0
  RC.set_span(cfg, a.span)
  if a.handle == "soft":  # as make_ys_rope_soft_env_cfg, at this span
    far = (RC.STEM_END_NOM[0] + a.span, 0.0, RC.STEM_END_NOM[2])
    cfg.scene.entities["soft_handle"] = EntityCfg(spec_fn=RC.soft_handle_spec, init_state=EntityCfg.InitialStateCfg(pos=far))
    cfg.scene.spec_fn = RC.connect_rope_soft
    cfg.events["reset_soft_handle"] = EventTermCfg(func=mjlab_events.reset_root_state_uniform, mode="reset",
                                                   params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("soft_handle")})
  c = cfg.commands["head_pose"]
  c.freeze_radius, c.reverse_prob = True, 0.0  # as ys_rope_duo: forward circle only
  cfg.rewards, cfg.curriculum = {}, {}
  cfg.terminations = {"time_out": cfg.terminations["time_out"]}  # as ys_rope_duo
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  venv = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
  policy = make_policy(TASK, a.policy, venv)
  robot, rope, cmd = env.scene["robot"], env.scene["rope"], env.command_manager.get_term("head_pose")
  sid = robot.find_sites("stem_end")[0][0]
  if a.handle == "soft":
    fe = env.scene["soft_handle"]; fid = fe.find_sites("far_site")[0][0]; far_yz = lambda: fe.data.site_pos_w[:, fid, 1:]
  else:
    mid = env.scene["far_handle"].indexing.mocap_id; far_yz = lambda: env.sim.data.mocap_pos[:, mid, 1:]
  fell = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device); trk = []
  obs = venv.get_observations()
  with torch.inference_mode():
    for _ in range(int(a.seconds / env.step_dt)):
      cmd.amp[:] = a.radius / hc.TIP_M_PER_RAD
      cmd.freq[:] = a.freq
      cmd._write()
      obs, *_ = venv.step(policy(obs))
      fell |= robot.data.projected_gravity_b[:, 2] > -0.5  # tilted > 60 deg
      trk.append(torch.cat([robot.data.site_pos_w[:, sid, None, 1:], far_yz()[:, None].clone(), rope.data.body_link_pos_w[..., 1:]], 1).cpu())
  f2 = torch.stack([fell, torch.zeros_like(fell)], 1)  # (n, 2) as ys_rope_duo: [L, far end]
  np.savez_compressed(a.track, p=torch.stack(trk).numpy(), dt=env.step_dt, fell=f2.cpu().numpy(), freq=a.freq)
  print(f"handle {a.handle}  fall% {100 * fell.float().mean().item():.1f}  saved {a.track}")
  env.close()


if __name__ == "__main__":
  main()
