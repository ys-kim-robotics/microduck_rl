"""ys_lf2_check <policy> [n] [secs] (7-38): the LF2 training env (play cfg, partner = the same policy, mirrored, as self-play) run with
<policy>; handle_sep (unweighted) on the trio's-circle envs after 4 s. rope14 at 20 g (YS_ROPE_MASS=0.020) must sit on SEP_REF
(mean |sep - ref| < 5 mm, asserted), 15 g wider and 25 g narrower at the top. DUMP=<npz> saves th and sep - ref (refitting SEP_REF). TASK=<task id>: another env with the same check (LF3, 7-39)."""
import math, os, sys, torch
from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from ys_eval import make_policy
import mjlab_microduck.tasks.ys_rope_duo_env_cfg as DU
def sepdev(env):
  cmd = env.command_manager.get_term("head_pose"); p = DU.hc.tip_feet(cmd._robot, cmd._sid, cmd._feet)[0]
  th = torch.atan2(p[:, 1] - cmd.center[:, 0], -(p[:, 2] - cmd.center[:, 1])); ref = torch.full_like(th, DU.SEP_REF[0])
  for k, (a, b) in enumerate(DU.SEP_REF[1], start=1): ref += a * torch.cos(k * th) + b * torch.sin(k * th)
  pa = env.scene["partner"]; sep = (cmd._robot.data.site_pos_w[:, cmd._sid] - pa.data.site_pos_w[:, pa.find_sites("stem_end")[0][0]]).norm(dim=-1)
  return sep - ref, th
T = os.environ.get("TASK", "Mjlab-YsRopeLF2-Flat-MicroDuck"); pol = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 64; secs = float(sys.argv[3]) if len(sys.argv) > 3 else 16
cfg = load_env_cfg(T, play=True); cfg.scene.num_envs = n; cfg.actions["partner_joint_pos"].checkpoint = pol
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0"); venv = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(T).clip_actions)
policy = make_policy(T, pol, venv); cmd = env.command_manager.get_term("head_pose"); obs = venv.get_observations(); r, on, dv, th_, lg, rtl, em = [], [], [], [], [], [], []
with torch.inference_mode():
  for k in range(int(secs / env.step_dt)):
    obs, *_ = venv.step(policy(obs))
    rt = DU.rope_timing(env)  # 7-40: its running mean needs every step (weight 0 here: the env does not call it)
    if k * env.step_dt >= 4: lg.append(DU.rope_lag(env)[0]); rtl.append(rt); em.append(torch.where(env._ys_lag['t'] >= DU.LAG_WAIT, torch.atan2(env._ys_lag['z'][:, 1], env._ys_lag['z'][:, 0]), math.nan)); r.append(DU.handle_sep(env)); dv.append(sepdev(env)[0]); th_.append(sepdev(env)[1]); on.append(((cmd.amp * DU.hc.TIP_M_PER_RAD - DU.TRIO_R).abs() < 1e-4) & ((cmd.freq - DU.TRIO_HZ).abs() < 1e-4))
r, on, dv, th_ = torch.stack(r), torch.stack(on), torch.stack(dv), torch.stack(th_)
lg = torch.stack(lg); z_ = torch.polar(torch.ones_like(lg), lg); ok_ = on.all(0)  # 7-40 (LF4): rope_lag, xana's lag per step; envs on the trio's circle throughout
lag_e = torch.rad2deg(z_[:, ok_].mean(0).angle()); R_ = z_[:, ok_].mean(0).abs()
em_ = torch.stack(em); em_ = torch.rad2deg(em_[~em_.isnan()]); rtl = torch.stack(rtl)  # every (step, env) where rope_timing pays
ref_ = torch.rad2deg(torch.polar(torch.ones_like(em_), torch.deg2rad(em_)).mean().angle())
print(f'rope_lag on trio envs ({int(ok_.sum())}): circular mean {torch.rad2deg(z_[:, ok_].mean().angle()):+.1f} deg (LAG_REF {math.degrees(DU.LAG_REF):+.1f}), per-env mean 10-90% {lag_e.quantile(0.1):+.1f} / {lag_e.quantile(0.9):+.1f}, per-env concentration {R_.mean():.3f}')
print(f'rope_timing running mean where it pays ({em_.numel()} env-steps): circular mean {ref_:+.1f} deg (-> LAG_REF), 10/50/90% {em_.quantile(0.1):+.1f} / {em_.quantile(0.5):+.1f} / {em_.quantile(0.9):+.1f}, rope_timing mean on trio envs {rtl[:, ok_].mean():.3f}')
top = (th_.abs() > 2.6)
import os, numpy as np
if os.environ.get('DUMP'): np.savez(os.environ['DUMP'], th=th_[on].cpu().numpy(), sepref=dv[on].cpu().numpy())
print(f'signed sep - ref on trio envs: mean {1000 * dv[on].mean():+.1f} mm, top +-30 deg {1000 * dv[on & top].mean():+.1f} mm, sd {1000 * dv[on].std():.1f} mm')
print(f"trio-circle share {on.float().mean():.2f}  handle_sep mean on those {r[on].mean():.3f}  (0 elsewhere: {r[~on].abs().max():.3f})")
if os.environ.get('YS_ROPE_MASS') == '0.020' and 'rope14' in pol: assert abs(1000 * dv[on].mean()) < 5, 'SEP_REF off at 20 g: refit it'
