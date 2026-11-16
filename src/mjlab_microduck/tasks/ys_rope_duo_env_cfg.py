"""Stage 5 (ys): train the turner against a REAL partner duck instead of the far-handle stand-in.

Scene per env: robot (learning) ══ rope ══ partner (frozen rope5 policy, mirrored, facing -x).
Every single-duck partner proxy (lag/scale, phase lock, compliant handle, slack/taut span) passed at
0% falls yet two rope5 ducks fell 25-75% at random times (2-15 s): the robot has to meet the real
coupled dynamics. The partner is an action term with action_dim 0 that runs the frozen policy on its
own mirrored obs (ys_rope_duo.py's wiring), so rsl_rl still sees one 14-D agent per env.
The far_handle mocap stays in the scene, unconnected: rope_turn's axis and rope_blew_up use its
nominal centre, which is where the partner's stem end sits.
"""

import copy
from dataclasses import dataclass, replace

import mujoco
import torch
from tensordict import TensorDict
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, RewardTermCfg, SceneEntityCfg
import mjlab_microduck.tasks.ys_head_circle_env_cfg as hc
import mjlab_microduck.tasks.ys_rope as R

from mjlab_microduck.tasks import MicroduckOnPolicyRunner  # defined before this module is imported
from mjlab_microduck.tasks.symmetry import _get_tensors
from mjlab_microduck.tasks.ys_rope_env_cfg import SPAN, STEM_END_NOM, TRIO_HZ, TRIO_R, make_ys_rope_bob_env_cfg, ramp_weight, set_span

PARTNER_CKPT = "logs/rsl_rl/ys_rope/2026-11-03_17-36-17_rope5_stay/model_18794.pt"
CMD = slice(51, 55)  # head-circle command slots: the partner gets the robot's (forward) command


def connect_duo(spec: mujoco.MjSpec) -> None:
  for a, b in (("rope/rope_start", "partner/stem_end"), ("rope/rope_end", "robot/stem_end")):
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE, name1=a, name2=b)
    eq.solref = [0.05, 1.0]  # as connect_rope


def load_frozen_actor(path: str, device) -> torch.nn.Module:
  """rsl_rl actor (61-512-256-128-14 ELU + obs normalizer) as a plain deterministic module."""
  sd = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
  dims = [sd[f"mlp.{i}.weight"].shape[1] for i in (0, 2, 4, 6)] + [sd["mlp.6.weight"].shape[0]]
  layers = []
  for i in range(4):
    layers += [torch.nn.Linear(dims[i], dims[i + 1])] + ([torch.nn.ELU()] if i < 3 else [])
  mlp = torch.nn.Sequential(*layers).to(device)
  mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
  mean, std = sd["obs_normalizer._mean"], sd["obs_normalizer._std"]

  class Actor(torch.nn.Module):
    def forward(self, x):
      return mlp((x - mean) / (std + 1e-2))  # EmpiricalNormalization, eps 1e-2

  return Actor().eval().requires_grad_(False)


class PartnerAction(JointPositionAction):
  """Takes no policy action: drives the partner with the frozen policy on its mirrored obs."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)  # buffers sized with the real 14 joints
    self._actor = load_frozen_actor(cfg.checkpoint, self.device)
    self._live = None  # self-play (rope11+): YsSelfPlayRunner points this at the actor being trained
    self._built = True

  @property
  def action_dim(self) -> int:  # 0 to the action manager: rsl_rl's agent stays 14-D
    return 0 if getattr(self, "_built", False) else self._action_dim

  def process_actions(self, actions: torch.Tensor):
    obs = self._env.obs_buf
    if "partner" not in obs:  # before the first observation compute
      return super().process_actions(torch.zeros_like(self._raw_actions))
    perm, sign, aperm, asign = _get_tensors(self.device)
    x = obs["partner"][:, perm] * sign
    x[:, CMD] = obs["actor"][:, CMD]
    with torch.no_grad():
      a = self._actor(x) if self._live is None else self._live(TensorDict({"actor": x}, batch_size=[x.shape[0]]))
      if self._live is not None and self.cfg.stochastic:  # explore like the learner (rope12+), same std
        a = a + self._live.distribution.std_param.abs() * torch.randn_like(a)
    super().process_actions(a[:, aperm] * asign)


@dataclass(kw_only=True)
class PartnerActionCfg(JointPositionActionCfg):
  checkpoint: str = PARTNER_CKPT
  stochastic: bool = False  # self-play: sample the partner's action with the learner's std

  def build(self, env):
    return PartnerAction(self, env)


def axis_low(env, z_target: float, std: float = 0.04):  # 0.02 paid 0.003 at the 5 cm start: no gradient
  """rope16 (7.2): 1 while the handle-circle centre (feet-frame z of the mouth tip's running mean) is at
  or below z_target, Gaussian above it. rope14 turns about 214 mm (stem ends 232 mm): the loop's
  bottom is a sharp V; a lower axis lays more rope on the floor -> a blunt, slower bottom (user)."""
  z = env.command_manager.get_term("head_pose").center[:, 1]
  return torch.exp(-((z - z_target).clamp(min=0) / std).square())


# LF2 (7-38): rope14@19200's stem-to-stem distance (m) at 20 g on the trio's circle, as a function of the robot's own stem angle th
# (0 = bottom, as the command): c0 + sum_k (a_k cos k th + b_k sin k th). Fitted in this env itself (play cfg, rope14 as both ducks,
# 256 envs x 20 s, trio's-circle envs only, residual sd 28.8 mm): ys_rope_duo.py's eval scene reads 3 mm wider overall, 9 mm at the top.
SEP_REF = (0.7750, ((-0.0702, -0.0213), (-0.0072, -0.0081), (-0.0030, 0.0007), (-0.0014, 0.0014)))


def handle_sep(env, std: float = 0.025):
  """LF2 (7-38): 1 when the two stem ends are as far apart as rope14's at 20 g at this point of the robot's own circle (Gaussian,
  std m), on the trio's circle only (0 elsewhere). At the top the rope is taut, and its pull, proportional to its mass, sets this
  distance: rope14 sits 28 mm wider at 15 g and 19 mm narrower at 25 g there (this env; 20 g +5 mm), and the rope's timing shifts by
  +21 / -14 deg (7-37). rope14 scores 0.56 / 0.73 / 0.69 at 15 / 20 / 25 g."""
  cmd, th, on = _trio(env)
  ref = torch.full_like(th, SEP_REF[0])
  for k, (a, b) in enumerate(SEP_REF[1], start=1):
    ref += a * torch.cos(k * th) + b * torch.sin(k * th)
  partner = env.scene["partner"]
  sid = env.__dict__.setdefault("_ys_partner_stem", partner.find_sites("stem_end")[0][0])
  sep = (cmd._robot.data.site_pos_w[:, cmd._sid] - partner.data.site_pos_w[:, sid]).norm(dim=-1)
  return torch.where(on, torch.exp(-((sep - ref) / std).square()), 0.0)


def _trio(env):
  """the robot's own stem angle th (rad, 0 = bottom, as the command) and the mask of envs on the trio's circle."""
  cmd = env.command_manager.get_term("head_pose")
  p = hc.tip_feet(cmd._robot, cmd._sid, cmd._feet)[0]
  th = torch.atan2(p[:, 1] - cmd.center[:, 0], -(p[:, 2] - cmd.center[:, 1]))
  on = ((cmd.amp * hc.TIP_M_PER_RAD - TRIO_R).abs() < 1e-4) & ((cmd.freq - TRIO_HZ).abs() < 1e-4)
  return cmd, th, on


ROPE_MID = R.ROPE_LINKS // 2  # link 6 of 12: the rope's middle as s35 duo_x / xana read it
LAG_REF = -0.302  # LF4 (7-40): rope14@19200 (lf3_init) at 20 g in this env: rope_timing's running mean -17.3 deg (ys_lf2_check, 1024 envs
# x 18 s, 326k paying env-steps; the per-step circular mean -18.8 deg; s35 duo_x's eval scene reads -12.5 deg)
LAG_TAU, LAG_WAIT = 1.6, 2.0  # s: rope_timing's running mean (two turns at 1.25 Hz), and how long on the trio's circle before it pays


def rope_lag(env):
  """(N,) rad, (N,) mask: the rope middle's phase about the line between the two stem ends minus the robot's own stem phase, per step
  (s35 xana's lag, which takes the mean over 10 s); the trio's-circle mask."""
  cmd, th, on = _trio(env)
  partner = env.scene["partner"]
  sid = env.__dict__.setdefault("_ys_partner_stem", partner.find_sites("stem_end")[0][0])
  m = env.scene["rope"].data.body_link_pos_w[:, ROPE_MID] - 0.5 * (cmd._robot.data.site_pos_w[:, cmd._sid] + partner.data.site_pos_w[:, sid])
  d = torch.atan2(m[:, 1], -m[:, 2]) - th
  return torch.atan2(d.sin(), d.cos()), on


def rope_timing(env, std: float = 0.26):
  """LF4 (7-40): 1 when the rope's timing is rope14's at 20 g (Gaussian on the angle, std rad ~15 deg), on the trio's circle only.
  The per-step rope_lag swings through the turn (rope14: concentration 0.3-0.5 over a game), so the term pays on its running mean
  (unit vectors, time constant LAG_TAU s), from LAG_WAIT s on the trio's circle. Stateful: call once per step (the reward manager does,
  once the weight is > 0). LF2 / LF3 paid the stem-end distance (handle_sep) and the rope came 7-25 deg late at every mass."""
  d, on = rope_lag(env)
  S = env.__dict__.get("_ys_lag")
  if S is None:
    S = env._ys_lag = {"z": torch.zeros(env.num_envs, 2, device=env.device), "t": torch.zeros(env.num_envs, device=env.device)}
  fresh = ~on | (env.episode_length_buf <= 1)
  a = env.step_dt / LAG_TAU
  S["z"] = torch.where(fresh[:, None], 0.0, (1 - a) * S["z"] + a * torch.stack([d.cos(), d.sin()], 1))
  S["t"] = torch.where(fresh, 0.0, S["t"] + env.step_dt)
  e = torch.atan2(S["z"][:, 1], S["z"][:, 0]) - LAG_REF
  return torch.where(S["t"] >= LAG_WAIT, torch.exp(-(torch.atan2(e.sin(), e.cos()) / std).square()), 0.0)


def make_ys_rope_duo_env_cfg(play: bool = False, stochastic: bool = False, bob: float = 1.0, span: float = SPAN, axis_z: float | None = None,
                             sep_w: float = 0.0, trio_frac: float = 0.0, mass_in: bool = False, time_w: float = 0.0):
  """rope10+: resume from rope5, rope5's rewards (stay_put 2.0), real partner duck. LF2 (7-38): sep_w > 0 adds handle_sep, ramped in
  from 0, and trio_frac of the commands become the trio's circle (where handle_sep pays). LF3 (3-a, 7-39): mass_in puts the rope's mass
  in every group's twist vx slot (ys_rope.add_mass_input), the partner's included. LF4 (7-40): time_w > 0 adds rope_timing the same way."""
  cfg = make_ys_rope_bob_env_cfg(play, stay=True)
  cfg.curriculum.pop("stay_put_ramp")
  cfg.rewards["stay_put"].weight = 2.0
  cfg.rewards["trunk_bob"].weight = bob  # rope13: 2.0 to win back the vertical swing (apex 431 -> 372 mm in rope11)
  ents = cfg.scene.entities
  robot = ents["robot"]
  partner_x = 2 * STEM_END_NOM[0] + span  # partner root: its stem end lands on the far handle spot
  partner = replace(robot, init_state=replace(robot.init_state, pos=(partner_x, 0.0, robot.init_state.pos[2]), rot=(0.0, 0.0, 0.0, 1.0)))
  # partner before rope: mjlab 1.3.0 slices default_joint_pos from the scene keyframe in entity order
  cfg.scene.entities = {"robot": robot, "partner": partner, "rope": ents["rope"], "far_handle": ents["far_handle"]}
  cfg.scene.spec_fn = connect_duo
  set_span(cfg, span)  # rope14+: shorter span, same rope
  cfg.sim.nconmax = 96

  for k, e in list(cfg.events.items()):  # partner copies of every robot event (reset pose, joints, DR)
    a = (e.params or {}).get("asset_cfg")
    if (a is not None and a.name == "robot") or k == "reset_base":
      e2 = copy.deepcopy(e)
      e2.params = dict(e2.params or {})
      e2.params["asset_cfg"] = replace(a, name="partner") if a is not None else SceneEntityCfg("partner")
      cfg.events[f"{k}_partner"] = e2

  act = cfg.actions["joint_pos"]
  cfg.actions["partner_joint_pos"] = PartnerActionCfg(**{k: getattr(act, k) for k in act.__dataclass_fields__ if k != "entity_name"}, entity_name="partner", stochastic=stochastic)
  if mass_in:  # before the partner group is copied from the actor's
    R.add_mass_input(cfg)
  actor = cfg.observations["actor"]
  part = copy.deepcopy(actor)
  for t in part.terms.values():
    if t.params and "asset_cfg" in t.params:
      t.params = {**t.params, "asset_cfg": replace(t.params["asset_cfg"], name="partner")}
  for t in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):  # default asset is robot: name it
    part.terms[t].params = {**(part.terms[t].params or {}), "asset_cfg": SceneEntityCfg("partner")}
  part.terms["actions"].params = {**(part.terms["actions"].params or {}), "action_name": "partner_joint_pos"}
  cfg.observations["partner"] = part

  if axis_z is not None:
    cfg.rewards["axis_low"] = RewardTermCfg(func=axis_low, weight=2.0, params={"z_target": axis_z})
  if sep_w > 0:  # new term: ramp in from 0, the critic has never seen it (as make_ys_rope_env_cfg's)
    cfg.commands["head_pose"].trio_frac = trio_frac
    cfg.rewards["handle_sep"] = RewardTermCfg(func=handle_sep, weight=0.0)
    cfg.curriculum["handle_sep_ramp"] = CurriculumTermCfg(func=ramp_weight, params={"reward_name": "handle_sep", "stages": [(10, sep_w / 2), (30, sep_w)]})
  if time_w > 0:
    cfg.commands["head_pose"].trio_frac = trio_frac
    cfg.rewards["rope_timing"] = RewardTermCfg(func=rope_timing, weight=0.0)
    cfg.curriculum["rope_timing_ramp"] = CurriculumTermCfg(func=ramp_weight, params={"reward_name": "rope_timing", "stages": [(10, time_w / 2), (30, time_w)]})
  # partner down: end the episode as a timeout (bootstrapped, no fall penalty for the robot)
  fell = copy.deepcopy(cfg.terminations["fell_over"])
  fell.params = {**fell.params, "asset_cfg": SceneEntityCfg("partner")}
  fell.time_out = True
  cfg.terminations["partner_fell"] = fell
  if R.ROPE_DR:  # 7-14 (user 2026-11-09): the rope's mass drawn per env (YS_ROPE_DR="lo,hi" kg)
    cfg.events["rope_mass_dr"] = EventTermCfg(func=R.rope_mass_scale, mode="startup",
                                              params={"asset_cfg": SceneEntityCfg("rope", body_names=("rope_.*",)), "mass_range": R.ROPE_DR})
  return cfg


class YsSelfPlayRunner(MicroduckOnPolicyRunner):
  """rope11+: the partner runs the live actor (deterministic, mirrored). With rope5 frozen as the
  partner, rope10 learned to let the partner turn the rope (two rope10 ducks: turn 0.19)."""

  def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
    super().__init__(env, train_cfg, log_dir, device, **kwargs)
    env.unwrapped.action_manager.get_term("partner_joint_pos")._live = self.alg.actor
    print("[ys self-play] partner -> live actor", type(self.alg.actor).__name__)
