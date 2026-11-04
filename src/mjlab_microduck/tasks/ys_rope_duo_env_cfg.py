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
from mjlab.managers import RewardTermCfg, SceneEntityCfg

from mjlab_microduck.tasks import MicroduckOnPolicyRunner  # defined before this module is imported
from mjlab_microduck.tasks.symmetry import _get_tensors
from mjlab_microduck.tasks.ys_rope_env_cfg import SPAN, STEM_END_NOM, make_ys_rope_bob_env_cfg, set_span

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


def make_ys_rope_duo_env_cfg(play: bool = False, stochastic: bool = False, bob: float = 1.0, span: float = SPAN, axis_z: float | None = None):
  """rope10+: resume from rope5, rope5's rewards (stay_put 2.0), real partner duck."""
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
  # partner down: end the episode as a timeout (bootstrapped, no fall penalty for the robot)
  fell = copy.deepcopy(cfg.terminations["fell_over"])
  fell.params = {**fell.params, "asset_cfg": SceneEntityCfg("partner")}
  fell.time_out = True
  cfg.terminations["partner_fell"] = fell
  return cfg


class YsSelfPlayRunner(MicroduckOnPolicyRunner):
  """rope11+: the partner runs the live actor (deterministic, mirrored). With rope5 frozen as the
  partner, rope10 learned to let the partner turn the rope (two rope10 ducks: turn 0.19)."""

  def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
    super().__init__(env, train_cfg, log_dir, device, **kwargs)
    env.unwrapped.action_manager.get_term("partner_joint_pos")._live = self.alg.actor
    print("[ys self-play] partner -> live actor", type(self.alg.actor).__name__)
