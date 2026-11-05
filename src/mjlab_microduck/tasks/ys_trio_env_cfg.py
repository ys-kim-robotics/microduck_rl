"""Stage 7-3 (ys): train the jumper inside the real three-duck scene.

ys_trio.py (eval only) put the side-jump policies between two rope14 turners: every (spot, clock
offset) still let the rope cross a foot almost every cycle (side6 facing -x: 92% of cycles, 22 mm
over the sole). The rope does not sweep under the feet as the lone-rope analysis assumed: it drags on
the floor beside the feet, then lifts diagonally through the footprint. So the jumper now learns
against the rope itself.

Scene per env, rotated 180 deg from ys_trio's frame so the learner keeps the jump env's conventions
(spawn at the env origin facing +x: centering, stay_put): jumper "robot" at the origin, turner
(rope14, as the duo's robot) at (+mid, Y) facing -x, partner (mirrored) at (-mid, Y) facing +x, the
rope between their stem ends. Y = ys_trio's best jumper spot (-80 mm off the rope line, mirrored by
the rotation). The turners are frozen action terms (action_dim 0, as ys_rope_duo_env_cfg's partner):
rsl_rl still sees one 14-D agent per env. Their clock is the jumper's minus DELTA0, so the rope stays
locked to what the jumper observes.
"""

import copy
import math
from dataclasses import dataclass, replace

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg

import mjlab_microduck.tasks.ys_head_circle_env_cfg as hc
import mjlab_microduck.tasks.ys_jump_env_cfg as J
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
from mjlab_microduck.tasks.symmetry import _get_tensors
from mjlab_microduck.tasks.ys_rope import ROPE_RADIUS
from mjlab_microduck.tasks.ys_rope_duo_env_cfg import load_frozen_actor
from mjlab_microduck.tasks.ys_rope_env_cfg import STEM_END_NOM, make_ys_rope_bob_env_cfg, set_span

TURNER_CKPT = "logs/rsl_rl/ys_rope/2026-11-03_23-03-17_rope14_short/model_19200.pt"
SPAN = 0.60  # rope14
TURNER_AMP = 0.12 / hc.TIP_M_PER_RAD  # the duo evals' 12 cm circle
DELTA0 = math.radians(60)  # jumper clock - turner phase (ys_trio sweep, side6 facing -x)
JUMPER_Y = 0.08  # ys_trio's spot y -80 mm, in the rotated frame the turners sit at -0.08 from the jumper
FOOT_BANDS = ((0.020, 0.064), (-0.065, -0.018))  # (left, right) sole y extents, jumper frame (side.py)
HEAD_TOP = 0.17  # rope above this passes over the head
ROPE_GRACE_STEPS = 100  # 2 s: the rope spins up after a reset


def turner_slots(env) -> torch.Tensor:
  """(N, 13) the command block the turners saw in ys_rope_duo (probe2): twist 0, whole-body circle
  head slots [neck f - 1.25 = 0, pitch amp cos, yaw amp sin, roll 0], body 0."""
  ph = env.command_manager.get_term("head_pose").phase - getattr(env, "_ys_delta", DELTA0)
  out = torch.zeros(ph.shape[0], 13, device=ph.device)
  out[:, 4], out[:, 5] = TURNER_AMP * torch.cos(ph), TURNER_AMP * torch.sin(ph)
  return out


def sample_delta(env, env_ids: torch.Tensor | None, half: float) -> None:
  """trio3: per-episode DELTA0 +- half. trio1 learned DELTA0 exactly (60 deg: 67% clean, 45/75: 27/43%)
  while the real rope phase spreads ~19 deg over envs: train a timing margin."""
  if not hasattr(env, "_ys_delta"):
    env._ys_delta = torch.full((env.num_envs,), DELTA0, device=env.device)
  ids = slice(None) if env_ids is None else env_ids
  env._ys_delta[ids] = DELTA0 + (2 * torch.rand_like(env._ys_delta[ids]) - 1) * half


class TurnerAction(JointPositionAction):
  """No policy action: runs the frozen turner on its own obs group (mirrored for the partner)."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._actor = load_frozen_actor(cfg.checkpoint, self.device)
    self._built = True

  @property
  def action_dim(self) -> int:
    return 0 if getattr(self, "_built", False) else self._action_dim

  def process_actions(self, actions: torch.Tensor):
    obs = self._env.obs_buf
    if self.cfg.obs_group not in obs:  # before the first observation compute
      return super().process_actions(torch.zeros_like(self._raw_actions))
    perm, sign, aperm, asign = _get_tensors(self.device)
    x = obs[self.cfg.obs_group].clone()
    if self.cfg.mirror:
      x = x[:, perm] * sign
    x[:, 48:61] = turner_slots(self._env)  # a mirrored forward circle = the reversed one (ys_rope_duo)
    with torch.no_grad():
      a = self._actor(x)
    super().process_actions(a[:, aperm] * asign if self.cfg.mirror else a)


@dataclass(kw_only=True)
class TurnerActionCfg(JointPositionActionCfg):
  checkpoint: str = TURNER_CKPT
  obs_group: str = "turner"
  mirror: bool = False

  def build(self, env):
    return TurnerAction(self, env)


def connect_trio(spec: mujoco.MjSpec) -> None:
  for a, b in (("rope/rope_start", "partner/stem_end"), ("rope/rope_end", "turner/stem_end")):
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE, name1=a, name2=b)
    eq.solref = [0.05, 1.0]  # as connect_rope


def rope_section(pts: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
  """rope polyline (N, L, 3) where it crosses x = xq (N,) -> (N, 2) y, z (nan where it does not)."""
  x = pts[..., 0] - xq[:, None]
  out = torch.full((pts.shape[0], 2), math.nan, device=pts.device)
  for i in range(pts.shape[1] - 1):
    a, b = x[:, i], x[:, i + 1]
    s = -a / torch.where(b != a, b - a, torch.full_like(a, 1e-9))
    hit = (s >= 0) & (s <= 1) & out[:, 0].isnan()
    out[hit] = (pts[:, i, 1:] + s[:, None] * (pts[:, i + 1, 1:] - pts[:, i, 1:]))[hit]
  return out


def rope_overlap(env: ManagerBasedRlEnv, cap: float = 0.05) -> torch.Tensor:
  """(>0, weight < 0) cm the rope top sits above the sole it passes under, at the jumper's x section
  (ys_trio's hit measure, capped). Only the feet collide with the rope: past a sole it would go
  through the shins, so the geometry, not the contact, is charged."""
  robot, rope = env.scene["robot"], env.scene["rope"]
  sec = rope_section(rope.data.body_link_pos_w, robot.data.root_link_pos_w[:, 0])
  w, x, y, z = robot.data.root_link_quat_w.unbind(-1)
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  yj = yaw.cos() * (sec[:, 0] - robot.data.root_link_pos_w[:, 1])  # rope y in the jumper frame at its x
  cmd = env.command_manager.get_term("head_pose")
  soles = J.foot_clearance(robot, cmd._sole) + J.FOOT_Z0  # (N, 2) lowest sole corner z
  pen = torch.zeros_like(yj)
  for j, (lo, hi) in enumerate(FOOT_BANDS):
    inb = (yj > lo) & (yj < hi) & (sec[:, 1] - ROPE_RADIUS < HEAD_TOP)
    pen = torch.where(inb, torch.maximum(pen, (sec[:, 1] + ROPE_RADIUS - soles[:, j]).clamp(0, cap)), pen)
  return torch.where(env.episode_length_buf > ROPE_GRACE_STEPS, pen.nan_to_num(0.0) / 0.01, 0.0)


def make_ys_trio_jump_env_cfg(play: bool = False, delta_half: float = 0.0):
  cfg = J.make_ys_side_jump_env_cfg(play, half_w=J.SIDE_OVERLAP_HALF_W, both=3.0)  # side6's task
  cfg.commands["head_pose"].zero_command_prob = 0.0  # the rope always turns: always jump
  rc = make_ys_rope_bob_env_cfg(play=True, stay=True, reverse=True)  # turners as in the duo evals (no pushes)
  set_span(rc, SPAN)
  mid = STEM_END_NOM[0] + SPAN / 2
  tr = rc.scene.entities["robot"]  # the T-handle duck
  flip = (0.0, 0.0, 0.0, 1.0)  # yaw pi
  turner = replace(tr, init_state=replace(tr.init_state, pos=(mid, -JUMPER_Y, tr.init_state.pos[2]), rot=flip))
  partner = replace(tr, init_state=replace(tr.init_state, pos=(-mid, -JUMPER_Y, tr.init_state.pos[2]), rot=(1.0, 0.0, 0.0, 0.0)))
  rope = rc.scene.entities["rope"]
  rope = replace(rope, init_state=replace(rope.init_state, pos=(-SPAN / 2, -JUMPER_Y, STEM_END_NOM[2]), rot=flip))
  # ducks before the rope: mjlab 1.3.0 slices default_joint_pos from the scene keyframe in entity order
  cfg.scene.entities = {"robot": cfg.scene.entities["robot"], "turner": turner, "partner": partner, "rope": rope}
  cfg.scene.spec_fn = connect_trio
  cfg.sim.nconmax = 128  # jumper 35 + rope-floor/rope-foot + turners' feet

  for k, e in rc.events.items():  # turner/partner copies of the rope task's robot events, and the rope's
    a = (e.params or {}).get("asset_cfg")
    if k in ("reset_rope", "reset_rope_joints"):
      cfg.events[k] = copy.deepcopy(e)
    elif (a is not None and a.name == "robot") or k == "reset_base":
      for who in ("turner", "partner"):
        e2 = copy.deepcopy(e)
        e2.params = dict(e2.params or {})
        e2.params["asset_cfg"] = replace(a, name=who) if a is not None else SceneEntityCfg(who)
        cfg.events[f"{who}_{k}"] = e2

  act = rc.actions["joint_pos"]
  fields = {k: getattr(act, k) for k in act.__dataclass_fields__ if k != "entity_name"}
  for who, mirror in (("turner", False), ("partner", True)):
    cfg.actions[f"{who}_joint_pos"] = TurnerActionCfg(**fields, entity_name=who, obs_group=who, mirror=mirror)
    g = copy.deepcopy(rc.observations["actor"])
    for t in g.terms.values():
      if t.params and "asset_cfg" in t.params:
        t.params = {**t.params, "asset_cfg": replace(t.params["asset_cfg"], name=who)}
    for t in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
      g.terms[t].params = {**(g.terms[t].params or {}), "asset_cfg": SceneEntityCfg(who)}
    g.terms["actions"].params = {**(g.terms["actions"].params or {}), "action_name": f"{who}_joint_pos"}
    cfg.observations[who] = g  # command slots are overwritten by TurnerAction

  for who in ("turner", "partner"):  # a turner down: end as a timeout (no fall penalty for the jumper)
    fell = copy.deepcopy(cfg.terminations["fell_over"])
    fell.params = {**fell.params, "asset_cfg": SceneEntityCfg(who)}
    fell.time_out = True
    cfg.terminations[f"{who}_fell"] = fell
  cfg.rewards["rope_overlap"] = RewardTermCfg(func=rope_overlap, weight=-3.0)
  if delta_half:
    cfg.events["sample_delta"] = EventTermCfg(func=sample_delta, mode="reset", params={"half": delta_half})
  return cfg


YsTrioJumpRlCfg = replace(MicroduckRlCfg, experiment_name="ys_trio_jump")
