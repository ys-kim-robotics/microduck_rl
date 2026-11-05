"""Stage 7-4 (3, ys): a follower turner that syncs to the leader by watching the leader's ball.

Until now both turners got one simulated clock: on real ducks that is a radio clock sync. Here the duo
scene (rope14's: span 0.60) keeps the frozen rope14 partner as the LEADER on the true clock, and the
learner (robot, from rope14) is the FOLLOWER: its own clock reads off by a per-episode offset, and its
head camera tracks the leader's ball (stem end), hidden behind its own ball / the rope. The rope rewards
are phase-free (circle radius and speed, rope turn): matching the leader is what turns the rope.
In the trio the follower then sits where rope14 is unmirrored (the duck the jumper faces).
"""

import math
from dataclasses import dataclass

import torch
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg

import mjlab_microduck.tasks.ys_trio_env_cfg as T
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab_microduck.tasks.symmetry import _get_tensors
from mjlab_microduck.tasks.ys_rope import ROPE_RADIUS
from mjlab_microduck.tasks.ys_rope_duo_env_cfg import CMD, PartnerAction, PartnerActionCfg, make_ys_rope_duo_env_cfg

LEADER_CKPT = T.TURNER_CKPT  # rope14
OFFSET_HALF = math.radians(60)  # follower clock error per episode, U[-half, half]


class LeaderAction(PartnerAction):
  """rope14 on the TRUE command (PartnerAction copies the learner's head slots: shifted here)."""

  def process_actions(self, actions: torch.Tensor):
    obs = self._env.obs_buf
    if "partner" not in obs:
      return super().process_actions(actions)
    perm, sign, aperm, asign = _get_tensors(self.device)
    x = obs["partner"][:, perm] * sign
    x[:, CMD] = self._env.command_manager.get_command("head_pose")
    with torch.no_grad():
      a = self._actor(x)
    super(PartnerAction, self).process_actions(a[:, aperm] * asign)


@dataclass(kw_only=True)
class LeaderActionCfg(PartnerActionCfg):
  def build(self, env):
    return LeaderAction(self, env)


def sample_offset(env, env_ids, half: float) -> None:
  if not hasattr(env, "_ys_off"):
    env._ys_off = torch.zeros(env.num_envs, device=env.device)
  ids = slice(None) if env_ids is None else env_ids
  env._ys_off[ids] = (2 * torch.rand_like(env._ys_off[ids]) - 1) * half


def shifted_head_command(env, command_name: str = "head_pose") -> torch.Tensor:
  """The follower's clock: [neck, amp cos, amp sin, roll] rotated by its episode offset (slots 1, 2 are
  amp cos / sin of the phase, hc._write)."""
  c = env.command_manager.get_command(command_name).clone()
  off = getattr(env, "_ys_off", None)
  if off is not None:
    co, so = off.cos(), off.sin()
    c[:, 1], c[:, 2] = c[:, 1] * co - c[:, 2] * so, c[:, 2] * co + c[:, 1] * so
  return c


def _leader_ball(env, me: str = "robot", leader: str = "partner") -> torch.Tensor:
  """(N, 3) leader's stem end in the follower's head_camera frame; hidden by the follower's own ball
  (it hangs right below the camera) or a rope segment across the line of sight."""
  rob, pa = env.scene[me], env.scene[leader]
  ids = env.__dict__.setdefault("_ys_fol_ids", {})
  if me not in ids:
    ids[me] = (rob.find_sites("head_camera")[0][0], pa.find_sites("stem_end")[0][0], rob.find_sites("stem_end")[0][0])
  i, j, own = ids[me]
  p0, p1 = rob.data.site_pos_w[:, i], pa.data.site_pos_w[:, j]
  R = matrix_from_quat(rob.data.site_quat_w[:, i])
  d = (R.transpose(1, 2) @ (p1 - p0)[..., None])[..., 0]
  x = d[:, 0].clamp(min=1e-3)
  u, v = -d[:, 1] / x / T.CAM_TAN_H, d[:, 2] / x / T.CAM_TAN_V
  ok = (d[:, 0] > 0.02) & (u.abs() <= 1) & (v.abs() <= 1)
  seg = p1 - p0
  L = env.scene["rope"].data.body_link_pos_w
  w = torch.linspace(0, 1, 5, device=L.device)[:-1]
  pts = torch.cat([(L[:, :-1, None] + w[None, None, :, None] * (L[:, 1:, None] - L[:, :-1, None])).flatten(1, 2),
                   rob.data.site_pos_w[:, own][:, None]], 1) - p0[:, None]
  rad = torch.full(pts.shape[:2], ROPE_RADIUS + 0.5 * T.BALL_R, device=L.device)
  rad[:, -1] = 1.5 * T.BALL_R  # own ball
  t = ((pts * seg[:, None]).sum(-1) / seg.square().sum(-1, keepdim=True)).clamp(0, 1)
  dist = (pts - t[..., None] * seg[:, None]).norm(dim=-1)
  ok &= ~((dist < rad) & (t > 0.02) & (t < 0.9)).any(-1)
  return torch.stack([u.clamp(-1, 1), v.clamp(-1, 1), ok.float()], -1)


def leader_ball_obs(env) -> torch.Tensor:
  """(N, 6) through the jumper's tracker model (30 Hz, lag, noise, drops)."""
  return T._cam_track(env, "_ys_fol_cam", "stem_end", False, true_fn=_leader_ball)


SYNC_STD = math.radians(20)
SYNC_TAU = 0.5  # s: EMA of each stem end = its circle centre


def sync_gap(env, std: float = SYNC_STD) -> torch.Tensor:
  """Follower stem end at the leader's place on the circle (world y-z angle about each one's EMA
  centre; two rope14 ducks on one clock: gap 0.9 deg). Privileged, reward only. rope_turn alone does
  not ask for it: rope14 with its clock 60 deg off still turns the rope (turn 1.00) 58 deg out of step."""
  rob, pa = env.scene["robot"], env.scene["partner"]
  if not hasattr(env, "_ys_sync"):
    env._ys_sync = dict(ids=(rob.find_sites("stem_end")[0][0], pa.find_sites("stem_end")[0][0]), c=None)
  st = env._ys_sync; i, j = st["ids"]
  p = torch.stack([rob.data.site_pos_w[:, i, 1:], pa.data.site_pos_w[:, j, 1:]], 1)  # (N, 2, y z)
  fresh = (env.episode_length_buf <= 1)[:, None, None]
  st["c"] = p.clone() if st["c"] is None else torch.where(fresh, p, st["c"] + (p - st["c"]) * env.step_dt / SYNC_TAU)
  a = torch.atan2(p[..., 1] - st["c"][..., 1], p[..., 0] - st["c"][..., 0])
  gap = torch.remainder(a[:, 0] - a[:, 1] + math.pi, 2 * math.pi) - math.pi
  return torch.where(env.episode_length_buf > 50, torch.exp(-(gap / std).square()), 0.0)


def make_ys_follow_env_cfg(play: bool = False, offset_half: float = OFFSET_HALF, sync_w: float = 2.0):
  cfg = make_ys_rope_duo_env_cfg(play, span=T.SPAN)
  act = cfg.actions["partner_joint_pos"]
  cfg.actions["partner_joint_pos"] = LeaderActionCfg(**{k: getattr(act, k) for k in act.__dataclass_fields__ if k != "checkpoint"}, checkpoint=LEADER_CKPT)
  cfg.observations["actor"].terms["head_command"].func = shifted_head_command
  for grp in ("actor", "critic"):  # appended last: actor 61:67, critic 76:82
    cfg.observations[grp].terms["leader_ball"] = ObservationTermCfg(func=leader_ball_obs)
  cfg.events["sample_offset"] = EventTermCfg(func=sample_offset, mode="reset", params={"half": offset_half})
  cfg.rewards["sync_gap"] = RewardTermCfg(func=sync_gap, weight=sync_w)
  return cfg
