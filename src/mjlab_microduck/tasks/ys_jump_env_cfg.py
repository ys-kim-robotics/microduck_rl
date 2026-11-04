"""Stage 7-1 (ys): the middle duck of the jump rope, alone: hop in place on a 1.25 Hz clock.

Built on the head-circle recipe (walking obs/action layout, warm start from full2500): the clock is
the head_pose command of HeadCircleCommand (head_yaw = sin φ, head_pitch = cos φ, amp 1, 0 = stand
still), so the 61-D layout is unchanged and the turners' rope phase can drive it later (7-3).

Reward design borrowed from (STAGES-ys.md 7-1):
  - ros-claw Circus ropehop: a phase-shaped FOOT height reference z*(φ) (parabolic bump, 0 outside the
    window) tracked every step -> pays the hop in the window AND charges feet up outside it, which
    rate-gates jitter (their ungated rhythm term was farmed by 7 Hz shaking).
  - chenp9527 microduck-jump: a linear, saturated height term (Gaussian alone has no gradient from
    flat feet), saturated at the same target as the tracking; target ramped from 1 cm (starting at
    3 cm was ~3x slower to learn there); in-place penalty.
Clearance = min over both feet of the foot site height above its standing height: both soles must be
over the rope (7 rope stage: ~26 mm p95 under the feet, ~60 ms per pass).
"""

import math
from dataclasses import dataclass, replace

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg, TerminationTermCfg

import mjlab_microduck.tasks.ys_head_circle_env_cfg as hc
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
from mjlab_microduck.tasks.ys_rope_env_cfg import stay_put

JUMP_FREQ_HZ = 1.25  # the rope's (stage 5-6)
JUMP_PHASE = math.pi  # apex of the foot reference, in command phase
JUMP_HALF_W = 0.8  # rad of phase on each side of the apex = 100 ms at 1.25 Hz (feet must be up ~60 ms)
H_START, H_MAX, H_STEP = 0.01, 0.05, 0.005  # m sole-clearance target: 1 -> 5 cm (jump12: 4 -> 5; ~35 mm apex keeps 30 mm through ±30 ms)
H_GROW = 0.8  # grow when the mean per-hop peak clearance reaches 80% of the target
H_CHECK_STEPS = 500  # env steps between checks (~20 iterations at 24 steps/iter)
CENTER_GAIN = 2.0  # 1/s: twist command = gain * offset to the spawn spot (body frame)
FOOT_Z0 = -0.002  # lowest sole-box corner when standing (m; the true mesh sits ~1.6 mm higher, rounded edges)


# foot collision mesh bounding box in its geom frame (m), both feet (mirrored); see sole_corners
_SOLE_LO, _SOLE_HI = (-0.00636, -0.02175, -0.02689), (0.00679, 0.02175, 0.02723)
_CORNERS = torch.tensor([[x, y, z] for x in (_SOLE_LO[0], _SOLE_HI[0]) for y in (_SOLE_LO[1], _SOLE_HI[1]) for z in (_SOLE_LO[2], _SOLE_HI[2])])


def jump_clearance(robot, feet) -> torch.Tensor:
  """lowest corner of BOTH foot collision boxes above the ground (m): >0 only when every part of both
  soles is up, toes included. jump2-5 used the foot SITE: the user saw toes still on the ground at
  "31 mm" (a tilted foot lifts its site while the tip stays down). feet = (left, right) geom ids."""
  q = robot.data.geom_quat_w[:, feet]  # (N, 2, 4) wxyz
  c = _CORNERS.to(q.device)  # (8, 3)
  w, v = q[..., :1, None], q[..., None, 1:]  # rotate the corners: c + 2w(v x c) + 2 v x (v x c)
  cc = c.expand(*q.shape[:2], 8, 3)
  t = 2 * torch.cross(v.expand_as(cc), cc, dim=-1)
  z = (cc + w * t + torch.cross(v.expand_as(cc), t, dim=-1))[..., 2] + robot.data.geom_pos_w[:, feet, 2, None]
  return z.amin((1, 2)) - FOOT_Z0


class JumpClockCommand(hc.HeadCircleCommand):
  """The circle command as a clock; adds the foot reference and its height curriculum."""

  cfg: "JumpClockCommandCfg"

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.h = cfg.h_start
    self._peak = torch.zeros(self.num_envs, device=self.device)  # per-hop max clearance in the window
    self._peaks, self._steps_h = [], 0
    self.metrics["h_target_cm"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["peak_clear_cm"] = torch.zeros(self.num_envs, device=self.device)
    self._last_peak = 0.0
    self._sole = self._robot.find_geoms(["left_foot_collision", "right_foot_collision"], preserve_order=True)[0]

  def window(self) -> torch.Tensor:
    """signed phase distance to the apex (rad, in [-π, π])."""
    return torch.remainder(self.phase - JUMP_PHASE + math.pi, 2 * math.pi) - math.pi

  def reference(self) -> torch.Tensor:
    d = self.window()
    return self.h * (1 - (d / JUMP_HALF_W).square()).clamp(min=0) * (self.amp > 0)

  def _center(self) -> None:
    """jump4: the twist command becomes 'walk back to the spot' (Circus centering v1). jump3 drifted
    1.9 m: proprio obs has no position, so stay_put was unobservable and its 2 cm Gaussian flat."""
    if CENTER_GAIN == 0:
      return
    w, x, y, z = self._robot.data.root_link_quat_w.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    dx, dy = (self._env.scene.env_origins[:, :2] - self._robot.data.root_link_pos_w[:, :2]).unbind(-1)
    c, s = yaw.cos(), yaw.sin()
    v = self._env.command_manager.get_term("twist").vel_command_b
    v[:, 0] = (CENTER_GAIN * (c * dx + s * dy)).clamp(-0.3, 0.3)
    v[:, 1] = (CENTER_GAIN * (-s * dx + c * dy)).clamp(-0.3, 0.3)
    v[:, 2] = (-CENTER_GAIN * yaw).clamp(-0.5, 0.5)

  def _update_metrics(self) -> None:
    c = jump_clearance(self._robot, self._sole)
    d = self.window()
    inw = (d.abs() < JUMP_HALF_W) & (self.amp > 0)
    self._peak = torch.where(inw, torch.maximum(self._peak, c), self._peak)
    done = (self.amp > 0) & (d > JUMP_HALF_W) & (d - 2 * math.pi * self.freq * self._env.step_dt <= JUMP_HALF_W)
    if done.any():  # the window just closed: one hop scored
      self._peaks.append(self._peak[done].mean().item())
      self._peak[done] = 0.0
    self._steps_h += 1
    if self._steps_h % H_CHECK_STEPS == 0 and self._peaks:
      self._last_peak = sum(self._peaks) / len(self._peaks)
      if self._last_peak >= H_GROW * self.h and not self.cfg.freeze_radius:
        self.h = min(self.h + H_STEP, H_MAX)
      self._peaks = []
    self._center()
    self.metrics["h_target_cm"][:] = 100 * self.h
    self.metrics["peak_clear_cm"][:] = 100 * self._last_peak


@dataclass(kw_only=True)
class JumpClockCommandCfg(hc.HeadCircleCommandCfg):
  h_start: float = H_START

  def build(self, env):
    return JumpClockCommand(self, env)


def jump_track(env: ManagerBasedRlEnv, std: float = 0.01) -> torch.Tensor:
  """Gaussian on clearance - z*(φ), every step (z* = 0 outside the window and for still envs)."""
  cmd = env.command_manager.get_term("head_pose")
  c = jump_clearance(cmd._robot, cmd._sole).clamp(min=0)
  return torch.exp(-((c - cmd.reference()) / std).square())


def jump_lift(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Linear, saturated at z*: the gradient from flat feet that the Gaussian lacks."""
  cmd = env.command_manager.get_term("head_pose")
  ref = cmd.reference()
  c = jump_clearance(cmd._robot, cmd._sole)
  return torch.where(ref > 0, torch.minimum(c.clamp(min=0), ref) / cmd.h, 0.0)


PUSH_LEAD = 0.5  # rad of phase before the window opens (~64 ms): the take-off


def jump_push(env: ManagerBasedRlEnv) -> torch.Tensor:
  """jump2: upward trunk velocity from take-off to the apex, capped at the ballistic v for h.
  jump1 stood still (jump_lift 0.01): flat feet earn ~90% of jump_track, and lift pays only once the
  feet are already up. vz pays for the push that gets them there (chenp's hop_pushoff)."""
  cmd = env.command_manager.get_term("head_pose")
  d = cmd.window()
  on = (d > -JUMP_HALF_W - PUSH_LEAD) & (d < 0) & (cmd.amp > 0)
  v_cap = math.sqrt(2 * 9.81 * cmd.h)
  vz = cmd._robot.data.root_link_lin_vel_w[:, 2]
  return torch.where(on, vz.clamp(0, v_cap) / v_cap, 0.0)


def foot_level(env: ManagerBasedRlEnv, std: float = 0.02) -> torch.Tensor:
  """jump6: soles level inside the window. jump5 'hopped' 30 mm of trunk on tiptoe: ankles extended,
  toe tips never left the ground (sole clearance -2 mm). Level soles turn trunk rise into clearance.
  Sole normal = geom x axis (the box is 13 mm thick in x); 1 - |n_z| ~ θ²/2, std 0.02 ~ 0.2 rad."""
  cmd = env.command_manager.get_term("head_pose")
  w, x, y, z = cmd._robot.data.geom_quat_w[:, cmd._sole].unbind(-1)
  nz = 2 * (x * z - w * y)  # world z of the geom x axis
  r = torch.exp(-(1 - nz.abs()) / std).mean(-1)
  return torch.where((cmd.window().abs() < JUMP_HALF_W) & (cmd.amp > 0), r, 0.0)


ROPE_CORE = 2 * math.pi * JUMP_FREQ_HZ * 0.03  # rad: ±30 ms, the rope's time under the feet
ROPE_GRACE_STEPS = 50  # 1 s after reset before the rope counts


def rope_hit(env: ManagerBasedRlEnv) -> torch.Tensor:
  """jump9: a virtual rope. In the ±30 ms core the lowest sole corner must clear h - 1 cm (the
  curriculum: 0 = just airborne at h 1 cm, 3 cm at h 4 cm), or the episode ends as a fall. jump2-8:
  standing or a tiptoe rise always out-scored a real hop under shaped rewards alone."""
  cmd = env.command_manager.get_term("head_pose")
  c = jump_clearance(cmd._robot, cmd._sole)
  core = (cmd.window().abs() < ROPE_CORE) & (cmd.amp > 0) & (env.episode_length_buf > ROPE_GRACE_STEPS)
  return core & (c < max(cmd.h - 0.01, 0.0))


AIR_HALF_W = math.pi * 0.18  # rad: air phase = 18% of the cycle (144 ms; a 25-30 mm hop flies ~140 ms)


def _air_phase(env):
  cmd = env.command_manager.get_term("head_pose")
  c = ((cmd.window().abs() < AIR_HALF_W) & (cmd.amp > 0)).float()
  s = env.scene["feet_ground_contact"].data
  return c, (s.found == 0).float(), s.force.norm(dim=-1)  # (N,), (N, 2) per-foot airborne, (N, 2) N


def contact_clock(env: ManagerBasedRlEnv) -> torch.Tensor:
  """jump10: Marope's 'align rhythm' (arXiv 2606.08064): per foot +1 if its contact state matches the
  clock, -1 if not, scaled 0.8 in the air phase and 0.2 on the ground. Standing nets ~0, a hop +.
  jump2-9 all paid standing (Gaussian 92%, tiptoe trunk z); here a foot down in the air phase costs."""
  c, air, _ = _air_phase(env)
  match = (air == c[:, None]).float()
  return ((2 * match - 1) * (0.8 * c + 0.2 * (1 - c))[:, None]).mean(-1)


def both_air(env: ManagerBasedRlEnv) -> torch.Tensor:
  """both feet off in the air phase: pays the hop, not a one-foot step."""
  c, air, _ = _air_phase(env)
  return c * air.prod(-1)


def air_force(env: ManagerBasedRlEnv, sigma: float = 13.0) -> torch.Tensor:
  """Siekmann/WTW swing force term (>0, weight < 0): foot force in the air phase, saturating at ~mg/2."""
  c, _, f = _air_phase(env)
  return c * (1 - torch.exp(-f.square() / sigma)).mean(-1)


def landing_impact(env: ManagerBasedRlEnv) -> torch.Tensor:
  """jump12: Marope's contact-force term: total foot force above 2.5 body weights, in body weights.
  jump11 landed at 59 N p95 (8x weight), too hard for xl330 gearboxes."""
  mg = 0.742 * 9.81
  return (env.scene["feet_ground_contact"].data.force.norm(dim=-1).sum(-1) / mg - 2.5).clamp(min=0)


TRUNK_Z0 = 0.116  # standing root height (m), measured with jump2 (sd 0.3 mm)
TRUNK_H, TRUNK_HALF_W, TRUNK_STD = 0.05, 0.65 * math.pi, 0.025  # Circus ropehop's body-z command


def trunk_track(env: ManagerBasedRlEnv) -> torch.Tensor:
  """jump3: Gaussian on root z - (Z0 + bump(φ)), the Circus recipe. Unlike foot clearance it pays
  from the ground up: crouch-and-extend already scores, so the gradient never sits at a flat zero."""
  cmd = env.command_manager.get_term("head_pose")
  bump = TRUNK_H * (1 - (cmd.window() / TRUNK_HALF_W).square()).clamp(min=0) * (cmd.amp > 0)
  z = cmd._robot.data.root_link_pos_w[:, 2]
  return torch.exp(-((z - TRUNK_Z0 - bump) / TRUNK_STD).square())


def make_ys_jump_env_cfg(play: bool = False):
  cfg = hc.make_ys_head_circle_env_cfg(play=play)  # standing, twist 0, head circle command
  old = cfg.commands["head_pose"]
  cfg.commands["head_pose"] = JumpClockCommandCfg(
    **{k: getattr(old, k) for k in hc.HeadCircleCommandCfg.__dataclass_fields__},
  )
  jc = cfg.commands["head_pose"]
  jc.freq_hz, jc.amp_range, jc.zero_command_prob = JUMP_FREQ_HZ, (1.0, 1.0), 0.2
  jc.resampling_time_range = (1e3, 1e3)  # the clock never jumps mid-episode (resampling re-draws φ)
  if play:
    jc.h_start, jc.freeze_radius = H_MAX, True

  r = cfg.rewards
  r["head_pose_tracking"].weight = 0.0  # the command is a clock now
  cfg.curriculum.pop("head_pose_bias_weight", None)
  r["head_pose_bias"].weight = 0.0
  # motion blockers for a ~4 cm hop (~0.9 m/s take-off): as the whole-body circle (ys_head_circle)
  r["track_linear_velocity"].params["std"] = 0.5  # jump4: 1.0 -> 0.5, it carries the centering now
  for k in ("air_time", "foot_clearance", "foot_swing_height"):  # walking terms, gated on |twist|:
    r[k].weight = 0.0  # the centering command would switch them on
  pose = r["pose"].params
  pose["std_standing"] = dict(pose["std_walking"])  # legs must crouch and extend
  r["jump_track"] = RewardTermCfg(func=jump_track, weight=2.0)  # jump3: 4 -> 0, standing earned 92% of it; jump7: 2, charges chenp's off-beat hops
  r["trunk_track"] = RewardTermCfg(func=trunk_track, weight=0.0)  # jump5: 20 -> 10; jump8: 0, it pays a tiptoe rise (jump7 lost chenp's hop 17 -> 8 mm)
  r["jump_lift"] = RewardTermCfg(func=jump_lift, weight=12.0)  # jump2: 4 -> 8; jump8: 12
  r["jump_push"] = RewardTermCfg(func=jump_push, weight=4.0)
  # jump10: contact-clock design (Marope / Siekmann / walk-these-ways pronk); jump3-9 shaping off
  r["contact_clock"] = RewardTermCfg(func=contact_clock, weight=8.0)
  r["both_air"] = RewardTermCfg(func=both_air, weight=3.0)
  r["air_force"] = RewardTermCfg(func=air_force, weight=-4.0)
  for k in ("jump_track", "trunk_track", "jump_push"):
    r[k].weight = 0.0
  r["jump_lift"].weight = 8.0  # sole height, saturating at the curriculum target (jump12: 4 -> 8)
  r["landing_impact"] = RewardTermCfg(func=landing_impact, weight=-3.0)  # jump13: -0.5 -> -3 (jump12 still landed 61 N p95)
  # cfg.terminations["rope_hit"] = TerminationTermCfg(func=rope_hit)  # jump9: no hop in 400 it
  # r["foot_level"] = RewardTermCfg(func=foot_level, weight=2.0)  # jump6: 0.01 of 0.5, no gradient
  r["stay_put"] = RewardTermCfg(func=stay_put, weight=3.0, params={"pos_std": 0.05, "yaw_std": 0.3})  # jump4: 2 -> 5 cm std (jump3 drifted 1.9 m); jump5: weight 2 -> 6, yaw 0.1 -> 0.3 rad (jump4 drifted 32 cm); jump10: 1, it out-paid the hop (jump8); jump11: 3 (jump10 drifted 93 cm)
  cfg.events["reset_base"].params["pose_range"] = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.12, 0.13), "yaw": (-0.03, 0.03)}
  return cfg


YsJumpRlCfg = replace(MicroduckRlCfg, experiment_name="ys_jump")
