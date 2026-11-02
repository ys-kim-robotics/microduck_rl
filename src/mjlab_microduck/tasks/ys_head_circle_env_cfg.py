"""Stage 4b (ys): stand still and trace a circle with the head.

Built on the walking recipe (make_microduck_velocity_env_cfg) so the obs/action layout is unchanged
(61D / 14D) and a walking checkpoint warm-starts it: `--init-checkpoint <velocity model.pt>` is a
plain resume, so the step counter continues and the inherited curricula stay at their final stage.

What changes vs the walking task (stage 4a, STAGES-ys.md, is the motivation):
  - twist is always zero (every env stands).
  - head_pose becomes a moving target: head_yaw = A·sin φ, head_pitch = A·cos φ, φ advancing at
    CIRCLE_FREQ_HZ. The command keeps its meaning (head joint deltas from HOME), so the warm-started
    policy tracks it from iteration 0 — with the 4a lag. Because φ advances at ONE fixed rate and
    direction, the current command determines the future one: the policy can learn to lead it.
  - head_pose_tracking gets a fine_std so a lag of tens of degrees is no longer cheap
    (4b-1: 0.1 left the pitch error in the flat tail -> 0.3, 4b-2).
  - tip_space=True (4b-3): the head command is only a phase/amplitude ENCODING; the reward is the
    mouth_tip position on a round circle in the trunk y-z plane. A circle in joint space is an
    ellipse at the tip (forward kinematics, round ~0.7 in 4a/4b-2) -> score the tip itself.
  - tip_load=True (4b-4): a rope-tension stand-in force at mouth_tip, resampled with the command.
    4b-3 survived radial/down 0.3 N but a 0.3 N pull toward the partner (+x) made it step (sway
    63 mm), 1 N felled all. Applied from the command term's per-step update, NOT a weight-0 reward
    hook: mjlab's RewardManager skips weight-0 terms, so such a hook never runs.
"""

import math
from dataclasses import dataclass, replace

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg

CIRCLE_FREQ_HZ = 1.5  # 4a: servo+policy gain 0.7 / lag ~80° here, feedforward fixed it -> learnable
AMP_RANGE = (0.2, 0.7)  # rad on head_yaw/head_pitch; 0.7 ≈ 4 cm mouth_tip swing (4a)
ZERO_PROB = 0.1  # still-head bucket: holding HOME stays trained
TRACK_FINE_STD = 0.3  # rad; 0.1 (4b-1) had no gradient at the 0.2-0.34 rad pitch error
# 4b-3 tip-space target, trunk frame. Measured with full2500 standing (scratch tipcal, 2026-11-02):
# HOME tip at (83.8, -0.1, 88.8) mm; yaw=A·sin -> y ≈ +54 mm/rad·A·sin, pitch=A·cos -> z ≈ -44 mm/rad·A·cos.
TIP_CENTER_B = (0.0838, 0.0, 0.0888)
TIP_M_PER_RAD = 0.05  # target radius R = 0.05·A -> 10-35 mm
TIP_STD, TIP_FINE_STD = 0.02, 0.005  # m
HEAD_PITCH, HEAD_YAW = 1, 2  # head_pose order: neck_pitch, head_pitch, head_yaw, head_roll


TIP_FX_RANGE = (0.0, 1.0)  # N toward the partner (trunk +x)
TIP_FR_RANGE = (0.0, 0.3)  # N outward along the tip radius (y-z)
TIP_FZ_RANGE = (-0.3, 0.0)  # N world vertical (rope weight)
TIP_LOAD_ZERO_PROB = 0.2  # unloaded bucket


def apply_tip_force(robot, bid, sid, fx, fr, fz):
  """Rope-tension stand-in at mouth_tip (N; floats or (n,) tensors): fx toward the partner (trunk +x),
  fr outward along the tip's radius in the trunk y-z plane, fz world vertical (negative = down).
  Applied as a wrench on the tip's body (force + lever torque) so the neck feels the right moment."""
  d = robot.data
  tb = quat_apply_inverse(d.root_link_quat_w, d.site_pos_w[:, sid] - d.root_link_pos_w)
  rad = torch.zeros_like(tb)
  rad[:, 1] = tb[:, 1] - TIP_CENTER_B[1]
  rad[:, 2] = tb[:, 2] - TIP_CENTER_B[2]
  fb = torch.as_tensor(fr, device=tb.device).reshape(-1, 1) * rad / rad.norm(dim=-1, keepdim=True).clamp_min(1e-3)
  fb[:, 0] += torch.as_tensor(fx, device=tb.device)
  f = quat_apply(d.root_link_quat_w, fb)
  f[:, 2] += torch.as_tensor(fz, device=tb.device)
  tau = torch.cross(d.site_pos_w[:, sid] - d.body_com_pos_w[:, bid], f, dim=-1)
  robot.write_external_wrench_to_sim(forces=f.unsqueeze(1), torques=tau.unsqueeze(1), body_ids=[bid])


class HeadCircleCommand(microduck_mdp.UniformPoseCommand):
  """head_pose on a circle in (head_yaw, head_pitch); amplitude and start phase resampled."""

  cfg: "HeadCircleCommandCfg"

  def __init__(self, cfg: "HeadCircleCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.amp = torch.zeros(self.num_envs, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)
    self.tip_force = torch.zeros(self.num_envs, 3, device=self.device)  # fx, fr, fz
    if cfg.tip_load:
      robot = env.scene["robot"]
      self._robot, self._sid, self._bid = robot, robot.find_sites("mouth_tip")[0][0], robot.find_bodies("jaw_soft")[0][0]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return
    self.amp[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.amp_range)
    self.amp[env_ids[torch.rand(n, device=self.device) < self.cfg.zero_command_prob]] = 0.0
    self.phase[env_ids] = torch.empty(n, device=self.device).uniform_(0.0, 2 * math.pi)
    if self.cfg.tip_load:
      for i, rng in enumerate((TIP_FX_RANGE, TIP_FR_RANGE, TIP_FZ_RANGE)):
        self.tip_force[env_ids, i] = torch.empty(n, device=self.device).uniform_(*rng)
      self.tip_force[env_ids[torch.rand(n, device=self.device) < TIP_LOAD_ZERO_PROB]] = 0.0
    self._write()

  def _update_command(self) -> None:
    self.phase += 2 * math.pi * self.cfg.freq_hz * self._env.step_dt
    self._write()
    if self.cfg.tip_load:  # every step: the radial direction follows the tip
      apply_tip_force(self._robot, self._bid, self._sid, *self.tip_force.unbind(1))

  def _write(self) -> None:
    self._command.zero_()
    self._command[:, HEAD_YAW] = self.amp * torch.sin(self.phase)
    self._command[:, HEAD_PITCH] = self.amp * torch.cos(self.phase)


@dataclass(kw_only=True)
class HeadCircleCommandCfg(microduck_mdp.UniformPoseCommandCfg):
  freq_hz: float = CIRCLE_FREQ_HZ
  amp_range: tuple[float, float] = AMP_RANGE
  tip_load: bool = False

  def build(self, env: ManagerBasedRlEnv) -> HeadCircleCommand:
    return HeadCircleCommand(self, env)


def mouth_circle_tracking(env: ManagerBasedRlEnv, command_name: str, std: float, fine_std: float) -> torch.Tensor:
  """exp-Gaussian (wide + fine, 50/50) on the mouth_tip y-z error to the circle target, trunk frame.
  x is free: pitching the head unavoidably moves the tip fore-aft a few mm, and the rope only sees y-z."""
  cmd = env.command_manager.get_term(command_name)
  d = env.scene["robot"].data
  sid = env.scene["robot"].find_sites("mouth_tip")[0][0]
  tip = quat_apply_inverse(d.root_link_quat_w, d.site_pos_w[:, sid] - d.root_link_pos_w)
  r = TIP_M_PER_RAD * cmd.amp
  ty = TIP_CENTER_B[1] + r * torch.sin(cmd.phase)
  tz = TIP_CENTER_B[2] - r * torch.cos(cmd.phase)
  err2 = (tip[:, 1] - ty) ** 2 + (tip[:, 2] - tz) ** 2
  return 0.5 * torch.exp(-err2 / std**2) + 0.5 * torch.exp(-err2 / fine_std**2)


def make_ys_head_circle_env_cfg(play: bool = False, tip_space: bool = False, tip_load: bool = False):
  cfg = make_microduck_velocity_env_cfg(play=play)

  twist = cfg.commands["twist"]
  twist.rel_standing_envs = 1.0
  twist.rel_turn_in_place_envs = 0.0
  cfg.curriculum.pop("standing_envs")  # would lower rel_standing_envs back to 0.25
  cfg.curriculum.pop("head_pose_range")  # ranges are the circle's now

  cfg.commands["head_pose"] = HeadCircleCommandCfg(
    resampling_time_range=(4.0, 8.0),
    ranges=((0.0, 0.0),) * 4,  # unused; keeps the 4D layout
    zero_command_prob=ZERO_PROB,
    tip_load=tip_load,
  )
  cfg.rewards["head_pose_tracking"].params["fine_std"] = TRACK_FINE_STD
  if tip_space:  # joint targets no longer the goal: free the head joints, score the tip
    cfg.rewards["head_pose_tracking"].weight = 0.0
    cfg.curriculum.pop("head_pose_bias_weight")
    cfg.rewards["head_pose_bias"].weight = 0.0
    cfg.rewards["mouth_circle"] = RewardTermCfg(
      func=mouth_circle_tracking,
      weight=2.0,  # same budget head_pose_tracking had
      params={"command_name": "head_pose", "std": TIP_STD, "fine_std": TIP_FINE_STD},
    )
  return cfg


YsHeadCircleRlCfg = replace(MicroduckRlCfg, experiment_name="ys_head_circle")
