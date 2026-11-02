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
  - head_pose_tracking gets a narrow fine_std so a lag of tens of degrees is no longer cheap.
"""

import math
from dataclasses import dataclass, replace

import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg

CIRCLE_FREQ_HZ = 1.5  # 4a: servo+policy gain 0.7 / lag ~80° here, feedforward fixed it -> learnable
AMP_RANGE = (0.2, 0.7)  # rad on head_yaw/head_pitch; 0.7 ≈ 4 cm mouth_tip swing (4a)
ZERO_PROB = 0.1  # still-head bucket: holding HOME stays trained
TRACK_FINE_STD = 0.1  # rad; a 80° lag at A=0.4 is ~0.5 rad error -> priced, not free
HEAD_PITCH, HEAD_YAW = 1, 2  # head_pose order: neck_pitch, head_pitch, head_yaw, head_roll


class HeadCircleCommand(microduck_mdp.UniformPoseCommand):
  """head_pose on a circle in (head_yaw, head_pitch); amplitude and start phase resampled."""

  cfg: "HeadCircleCommandCfg"

  def __init__(self, cfg: "HeadCircleCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.amp = torch.zeros(self.num_envs, device=self.device)
    self.phase = torch.zeros(self.num_envs, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return
    self.amp[env_ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.amp_range)
    self.amp[env_ids[torch.rand(n, device=self.device) < self.cfg.zero_command_prob]] = 0.0
    self.phase[env_ids] = torch.empty(n, device=self.device).uniform_(0.0, 2 * math.pi)
    self._write()

  def _update_command(self) -> None:
    self.phase += 2 * math.pi * self.cfg.freq_hz * self._env.step_dt
    self._write()

  def _write(self) -> None:
    self._command.zero_()
    self._command[:, HEAD_YAW] = self.amp * torch.sin(self.phase)
    self._command[:, HEAD_PITCH] = self.amp * torch.cos(self.phase)


@dataclass(kw_only=True)
class HeadCircleCommandCfg(microduck_mdp.UniformPoseCommandCfg):
  freq_hz: float = CIRCLE_FREQ_HZ
  amp_range: tuple[float, float] = AMP_RANGE

  def build(self, env: ManagerBasedRlEnv) -> HeadCircleCommand:
    return HeadCircleCommand(self, env)


def make_ys_head_circle_env_cfg(play: bool = False):
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
  )
  cfg.rewards["head_pose_tracking"].params["fine_std"] = TRACK_FINE_STD
  return cfg


YsHeadCircleRlCfg = replace(MicroduckRlCfg, experiment_name="ys_head_circle")
