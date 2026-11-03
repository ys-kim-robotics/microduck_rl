"""Stage 5c (ys): turn a real (simulated) jump rope with a T-handle welded to the beak.

Scene, per env (robot spawns at the env origin facing +x):
  robot ── T-handle (child of jaw_soft: crossbar in the beak, stem STEM_LEN forward)
        └─ stem_end ══CONNECT══ rope (capsule chain, ys_rope.py) ══CONNECT══ far_handle (mocap)
The far handle is the partner stand-in: it copies the stem end's motion about its own running-mean
centre (same circle, same phase, user's design 5a-3), so the rope is turned from both ends.

Why the stem: the beak tip sits ~5.4 cm from the head joints, so the head alone drew <= ~6 cm
circles (5b). A stem of length L moves (5.4 + L) cm per rad: same head motion, bigger handle circle
(5c-0: L = 8 cm turned bc5's 35 mm tip circle into 82 x 67 mm). Cost: rope force x a longer lever on
the neck servos -> neck_torque penalty, so the policy shares the work with the body (user: a person
with weak arms uses the waist).

Policy obs are unchanged (61D; the rope is felt, not seen, like the ball in BallKick). The circle
command is 5b's whole_body one, scored at stem_end instead of mouth_tip; rope_turn rewards the rope's
middle actually going around.
"""

import math
from dataclasses import replace

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from mjlab.envs.mdp import events as mjlab_events

from mjlab_microduck.tasks import ys_head_circle_env_cfg as hc
from mjlab_microduck.tasks.ys_rope import ROPE_LEN, ROPE_LINKS, add_rope

STEM_LEN = 0.10  # m past the beak tip (5c-0: 8 cm -> ~8 cm circles, 12 cm -> ~10 cm)
SPAN = 0.70  # m between the two handle centres (rope 0.8 m: rests ~3 cm above the floor, 5a)
# Nominal stem end in the env frame at HOME standing: root 0.116 m above the feet, mouth_tip at
# (83.8, 0, 88.8) mm in the trunk frame (4b-3), beak pointing +x.
STEM_END_NOM = (0.0838 + STEM_LEN, 0.0, 0.116 + 0.0888)
FAR_NOM = (STEM_END_NOM[0] + SPAN, 0.0, STEM_END_NOM[2])
MIRROR_TAU = 1.0  # s; running-mean centre of the stem end that the far handle copies around
NECK_ACTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def add_t_handle(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Crossbar held in the beak + a stem along the beak (jaw_soft -z), 5 g, no collisions."""
  jaw = spec.body("jaw_soft")
  h = jaw.add_body(name="t_handle", pos=[-0.0081, 0.0, -0.0777])  # = mouth_tip in the jaw frame
  kw = dict(type=mujoco.mjtGeom.mjGEOM_CAPSULE, contype=0, conaffinity=0, rgba=[0.2, 0.4, 0.9, 1])
  h.add_geom(fromto=[0, -0.02, 0, 0, 0.02, 0], size=[0.004, 0, 0], mass=0.002, **kw)
  h.add_geom(fromto=[0, 0, 0, 0, 0, -STEM_LEN], size=[0.003, 0, 0], mass=0.003, **kw)
  h.add_site(name="stem_end", pos=[0, 0, -STEM_LEN], size=[0.005, 0, 0])
  return spec


def rope_spec() -> mujoco.MjSpec:
  """Rope entity: free root at the far end (entity origin), chain toward -x to the duck's stem end."""
  spec = mujoco.MjSpec()
  add_rope(spec, (0, 0, 0), (-SPAN, 0, 0), length=ROPE_LEN, free=True)
  m = spec.compile()
  spec.add_key(name="init", qpos=m.qpos0.tolist())  # the sagging V; entity joint_pos=None reads it
  return spec


def far_handle_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  b = spec.worldbody.add_body(name="far_handle", mocap=True)
  b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], contype=0, conaffinity=0, rgba=[0.9, 0.6, 0.1, 1])
  b.add_site(name="far_site")
  return spec


def connect_rope(spec: mujoco.MjSpec) -> None:
  """Scene spec_fn (runs after the entities are attached): pin both rope ends to their handles."""
  for a, b in (("rope/rope_start", "far_handle/far_site"), ("rope/rope_end", "robot/stem_end")):
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE, name1=a, name2=b)
    # 0.01 (= 2 x dt) blew up against the 1.7 g links once the robot pulled on it (16/16 envs fell,
    # 5c-2); 0.05 held with a ~1 mm mean gap and no falls. The rope needs a little give at its ends.
    eq.solref = [0.05, 1.0]


def reset_rope_joints(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
  """Back to the sagging V. Not rope.data.default_joint_pos: with several entities mjlab 1.3.0 slices
  that from the SCENE keyframe (65 values here vs the rope's 44), so the rope's own qpos0 is used."""
  rope = env.scene["rope"]
  ids = slice(None) if env_ids is None else env_ids
  n = rope.data.joint_pos[ids].shape[0]
  q0 = torch.tensor(_rope_q0(), device=env.device).expand(n, -1)
  rope.write_joint_state_to_sim(q0, torch.zeros_like(rope.data.joint_vel[ids]), env_ids=env_ids)


_Q0 = None


def _rope_q0():
  global _Q0
  if _Q0 is None:
    _Q0 = rope_spec().compile().qpos0[7:].tolist()  # after the free root
  return _Q0


class RopeCircleCommand(hc.HeadCircleCommand):
  """5b's whole-body circle, scored at the T-handle stem end, plus the mirroring far handle."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._sid = self._robot.find_sites("stem_end")[0][0]  # circle score / centre now at the stem end
    self._far = env.scene["far_handle"]
    self._stem_c = torch.zeros(self.num_envs, 3, device=self.device)
    self._far_c = torch.tensor(FAR_NOM, device=self.device) + env.scene.env_origins
    self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

  def reset(self, env_ids):
    self._fresh[env_ids] = True
    return super().reset(env_ids)

  def _update_command(self) -> None:
    super()._update_command()
    stem = self._robot.data.site_pos_w[:, self._sid]
    self._stem_c = torch.where(self._fresh[:, None], stem, self._stem_c + (stem - self._stem_c) * (self._env.step_dt / MIRROR_TAU))
    self._fresh[:] = False
    far = self._far_c.clone()
    far[:, 1:] += stem[:, 1:] - self._stem_c[:, 1:]  # same y-z offset = same circle, same phase
    # direct write: Entity.write_mocap_pose_to_sim breaks on a plain int mocap_id (mjlab 1.3.0, [n,1,3] vs [n,3])
    self._env.sim.data.mocap_pos[:, self._far.indexing.mocap_id] = far


class RopeCircleCommandCfg(hc.HeadCircleCommandCfg):
  def build(self, env):
    return RopeCircleCommand(self, env)


def _axis_frame(env):
  """Rope axis centre (y, z) between the far handle centre and the duck's stem-end centre."""
  cmd = env.command_manager.get_term("head_pose")
  return 0.5 * (cmd._far_c[:, 1:] + cmd._stem_c[:, 1:])


def rope_turn(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Angular progress of the rope's middle link about the rope axis, per step, normalised by the
  commanded rate: ~1 when the rope goes around with the handle, 0 when it just swings (back-and-forth
  cancels), negative when it turns the wrong way. Clamped to [-1, 1.5]."""
  rope = env.scene["rope"]
  cmd = env.command_manager.get_term("head_pose")
  p = rope.data.body_link_pos_w[:, ROPE_LINKS // 2, 1:] - _axis_frame(env)
  v = rope.data.body_link_lin_vel_w[:, ROPE_LINKS // 2, 1:]
  r2 = (p**2).sum(-1).clamp_min(1e-4)
  w = (p[:, 0] * v[:, 1] - p[:, 1] * v[:, 0]) / r2  # same sense as the handle circle (y = sin, z = -cos)
  return torch.clamp(w / (2 * math.pi * cmd.freq), -1.0, 1.5) * (cmd.amp > 0)


def neck_torque(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Cost >= 0 (negative weight): sum of squared neck/head servo torques (N·m)²."""
  robot = env.scene["robot"]
  ids = [robot.actuator_names.index(n) for n in NECK_ACTS]
  return robot.data.actuator_force[:, ids].square().sum(1)


def rope_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Critic only: rope middle link y-z about the axis and its velocity (the policy stays rope-blind)."""
  rope = env.scene["rope"]
  p = rope.data.body_link_pos_w[:, ROPE_LINKS // 2, 1:] - _axis_frame(env)
  return torch.cat([p, rope.data.body_link_lin_vel_w[:, ROPE_LINKS // 2, 1:]], dim=1)


def make_ys_rope_env_cfg(play: bool = False):
  cfg = hc.make_ys_head_circle_env_cfg(play=play, tip_space=True, tip_load=False, whole_body=True)
  robot = cfg.scene.entities["robot"]
  get_spec = robot.spec_fn
  cfg.scene.entities = {
    "robot": replace(robot, spec_fn=lambda: add_t_handle(get_spec())),
    "rope": EntityCfg(spec_fn=rope_spec, init_state=EntityCfg.InitialStateCfg(pos=FAR_NOM, joint_pos=None)),
    "far_handle": EntityCfg(spec_fn=far_handle_spec, init_state=EntityCfg.InitialStateCfg(pos=FAR_NOM)),
  }
  cfg.scene.spec_fn = connect_rope
  cfg.sim.nconmax = 64  # + up to ROPE_LINKS rope-floor contacts per env

  # Spawn at the env origin facing +x (the rope and far handle are laid out along +x).
  cfg.events["reset_base"].params["pose_range"] = {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (0.12, 0.13), "yaw": (-0.03, 0.03)}
  cfg.events.pop("push_robot")
  cfg.events["reset_rope"] = EventTermCfg(func=mjlab_events.reset_root_state_uniform, mode="reset",
                                          params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("rope")})
  cfg.events["reset_rope_joints"] = EventTermCfg(func=reset_rope_joints, mode="reset")
  cfg.events["reset_far_handle"] = EventTermCfg(func=mjlab_events.reset_root_state_uniform, mode="reset",
                                                params={"pose_range": {}, "asset_cfg": SceneEntityCfg("far_handle")})

  old = cfg.commands["head_pose"]
  cfg.commands["head_pose"] = RopeCircleCommandCfg(**{k: getattr(old, k) for k in old.__dataclass_fields__})

  cfg.rewards["mouth_circle_feet"].weight = 1.0  # handle circle: shaping now, the rope is the goal
  cfg.rewards["rope_turn"] = RewardTermCfg(func=rope_turn, weight=2.0)
  cfg.rewards["neck_torque"] = RewardTermCfg(func=neck_torque, weight=-0.5)  # head_roll alone held 0.64 N·m (5c-0)
  cfg.observations["critic"].terms["rope"] = ObservationTermCfg(func=rope_obs)
  return cfg


YsRopeRlCfg = replace(hc.YsHeadCircleRlCfg, experiment_name="ys_rope")
