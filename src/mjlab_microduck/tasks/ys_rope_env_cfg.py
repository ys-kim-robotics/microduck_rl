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
from dataclasses import dataclass, replace

import mujoco
import torch
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
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


def rope_spec(span: float = SPAN) -> mujoco.MjSpec:
  """Rope entity: free root at the far end (entity origin), chain toward -x to the duck's stem end."""
  spec = mujoco.MjSpec()
  add_rope(spec, (0, 0, 0), (-span, 0, 0), length=ROPE_LEN, free=True)
  m = spec.compile()
  spec.add_key(name="init", qpos=m.qpos0.tolist())  # the sagging V; entity joint_pos=None reads it
  return spec


def far_handle_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  b = spec.worldbody.add_body(name="far_handle", mocap=True)
  b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], contype=0, conaffinity=0, rgba=[0.9, 0.6, 0.1, 1])
  b.add_site(name="far_site")
  return spec


def soft_handle_spec() -> mujoco.MjSpec:
  """rope9+: the far handle as a free body (gravity-compensated) instead of the mocap itself."""
  spec = mujoco.MjSpec()
  b = spec.worldbody.add_body(name="soft_handle", gravcomp=1.0)
  b.add_freejoint()
  b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.01, 0, 0], mass=SOFT_MASS, contype=0, conaffinity=0, rgba=[0.2, 0.6, 0.9, 1])
  b.add_site(name="far_site")
  return spec


SOFT_MASS = 0.1  # kg; stand-in for the partner's head + handle
SOFT_TC = 0.08  # s; CONNECT timeconst holding the handle to the mocap target ~ a 2 Hz spring: it lags and is dragged


def connect_rope_soft(spec: mujoco.MjSpec) -> None:
  """Scene spec_fn for rope9+: rope -> free soft_handle -> soft spring -> far_handle (mocap target).
  A rigid mocap absorbs any rope pull; a real partner duck yields, and two rope5 ducks fell together
  (Circus measured the same 'two-sided collapse' and fixed it with a compliant carrier)."""
  for a, b, tc in (("rope/rope_start", "soft_handle/far_site", 0.05), ("rope/rope_end", "robot/stem_end", 0.05),
                   ("soft_handle/far_site", "far_handle/far_site", SOFT_TC)):
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE, name1=a, name2=b)
    eq.solref = [tc, 1.0]


def connect_rope(spec: mujoco.MjSpec) -> None:
  """Scene spec_fn (runs after the entities are attached): pin both rope ends to their handles."""
  for a, b in (("rope/rope_start", "far_handle/far_site"), ("rope/rope_end", "robot/stem_end")):
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE, name1=a, name2=b)
    # 0.01 (= 2 x dt) blew up against the 1.7 g links once the robot pulled on it (16/16 envs fell,
    # 5c-2); 0.05 held with a ~1 mm mean gap and no falls. The rope needs a little give at its ends.
    eq.solref = [0.05, 1.0]


def reset_rope_joints(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, span: float = SPAN) -> None:
  """Back to the sagging V. Not rope.data.default_joint_pos: with several entities mjlab 1.3.0 slices
  that from the SCENE keyframe (65 values here vs the rope's 44), so the rope's own qpos0 is used."""
  rope = env.scene["rope"]
  ids = slice(None) if env_ids is None else env_ids
  n = rope.data.joint_pos[ids].shape[0]
  q0 = torch.tensor(_rope_q0(span), device=env.device).expand(n, -1)
  rope.write_joint_state_to_sim(q0, torch.zeros_like(rope.data.joint_vel[ids]), env_ids=env_ids)


_Q0 = {}


def _rope_q0(span: float = SPAN):
  if span not in _Q0:
    _Q0[span] = rope_spec(span).compile().qpos0[7:].tolist()  # after the free root
  return _Q0[span]


def set_span(cfg, span: float) -> None:
  """rope14+: handle centres `span` apart (rope length unchanged): rope layout, its reset, the far handle."""
  far = (STEM_END_NOM[0] + span, 0.0, STEM_END_NOM[2])
  ents = cfg.scene.entities
  ents["rope"] = replace(ents["rope"], spec_fn=lambda: rope_spec(span), init_state=replace(ents["rope"].init_state, pos=far))
  if "far_handle" in ents:
    ents["far_handle"] = replace(ents["far_handle"], init_state=replace(ents["far_handle"].init_state, pos=far))
  cfg.events["reset_rope_joints"].params = {"span": span}
  if hasattr(cfg.commands["head_pose"], "span"):
    cfg.commands["head_pose"].span = span


class RopeCircleCommand(hc.HeadCircleCommand):
  """5b's whole-body circle, scored at the T-handle stem end, plus the mirroring far handle."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._sid = self._robot.find_sites("stem_end")[0][0]  # circle score / centre now at the stem end
    self._far = env.scene["far_handle"]
    self._stem_c = torch.zeros(self.num_envs, 3, device=self.device)
    self._far_c = torch.tensor((STEM_END_NOM[0] + cfg.span, 0.0, STEM_END_NOM[2]), device=self.device) + env.scene.env_origins
    self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    # imperfect partner (rope7+): per-env lag (steps), circle scale, and a wandering centre offset
    L = cfg.partner_lag_steps + 1
    self._hist = torch.zeros(L, self.num_envs, 2, device=self.device)  # ring buffer of stem y-z offsets
    self._k = 0
    self._lag = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._scale = torch.ones(self.num_envs, device=self.device)
    self._wander = torch.zeros(self.num_envs, 2, device=self.device)
    self._dx = torch.zeros(self.num_envs, device=self.device)  # rope9+: far handle x offset (episode + drift)
    self._dx_wander = torch.zeros(self.num_envs, device=self.device)

  def reset(self, env_ids):
    ids = slice(None) if env_ids is None else env_ids
    self._fresh[ids] = True
    n = self._lag[ids].shape[0]
    self._lag[ids] = torch.randint(0, self.cfg.partner_lag_steps + 1, (n,), device=self.device)
    self._scale[ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.partner_scale)
    self._wander[ids] = 0.0
    self._hist[:, ids] = 0.0
    self._dx[ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.partner_x_range)
    self._dx_wander[ids] = 0.0
    return super().reset(env_ids)

  def _update_command(self) -> None:
    super()._update_command()
    stem = self._robot.data.site_pos_w[:, self._sid]
    self._stem_c = torch.where(self._fresh[:, None], stem, self._stem_c + (stem - self._stem_c) * (self._env.step_dt / MIRROR_TAU))
    self._fresh[:] = False
    far = self._far_c.clone()
    off = stem[:, 1:] - self._stem_c[:, 1:]  # same y-z offset = same circle, same phase
    L = self._hist.shape[0]
    self._k = (self._k + 1) % L
    self._hist[self._k] = off
    off = self._hist[(self._k - self._lag) % L, torch.arange(self.num_envs, device=self.device)]  # lagged
    if self.cfg.partner_wander > 0:  # Ornstein-Uhlenbeck drift of the partner's circle centre
      dt, tau = self._env.step_dt, 2.0
      self._wander += -self._wander * dt / tau + self.cfg.partner_wander * math.sqrt(2 * dt / tau) * torch.randn_like(self._wander)
    far[:, 1:] += self._scale[:, None] * off + self._wander
    if self.cfg.partner_x_wander > 0:  # slack <-> taut: the duck must not lean on the rope's pull
      dt, tau = self._env.step_dt, 1.0
      self._dx_wander += -self._dx_wander * dt / tau + self.cfg.partner_x_wander * math.sqrt(2 * dt / tau) * torch.randn_like(self._dx_wander)
    far[:, 0] += self._dx + self._dx_wander
    # direct write: Entity.write_mocap_pose_to_sim breaks on a plain int mocap_id (mjlab 1.3.0, [n,1,3] vs [n,3])
    self._env.sim.data.mocap_pos[:, self._far.indexing.mocap_id] = far


@dataclass(kw_only=True)
class RopeCircleCommandCfg(hc.HeadCircleCommandCfg):
  span: float = SPAN  # far handle centre = stem end nominal + span along x
  partner_lag_steps: int = 0  # far handle copies the stem offset this many 20 ms steps late (U[0, max] per env)
  partner_scale: tuple[float, float] = (1.0, 1.0)  # far circle = this x the duck's (U per env)
  partner_wander: float = 0.0  # std (m) of the far circle centre's slow drift
  partner_x_range: tuple[float, float] = (0.0, 0.0)  # far handle x offset (m) per episode, U; < 0 = toward the duck (slack)
  partner_x_wander: float = 0.0  # std (m) of its drift along x (OU, tau 1 s)

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


def rope_loop(env: ManagerBasedRlEnv, r_full: float = 0.2) -> torch.Tensor:
  """A big loop, not just a turning one: the middle link's distance from the rope axis (r_full m pays 1)
  times its forward turn rate. rope2/3 turned the rope ~at handle height, no room for a jumper. The
  rope length caps the loop: 0.85 m over a 0.7 m span sags at most 0.24 m."""
  rope = env.scene["rope"]
  r = (rope.data.body_link_pos_w[:, ROPE_LINKS // 2, 1:] - _axis_frame(env)).norm(dim=-1)
  return (r / r_full).clamp(0.0, 1.0) * rope_turn(env).clamp(0.0, 1.0)


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


def rope_blew_up(env: ManagerBasedRlEnv, max_speed: float = 15.0, max_dist: float = 1.0) -> torch.Tensor:
  """Rare chain divergence: any rope link faster than max_speed (m/s) or farther than max_dist (m)
  from the far handle. Terminating resets the env in the same step, so the exploded state never
  reaches the obs normalizer: rope1 inflated base_ang_vel's running variance 19x in 3 iterations
  and the policy collapsed (234 falls / 256 envs / 6 s vs 3 for its parent bc5)."""
  rope = env.scene["rope"]
  cmd = env.command_manager.get_term("head_pose")
  fast = rope.data.body_link_lin_vel_w.norm(dim=-1).max(1).values > max_speed
  far = (rope.data.body_link_pos_w - cmd._far_c[:, None]).norm(dim=-1).max(1).values > max_dist
  return fast | far | ~torch.isfinite(rope.data.body_link_pos_w).all(-1).all(-1)


def ramp_weight(env: ManagerBasedRlEnv, env_ids, reward_name: str, stages: list[tuple[int, float]]) -> torch.Tensor:
  """reward_weight with ITERATIONS counted from this run's first step (a plain resume keeps the parent's
  step counter): stages = [(iteration, weight), ...], latest elapsed wins."""
  del env_ids
  if env.common_step_counter == 0:  # before the checkpoint load restores the parent's counter
    return torch.tensor([env.reward_manager.get_term_cfg(reward_name).weight])
  start = env.__dict__.setdefault("_ys_ramp_start", env.common_step_counter)
  it = (env.common_step_counter - start) // 24
  term = env.reward_manager.get_term_cfg(reward_name)
  for i, w in stages:
    if it >= i:
      term.weight = w
  return torch.tensor([term.weight])


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
  if play:  # show the full-size circle the curriculum ends at, not its 3.5 cm start
    cfg.commands["head_pose"].r_start, cfg.commands["head_pose"].freeze_radius = hc.WB_R_MAX, True

  # New terms ramp in from 0 (iterations from this run's start): the critic comes from bc5's reward
  # stack, and full weights from step 0 would score the inherited policy against a value function
  # that never saw them.
  cfg.rewards["rope_turn"] = RewardTermCfg(func=rope_turn, weight=0.0)
  cfg.rewards["neck_torque"] = RewardTermCfg(func=neck_torque, weight=0.0)  # head_roll alone held 0.64 N·m (5c-0)
  for name, stages in (("rope_turn", [(100, 0.5), (300, 1.0), (600, 2.0)]),
                       ("neck_torque", [(200, -0.1), (400, -0.25), (700, -0.5)]),
                       ("mouth_circle_feet", [(0, 2.0), (600, 1.0)])):  # handle circle -> shaping once the rope pays
    cfg.curriculum[f"{name}_ramp"] = CurriculumTermCfg(func=ramp_weight, params={"reward_name": name, "stages": stages})
  cfg.terminations["rope_blew_up"] = TerminationTermCfg(func=rope_blew_up)
  # Critic rope obs (rope_obs) left out for now: it changes the critic input size, so a 5b checkpoint
  # no longer loads (strict state_dict). Add with a widened-critic checkpoint if value learning lags.
  return cfg


def trunk_bob(env: ManagerBasedRlEnv, v0sq: float = 0.05) -> torch.Tensor:
  """Legs, not just the neck, lift the handle: trunk vertical velocity times the stem end's vertical
  velocity (both world z), squashed by tanh. Phase-free like circle_score: pays bobbing WITH the
  handle, charges bobbing against it, 0 if either is still. v0sq (m²/s²): 2 cm trunk bob at 1.25 Hz
  (0.16 m/s) under a 0.3 m/s stem pays ~tanh(1) at the peak."""
  robot = env.scene["robot"]
  cmd = env.command_manager.get_term("head_pose")
  vt = robot.data.root_link_lin_vel_w[:, 2]
  vs = robot.data.site_lin_vel_w[:, cmd._sid, 2]
  return torch.tanh(vt * vs / v0sq) * (cmd.amp > 0)


def stay_put(env: ManagerBasedRlEnv, pos_std: float = 0.02, yaw_std: float = 0.1) -> torch.Tensor:
  """1 at the spawn spot facing +x (the rope axis), Gaussian in trunk xy offset (m) and heading (rad).
  rope3 drifted 55 mm and turned 25 deg in 19 s under the rope's pull (ys_rope_check); the stock
  velocity-tracking terms miss a drift that slow."""
  robot = env.scene["robot"]
  dxy = (robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).square().sum(-1)
  w, x, y, z = robot.data.root_link_quat_w.unbind(-1)
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  return torch.exp(-dxy / pos_std**2 - yaw.square() / yaw_std**2)


def make_ys_rope_bob_env_cfg(play: bool = False, loop: bool = False, stay: bool = False, reverse: bool = False):
  """rope3+ : resume from a rope2 policy. Radius fixed at the curriculum's end (rope2 cycled 12 -> 4.5
  -> 12 cm: the shrink gate fired whenever the turning rope bent the circle), the rope2 weights at
  their final values, and trunk_bob ramped in so the legs share the vertical stroke."""
  cfg = make_ys_rope_env_cfg(play)
  cmd = cfg.commands["head_pose"]
  cmd.r_start, cmd.freeze_radius = hc.WB_R_MAX, True
  for name, w in (("rope_turn", 2.0), ("neck_torque", -0.5), ("mouth_circle_feet", 1.0)):
    cfg.curriculum.pop(f"{name}_ramp")
    cfg.rewards[name].weight = w
  cfg.rewards["trunk_bob"] = RewardTermCfg(func=trunk_bob, weight=0.0)
  cfg.curriculum["trunk_bob_ramp"] = CurriculumTermCfg(func=ramp_weight, params={"reward_name": "trunk_bob", "stages": [(30, 0.5), (100, 1.0)]})
  if loop:  # rope4+: resume from rope3. Higher loop for a jumper, and the neck hands more of the work to the body
    cfg.curriculum.pop("trunk_bob_ramp")
    cfg.rewards["trunk_bob"].weight = 1.0
    cfg.rewards["rope_loop"] = RewardTermCfg(func=rope_loop, weight=0.0)
    for name, stages in (("rope_loop", [(30, 1.0), (100, 2.0)]), ("neck_torque", [(50, -1.0), (150, -1.5)])):
      cfg.curriculum[f"{name}_ramp"] = CurriculumTermCfg(func=ramp_weight, params={"reward_name": name, "stages": stages})
  if stay:  # rope5+: resume from rope3 (bob weights at their final values), add stay_put
    cfg.curriculum.pop("trunk_bob_ramp")
    cfg.rewards["trunk_bob"].weight = 1.0
    cfg.rewards["stay_put"] = RewardTermCfg(func=stay_put, weight=0.0)
    cfg.curriculum["stay_put_ramp"] = CurriculumTermCfg(func=ramp_weight, params={"reward_name": "stay_put", "stages": [(30, 1.0), (100, 2.0)]})
  if reverse:  # rope6+: resume from rope5. Half the commands turn the other way: the turner facing the
    # duck sees the same rope rotation mirrored, so one policy serves both ends (as Circus does)
    cfg.curriculum.pop("stay_put_ramp")
    cfg.rewards["stay_put"].weight = 2.0
    cfg.commands["head_pose"].reverse_prob = 0.5
  return cfg


def phase_lock(env: ManagerBasedRlEnv, sigma2: float = 0.2, tau: float = 2.0) -> torch.Tensor:
  """Steady phase against the command clock, any constant lag allowed: the stem end's angle about its
  running centre minus the commanded phase, scored against that difference's own running mean
  (per-env circular EMA, tau s): exp((cos(d - mean) - 1) / sigma2), 30 deg off pays ~0.5.
  Two rope5 ducks on one rope wobbled 30-83 deg (std) against each other: the mocap partner
  copied the duck, so nothing tied it to the clock. Ducks sharing a clock and a policy share the
  mean lag, so only the jitter has to go."""
  cmd = env.command_manager.get_term("head_pose")
  p, _ = hc.tip_feet(cmd._robot, cmd._sid, cmd._feet)
  a = torch.atan2(p[:, 1] - cmd.center[:, 0], -(p[:, 2] - cmd.center[:, 1]))  # same sense as the command
  d = a - cmd.phase
  z = torch.stack([torch.cos(d), torch.sin(d)], -1)
  m = env.__dict__.get("_ys_lock_mean")
  if m is None or m.shape != z.shape:
    m = z.clone()
  m = m + (z - m) * (env.step_dt / tau)
  m = torch.where(env.episode_length_buf[:, None] <= 1, z, m)  # fresh episode: restart the mean
  env.__dict__["_ys_lock_mean"] = m
  mean = torch.atan2(m[:, 1], m[:, 0])
  return torch.exp((torch.cos(d - mean) - 1.0) / sigma2) * (cmd.amp > 0)


def make_ys_rope_lock_env_cfg(play: bool = False):
  """rope8+: resume from rope5, add phase_lock (ramped in)."""
  cfg = make_ys_rope_bob_env_cfg(play, stay=True)
  cfg.curriculum.pop("stay_put_ramp")
  cfg.rewards["stay_put"].weight = 2.0
  cfg.rewards["phase_lock"] = RewardTermCfg(func=phase_lock, weight=0.0)
  cfg.curriculum["phase_lock_ramp"] = CurriculumTermCfg(func=ramp_weight, params={"reward_name": "phase_lock", "stages": [(30, 1.0), (100, 2.0)]})
  return cfg


def make_ys_rope_partner_env_cfg(play: bool = False):
  """rope7+: resume from rope5. The far handle is an imperfect partner: 0-100 ms late, a 0.7-1.3x
  circle, a centre drifting ~2 cm. Two rope5 ducks on one rope fell 25-44% (ys_rope_duo): the
  mocap partner had mirrored the duck perfectly, a real one does not."""
  cfg = make_ys_rope_bob_env_cfg(play, stay=True)
  cfg.curriculum.pop("stay_put_ramp")
  cfg.rewards["stay_put"].weight = 2.0
  c = cfg.commands["head_pose"]
  c.partner_lag_steps, c.partner_scale, c.partner_wander = 5, (0.7, 1.3), 0.02
  return cfg


def make_ys_rope_slack_env_cfg(play: bool = False):
  """rope9+: resume from rope5. The far handle's x moves (per episode -6..+3 cm, drifting ~2 cm), so the
  rope goes slack and taut. rope5 leaned on the rope's pull: detached, it fell 100% within ~1 s, and two
  rope5 ducks leaning on each other collapsed together."""
  cfg = make_ys_rope_bob_env_cfg(play, stay=True)
  cfg.curriculum.pop("stay_put_ramp")
  cfg.rewards["stay_put"].weight = 2.0
  c = cfg.commands["head_pose"]
  c.partner_x_range, c.partner_x_wander = (-0.06, 0.03), 0.02
  return cfg


def make_ys_rope_soft_env_cfg(play: bool = False):
  """rope9+: resume from rope5 with a compliant far handle (connect_rope_soft)."""
  cfg = make_ys_rope_bob_env_cfg(play, stay=True)
  cfg.curriculum.pop("stay_put_ramp")
  cfg.rewards["stay_put"].weight = 2.0
  cfg.scene.entities["soft_handle"] = EntityCfg(spec_fn=soft_handle_spec, init_state=EntityCfg.InitialStateCfg(pos=FAR_NOM))
  cfg.scene.spec_fn = connect_rope_soft
  cfg.events["reset_soft_handle"] = EventTermCfg(func=mjlab_events.reset_root_state_uniform, mode="reset",
                                                 params={"pose_range": {}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("soft_handle")})
  return cfg


YsRopeRlCfg = replace(hc.YsHeadCircleRlCfg, experiment_name="ys_rope")
