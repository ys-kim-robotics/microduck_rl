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
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg

import mjlab_microduck.tasks.ys_head_circle_env_cfg as hc
import mjlab_microduck.tasks.ys_jump_env_cfg as J
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse
from mjlab_microduck.tasks.symmetry import _get_tensors
import mjlab_microduck.tasks.ys_beat as B
from mjlab_microduck.tasks.ys_beat import Beat
from mjlab_microduck.tasks.ys_rope import ROPE_RADIUS
from mjlab_microduck.tasks.ys_rope_duo_env_cfg import load_frozen_actor
from mjlab_microduck.tasks.ys_rope_env_cfg import STEM_END_NOM, STEM_LEN, make_ys_rope_bob_env_cfg, set_span

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


def rope_phase_obs(env) -> torch.Tensor:
  """trio4: (N, 2) cos, sin of the rope's real angle about the stem-end axis minus the jumper clock.
  trio1/3 jumped on their clock alone; the real rope phase sits ~19 deg apart over envs (p10-p90)."""
  tu, pa, rope = env.scene["turner"], env.scene["partner"], env.scene["rope"]
  if not hasattr(env, "_ys_stem"):
    env._ys_stem = (tu.find_sites("stem_end")[0][0], pa.find_sites("stem_end")[0][0])
  s1, s2 = env._ys_stem
  axis = 0.5 * (tu.data.site_pos_w[:, s1, 1:] + pa.data.site_pos_w[:, s2, 1:])
  m = rope.data.body_link_pos_w[:, 6, 1:] - axis  # middle link
  rel = -torch.atan2(m[:, 0], -m[:, 1]) - env.command_manager.get_term("head_pose").phase  # rope turns -clock (trio_diag)
  return torch.stack([rel.cos(), rel.sin()], -1)


# Head camera (runtime/deploy/robotd.toml, mediad/src/camera.rs): IMX219, 62 deg horizontal, 16:9, 30 fps.
# A colour-marker tracker on the turner's head (the detector NN runs at 2 Hz: too slow for a rope). The head
# circles with the rope and stays in frame 89% of the time (handle tip 45%: the jumper's head swings).
CAM_TAN_H = math.tan(math.radians(31.0))
CAM_TAN_V = CAM_TAN_H * 9 / 16
CAM_HZ = 30.0
CAM_LAG = (3, 5)  # policy steps (60-100 ms): capture + ISP + tracker, unmeasured -> randomized per env
CAM_NOISE = 0.01  # normalized image units (~6 px at 1280)
CAM_DROP = 0.05


def _cam_vec(env, site: str):
  """(d, p0, p1): a leader (front turner) site in the jumper's head_camera frame, and both world points."""
  rob, tu = env.scene["robot"], env.scene["turner"]
  ids = env.__dict__.setdefault("_ys_cam_ids", {})
  if site not in ids:
    ids[site] = (rob.find_sites("head_camera")[0][0], tu.find_sites(site)[0][0])
  i, j = ids[site]
  p0, p1 = rob.data.site_pos_w[:, i], tu.data.site_pos_w[:, j]
  R = matrix_from_quat(rob.data.site_quat_w[:, i])
  return (R.transpose(1, 2) @ (p1 - p0)[..., None])[..., 0], p0, p1


def _cam_true(env, site: str = "head_imu", occlude: bool = False) -> torch.Tensor:
  """(N, 3) u, v in [-1, 1] (right, up) of a turner site (head_imu: its head; stem_end: the red ball) in
  the jumper's head_camera site frame (x forward, y left, z up: probed), valid = in front and in the
  frame (and, occlude, not behind a rope link)."""
  d, p0, p1 = _cam_vec(env, site)
  x = d[:, 0].clamp(min=1e-3)
  u, v = -d[:, 1] / x / CAM_TAN_H, d[:, 2] / x / CAM_TAN_V
  ok = (d[:, 0] > 0.02) & (u.abs() <= 1) & (v.abs() <= 1)
  if occlude:  # a rope link across the line of sight hides the ball. ponytail: the turner's own head/body never does
    seg = p1 - p0
    L = env.scene["rope"].data.body_link_pos_w  # link origins are ~7 cm apart: test 4 points per link
    w = torch.linspace(0, 1, 5, device=L.device)[:-1]
    q = (L[:, :-1, None] + w[None, None, :, None] * (L[:, 1:, None] - L[:, :-1, None])).flatten(1, 2) - p0[:, None]
    t = ((q * seg[:, None]).sum(-1) / seg.square().sum(-1, keepdim=True)).clamp(0, 1)
    dist = (q - t[..., None] * seg[:, None]).norm(dim=-1)
    ok &= ~((dist < ROPE_RADIUS + 0.5 * BALL_R) & (t > 0.05) & (t < 0.9)).any(-1)
  return torch.stack([u.clamp(-1, 1), v.clamp(-1, 1), ok.float()], -1)


def _cam_track(env, key: str, site: str, occlude: bool, true_fn=None) -> torch.Tensor:
  """(N, 6) the tracker's last two delivered frames (u, v, valid) of one marker. 30 Hz sample-and-hold
  at a per-env camera phase, 60-100 ms lag (one camera: shared by every marker), noise, 5% dropped
  frames (valid 0, last u v held). Advanced once per env step (actor and critic both call it).
  true_fn(env) -> (N, 3) overrides the jumper-to-turner geometry (the follower's camera)."""
  n, dev = env.num_envs, env.device
  meas = true_fn or (lambda e: _cam_true(e, site, occlude))
  clk = env.__dict__.get("_ys_cam_clock")
  if clk is None:
    clk = env._ys_cam_clock = dict(lag=torch.randint(CAM_LAG[0], CAM_LAG[1] + 1, (n,), device=dev), ph=torch.rand(n, device=dev))
  st = env.__dict__.get(key)
  if st is None:
    st = dict(step=-1, ring=meas(env)[:, None].repeat(1, CAM_LAG[1] + 1, 1),
              cur=torch.zeros(n, 3, device=dev), prev=torch.zeros(n, 3, device=dev), age=torch.ones(n, device=dev))
    setattr(env, key, st)
  if st["step"] != env.common_step_counter:
    st["step"] = env.common_step_counter
    st["ring"] = torch.cat([meas(env)[:, None], st["ring"][:, :-1]], 1)
    t = env.episode_length_buf * env.step_dt * CAM_HZ + clk["ph"]
    new = (t.floor() != (t - env.step_dt * CAM_HZ).floor()) | (env.episode_length_buf <= 1)
    m = st["ring"][torch.arange(n, device=dev), clk["lag"]].clone()
    m[:, :2] += CAM_NOISE * torch.randn_like(m[:, :2])
    drop = torch.rand(n, device=dev) < CAM_DROP
    m[drop, :2], m[drop, 2] = st["cur"][drop, :2], 0.0
    st["prev"] = torch.where(new[:, None], st["cur"], st["prev"])
    st["cur"] = torch.where(new[:, None], m, st["cur"])
    seen = new & (m[:, 2] > 0)  # age: s since the last delivered valid sighting (capped 1 s; 1 at reset)
    st["age"] = torch.where(seen, torch.zeros_like(st["age"]), (st["age"] + env.step_dt).clamp(max=1.0))
    st["age"][env.episode_length_buf <= 1] = 1.0
  return torch.cat([st["cur"], st["prev"]], -1)


def cam_tip_obs(env) -> torch.Tensor:
  """trio4+: the leader's head through the jumper's tracker (6)."""
  return _cam_track(env, "_ys_cam", "head_imu", False)


def cam_ball_obs(env) -> torch.Tensor:
  """stage 7-4 (2): the leader's red ball (the rope's attach point) through the same tracker (6), hidden
  by rope links crossing the line of sight. In frame ~32% of the time (trio6 posture)."""
  return _cam_track(env, "_ys_cam_ball", "stem_end", True)


def widen_checkpoint(src: str, dst: str, add: dict) -> None:
  """trio1 checkpoint -> one whose actor/critic take add[net] more obs at the end, zero-weighted
  (same behaviour at the start), unit normalizer stats, Adam moments padded with zeros."""
  sd = torch.load(src, map_location="cpu", weights_only=False)
  pad = lambda t, k, v=0.0: torch.cat([t, torch.full((*t.shape[:-1], k), v, dtype=t.dtype)], -1)
  old = {}
  for net, k in add.items():
    d = sd[f"{net}_state_dict"]
    old[d["mlp.0.weight"].shape[1]] = k
    d["mlp.0.weight"] = pad(d["mlp.0.weight"], k)
    d["obs_normalizer._mean"] = pad(d["obs_normalizer._mean"], k)
    for n in ("_var", "_std"):
      d["obs_normalizer." + n] = pad(d["obs_normalizer." + n], k, 1.0)
  for st in sd["optimizer_state_dict"]["state"].values():
    for n in ("exp_avg", "exp_avg_sq"):
      if st[n].dim() == 2 and st[n].shape[0] == 512 and st[n].shape[1] in old:
        st[n] = pad(st[n], old[st[n].shape[1]])
  torch.save(sd, dst)


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
    if self.cfg.mirror:  # extra (camera) dims after the 61-D layout mirror by cfg.extra_sign (u -> -u)
      x = torch.cat([x[:, :61][:, perm] * sign, x[:, 61:] * torch.tensor(self.cfg.extra_sign, device=x.device)], 1)
    sl = turner_slots(self._env)  # a mirrored forward circle = the reversed one (ys_rope_duo)
    if self.cfg.clock_off:  # follower (7-4 stage 3): its own clock is off by an episode offset
      off = self._env.__dict__.get("_ys_off")
      amp = self._env.__dict__.get("_ys_famp")  # (N,) 0..1 circle size (7-6: hold the handle still, listen)
      if self.cfg.beat:  # 7-6: ...corrected by ear (ys_beat)
        if not hasattr(self, "_beat"):
          self._beat = Beat(self._env)
        c, a = self._beat.step()
        off, amp = (c if off is None else off + c), (amp if a is None else a)
      if off is not None:
        co, so = off.cos(), off.sin()
        sl[:, 4], sl[:, 5] = sl[:, 4] * co - sl[:, 5] * so, sl[:, 5] * co + sl[:, 4] * so
      if amp is not None:
        sl[:, 4:6] *= amp[:, None]
    x[:, 48:61] = sl
    lock = self._beat.f.lock if self.cfg.beat else None
    if lock is not None and x.shape[1] > 61 and B.AFTER_LOCK == "mask":  # 7-6: camera blocked by the jumper
      x[lock, 61:] = 0.0
    with torch.no_grad():
      a = self._actor(x)
      if lock is not None and x.shape[1] > 61 and B.AFTER_LOCK == "switch":
        if not hasattr(self, "_actor2"):
          self._actor2 = load_frozen_actor(TURNER_CKPT, self.device)
        a = torch.where(lock[:, None], self._actor2(x[:, :61]), a)
    super().process_actions(a[:, aperm] * asign if self.cfg.mirror else a)


@dataclass(kw_only=True)
class TurnerActionCfg(JointPositionActionCfg):
  checkpoint: str = TURNER_CKPT
  obs_group: str = "turner"
  mirror: bool = False
  clock_off: bool = False
  beat: bool = False
  extra_sign: tuple = ()

  def build(self, env):
    return TurnerAction(self, env)


def cam_age_obs(env) -> torch.Tensor:
  """(N, 2) s since the jumper's tracker last saw the leader's head / ball (capped 1 s). B1 saw the
  ball 15% of steps, the head 0%. Must come after cam_tip and cam_ball (they advance the tracker)."""
  return torch.stack([env._ys_cam["age"], env._ys_cam_ball["age"]], -1)


def follower_ball_obs(env) -> torch.Tensor:
  """The back follower's tracker (6) on the leader's red ball, through the jumper's body (blocker)."""
  import mjlab_microduck.tasks.ys_follow_env_cfg as F  # circular at module level
  return _cam_track(env, "_ys_trio_fol", "stem_end", False, true_fn=lambda e: F._leader_ball(e, "partner", "turner", ("robot",)))


CLOCK_FADE_STEPS = 24 * 200  # trio6: env steps (24 per iteration) over which the actor's clock fades out


def faded_head_command(env, command_name: str = "head_pose", fade_steps: int = CLOCK_FADE_STEPS) -> torch.Tensor:
  """trio6: the actor's head-command slots (the jump clock, amp cos/sin) scaled 1 -> 0 over
  fade_steps (0: off from the start, play): the jumper then only knows when to jump from the camera
  (no clock sync between real ducks). The critic keeps the clock."""
  a = max(0.0, 1.0 - env.common_step_counter / fade_steps) if fade_steps else 0.0
  return a * env.command_manager.get_command(command_name)


class TrioClockCommand(J.JumpClockCommand):
  """trio5: the jump windows (every clock-based reward) follow the rope, not the jumper's clock.
  trio3/4 randomized the rope offset but rewarded the clock's windows, so tracking the rope was
  penalized: trio4's actor ignored its camera (camera zeroed: same clean %). The observed clock is
  unchanged: only the camera tells the jumper where the windows are."""

  def window(self) -> torch.Tensor:
    shift = getattr(self._env, "_ys_delta", None)
    shift = 0.0 if shift is None else shift - DELTA0  # turner phase = clock - delta: a later rope
    return torch.remainder(self.phase - J.JUMP_PHASE - shift + math.pi, 2 * math.pi) - math.pi


@dataclass(kw_only=True)
class TrioClockCommandCfg(J.JumpClockCommandCfg):
  def build(self, env):
    return TrioClockCommand(self, env)


BALL_R = 0.010  # stage 7-4: tracking markers on the stem ends (rope attach point): T, ball, rope
BALL_RGBA = {"turner": (0.95, 0.05, 0.05, 1), "partner": (0.05, 0.8, 0.1, 1)}  # leader red, follower green
ROPE_RGBA = (0.95, 0.95, 0.95, 1)  # white: the red rope would read as the leader's ball


def with_ball(spec_fn, rgba):
  def fn():
    spec = spec_fn()
    spec.body("t_handle").add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0, 0, -STEM_LEN], size=[BALL_R, 0, 0],
                                   mass=0.001, contype=0, conaffinity=0, rgba=list(rgba))
    return spec
  return fn


def with_rope_color(spec_fn, rgba):
  def fn():
    spec = spec_fn()
    for g in spec.geoms:
      g.rgba = list(rgba)
    return spec
  return fn


HEAD_X0 = 0.042  # m: head_imu ahead of the trunk root in the standing pose (trio6 jumps at 0.104)


def head_back(env: ManagerBasedRlEnv, tol: float = 0.015, cap: float = 0.06) -> torch.Tensor:
  """(>0, weight < 0) cm the jumper's head sits ahead of its standing spot beyond tol (user: no need to
  thrust the head forward; trio6 holds it 62 mm ahead, neck and head pitch ~-1.7 rad each)."""
  rob = env.scene["robot"]
  if not hasattr(env, "_ys_head_site"):
    env._ys_head_site = rob.find_sites("head_imu")[0][0]
  d = quat_apply_inverse(rob.data.root_link_quat_w, rob.data.site_pos_w[:, env._ys_head_site] - rob.data.root_link_pos_w)
  return (d[:, 0] - HEAD_X0 - tol).clamp(0, cap) / 0.01


GAZE_V, GAZE_H = math.radians(14), math.radians(24)  # free band: frame half-angles 18.6 / 31 deg minus the
# leader head's own swing (trio6, camera level: -12..+9 deg vertical p10-p90)


def gaze(env: ManagerBasedRlEnv, v_tol: float = GAZE_V, h_tol: float = GAZE_H, cap: float = math.radians(30)) -> torch.Tensor:
  """(>0, weight < 0) per 10 deg the leader's head sits outside the central band of the jumper's camera.
  head_back alone was met by pitching the head up: camera +3 deg (trio6) -> +35 (B1), the leader's head
  36 deg below the axis, out of frame 100% (trio_cams.py, 2026-11-06). Behind the camera = both caps."""
  d, _, _ = _cam_vec(env, "head_imu")
  el, az = torch.atan2(d[:, 2], d[:, 0]), torch.atan2(-d[:, 1], d[:, 0])
  return ((el.abs() - v_tol).clamp(0, cap) + (az.abs() - h_tol).clamp(0, cap)) / math.radians(10)


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


def make_ys_trio_jump_env_cfg(play: bool = False, delta_half: float = 0.0, rope_obs: bool = False, cam_obs: bool = False, rope_windows: bool = False, no_clock: bool = False, head_w: float = 0.0, head_tol: float = 0.015, clock_fade: int = CLOCK_FADE_STEPS, ball_obs: bool = False, cam_age: bool = False, follower: str | None = None, follower_off: float = math.radians(60), gaze_w: float = 0.0, beat: bool = False, follower_cam: bool = False):
  cfg = J.make_ys_side_jump_env_cfg(play, half_w=J.SIDE_OVERLAP_HALF_W, both=3.0)  # side6's task
  cfg.commands["head_pose"].zero_command_prob = 0.0  # the rope always turns: always jump
  rc = make_ys_rope_bob_env_cfg(play=True, stay=True, reverse=True)  # turners as in the duo evals (no pushes)
  set_span(rc, SPAN)
  mid = STEM_END_NOM[0] + SPAN / 2
  tr = rc.scene.entities["robot"]  # the T-handle duck
  flip = (0.0, 0.0, 0.0, 1.0)  # yaw pi
  turner = replace(tr, spec_fn=with_ball(tr.spec_fn, BALL_RGBA["turner"]), init_state=replace(tr.init_state, pos=(mid, -JUMPER_Y, tr.init_state.pos[2]), rot=flip))
  partner = replace(tr, spec_fn=with_ball(tr.spec_fn, BALL_RGBA["partner"]), init_state=replace(tr.init_state, pos=(-mid, -JUMPER_Y, tr.init_state.pos[2]), rot=(1.0, 0.0, 0.0, 0.0)))
  rope = rc.scene.entities["rope"]
  rope = replace(rope, spec_fn=with_rope_color(rope.spec_fn, ROPE_RGBA), init_state=replace(rope.init_state, pos=(-SPAN / 2, -JUMPER_Y, STEM_END_NOM[2]), rot=flip))
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
  if follower:  # the back turner (partner, mirrored, green ball) follows; the front one (red ball, the duck
    # the jumper watches) leads on the true clock. follower_cam: a 67-D camera follower (F1) watching the
    # leader's red ball, mirrored with it (u -> -u); the jumper in between blocks its view
    a = cfg.actions["partner_joint_pos"]
    a.checkpoint, a.clock_off, a.beat = follower, True, beat
    import mjlab_microduck.tasks.ys_follow_env_cfg as F
    if follower_cam:
      cfg.observations["partner"].terms["leader_ball"] = ObservationTermCfg(func=follower_ball_obs)
      a.extra_sign = (-1.0, 1.0, 1.0, -1.0, 1.0, 1.0)
    cfg.events["sample_offset"] = EventTermCfg(func=F.sample_offset, mode="reset", params={"half": follower_off})

  for who in ("turner", "partner"):  # a turner down: end as a timeout (no fall penalty for the jumper)
    fell = copy.deepcopy(cfg.terminations["fell_over"])
    fell.params = {**fell.params, "asset_cfg": SceneEntityCfg(who)}
    fell.time_out = True
    cfg.terminations[f"{who}_fell"] = fell
  cfg.rewards["rope_overlap"] = RewardTermCfg(func=rope_overlap, weight=-3.0)
  if rope_obs:  # sim-only (privileged): appended last, slots 61:63 (actor), 76:78 (critic)
    for grp in ("actor", "critic"):
      cfg.observations[grp].terms["rope_phase"] = ObservationTermCfg(func=rope_phase_obs)
  if cam_obs:  # real-feasible: tracker on the actor (61:67); the critic also gets the true rope phase (76:84)
    for grp in ("actor", "critic"):
      cfg.observations[grp].terms["cam_tip"] = ObservationTermCfg(func=cam_tip_obs)
    cfg.observations["critic"].terms["rope_phase"] = ObservationTermCfg(func=rope_phase_obs)
  if ball_obs:  # appended after cam_tip / rope_phase: actor 67:73, critic 84:90
    for grp in ("actor", "critic"):
      cfg.observations[grp].terms["cam_ball"] = ObservationTermCfg(func=cam_ball_obs)
  if cam_age:  # B2+: actor 73:75, critic 90:92
    for grp in ("actor", "critic"):
      cfg.observations[grp].terms["cam_age"] = ObservationTermCfg(func=cam_age_obs)
  if gaze_w:
    cfg.rewards["gaze"] = RewardTermCfg(func=gaze, weight=-gaze_w)
  if head_w:
    cfg.rewards["head_back"] = RewardTermCfg(func=head_back, weight=-head_w, params={"tol": head_tol})
  if no_clock:
    t = cfg.observations["actor"].terms["head_command"]
    t.func, t.params = faded_head_command, {**t.params, "fade_steps": 0 if play else clock_fade}  # 0: resuming a clock-free policy
  if rope_windows:
    c = cfg.commands["head_pose"]
    cfg.commands["head_pose"] = TrioClockCommandCfg(**{f: getattr(c, f) for f in c.__dataclass_fields__})
  if delta_half:
    cfg.events["sample_delta"] = EventTermCfg(func=sample_delta, mode="reset", params={"half": delta_half})
  return cfg


YsTrioJumpRlCfg = replace(MicroduckRlCfg, experiment_name="ys_trio_jump")
