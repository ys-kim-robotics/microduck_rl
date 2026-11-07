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
from mjlab.entity import EntityCfg
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
ROPE_GRACE_STEPS = int(__import__("os").environ.get("YS_ROPE_GRACE", "100"))  # 2 s: the rope spins up after a reset. E4 (7-11):
# 15 (0.3 s) so the spin-up's first passes count: G1 never had to clear them (J walks in before the rope turns, user task 1)


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
    stop = self._env.__dict__.get("_ys_amp")  # (N,) 0..1 both turners' circle size (7-8: stop the rope after a hit;
    if stop is not None:  # rope14 saw circle-off commands 10% of the time in training: zero_command_prob 0.1)
      sl[:, 4:6] *= stop[:, None]
    x[:, 48:61] = sl
    walk = self._env.__dict__.get("_ys_walk", {}).get(self.cfg.obs_group)  # 7-10: (N, 13) command block for the standing ONNX
    # while it takes the handle and walks to its place (NaN rows: rope14 turns); commands in the duck's own body frame
    lock = self._beat.f.lock if self.cfg.beat else None
    if lock is not None and x.shape[1] > 61 and B.AFTER_LOCK == "mask":  # 7-6: camera blocked by the jumper
      x[lock, 61:] = 0.0
    with torch.no_grad():
      a = self._actor(x)
      if lock is not None and x.shape[1] > 61 and B.AFTER_LOCK == "switch":
        if not hasattr(self, "_actor2"):
          self._actor2 = load_frozen_actor(TURNER_CKPT, self.device)
        a = torch.where(lock[:, None], self._actor2(x[:, :61]), a)
    if walk is not None and (m := ~walk[:, 0].isnan()).any():  # ...on the standing policy (env._ys_pick_onnx, 61-D, e.g. velstand)
      c = walk[m].clone()
      if self.cfg.mirror:  # the obs is mirrored: so are the lateral / turning command slots
        c[:, [1, 2, 5, 6, 8, 10, 12]] *= -1
      a = a.clone(); a[m] = pick_actions(self._env, x[m, :61], c)
    super().process_actions(a[:, aperm] * asign if self.cfg.mirror else a)


def pick_actions(env, x: torch.Tensor, cmd: torch.Tensor) -> torch.Tensor:
  """(n, 61) turner obs, (n, 13) command block [twist 3, head 4, body 6] -> (n, 14) actions of the standing ONNX."""
  import numpy as np
  import onnxruntime as ort
  if "_ys_pick_sess" not in env.__dict__:
    env._ys_pick_sess = ort.InferenceSession(env._ys_pick_onnx)
  z = x.clone(); z[:, 48:61] = cmd
  z = z.cpu().numpy().astype(np.float32)
  return torch.from_numpy(np.concatenate([env._ys_pick_sess.run(None, {"obs": z[i: i + 1]})[0] for i in range(len(z))])).to(x.device)


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


CLOCK_IN = int(__import__("os").environ.get("YS_J_CLOCK_IN", "0"))


def faded_head_command(env, command_name: str = "head_pose", fade_steps: int = CLOCK_FADE_STEPS) -> torch.Tensor:
  """trio6: the actor's head-command slots (the jump clock, amp cos/sin) scaled 1 -> 0 over
  fade_steps (0: off from the start, play): the jumper then only knows when to jump from the camera
  (no clock sync between real ducks). The critic keeps the clock."""
  a = max(0.0, 1.0 - env.common_step_counter / fade_steps) if fade_steps else 0.0
  if CLOCK_IN:  # 7-12: J's own clock back (v2: all three run on J's start call), faded in over CLOCK_IN steps (eval: 1)
    a = min(1.0, (env.common_step_counter + 1 - env.__dict__.setdefault("_ys_clock0", env.common_step_counter)) / CLOCK_IN)
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
  pen = ((el.abs() - v_tol).clamp(0, cap) + (az.abs() - h_tol).clamp(0, cap)) / math.radians(10)
  e = env.__dict__.get("_ys_enter")
  return pen if e is None or ENTER_MODE == "side" else pen * ~(e["fw"] & (_yaw(env.scene["robot"]).abs() > 0.3))  # 7-12 A/B


def connect_trio(spec: mujoco.MjSpec) -> None:
  for a, b in (("rope/rope_start", "partner/stem_end"), ("rope/rope_end", "turner/stem_end")):
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE, name1=a, name2=b)
    eq.solref = [0.05, 1.0]  # as connect_rope


def holder_spec(name: str):
  """7-10 (user: the handle waits in a holder, 2 (a)): where a rope end rests before its turner takes the handle."""
  spec = mujoco.MjSpec(); b = spec.worldbody.add_body(name=name, mocap=True)
  b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.008, 0.008, 0.008], contype=0, conaffinity=0, rgba=[0.45, 0.45, 0.5, 1]); b.add_site(name="s")
  return spec


PICK_EQ = ("rope_p_stem", "rope_t_stem", "rope_p_hold", "rope_t_hold")  # equality names (connect_trio_pick order)


def connect_trio_pick(spec: mujoco.MjSpec) -> None:
  """Each rope end is held by its holder until the turner bites the handle, then by the stem end (env.sim.data.eq_active):
  the holder sits where the stem end is at the full bow, so the switch moves nothing."""
  for nm, a, b, on in (("rope_p_stem", "rope/rope_start", "partner/stem_end", False), ("rope_t_stem", "rope/rope_end", "turner/stem_end", False),
                       ("rope_p_hold", "rope/rope_start", "hold_p/s", True), ("rope_t_hold", "rope/rope_end", "hold_t/s", True)):
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_SITE, name1=a, name2=b, name=nm)
    eq.solref, eq.active = [0.05, 1.0], on


def rope_section(pts: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
  """rope polyline (N, L, 3) where it crosses x = xq (N,) -> (N, 2) y, z (nan where it does not)."""
  x = pts[..., 0] - xq[:, None]
  a, b = x[:, :-1], x[:, 1:]  # all segments at once: the first one crossing x = xq (a per-segment loop with masked
  s = -a / torch.where(b != a, b - a, torch.full_like(a, 1e-9))  # writes synced the GPU 11 times per call)
  hit = (s >= 0) & (s <= 1)
  i = hit.float().argmax(-1)  # first crossing
  idx = i[:, None, None].expand(-1, 1, 2)
  p0, p1 = pts[:, :-1, 1:].gather(1, idx)[:, 0], pts[:, 1:, 1:].gather(1, idx)[:, 0]
  out = p0 + s.gather(1, i[:, None]) * (p1 - p0)
  return torch.where(hit.any(-1, keepdim=True), out, torch.full_like(out, math.nan))


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
  rel, c, s = sec[:, 0] - robot.data.root_link_pos_w[:, 1], yaw.cos(), yaw.sin()
  for j, (lo, hi) in enumerate(FOOT_BANDS):
    inb = (yj > lo) & (yj < hi) & (sec[:, 1] - ROPE_RADIUS < HEAD_TOP)
    if ENTER_MODE != "side":  # 7-12 A/B: a turned jumper's sole box (body x +-FOOT_HX, y lo..hi) spans these world y
      ys = torch.stack([bx * s + by * c for bx in (-FOOT_HX, FOOT_HX) for by in (lo, hi)], -1)
      inb = (rel > ys.amin(-1)) & (rel < ys.amax(-1)) & (sec[:, 1] - ROPE_RADIUS < HEAD_TOP)
    pen = torch.where(inb, torch.maximum(pen, (sec[:, 1] + ROPE_RADIUS - soles[:, j]).clamp(0, cap)), pen)
  return torch.where(env.episode_length_buf > ROPE_GRACE_STEPS, pen.nan_to_num(0.0) / 0.01, 0.0)


# 7-12 hit review (user 2026-11-07 night: "judged hits while the rope keeps turning?"): rope_overlap is a proxy (the
# rope in fixed bands beside J's ROOT, above the sole, below 17 cm). Against the actual geometry (s10/hit_review.py,
# N2Cf2 / N2Af): of the steps it calls a hit 37-39% are a real foot touch, 14% pass the leg, 47-49% neither (the feet
# move off the nominal bands), and it misses ~2000 real touches per 128 x 20 s. rope_hit_real: a physical foot-rope
# contact (the sim feet collide with the rope; a 20 g rope dragged by the turners slides off and keeps turning) or the
# rope within LEG_R of the actual leg polyline (hip-knee-ankle-foot: the sim legs have no rope collision; the real ones do).
LEG_R = 0.015  # m leg half-thickness
_LEGS = (("upper_leg_left", "leg", "ankle_left"), ("upper_leg_right", "leg_2", "ankle_right"))


def _seg_dist(p, a, b) -> torch.Tensor:
  ab = b - a; t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)).clamp(0, 1)
  return (p - (a + t[:, None] * ab)).norm(dim=-1)


def rope_leg_pass(env) -> torch.Tensor:
  """(N,) the rope (at each foot's x) within LEG_R + rope radius of that leg's hip-knee-ankle-foot polyline."""
  rob, rope = env.scene["robot"], env.scene["rope"]
  if "_ys_leg_ids" not in env.__dict__:
    names = list(rob.body_names); env._ys_leg_ids = [[names.index(b) for b in leg] for leg in _LEGS]
  feet = env.command_manager.get_term("head_pose")._sole; out = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for f in range(2):
    c = rob.data.geom_pos_w[:, feet[f]]; sec = rope_section(rope.data.body_link_pos_w, c[:, 0]); ok = torch.isfinite(sec[:, 0])
    pr = torch.stack([c[:, 0], sec[:, 0].nan_to_num(9.0), sec[:, 1].nan_to_num(9.0)], -1)
    pts = [rob.data.body_link_pos_w[:, i] for i in env._ys_leg_ids[f]] + [c]
    d = torch.stack([_seg_dist(pr, pts[i], pts[i + 1]) for i in range(3)], -1).amin(-1)
    out |= ok & (d < LEG_R + ROPE_RADIUS)
  return out


def rope_foot_touch(env) -> torch.Tensor:
  """(N,) J's foot and the rope in physical contact (dist <= 0), read off the simulator's contact list (a contact
  sensor on rope bodies reported nothing; the list has the pairs: 426 foot-rope contacts in 16 envs x 8 s)."""
  import mujoco
  d, m = env.sim.data, env.sim.mj_model
  if "_ys_touch_ids" not in env.__dict__:
    gn = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    bn = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[i]) or ""
    feet = [i for i in range(m.ngeom) if gn(i) in ("robot/left_foot_collision", "robot/right_foot_collision")]
    rope = [i for i in range(m.ngeom) if bn(i).startswith("rope/rope_")]
    env._ys_touch_ids = (torch.tensor(feet, device=env.device), torch.tensor(rope, device=env.device))
  feet, rope = env._ys_touch_ids
  g = d.contact.geom[:].long(); w = d.contact.worldid[:].long(); dist = d.contact.dist[:]
  live = torch.arange(g.shape[0], device=g.device) < d.nacon[:][0]  # everything stays on the GPU (no host sync)
  isf, isr = torch.isin(g, feet), torch.isin(g, rope)
  pair = ((isf[:, 0] & isr[:, 1]) | (isf[:, 1] & isr[:, 0])) & (dist <= 0) & live
  out = torch.zeros(env.num_envs, device=env.device).scatter_reduce_(0, w.clamp(0, env.num_envs - 1), pair.float(), reduce="amax")
  return out > 0


def rope_hit_real(env) -> torch.Tensor:
  return rope_foot_touch(env) | rope_leg_pass(env)


# The user's rule (2026-11-07 night, "middle"): caught = a real touch (foot contact or leg pass) lasting >= 60 ms, or
# the rope held back > 30 deg within 0.5 s of a touch; a brief brush is allowed. Lag = the rope's middle-link angle
# minus the clock (rope_phase_obs) against its quiet pattern ROPE_REF (per 10-deg clock bin; set by the turners: N2Cf2
# and N2Af runs agree within 3.4 deg) and the env's own offset (running mean over quiet steps: _ys_delta moves it).
ROPE_REF = torch.tensor([-0.0871, 0.1829, 0.4355, 0.6508, 0.8393, 0.9842, 1.0824, 1.1496, 1.1848, 1.2271, 1.2601, 1.2395,
                         1.1836, 1.0318, 0.8050, 0.5412, 0.2295, -0.0781, -0.3880, -0.6580, -0.8896, -1.1058, -1.3264, -1.4981,
                         -1.6188, -1.6974, -1.7398, -1.7275, -1.6682, -1.5439, -1.3938, -1.2354, -1.0317, -0.8151, -0.5823, -0.3369])
HIT_RULE = __import__("os").environ.get("YS_HIT_RULE", "proxy")  # "caught": the rule above in training and evals
# CAUGHT_LAG: the user's "held back > 30 deg" was inside the rope's own wobble: with nothing touching, a 0.5 s window
# peaks above 30 deg 49-63% of the time (after a 20-40 ms brush 71-74%), so brushes counted as catches and their
# random 3.0/step taught N3Cf to brush more (1-step touches 42 -> 60%, in place >= 20 passes 25 -> 9%). 90 deg is
# past the quiet window's 97-99th percentile (1-3%; after a >= 60 ms touch 21-26%): only a real drag counts.
CAUGHT_STEPS, CAUGHT_LAG, QUIET_S = 3, math.radians(float(__import__("os").environ.get("YS_CAUGHT_LAG", "90"))), 0.5


def _wrap(a):
  return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def rope_state(env) -> dict:
  """once per env step: touch, its run length (steps), s since the last touch, rope lag (rad), caught."""
  n, dev = env.num_envs, env.device
  st = env.__dict__.get("_ys_hit")
  if st is None:
    st = env._ys_hit = dict(step=-1, run=torch.zeros(n, dtype=torch.long, device=dev), since=torch.full((n,), 9.0, device=dev),
                            off=torch.zeros(n, device=dev), nq=torch.zeros(n, device=dev), lag=torch.zeros(n, device=dev),
                            touch=torch.zeros(n, dtype=torch.bool, device=dev), caught=torch.zeros(n, dtype=torch.bool, device=dev))
  if st["step"] == env.common_step_counter:
    return st
  st["step"] = env.common_step_counter
  fresh = env.episode_length_buf <= 1  # torch.where, not masked writes: no host sync
  st["run"] = torch.where(fresh, 0, st["run"]); st["since"] = torch.where(fresh, 9.0, st["since"])
  st["nq"] = torch.where(fresh, 0.0, st["nq"]); st["off"] = torch.where(fresh, 0.0, st["off"])
  touch = rope_hit_real(env)
  st["run"] = torch.where(touch, st["run"] + 1, 0); st["since"] = torch.where(touch, 0.0, st["since"] + env.step_dt)
  cmd = env.command_manager.get_term("head_pose")
  c, sn = rope_phase_obs(env).unbind(-1); rel = torch.atan2(sn, c)
  b = (torch.remainder(cmd.phase, 2 * math.pi) / math.radians(10)).long().clamp(0, 35)
  d = _wrap(rel - ROPE_REF.to(dev)[b])
  quiet = (st["since"] > QUIET_S) & (env.episode_length_buf * env.step_dt > 3.0)
  st["nq"] = st["nq"] + quiet.float(); a = torch.where(quiet, (1.0 / st["nq"].clamp(min=1)).clamp(min=0.02), 0.0)
  st["off"] = _wrap(st["off"] + a * _wrap(d - st["off"]))
  st["lag"] = _wrap(d - st["off"]).abs() * (st["nq"] > 20)
  st["touch"] = touch
  st["caught"] = (st["since"] < QUIET_S) & ((st["run"] >= CAUGHT_STEPS) | (st["lag"] > CAUGHT_LAG))
  return st


def rope_caught(env) -> torch.Tensor:
  return rope_state(env)["caught"]


def rope_bad(env) -> torch.Tensor:
  """the hit test the entry machinery uses: the user's rule with YS_HIT_RULE=caught / hybrid, else the old proxy."""
  return rope_caught(env) if HIT_RULE in ("caught", "hybrid") else rope_overlap(env) > 0


def rope_caught_cost(env) -> torch.Tensor:
  """(>= 0, weight < 0) hybrid: 3.0 per caught step on top of the proxy's dense shaping (spin-up grace kept)."""
  return 3.0 * rope_caught(env).float() * (env.episode_length_buf > ROPE_GRACE_STEPS)


def rope_hit_cost(env) -> torch.Tensor:
  """(>= 0, weight < 0) training penalty for the user's rule: 0.3 per step of a brief touch (allowed but not free),
  3.0 per step once caught. rope_overlap's spin-up grace kept."""
  st = rope_state(env)
  return (0.3 * st["touch"].float() + 2.7 * st["caught"].float()) * (env.episode_length_buf > ROPE_GRACE_STEPS)


import os as _os
_E2 = _os.environ.get("YS_ENTER_V", "2") == "2"
# E1: waiting place U[0, 0.25] m widened over 200 iterations, set-off at leader phase 120-200 deg, 4/s, 0.6 m/s: no falls,
# arrival 96-100% (0.58-0.90 s from 12+ cm) but 62-76% hit on the way: the rope sweeps the path at foot level from +0.09
# to -0.19 m and the clear gap is ~0.5 s. E2 (YS_ENTER_V=2): wait 12-20 cm away (the rope reaches ~+0.09-0.13 m at foot
# level), set off at 140-170 deg, 6/s up to 0.8 m/s; 20% of episodes start at the spot (keeps the in-place jump).
ENTER_Y, ENTER_RAMP = (0.20, 0) if _E2 else (0.25, 200)  # m: farthest waiting place (+y); iterations to widen U[.., y] to it
ENTER_RAMP = int(_os.environ.get("YS_ENTER_RAMP", ENTER_RAMP))  # 7-12 A/B: facing -y while waiting is new
ENTER_YMIN, ENTER_NEAR_P = (0.12, 0.2) if _E2 else (0.0, 0.0)  # far waits from ENTER_YMIN; share of episodes at the spot
ENTER_NEAR_P = float(_os.environ.get("YS_ENTER_NEAR_P", ENTER_NEAR_P))  # E4: 1.0, J at its spot from the reset (stay_g1 anchor only)
ENTER_PH = tuple(math.radians(v) for v in ((140, 170) if _E2 else (120, 200)))  # leader phase to set off
ENTER_GAIN, ENTER_VMAX, ENTER_NEAR = (6.0, 0.8, 0.03) if _E2 else (4.0, 0.6, 0.03)  # 1/s, m/s, m: walk-in command
# 7-12 dash (user 2026-11-07: dart into the TURNING rope): YS_ENTER_DASH=v -> full v until ENTER_NEAR (E3b's P command arrived
# 0.46-0.50 s = when the rope comes down on the entry side; its gait peaks at 0.52 m/s), YS_ENTER_PH="lo,hi" deg set-off window
ENTER_DASH = float(_os.environ.get("YS_ENTER_DASH", 0))  # gain = v / distance: |v| = v toward the spot (a flat gain of 100 also
# slammed the x axis to the cap on 1 cm errors: 62-68% falls)
if _os.environ.get("YS_ENTER_PH"): ENTER_PH = tuple(math.radians(float(v)) for v in _os.environ["YS_ENTER_PH"].split(","))
# 7-12 entry scenarios (user 2026-11-07, YS_ENTER_MODE): side (C) = side-run facing L (E1-E6); turn (A) = wait and run in
# facing the spot (-y), turn to face L over the last ENTER_TURN_D (90 deg at 3 rad/s = 0.5 s: most of the run), side-jump; fwd (B) = run in facing -y and jump the first
# FWD_PASSES passes forward at FWD_Y (the rope lies 80-180 deg just behind the heels: one two-foot jump ~190 deg), then
# turn to face L at FWD_TURN_RATE (90 deg in 2.6 s, ~3 passes) while the spot slides to 0 (door map: logs/ys_trio/media/ENTER_door.png)
ENTER_MODE = _os.environ.get("YS_ENTER_MODE", "side")
# N1 (user 2026-11-07: a fresh J that learns entry + jumping, all three ways): mix = each entry draws side / turn / fwd
MODES = {"side": 0, "turn": 1, "fwd": 2}


def _mode(n: int, dev) -> torch.Tensor:
  return torch.randint(0, 3, (n,), device=dev) if ENTER_MODE == "mix" else torch.full((n,), MODES[ENTER_MODE], device=dev)


# 7-12 entry side (user 2026-11-07 evening: where you come in from changes the odds), YS_ENTER_SIDE: "+" = J's left (the rope
# comes over the top toward J, down in front of it at 0-60 deg, then away along the floor: front door, waits 12-20 cm);
# "-" = the far side (the rope sweeps along the floor TOWARD J out to -28 cm at 180-260 deg and rises in front of it: back
# door; J's spot is 8 cm off the rope axis toward +y, so J waits 36-44 cm out and sets off as the rope rises past it,
# 250-290 deg: ~0.62 s until it sweeps the floor again at 180); "mix" = per entry
ENTER_SIDE = _os.environ.get("YS_ENTER_SIDE", "+")
ENTER_YNEG = tuple(float(v) for v in _os.environ.get("YS_ENTER_YNEG", "0.36,0.44").split(","))
ENTER_PH_NEG = tuple(math.radians(float(v)) for v in _os.environ.get("YS_ENTER_PH_NEG", "250,290").split(","))
# 7-12 B (user 2026-11-07: B jumps its first passes FORWARD on two feet): the side jump's per-foot windows (left, then
# right 160 ms later) also paid B facing forward, so N1c's B lifted one foot at the sweep (98% hit at ~185 deg, right
# sole 41 mm up, left down). YS_B_TWOFOOT=1: while B faces forward both feet share the right foot's window.
B_TWOFOOT = _os.environ.get("YS_B_TWOFOOT", "0") == "1"
ENTER_TURN_D, FWD_Y, FWD_PASSES, FWD_TURN_RATE, TURN_GAIN, TURN_WMAX = 0.12, -0.05, 3, 0.6, 6.0, 3.0  # m, m, -, rad/s, 1/s, rad/s
FOOT_HX = 0.027  # m: sole half-length (J._CORNERS)


def _yaw(rob) -> torch.Tensor:
  w, x, y, z = rob.data.root_link_quat_w.unbind(-1)
  return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _face(st: torch.Tensor, turn: torch.Tensor) -> None:
  """root states (n, 13) turned by yaw turn (n,) rad about z, in place."""
  h = 0.5 * turn; qw, qz = h.cos(), h.sin(); w, x, y, z = st[:, 3:7].unbind(-1)
  st[:, 3:7] = torch.stack([qw * w - qz * z, qw * x - qz * y, qw * y + qz * x, qw * z + qz * w], -1)


ENTER_WAIT = (2.5, 4.0)  # s: earliest set-off (the rope has spun up: ROPE_GRACE_STEPS 2 s)


def _draw(e, ids, near, ymax, dev) -> None:
  """one entry's draw: side (+1 / -1), signed waiting place y0, set-off phase, scenario, facing (A/B face the spot)."""
  u = lambda lo, hi: torch.empty(len(ids), device=dev).uniform_(lo, hi)
  sd = torch.full((len(ids),), -1.0 if ENTER_SIDE == "-" else 1.0, device=dev)
  if ENTER_SIDE == "mix": sd = torch.where(torch.rand(len(ids), device=dev) < 0.5, 1.0, -1.0)
  sd = torch.where(near, 1.0, sd); e["side"][ids] = sd
  e["y0"][ids] = torch.where(near, u(0.0, 0.03), torch.where(sd > 0, u(min(ENTER_YMIN, ymax), ymax), -u(*ENTER_YNEG)))
  e["ph"][ids] = torch.where(sd > 0, u(*ENTER_PH), u(*ENTER_PH_NEG))
  e["mode"][ids] = _mode(len(ids), dev); e["fw"][ids] = (e["mode"][ids] > 0) & ~near
  e["d_prev"][ids] = e["y0"][ids].abs()


def enter_reset(env, env_ids: torch.Tensor | None, full: bool = False) -> None:
  """7-9 reset event (after reset_base): the jumper stands y0 ~ U[0, ymax] beside its spot and waits there (its
  centering command holds it) until enter_hook sends it in. User 2026-11-06: walk in later; this is that skill."""
  n, dev = env.num_envs, env.device
  ids = torch.arange(n, device=dev) if env_ids is None else env_ids
  if not hasattr(env, "_ys_enter"):
    z = lambda: torch.zeros(n, device=dev)
    env._ys_enter = {"y0": z(), "t": z(), "ph": z(), "t_near": z(), "go": z().bool(), "near": z().bool(), "t_go": z(),
                     "d_prev": z(), "pk": z().long(), "bad": z().bool(), "t_hit": z(), "fw": z().bool(), "t_turn": z(), "ent": z().bool(), "mode": z().long(), "d": z(), "side": z()}
    env._ys_center_hook = enter_hook
  start = env.__dict__.setdefault("_ys_enter_start", env.common_step_counter)
  ymax = ENTER_Y if (full or ENTER_RAMP == 0) else ENTER_Y * min(1.0, max(0.1, (env.common_step_counter - start) / (ENTER_RAMP * 24)))
  u = lambda lo, hi: torch.empty(len(ids), device=dev).uniform_(lo, hi)
  e, rob = env._ys_enter, env.scene["robot"]
  near = torch.rand(len(ids), device=dev) < ENTER_NEAR_P
  _draw(e, ids, near, ymax, dev); e["t"][ids] = u(*ENTER_WAIT)
  e["go"][ids], e["near"][ids], e["t_near"][ids], e["t_go"][ids], e["pk"][ids], e["bad"][ids] = False, False, 0.0, torch.inf, 0, False
  e["t_hit"][ids] = torch.inf
  e["ent"][ids] = ~near; e["t_turn"][ids] = torch.inf
  # reset_base's pose again from the default state (its write is not in root_link_pos_w until the next forward: reading
  # that gave the previous episode's fallen pose, smoke test 63/64 falls per iteration), plus y0
  pr = env.event_manager.get_term_cfg("reset_base").params["pose_range"]
  st = rob.data.default_root_state[ids].clone(); st[:, 7:] = 0.0
  st[:, :3] += env.scene.env_origins[ids] + torch.stack([u(*pr.get(k, (0.0, 0.0))) for k in ("x", "y", "z")], 1)
  st[:, 1] += e["y0"][ids]  # the jumper faces +x: its left (+y) is the side the rope comes down on, then sweeps away from
  _face(st, -0.5 * math.pi * e["side"][ids] * e["fw"][ids].float())  # A/B: facing the spot
  rob.write_root_state_to_sim(st, ids)


def enter_hook(cmd):
  """JumpClockCommand._center's spot / gain / speed cap: the waiting place until the set-off time and phase, then the
  spot with the walk-in command (ENTER_GAIN, ENTER_VMAX) until within ENTER_NEAR, then the usual centering."""
  env, e = cmd._env, cmd._env._ys_enter
  t = env.episode_length_buf * env.step_dt
  lead = cmd.phase - env.__dict__.get("_ys_delta", DELTA0)
  go = ~e["go"] & (t >= e["t"]) & (torch.remainder(lead - e["ph"], 2 * math.pi) < 2 * math.pi * cmd.freq.abs() * env.step_dt)
  e["go"] |= go; e["t_go"] = torch.where(go, t, e["t_go"])
  fw, h = e["fw"], -0.5 * math.pi * e["side"]  # A/B run-in heading: facing the spot
  fwA, fwB = fw & (e["mode"] == 1), fw & (e["mode"] == 2)
  # B: forward-facing until FWD_PASSES passes at FWD_Y, then a slow turn while the spot slides to 0
  e["t_turn"] = torch.where(fwB & e["near"] & e["t_turn"].isinf(), t + 0.8 * FWD_PASSES, e["t_turn"])
  yB = h * (1 - FWD_TURN_RATE * (t - e["t_turn"]).clamp(min=0) / (0.5 * math.pi)).clamp(min=0)
  sy = torch.where(fwB, FWD_Y * yB / h, 0.0)
  spot = torch.stack([torch.zeros_like(t), torch.where(e["go"], sy, e["y0"])], 1)
  d = (env.scene.env_origins[:, :2] + torch.stack([torch.zeros_like(t), sy], 1) - cmd._robot.data.root_link_pos_w[:, :2]).norm(dim=-1)
  e["d"] = d
  new = e["go"] & ~e["near"] & (d < ENTER_NEAR)
  e["t_near"] = torch.where(new, t, e["t_near"]); e["near"] |= new
  walk = e["go"] & ~e["near"]
  g = ENTER_DASH / d.clamp(min=ENTER_NEAR) if ENTER_DASH else ENTER_GAIN
  out = (spot, torch.where(walk, g, J.CENTER_GAIN), torch.where(walk, max(ENTER_VMAX, ENTER_DASH), 0.3))
  if ENTER_MODE == "side":
    return out
  # A: -y until the last ENTER_TURN_D (from ENTER_NEAR), then face L
  yref = torch.where(fwB, yB, torch.where(fwA & ~e["near"], h * ((d - ENTER_NEAR) / ENTER_TURN_D).clamp(0, 1), 0.0))
  dy = torch.remainder(yref - _yaw(cmd._robot) + math.pi, 2 * math.pi) - math.pi
  if B_TWOFOOT:  # B facing forward: both feet in one window (ys_jump_env_cfg.foot_window)
    env._ys_two_foot = fwB & (_yaw(cmd._robot).abs() > 0.3)
  wz = torch.where(fw & e["go"], (TURN_GAIN * dy).clamp(-TURN_WMAX, TURN_WMAX), (J.CENTER_GAIN * dy).clamp(-0.5, 0.5))  # waiting: the usual hold
  return (*out, wz)


ENTER_AGAIN_PASSES, ENTER_AGAIN_WAIT = int(_os.environ.get("YS_ENTER_AGAIN_PASSES", 3)), (1.0, 2.0)  # passes at the spot before J is put back out; s it waits there first


def enter_again(env, env_ids) -> None:
  """7-12 step event (user 2026-11-07: dart into the TURNING rope): E1-E3b had one walk-in per 19 s episode (~5% of the
  time, lesson 8). The rope keeps turning; J alone is put back to a new waiting place (standing, default joints) after
  ENTER_AGAIN_PASSES passes at its spot or 0.5 s after a hit, and walks in again: ~3-4 entries per episode. Episodes that
  started at the spot (the ENTER_NEAR_P draw) stay there. Sim-only training trick: the actor never sees it."""
  e, rob = env._ys_enter, env.scene["robot"]; t = env.episode_length_buf * env.step_dt
  new_hit = e["go"] & e["t_hit"].isinf() & (rope_bad(env) | (rob.data.projected_gravity_b[:, 2] > -0.5))
  e["t_hit"] = torch.where(new_hit, t, e["t_hit"])
  stay = 0.8 * ENTER_AGAIN_PASSES + (0.8 * (FWD_PASSES - 1) + 0.5 * math.pi / FWD_TURN_RATE) * (e["mode"] == 2)  # B: through its turn
  out = e["ent"] & e["go"] & ((e["near"] & (t - e["t_near"] > stay)) | (t - e["t_hit"] > 0.5))
  ids = out.nonzero().flatten()
  if len(ids) == 0:
    return
  u = lambda lo, hi: torch.empty(len(ids), device=env.device).uniform_(lo, hi)
  ymax = ENTER_Y if ENTER_RAMP == 0 else ENTER_Y * min(1.0, max(0.1, (env.common_step_counter - env._ys_enter_start) / (ENTER_RAMP * 24)))
  _draw(e, ids, torch.zeros(len(ids), dtype=torch.bool, device=env.device), ymax, env.device); e["t"][ids] = t[ids] + u(*ENTER_AGAIN_WAIT)
  e["go"][ids], e["near"][ids], e["t_near"][ids], e["t_go"][ids], e["pk"][ids], e["bad"][ids] = False, False, 0.0, torch.inf, 0, False
  e["t_hit"][ids], e["t_turn"][ids] = torch.inf, torch.inf
  st = rob.data.default_root_state[ids].clone(); st[:, 7:] = 0.0
  st[:, :3] += env.scene.env_origins[ids]; st[:, 1] += e["y0"][ids]
  _face(st, -0.5 * math.pi * e["side"][ids] * e["fw"][ids].float())
  rob.write_root_state_to_sim(st, ids)
  rob.write_joint_state_to_sim(rob.data.default_joint_pos[ids].clone(), torch.zeros_like(rob.data.default_joint_pos[ids]), env_ids=ids)
  env.sim.forward()  # this step's obs read the new pose (step events run after the step's forward)


ENTER_BANK = _os.environ.get("YS_ENTER_BANK")  # (2) (user 2026-11-07): the entry policy's arrival states (s10/bank_rec.py)
BANK_TRACK = ("_ys_cam", "_ys_cam_ball")  # the jumper's camera trackers (_cam_track state)


def bank_put(env, env_ids) -> None:
  """(2) step event, J at its spot from the reset (YS_ENTER_NEAR_P=1): the in-place jumper learns to take over from
  where the entry policy (N1c) leaves it. N1c entering then E5 from arrival: >=10 passes 0-2% though E5 in place does
  74%: the arrival states are not ones E5 has seen. Each env, once due, is put into a random recorded arrival state
  (root, joints, camera-tracker memory) when the leader phase reaches that state's own phase; due again
  ENTER_AGAIN_PASSES passes later or 0.5 s after a hit. Sim-only training trick: the actor never sees it."""
  e, rob = env._ys_enter, env.scene["robot"]; t = env.episode_length_buf * env.step_dt; dev = env.device
  if "_ys_bank" not in env.__dict__:
    env._ys_bank = torch.load(ENTER_BANK, map_location=dev, weights_only=False)
    env._ys_bank_i = torch.randint(0, env._ys_bank["ph"].shape[0], (env.num_envs,), device=dev)
    env._ys_bank_due = torch.empty(env.num_envs, device=dev).uniform_(*ENTER_WAIT)
  bk, due = env._ys_bank, env._ys_bank_due
  fresh = env.episode_length_buf <= 1
  due[fresh] = torch.empty(int(fresh.sum()), device=dev).uniform_(*ENTER_WAIT)
  hit = e["go"] & e["t_hit"].isinf() & (rope_bad(env) | (rob.data.projected_gravity_b[:, 2] > -0.5))
  e["t_hit"] = torch.where(hit, t, e["t_hit"]); due[:] = torch.where(hit, torch.minimum(due, t + 0.5), due)
  cmd = env.command_manager.get_term("head_pose")
  lead = cmd.phase - env.__dict__.get("_ys_delta", DELTA0)
  ph = bk["ph"][env._ys_bank_i]
  ids = ((t >= due) & (torch.remainder(lead - ph, 2 * math.pi) < 2 * math.pi * cmd.freq.abs() * env.step_dt)).nonzero().flatten()
  if len(ids) == 0:
    return
  i = env._ys_bank_i[ids]
  rob.write_root_state_to_sim(torch.cat([bk["pos"][i] + env.scene.env_origins[ids], bk["quat"][i], bk["lin"][i], bk["ang"][i]], 1), ids)
  rob.write_joint_state_to_sim(bk["q"][i].clone(), bk["qd"][i].clone(), env_ids=ids)
  for kf, v in bk.items():
    key, _, f = kf.partition(".")
    if f and key in env.__dict__ and f in env.__dict__[key]: env.__dict__[key][f][ids] = v[i].to(env.__dict__[key][f].dtype)
  e["go"][ids], e["near"][ids], e["t_go"][ids], e["t_near"][ids], e["pk"][ids], e["bad"][ids] = True, True, t[ids], t[ids], 0, False
  e["t_hit"][ids], e["fw"][ids], e["mode"][ids] = torch.inf, False, 0
  due[ids] = t[ids] + 0.8 * ENTER_AGAIN_PASSES + torch.rand(len(ids), device=dev)
  env._ys_bank_i[ids] = torch.randint(0, bk["ph"].shape[0], (len(ids),), device=dev)
  env.sim.forward()  # this step's obs read the new pose (step events run after the step's forward)


def rope_overlap_early(env, t0: float = 0.3, t1: float = 3.0) -> torch.Tensor:
  """7-11 E5: rope_overlap in the spin-up only (t0..t1 s after the reset): ~10% of an episode, so E4's plain
  overlap barely moved (-0.35 flat over 300 it) while J walking in first is hit there 92% (passes 1-3)."""
  t = env.episode_length_buf * env.step_dt
  return rope_overlap(env) * ((t >= t0) & (t < t1))


def enter_approach(env, v_full: float = 0.5) -> torch.Tensor:
  """7-9 E3 (user (a): hurry in): speed toward the spot (m/s, / v_full, clamped 0..1) from set-off until arrival: E1/E2
  took 0.50-0.58 s from 12-20 cm and the rope is back at the feet ~0.5 s after the set-off."""
  e = env._ys_enter; d = (env.scene.env_origins[:, :2] - env.scene["robot"].data.root_link_pos_w[:, :2]).norm(dim=-1)
  v = (e["d_prev"] - d) / env.step_dt; e["d_prev"] = d
  return (v / v_full).clamp(0, 1) * (e["go"] & ~e["near"])


def enter_clear(env, n_pass: int = 2) -> torch.Tensor:
  """7-9 E3 (user (a)): +1/dt at the end of each of the first n_pass 0.8 s windows after the set-off with no rope over a
  sole and the jumper up (hit = end: the first passes are the ones E1/E2 lost, 87-96% at the first pass)."""
  e = env._ys_enter; t = env.episode_length_buf * env.step_dt
  on = e["go"] & (e["pk"] < n_pass)
  e["bad"] |= on & (rope_bad(env) | (env.scene["robot"].data.projected_gravity_b[:, 2] > -0.5))
  end = on & (t >= e["t_go"] + 0.8 * (e["pk"] + 1).float())
  r = (end & ~e["bad"]).float() / env.step_dt
  e["pk"] = torch.where(end, e["pk"] + 1, e["pk"]); e["bad"] = torch.where(end, False, e["bad"])
  e["pk"] = torch.where(end & (r == 0), n_pass, e["pk"])  # a hit ends the bonuses for that episode
  return r


def enter_late(env, after: float = 0.6) -> torch.Tensor:
  """N1 cost >= 0 (negative weight): cm J stands off its spot (beyond ENTER_NEAR) from `after` s past the set-off on,
  arrived or not. N1@3000 (A/B) backed away from the turning rope against a +0.8 m/s command (19 -> 37 cm in 3 s):
  outside the rope its jump rewards cost nothing. At -1/cm staying out outweighs a hit (rope_overlap -3/cm, a few steps)."""
  e, t = env._ys_enter, env.episode_length_buf * env.step_dt
  after = after + (e["y0"].abs() - 0.20).clamp(min=0) / 0.5  # the far side's 36-44 cm run: ~1 s
  return (e["d"] - ENTER_NEAR).clamp(min=0) / 0.01 * (e["go"] & (t - e["t_go"] > after))


def stay_g1(env, ckpt: str, settle: float = 1.0, waiting: bool = False) -> torch.Tensor:
  """7-9 cost >= 0 (negative weight): squared gap between the jumper's actions and the frozen accepted jumper's on the
  same obs, once it has been at its spot settle s (user: keep the trained motions; only the walk-in is new)."""
  if "_ys_g1" not in env.__dict__:
    env._ys_g1 = load_frozen_actor(ckpt, env.device)
  e, t = env._ys_enter, env.episode_length_buf * env.step_dt
  on = e["near"] & (t - e["t_near"] > settle)
  if ENTER_MODE != "side": on &= _yaw(env.scene["robot"]).abs() < 0.2  # A/B: E5's motion is the side jump
  if waiting: on |= ~e["go"] & (t > 1.0)  # E3 (user (b)): also while it hops in place at the waiting place
  with torch.no_grad():
    ref = env._ys_g1(env.obs_buf["actor"])
  return (env.action_manager.action - ref).square().mean(-1) * on


# 7-12 duck (user 2026-11-07 night: "what if J comes in with its head down?"), YS_DUCK=1: while running in (set off,
# not at the spot yet) the head-up / gaze / head-forward terms are off and `duck` pays the camera pitched DUCK_PITCH
# down. Only the feet meet the rope in sim, so any gain is through the body (the head is ~38% of the mass: a lower
# centre of mass for the dash); on the real robot it also keeps the head under the rope. J times the run on its clock.
DUCK = _os.environ.get("YS_DUCK", "0") == "1"
DUCK_PITCH = math.radians(float(_os.environ.get("YS_DUCK_PITCH", "-35")))


def _entering(env) -> torch.Tensor:
  e = env.__dict__.get("_ys_enter")
  return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device) if e is None else e["go"] & ~e["near"]


def cam_pitch(env) -> torch.Tensor:
  """(N,) head camera axis pitch, rad (up +), as J.head_up measures it."""
  rob = env.scene["robot"]
  if not hasattr(env, "_ys_cam_site"):
    env._ys_cam_site = rob.find_sites(["head_camera"])[0][0]
  w, x, y, z = rob.data.site_quat_w[:, env._ys_cam_site].unbind(-1)
  return torch.asin((2 * (x * z - w * y)).clamp(-1, 1))


def duck(env, std: float = 0.5) -> torch.Tensor:  # wide as head_up (pitch_std 0.5): J looks level-up now, a 0.15 Gaussian at -35 deg pays ~0
  return torch.exp(-((cam_pitch(env) - DUCK_PITCH) / std).square()) * _entering(env)


def head_up_free(env, **kw) -> torch.Tensor:
  return J.head_up(env, **kw) * ~_entering(env)


def gaze_free(env, **kw) -> torch.Tensor:
  return gaze(env, **kw) * ~_entering(env)


def head_back_free(env, **kw) -> torch.Tensor:
  return head_back(env, **kw) * ~_entering(env)


# (3) distillation (user 2026-11-07 evening): one student (from N1c), PPO + behavior cloning, routed per frame. Settled
# at the spot facing L (>= BC_SETTLE s after arriving): taught by E5 (its clock slots zeroed: E5 is clock-free). Waiting or
# running in: anchored to N1c, the student's start (velstand run-3 lesson, distill.py: BC on one slice rewrites the
# shared weights unless the rest is anchored). The first BC_SETTLE s at the spot, and B facing -y, belong to PPO alone.
BC_SETTLE = 1.0


def bc_gate(env) -> torch.Tensor:
  """(N, 2) [E5 teaches, N1c anchors]: sim-only routing for PpoWithJumpBc in its own obs group ("bc": stored in the
  rollout, never an actor or critic input)."""
  e = env.__dict__.get("_ys_enter")
  if e is None:
    return torch.zeros(env.num_envs, 2, device=env.device)
  t = env.episode_length_buf * env.step_dt
  teach = e["go"] & e["near"] & (t - e["t_near"] > BC_SETTLE) & (_yaw(env.scene["robot"]).abs() < 0.3)
  return torch.stack([teach, ~e["go"] | ~e["near"]], -1).float()


from mjlab_microduck.tasks.distill import PpoWithExpertBc, PpoWithExpertBcCfg, _with_flat, default_bc_cfg  # noqa: E402


class PpoWithJumpBc(PpoWithExpertBc):
  """(3) PpoWithExpertBc routed by the "bc" obs group (bc_gate) instead of the tilt: expert = E5 on settled frames
  (clock_slice zeroed in its input), anchor = N1c on entry frames, everything else PPO only."""

  def _bc_update(self) -> dict[str, float]:
    cfg = self.bc_cfg
    obs_td = self.storage.observations.flatten(0, 1)
    gate = obs_td["bc"]; teach, anchor = gate[:, 0] > 0.5, gate[:, 1] > 0.5
    stats = {"jump_bc_teach_frac": teach.float().mean().item(), "jump_bc_anchor_frac": anchor.float().mean().item()}
    if teach.sum().item() < cfg["min_samples"]:
      teach = torch.zeros_like(teach)
    idx = (teach | (anchor & (self.anchor is not None))).nonzero().flatten()
    if idx.numel() == 0:
      stats["expert_bc"] = 0.0
      return stats
    groups = list(self.actor.obs_groups)
    obs_td, teach = obs_td[idx], teach[idx]
    weight = torch.where(teach, float(cfg["coef"]), float(cfg.get("anchor_coef", 0.0)))
    with torch.no_grad():
      target = torch.zeros(idx.numel(), self.storage.actions.shape[-1], device=self.device)
      if teach.any():
        td = obs_td[teach].clone(); flat = torch.cat([td[g] for g in groups], -1).clone()
        lo, hi = cfg["clock_slice"]; flat[:, lo:hi] = 0.0
        target[teach] = self.expert(_with_flat(td, groups, flat))
      if (~teach).any():
        target[~teach] = self.anchor(obs_td[~teach])
    n = idx.numel(); mb = max(cfg.get("min_mini_batch", 1), n // cfg["mini_batches"]); total, steps = 0.0, 0
    for _ in range(cfg["epochs"]):
      perm = torch.randperm(n, device=self.device)
      for st in range(0, n, mb):
        sel = perm[st:st + mb]
        loss = (weight[sel] * (self.actor(obs_td[sel]) - target[sel]).pow(2).mean(dim=1)).mean()
        self.bc_optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm); self.bc_optimizer.step()
        total += loss.item(); steps += 1
    stats["expert_bc"] = total / max(steps, 1)
    return stats


@dataclass
class PpoWithJumpBcCfg(PpoWithExpertBcCfg):
  class_name: str = "mjlab_microduck.tasks.ys_trio_env_cfg.PpoWithJumpBc"


def make_ys_trio_jump_env_cfg(play: bool = False, delta_half: float = 0.0, rope_obs: bool = False, cam_obs: bool = False, rope_windows: bool = False, no_clock: bool = False, head_w: float = 0.0, head_tol: float = 0.015, clock_fade: int = CLOCK_FADE_STEPS, ball_obs: bool = False, cam_age: bool = False, follower: str | None = None, follower_off: float = math.radians(60), gaze_w: float = 0.0, beat: bool = False, follower_cam: bool = False, beat_start: str = "ear", enter: str | None = None, enter_w: float = 1.0, pick: bool = False, enter3: bool = False, bc: bool = False):
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
  if pick:  # 7-10: the rope ends wait in holders (mocaps placed by the scenario script) until the turners take them
    cfg.scene.entities.update({"hold_p": EntityCfg(spec_fn=lambda: holder_spec("hold_p")), "hold_t": EntityCfg(spec_fn=lambda: holder_spec("hold_t"))})
    cfg.scene.spec_fn = connect_trio_pick
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
    if beat and beat_start == "ear":  # both turners start on the jumper's start sound: a few deg off, not +-180
      cfg.events["sample_offset"] = EventTermCfg(func=B.start_by_ear, mode="reset")

  for who in ("turner", "partner"):  # a turner down: end as a timeout (no fall penalty for the jumper)
    fell = copy.deepcopy(cfg.terminations["fell_over"])
    fell.params = {**fell.params, "asset_cfg": SceneEntityCfg(who)}
    fell.time_out = True
    cfg.terminations[f"{who}_fell"] = fell
  cfg.rewards["rope_overlap"] = RewardTermCfg(func=rope_overlap, weight=-3.0)
  if HIT_RULE == "caught":  # the user's rule (rope_state) replaces the proxy penalty
    del cfg.rewards["rope_overlap"]; cfg.rewards["rope_hit"] = RewardTermCfg(func=rope_hit_cost, weight=-3.0)
  elif HIT_RULE == "hybrid":  # N5: trained with the user's rule alone (N3Cf, N4Cf) in place >= 20 passes fell to 13 / 2%
    # (N2Cf2 on the proxy: 30%, same eval): its 0.3 / 3.0 per step was much sparser than the proxy's cm-graded cost
    # (episode -0.19 vs -0.5). Keep the proxy as dense shaping, add the caught cost; evals use the user's rule.
    cfg.rewards["rope_caught"] = RewardTermCfg(func=rope_caught_cost, weight=-3.0)
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
  if enter:  # 7-9 walk-in, warm from the accepted jumper `enter` (its actions anchor the in-place motion)
    cfg.events["enter_reset"] = EventTermCfg(func=enter_reset, mode="reset", params={"full": play})
    if enter_w:  # N1 (YS_ENTER_W3=0): no anchor, a fresh J
      cfg.rewards["stay_g1"] = RewardTermCfg(func=stay_g1, weight=-enter_w, params={"ckpt": enter, "waiting": enter3 and _os.environ.get("YS_ENTER3_WAIT", "0") == "1"})
    # E3 (anchored while waiting too, from G1): falls 5x E1's, episode length stuck ~300 by 250 iterations (G1 falls 77%
    # while waiting, so anchoring there anchors a faller); E3b: from E1@13199, anchor at the spot only
    if float(_os.environ.get("YS_EARLY_W", "0")) > 0:  # E5: the spin-up's overlap, extra (x YS_EARLY_W of rope_overlap's -3)
      cfg.rewards["rope_overlap_early"] = RewardTermCfg(func=rope_overlap_early, weight=-3.0 * float(_os.environ["YS_EARLY_W"]))
    if ENTER_BANK:  # (2): put into the entry policy's arrival states
      cfg.events["bank_put"] = EventTermCfg(func=bank_put, mode="step")
    if _os.environ.get("YS_ENTER_AGAIN") == "1":  # E6: several walk-ins per episode
      cfg.events["enter_again"] = EventTermCfg(func=enter_again, mode="step")
    if enter3:  # E3 (user 2026-11-06 (a)(b)(c)): hurry in, clear the first passes, anchored harder (enter_w 3)
      cfg.rewards["enter_approach"] = RewardTermCfg(func=enter_approach, weight=2.0)
      cfg.rewards["enter_clear"] = RewardTermCfg(func=enter_clear, weight=3.0)
      if float(_os.environ.get("YS_ENTER_LATE_W", "0")) > 0:  # N1b: staying out of the rope costs
        cfg.rewards["enter_late"] = RewardTermCfg(func=enter_late, weight=-float(_os.environ["YS_ENTER_LATE_W"]))
  if enter and DUCK:  # head down while running in
    for k, f in (("head_up", head_up_free), ("gaze", gaze_free), ("head_back", head_back_free)):
      if k in cfg.rewards: cfg.rewards[k].func = f
    cfg.rewards["duck"] = RewardTermCfg(func=duck, weight=2.0)
  if bc:  # (3): the BC routing (sim-only obs group, see bc_gate)
    from mjlab.managers import ObservationGroupCfg
    cfg.observations["bc"] = ObservationGroupCfg(terms={"gate": ObservationTermCfg(func=bc_gate)}, concatenate_terms=True, enable_corruption=False)
  return cfg


YsTrioJumpRlCfg = replace(MicroduckRlCfg, experiment_name="ys_trio_jump")
_alg = MicroduckRlCfg.algorithm  # (3): the same PPO, + routed BC (expert E5, anchor N1c: YS_BC_EXPERT / YS_BC_ANCHOR)
YsTrioJumpBcRlCfg = replace(YsTrioJumpRlCfg, algorithm=PpoWithJumpBcCfg(
  **{f: getattr(_alg, f) for f in _alg.__dataclass_fields__ if f not in ("class_name", "bc_cfg")},
  bc_cfg={**default_bc_cfg(), "checkpoint_path": _os.environ.get("YS_BC_EXPERT", "logs/rsl_rl/ys_trio_jump/2026-11-07_01-53-39_E5/model_13199.pt"),
          "anchor_checkpoint_path": _os.environ.get("YS_BC_ANCHOR"), "coef": 1.0, "anchor_coef": float(_os.environ.get("YS_BC_ANCHOR_W", "0.5")),
          "clock_slice": (51, 55)}))
