"""ys_trio_scn.py <jumper ckpt> [starts deg 60,120,180 | ear] : the J/F/L scenario on Mjlab-YsTrioJumpBeat (128 envs, 30 s).
The jumper waits 0.8 m off to the side (its fall terminations off) while the two turners start; at T_ENTER it is put
in its spot. All three run the start sound's clock; the jumper calls at ys_beat.BEEP_PH on it once in, and the
follower's clock is dragged by ear only with BEAT_CORRECT=1 (v1: F1 camera before, left-landing beeps after). Reports
per start offset X (random sign):
  before (t 4-6): follower MOTION error |gap - ref| (stem-end angles about EMA centres, ref = X 0 run), camera valid %
  after: clock-error converge time after entry, motion error p50/p90 (t >= 16), clean (t >= 16), falls, camera valid %.
Env knobs: BEAT_MISS/BEAT_FALSE/BEAT_CALERR/BEAT_AFTER/BEAT_CORRECT (ys_beat), BEAT_N, SEED, BEAT_DRIFT, BEAT_SLIP;
YS_BEAT_FOLLOWER/YS_BEAT_CAM choose the follower.
Hit = end (user 2026-11-06): STREAK = rope passes (0.8 s windows from entry) before the first overlap > 0 or a
jumper fall; a turner restart censors. BEAT_ENTER_PH="deg,..." (leader phase at the put-in; default ENTER_PH) sweeps
the entry; BEAT_NOREF=1 skips the start-0 reference run. Also prints the leader phase at J's foot events.
BEAT_WALKIN="y0,ph" (with BEAT_JSTAND): a stand-in walk-in, see WALKIN. BEAT_JSTAND=<onnx>: J switches to that standing policy at the stop. BEAT_STOP="delay,ramp,force": after J's first hit (+ delay s, J's call to the turners' ears) both turners' circle
command ramps to 0 over ramp s and stays there (env._ys_amp); envs not hit by force s stop then. Reports L/F falls and
the rope's speed after the stop, and the first 5 s of it per env.
BEAT_LOAD=1: per-joint actuator torque of J / L / F from J's entry + 0.5 s (p99, max, rms vs the BAM xl330 stall).
Also importable: away(env), enter(env) for videos."""
import sys, math, os; sys.path.insert(0, "scripts")
import numpy as np, torch
T_ENTER, AWAY, EPISODE = 6.0, (0.0, 0.8, 0.0), float(os.environ.get("BEAT_EPISODE", "30"))  # s, m, s (trained at 20 s)
WALKIN = tuple(float(v) for v in os.environ["BEAT_WALKIN"].split(",")) if os.environ.get("BEAT_WALKIN") else None
# BEAT_WALKIN="y0,ph": J waits standing (BEAT_JSTAND policy, not pinned) y0 m to the side and at leader phase ph (deg)
# from T_ENTER switches to its jumper policy, whose centering command (twist = 2/s x offset to the spot) brings it in
if WALKIN: AWAY = (0.0, WALKIN[0], 0.0)
OMEGA_DT = 2 * math.pi * 1.25 * 0.02  # rad of rope per step
ENTER_PH = math.radians(225)  # rad, leader phase at the put-in: the rope just passed (scn_v2_sweep.eval: standing
# put-in, 1st pass hit 63% here vs 89-100% elsewhere; was 154, the left landing: 100%)
TRACK = ("_ys_cam", "_ys_cam_ball")  # ys_trio_env_cfg._cam_track state of the jumper's head / ball trackers
INPLACE = os.environ.get("BEAT_INPLACE") == "1"  # the jumper jumps in its spot from t 0 (training-like): the 'entry' only starts the count
BANK_OUT, BANK_IN = os.environ.get("BEAT_BANK_OUT"), os.environ.get("BEAT_BANK_IN")  # INPLACE: save its state at the
# entry; else: put it in with that state (already jumping: a stand-in for walking in) instead of standing

LOAD = os.environ.get("BEAT_LOAD") == "1"
REFA = os.environ.get("BEAT_REF_ACTOR")  # a checkpoint whose actor J's actions are compared with (motion kept? RMS rad, in play)
PICK = os.environ.get("BEAT_PICK_ONNX")  # 7-10 (user: 2 (a), holders): the turners take the handles first, on this standing ONNX
P_BOW0, P_BOW1, P_GRASP, P_LIFT1, P_START, P_TOL = 1.0, 2.5, 3.0, 4.0, 4.5, 0.010  # s ..., m: bow, bite, lift, start call
WALKOUT = PICK and os.environ.get("BEAT_WALKOUT") == "1"  # user 2026-11-06: one holder in the middle, both handles level on it;
# L and F stand at it facing each other, bite, and walk backwards to their places in turn (no bow)
W_GAP, W_BITE, W_GAP_T, W_SET = 0.03, 0.8, 0.3, 0.6  # m stem end to the middle; s bite; s between L and F; s holder set
HOLD_Y = float(os.environ.get("BEAT_HOLD_Y", "-0.5"))  # m (env frame): the holder line, out of the work area (user): the
# rope sweeps y -0.28..+0.16 at J (axis y -0.08) and J waits at +0.8; L and F come in sideways from it
W_STILL, W_STILL_V, W_STILL_W, W_MIN_WAIT, J_AFTER = float(os.environ.get("BEAT_W_STILL", "1.0")), 0.02, 0.1, float(os.environ.get("BEAT_W_MINWAIT", "1.5")), 4.0  # s still, m/s, rad/s, s, s: J's start call
# once both have stood still W_STILL s (and >= W_MIN_WAIT after arriving); J comes in J_AFTER s after its call
# velstand has a dead band (|vx| <= 0.15 m/s: no steps) and walks slower than told: -0.4 -> -0.126 m/s backwards, then
# coasts 39 mm after the command stops (16 envs, flat); sideways 0.5 -> 0.081 m/s (0.3: 0.034), coast 11 mm.
# So: a fixed command until the coast distance, then stop.
W_VX, W_VY, W_STOP_F, W_STOP_L = 0.4, 0.5, 0.04, 0.03
W_WZ, W_TURN_TOL, W_BACK0, W_STOP_FWD = float(os.environ.get("BEAT_W_WZ", "1.0")), math.radians(5), 0.10, 0.025  # rad/s told (90 deg in 1.56 s, err 2-4 deg, 16 envs),
# rad done, m backed off the holder first, m it coasts walking forward (0.4 -> 0.42 m in 423 +- 12 mm)  # m/s told, m left when it stops (forward, sideways)
W_STAY = os.environ.get("BEAT_WALKOUT_STAY") == "1"  # control run: start at the places (bite, then the start call)
W_WAIT = float(os.environ.get("BEAT_WALKOUT_WAIT", "0.5"))  # s from F's arrival to the start call
W_RELAY = os.environ.get("BEAT_WALKOUT_RELAY") == "1"
W_CAM = os.environ.get("BEAT_WALKOUT_CAM") == "1"  # user 2026-11-06: stop on the other duck's ball size in the head camera
CAM_F, CAM_HZ, CAM_LAG, CAM_SR, CAM_SU, W_COAST = 640 / math.tan(math.radians(31)), 30.0, 4, 0.2, 1.0, 0.04
# px focal length (1280 px wide, 62 deg), frames/s, steps of latency, px radius / centre noise, m it coasts after stop  # diagnostic (not physical): at the start call the rope is laid as a fresh V
if PICK: T_ENTER = float(os.environ.get("BEAT_T_ENTER", "12.0" if WALKOUT else "9.0"))


def new_ws(env, n):
  """Walk-out state for n envs (+ the head-camera site in each turner's root frame, for W_CAM)."""
  from mjlab.utils.lab_api.math import quat_apply_inverse as qai
  dev, wh = env.device, ("turner", "partner")
  W = {"yaw0": {w: torch.zeros(n, device=dev) for w in wh}, "spot": {w: torch.zeros(n, 2, device=dev) for w in wh},
       "arr": {w: torch.zeros(n, dtype=torch.bool, device=dev) for w in wh}, "t_arr": {w: torch.zeros(n, device=dev) for w in wh},
       "r_t": {w: torch.zeros(n, device=dev) for w in wh}, "r_stop": {w: torch.zeros(n, device=dev) for w in wh},
       "cam": {w: torch.full((CAM_LAG + 1, n, 2), 1e3, device=dev) for w in wh}, "meas": {w: torch.full((n, 2), 1e3, device=dev) for w in wh},
       "ph": torch.zeros(n, dtype=torch.long, device=dev), "t_mark": torch.zeros(n, device=dev), "still": torch.zeros(n, device=dev),
       "t_start": torch.full((n,), torch.inf, device=dev), "hold_p": {w: torch.zeros(n, 2, device=dev) for w in wh}}
  W["rel_cam"] = {w: qai(env.scene[w].data.root_link_quat_w, env.scene[w].data.site_pos_w[:, env.scene[w].find_sites("head_camera")[0][0]] - env.scene[w].data.root_link_pos_w).median(0).values for w in wh}
  return W


def setup_walkout(env, ids, rel, W):
  """Walk-out (re)start for ids: remember each duck's place (its reset pose), stand it at the middle holder facing the
  other (stem ends W_GAP from the middle), the rope ends in the holder; W: per-env state."""
  from mjlab.utils.lab_api.math import quat_apply
  roots = {}
  for who in ("turner", "partner"):
    e = env.scene[who]; p, q = e.data.root_link_pos_w[ids].clone(), e.data.root_link_quat_w[ids].clone()
    w, x, y, z = q.unbind(-1); W["yaw0"][who][ids] = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)); W["spot"][who][ids] = p[:, :2]
    f = quat_apply(q, torch.tensor((1.0, 0.0, 0.0), device=env.device).expand(len(ids), 3)); f[:, 2] = 0.0  # toward the middle
    newp = p.clone()
    if not W_STAY: newp[:, 0] = env.scene.env_origins[ids, 0] - f[:, 0] * (W_GAP + float(rel[who][0])); newp[:, 1] = env.scene.env_origins[ids, 1] + HOLD_Y
    e.write_root_state_to_sim(torch.cat([newp, q, torch.zeros(len(ids), 6, device=env.device)], 1), ids)
    e.write_joint_state_to_sim(e.data.default_joint_pos[ids].clone(), torch.zeros_like(e.data.default_joint_vel[ids]), env_ids=ids)
    roots[who] = (newp, q)
  W["ph"][ids], W["t_mark"][ids], W["still"][ids], W["t_start"][ids] = 0, 0.0, 0.0, torch.inf
  for w in ("turner", "partner"): W["hold_p"][w][ids] = roots[w][0][:, :2]
  for who in ("turner", "partner"):
    W["arr"][who][ids] = False; W["t_arr"][who][ids] = torch.inf; W["meas"][who][ids] = 1e3; W["cam"][who][:, ids] = 1e3
  if W_CAM:  # ball size at the place: L's camera on F's ball while F still waits at the middle; F's on L's ball, L at its place
    import mjlab_microduck.tasks.ys_trio_env_cfg as T
    def at(who, rel_, pose): return pose[0] + quat_apply(pose[1], rel_.expand(len(ids), 3))
    spot_pose = {w: (torch.cat([W["spot"][w][ids], roots[w][0][:, 2:3]], 1), roots[w][1]) for w in ("turner", "partner")}
    side_pose = {w: (torch.cat([roots[w][0][:, :1], W["spot"][w][ids][:, 1:2], roots[w][0][:, 2:3]], 1), roots[w][1]) for w in ("turner", "partner")}  # after the sideways step
    for who, other, other_pose in (("partner", "turner", spot_pose["turner"]),):  # F's camera on L's ball, both at their places
      d_t = (at(other, rel[other], other_pose) - at(who, W["rel_cam"][who], spot_pose[who])).norm(dim=-1)
      W["r_t"][who][ids] = CAM_F * T.BALL_R / d_t; W["r_stop"][who][ids] = CAM_F * T.BALL_R / (d_t - W_COAST)
  if W_STAY:  # control: already at their places, the rope hanging between as at a reset (no walk, no rope on the floor)
    from mjlab.utils.lab_api.math import quat_apply as qa
    pts = [roots[w][0] + qa(roots[w][1], rel[w].expand(len(ids), 3)) for w in ("partner", "turner")]
    return place_holders(env, ids, rel, float((pts[1] - pts[0]).norm(dim=-1).median()), side=False, roots=roots)
  return place_holders(env, ids, rel, 2 * W_GAP, side=True, roots=roots)


def do_bite(env, bite, EQI, got, gerr):
  """The beaks close where bite: a rope end moves from its holder to the stem end if that is within P_TOL."""
  for c_, (who, hk) in enumerate((("partner", "hold_p"), ("turner", "hold_t"))):
    e = env.scene[who]; sid_ = e.find_sites("stem_end")[0][0]
    d_ = (e.data.site_pos_w[:, sid_] - env.sim.data.mocap_pos[:, env.scene[hk].indexing.mocap_id]).norm(dim=-1)
    ok_ = bite & (d_ < P_TOL); first = bite & gerr[:, c_].isnan(); gerr[first, c_] = 1000 * d_[first]; got[:, c_] |= ok_
    env.sim.data.eq_active[ok_, EQI[c_]] = True; env.sim.data.eq_active[ok_, EQI[c_ + 2]] = False


def ball_seen(env, who, other):
  """`who`'s head camera on `other`'s ball (stem end): (N,) apparent radius px and horizontal centre px (+ right), true geometry."""
  import mjlab_microduck.tasks.ys_trio_env_cfg as T
  from mjlab.utils.lab_api.math import quat_apply_inverse as qai
  e, o = env.scene[who], env.scene[other]
  c = e.data.site_pos_w[:, e.find_sites("head_camera")[0][0]]; q = e.data.site_quat_w[:, e.find_sites("head_camera")[0][0]]
  d = qai(q, o.data.site_pos_w[:, o.find_sites("stem_end")[0][0]] - c)  # camera frame: x forward, y left, z up
  return CAM_F * T.BALL_R / d.norm(dim=-1), -CAM_F * d[:, 1] / d[:, 0].clamp(min=1e-3)


def cam_walk_cmd(env, WS, who, other, go):
  """Back out until the other's ball looks as small as at the place (minus the coast), keeping it centred; (N, 13), (N,) stop."""
  k = int(env.common_step_counter)
  r_true, u_true = ball_seen(env, who, other)
  buf = WS["cam"][who]; buf[k % (CAM_LAG + 1)] = torch.stack([r_true + CAM_SR * torch.randn_like(r_true), u_true + CAM_SU * torch.randn_like(u_true)], 1)
  t = env.episode_length_buf * env.step_dt
  new_frame = torch.floor(t * CAM_HZ) != torch.floor((t - env.step_dt) * CAM_HZ)  # a camera frame arrives (30 Hz)
  WS["meas"][who] = torch.where(new_frame[:, None], buf[(k - CAM_LAG) % (CAM_LAG + 1)], WS["meas"][who])
  r, u = WS["meas"][who].unbind(1)
  stop = r <= WS["r_stop"][who]
  cmd = torch.zeros(len(r), 13, device=r.device)
  cmd[:, 0] = torch.where(stop, 0.0, -W_VX); cmd[:, 2] = (-2.0 * u / CAM_F).clamp(-0.6, 0.6)  # u > 0: ball right -> turn right
  cmd[~go] = 0.0
  return cmd, stop


def spot_from_turners(env, rel):
  """(N, 3) J's spot as the turners make it: the line between their handle circles' centres (each duck's standing stem
  end, rel in its root frame: the stem itself circles ~10 cm while turning), J_Y to the left of F->L at the middle (as in
  the trio layout: J at the origin, the turners at y -0.08), heading along F->L. Also J's centering target."""
  import mjlab_microduck.tasks.ys_trio_env_cfg as T
  from mjlab.utils.lab_api.math import quat_apply
  ctr = {w: (env.scene[w].data.root_link_pos_w + quat_apply(env.scene[w].data.root_link_quat_w, rel[w].expand(env.num_envs, 3)))[:, :2] for w in ("turner", "partner")}
  a, b = ctr["turner"], ctr["partner"]
  ax = (a - b) / (a - b).norm(dim=-1, keepdim=True).clamp(min=1e-6); nrm = torch.stack([-ax[:, 1], ax[:, 0]], 1)
  spot = (a + b) / 2 + T.JUMPER_Y * nrm - env.scene.env_origins[:, :2]
  env._ys_spot_rel = torch.cat([spot, torch.atan2(ax[:, 1], ax[:, 0])[:, None]], 1)
  env._ys_center_hook = lambda cmd: (env._ys_spot_rel[:, :2], 2.0, 0.3)  # J's centering pulls to that spot
  return env._ys_spot_rel


def _yaw(e):
  w, x, y, z = e.data.root_link_quat_w.unbind(-1); return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def turn_cmd(env, who, yaw_t, go):
  """(N, 13) turn in place to yaw_t (fixed rate past velstand's dead band), (N,) done."""
  e = torch.atan2(torch.sin(yaw_t - _yaw(env.scene[who])), torch.cos(yaw_t - _yaw(env.scene[who])))
  cmd = torch.zeros(len(e), 13, device=e.device); cmd[:, 2] = torch.where(e.abs() > W_TURN_TOL, W_WZ * e.sign(), 0.0); cmd[~go] = 0.0
  return cmd, e.abs() <= W_TURN_TOL


def walkout_pre(env, WS):
  """Before a step (user 2026-11-06). Phases (WS["ph"]): 0 bite at the holder (out of the work area), 1 both back off
  W_BACK0, 2 both turn to face +y, 3 both walk forward to the work line, 4 both turn back to face each other, 5 L backs
  out to its place (odometry: the true distance here), 6 F backs out until L's ball looks as at the places (W_CAM) / to its
  place, 7 both stand still (J's start call once both have been still W_STILL s and >= W_MIN_WAIT), 8 turning."""
  te, dt, ph, wh = env.episode_length_buf * env.step_dt, env.step_dt, WS["ph"], ("turner", "partner")
  side_yaw = torch.full_like(te, math.pi / 2)  # facing +y: toward the work line
  ph[(ph == 0) & (te >= W_BITE + 0.2)] = 1
  C = {w: torch.zeros(len(te), 13, device=te.device) for w in wh}; ok = {}
  for w in wh:  # 1: back off the holder
    e = env.scene[w]; f = torch.stack([WS["yaw0"][w].cos(), WS["yaw0"][w].sin()], 1)
    c, ok[w] = walk_cmd(env, w, WS["hold_p"][w] - W_BACK0 * f, WS["yaw0"][w], ph == 1); C[w] = torch.where((ph == 1)[:, None], c, C[w])
  m = (ph == 1) & ok["turner"] & ok["partner"]; ph[m] = 2
  for w in wh:  # 2: turn to +y
    c, ok[w] = turn_cmd(env, w, side_yaw, ph == 2); C[w] = torch.where((ph == 2)[:, None], c, C[w])
  m = (ph == 2) & ok["turner"] & ok["partner"]; ph[m] = 3
  for w in wh:  # 3: forward to the work line, keeping x
    cur = env.scene[w].data.root_link_pos_w[:, :2]; tgt = torch.stack([cur[:, 0], WS["spot"][w][:, 1]], 1)
    c, nr = walk_cmd(env, w, tgt, side_yaw, ph == 3)
    ok[w] = (WS["spot"][w][:, 1] - cur[:, 1]) <= W_STOP_F + 0.005; c[:, 0] = torch.where(ok[w], 0.0, c[:, 0]); C[w] = torch.where((ph == 3)[:, None], c, C[w])
  m = (ph == 3) & ok["turner"] & ok["partner"]; ph[m] = 4
  for w in wh:  # 4: turn back to face each other
    c, ok[w] = turn_cmd(env, w, WS["yaw0"][w], ph == 4); C[w] = torch.where((ph == 4)[:, None], c, C[w])
  m = (ph == 4) & ok["turner"] & ok["partner"]; WS["t_mark"][m] = te[m]; ph[m] = 5
  goL = (ph == 5) & (te >= WS["t_mark"] + W_GAP_T)  # 5: L to its place along its facing only (odometry)
  curL = env.scene["turner"].data.root_link_pos_w[:, :2]; fL = torch.stack([WS["yaw0"]["turner"].cos(), WS["yaw0"]["turner"].sin()], 1)
  tgtL = curL + ((WS["spot"]["turner"] - curL) * fL).sum(-1, keepdim=True) * fL
  cL, nL = walk_cmd(env, "turner", tgtL, WS["yaw0"]["turner"], goL); C["turner"] = torch.where((ph == 5)[:, None], cL, C["turner"])
  m = goL & nL; WS["arr"]["turner"] |= m; WS["t_arr"]["turner"][m] = te[m]; WS["t_mark"][m] = te[m]; ph[m] = 6
  goF = (ph == 6) & (te >= WS["t_mark"] + W_GAP_T)  # 6: F sets the distance on L's ball
  cF, nF = cam_walk_cmd(env, WS, "partner", "turner", goF) if W_CAM else walk_cmd(env, "partner", WS["spot"]["partner"], WS["yaw0"]["partner"], goF)
  C["partner"] = torch.where((ph == 6)[:, None], cF, C["partner"])
  m = goF & nF; WS["arr"]["partner"] |= m; WS["t_arr"]["partner"][m] = te[m]; WS["t_mark"][m] = te[m]; ph[m] = 7
  if W_CAM: cam_walk_cmd(env, WS, "turner", "partner", torch.zeros_like(goF))  # keep L's camera buffer fed (unused)
  v = torch.stack([env.scene[w].data.root_link_lin_vel_w[:, :2].norm(dim=-1) for w in wh], 1).amax(1)
  wz = torch.stack([env.scene[w].data.root_link_ang_vel_w[:, 2].abs() for w in wh], 1).amax(1)
  WS["still"] = torch.where((ph == 7) & (v < W_STILL_V) & (wz < W_STILL_W), WS["still"] + dt, 0.0)
  m = (ph == 7) & (WS["still"] >= W_STILL) & (te >= WS["t_mark"] + W_MIN_WAIT); WS["t_start"][m] = te[m]; ph[m] = 8
  for w in wh: C[w][ph == 8] = torch.nan
  env._ys_walk = C; env._ys_amp = ((te - WS["t_start"]) / 1.0).clamp(0, 1).nan_to_num(0.0)
  env._ys_tenter = WS["t_start"] + J_AFTER
  return WS["t_start"]


def walkout_post(env, WS, EQI, REL, re, got, gerr):
  """After a step: a restarted env is set up again; at W_SET the handles are put where the beaks are; bite at W_BITE."""
  te, dt = env.episode_length_buf * env.step_dt, env.step_dt
  if re.any(): setup_walkout(env, re.nonzero()[:, 0], REL, WS)
  mset = (te >= W_SET - dt / 2) & (te < W_SET + dt / 2)
  if mset.any():
    for who, hk in (("partner", "hold_p"), ("turner", "hold_t")):
      e = env.scene[who]; sid_ = e.find_sites("stem_end")[0][0]; mi = env.scene[hk].indexing.mocap_id
      mp = env.sim.data.mocap_pos[:, mi].clone(); mp[mset] = e.data.site_pos_w[mset, sid_]; env.sim.data.mocap_pos[:, mi] = mp
  bite = (te >= W_BITE - dt / 2) & (te < W_BITE + dt / 2)
  if bite.any(): do_bite(env, bite, EQI, got, gerr)


def walk_cmd(env, who, spot, yaw0, go):
  """(N, 13) standing-ONNX command block that walks `who` to spot (N, 2) facing yaw0 (N,), only where go; (N,) arrived."""
  e = env.scene[who]; p = e.data.root_link_pos_w[:, :2]; w, x, y, z = e.data.root_link_quat_w.unbind(-1)
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)); c, s_ = yaw.cos(), yaw.sin()
  err = spot - p; ef, el = err[:, 0] * c + err[:, 1] * s_, -err[:, 0] * s_ + err[:, 1] * c
  cmd = torch.zeros(len(p), 13, device=p.device)
  cmd[:, 0] = torch.where(ef.abs() > W_STOP_F, W_VX * ef.sign(), 0.0); cmd[:, 1] = torch.where(el.abs() > W_STOP_L, W_VY * el.sign(), 0.0)
  cmd[:, 2] = (-3.0 * torch.atan2(torch.sin(yaw - yaw0), torch.cos(yaw - yaw0))).clamp(-0.6, 0.6)
  near = (ef.abs() <= W_STOP_F) & (el.abs() <= W_STOP_L)
  cmd[~go | near] = 0.0
  return cmd, near


def pick_bow(te):
  """(N,) neck/head pitch (rad) of the pick schedule; NaN from the start call on (rope14 turns)."""
  b = 1.1 * ((te - P_BOW0) / (P_BOW1 - P_BOW0)).clamp(0, 1) * (1 - ((te - P_GRASP) / (P_LIFT1 - P_GRASP)).clamp(0, 1))
  return torch.where(te < P_START, b, torch.nan)


def bow_cmd(b):
  """(N,) bow (NaN: turning) -> (N, 13) command block for the standing ONNX: stand, neck and head pitch +b."""
  c = torch.zeros(len(b), 13, device=b.device); c[:, 3] = b; c[:, 4] = b; c[b.isnan()] = torch.nan
  return c


def place_holders(env, ids, rel, span, side=False, roots=None):
  """Holders where each stem end is at the full bow (rel: stem end in its turner's root frame), the rope lying between."""
  from mjlab.utils.lab_api.math import quat_apply
  pts = []
  for who, hk in (("partner", "hold_p"), ("turner", "hold_t")):
    e = env.scene[who]; r_, q = roots[who] if roots else (e.data.root_link_pos_w[ids], e.data.root_link_quat_w[ids])
    p = r_ + quat_apply(q, rel[who].expand(len(ids), 3))
    env.sim.data.mocap_pos[ids, env.scene[hk].indexing.mocap_id] = p; pts.append(p)
  rope = env.scene["rope"]; flip = torch.tensor((0.0, 0.0, 0.0, 1.0), device=env.device).expand(len(ids), 4)
  if side:  # walk-out: the two ends 6 cm apart at the holder; the V hangs sideways (a deep V down would sink into the floor)
    from mjlab.utils.lab_api.math import quat_mul
    flip = quat_mul(flip, torch.tensor((0.7071068, 0.7071068, 0.0, 0.0), device=env.device).expand(len(ids), 4))
  rope.write_root_state_to_sim(torch.cat([pts[0], flip, torch.zeros(len(ids), 6, device=env.device)], 1), ids)
  import mjlab_microduck.tasks.ys_rope_env_cfg as RE
  RE.reset_rope_joints(env, ids, span)
  m = env.sim.mj_model
  import mujoco
  eqi = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, n) for n in ("rope_p_stem", "rope_t_stem", "rope_p_hold", "rope_t_hold")]
  for i, on in zip(eqi, (False, False, True, True)): env.sim.data.eq_active[ids, i] = on
  return eqi
STOP = tuple(float(v) for v in os.environ["BEAT_STOP"].split(",")) if os.environ.get("BEAT_STOP") else None
JSTAND = os.environ.get("BEAT_JSTAND")  # with BEAT_STOP: J switches to this 61-D ONNX (e.g. policies/velstand.onnx) at the stop


def jstand(x):
  """(n, 73) J actor obs -> (n, 14) actions of the standing ONNX: its 61-D layout, the command block zeroed (stand still)."""
  import onnxruntime as ort
  if "sess" not in jstand.__dict__: jstand.sess = ort.InferenceSession(JSTAND)
  z = x[:, :61].clone(); z[:, 48:61] = 0.0; z = z.cpu().numpy().astype(np.float32)
  return torch.from_numpy(np.concatenate([jstand.sess.run(None, {"obs": z[i: i + 1]})[0] for i in range(len(z))])).to(x.device)
KT, R_OHM, V_NOM, I_MAX = 0.366, 2.811, 7.4, 1.75  # bam/params/xl330/m6.json; battery DR 6.5-8.2 V; firmware current
# limit (bam/dynamixel/actuator.py max_current): |tau| <= KT * I_MAX = 0.64 N m whatever the voltage


def print_load(tau, inn, names, settle):
  """tau (T, N, 3, nj), inn (T, N): torque while the game is on (from J's entry + settle steps)."""
  on = inn & (torch.cumsum(inn.int(), 0) > settle)
  stall = V_NOM * KT / R_OHM
  print(f"  actuator torque |tau| N m in play (J entry + {settle} steps), BAM xl330 stall ~{stall:.2f} N m at {V_NOM} V (6.5-8.2 V: {6.5 * KT / R_OHM:.2f}-{8.2 * KT / R_OHM:.2f})")
  cap = KT * I_MAX; print(f"  current limit {I_MAX} A -> {cap:.3f} N m cap; sat = % of play time at >= 98% of the cap")
  print(f"  {'joint':18s}" + "".join(f" {w + ' p99':>8s} {w + ' max':>8s} {w + ' rms':>7s} {w + ' sat':>6s}" for w in ("J", "L", "F")))
  for j, nm in enumerate(names):
    if nm.startswith("passive_"):
      continue
    row = []
    for d in range(3):
      x = tau[..., d, j][on].abs()
      row.append(f" {torch.quantile(x, 0.99):8.3f} {x.max():8.3f} {x.pow(2).mean().sqrt():7.3f} {100 * (x >= 0.98 * cap).float().mean():5.1f}%")
    print(f"  {nm:18s}" + "".join(row))
  for d, w in enumerate(("J", "L", "F")):
    x = tau[..., d, :][on].abs(); q = torch.quantile(x, 0.99, dim=0)
    heat = ((x / KT) ** 2 * R_OHM).mean(0)  # W of Joule heat per motor
    print(f"  {w}: worst p99 {q.max():.3f} N m ({names[int(q.argmax())]}), any joint at the cap {100 * (x.max(-1).values >= 0.98 * cap).float().mean():.1f}% of play, hottest motor I2R {heat.max():.2f} W ({names[int(heat.argmax())]}), all {heat.sum():.1f} W")


def print_stop(S, dt, win=5.0):
  """S (T, N, 14): time since the stop began, rope middle speed, L fallen, F fallen, J gravity z, restart,
  J left/right sole z (m), L stem end xyz, F stem end xyz."""
  T_, N_ = S.shape[:2]; rows, curves = [], []
  for n in range(N_):
    k0 = np.flatnonzero(S[:, n, 0] >= 0)
    if not len(k0): continue
    k0 = k0[0]; k1 = min(T_, k0 + int(win / dt)); seg = S[k0:k1, n]
    if S[k0:k1, n, 5].any(): seg = seg[: int(np.argmax(S[k0:k1, n, 5]))]  # an episode restart (a turner down) ends it
    if len(seg) < int(win / dt): rows.append((True, None, len(seg) * dt)); continue  # ran out or restarted
    still = seg[:, 1] < 0.1; settle = next((i * dt for i in range(len(seg)) if still[i:].all()), np.inf)
    rows.append((bool(seg[:, 2].any() or seg[:, 3].any()), settle, None)); curves.append(seg[:, 1])
  full = [r for r in rows if r[2] is None]; cut = [r for r in rows if r[2] is not None]
  st = np.array([r[1] for r in full]); fell = np.array([r[0] for r in full])
  C = np.array(curves) if curves else np.zeros((0, 1))
  print(f"  STOP: {len(full)} envs with a full {win:.0f} s after the stop ({len(cut)} cut short by the episode end or a restart: {sum(1 for r in cut if r[0])})")
  print(f"  turner fell within {win:.0f} s: {100 * fell.mean():.0f}%   rope still (< 0.1 m/s for good) after p50 {np.median(st):.1f} s, p90 {np.percentile(st, 90):.1f} s, never {100 * np.isinf(st).mean():.0f}%")
  if len(C): print("  rope middle speed m/s p50 at +0/0.5/1/2/3/5 s: " + " ".join(f"{np.median(C[:, min(int(t / dt), C.shape[1] - 1)]):.2f}" for t in (0, 0.5, 1, 2, 3, 4.98)))
  hop = []  # J after the stop: highest sole in each second (mm)
  for n in range(N_):
    k0 = np.flatnonzero(S[:, n, 0] >= 0)
    if len(k0) and k0[0] + int(win / dt) <= T_ and not S[k0[0]: k0[0] + int(win / dt), n, 5].any():
      seg = S[k0[0]: k0[0] + int(win / dt), n, 6:8].min(1) * 1000; hop.append([seg[int(i / dt): int((i + 1) / dt)].max() for i in range(int(win))])
  if hop: print("  J's lower sole, max per second after the stop (mm) p50: " + " ".join(f"{v:.0f}" for v in np.median(hop, 0)) + f"  (>= 15 mm = still hopping: {100 * (np.array(hop)[:, -1] >= 15).mean():.0f}% in the last second)")


def patch_cfg():  # the jumper may fall while it waits outside: drop its terminations
  import ys_eval
  orig = ys_eval.load_env_cfg
  def load(task, play=True):
    cfg = orig(task, play=play)
    for k in ("fell_over", "out_of_terrain_bounds"):
      cfg.terminations.pop(k, None)
    cfg.episode_length_s = EPISODE
    return cfg
  ys_eval.load_env_cfg = load

def away(env):
  rob = env.scene["robot"]
  env._ys_spot = torch.cat([rob.data.root_link_pos_w, rob.data.root_link_quat_w], 1).clone()
  st = torch.cat([env._ys_spot, torch.zeros(env.num_envs, 6, device=env.device)], 1); st[:, :3] += torch.tensor(AWAY, device=env.device)
  if not INPLACE: rob.write_root_state_to_sim(st)
  if WALKIN: env._ys_walk_ph = math.radians(WALKIN[1])
  if BANK_IN: env._ys_bank = torch.load(BANK_IN, map_location=env.device, weights_only=False)
  env._ys_in = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  env._ys_tin = torch.full((env.num_envs,), math.inf, device=env.device)

def enter(env):
  """Call before every step. From T_ENTER (episode time), each env's jumper steps in when the rope has just
  passed (leader phase at the left-landing phase): a human jumper waits for the rope too. Teleport: it lands
  in its spot standing. An episode restart (a turner fell) sends it back to wait."""
  import mjlab_microduck.tasks.ys_beat as B
  rob = env.scene["robot"]; done = env._ys_in; te = env.episode_length_buf * env.step_dt
  fresh = env.episode_length_buf == 0
  done[fresh] = False; env._ys_tin[fresh] = math.inf
  out = (~done).nonzero()[:, 0]  # still waiting: held standing where it waits (a statue, not a policy at the side)
  if len(out) and not INPLACE and not WALKIN:
    st = torch.cat([env._ys_spot[out], torch.zeros(len(out), 6, device=env.device)], 1); st[:, :3] += torch.tensor(AWAY, device=env.device)
    rob.write_root_state_to_sim(st, out)
    rob.write_joint_state_to_sim(rob.data.default_joint_pos[out].clone(), torch.zeros_like(rob.data.default_joint_pos[out]), env_ids=out)
  lead = env.command_manager.get_term("head_pose").phase - getattr(env, "_ys_delta", math.radians(60))
  ids = (~done & (te >= env.__dict__.get("_ys_tenter", T_ENTER)) & (torch.remainder(lead - (env._ys_walk_ph if WALKIN else ENTER_PH), 2 * math.pi) < OMEGA_DT)).nonzero()[:, 0]
  env._ys_tin[ids] = te[ids]
  if len(ids) and INPLACE and BANK_OUT:
    if not hasattr(env, "_ys_bank"):
      env._ys_bank = {k: torch.zeros(env.num_envs, n, device=env.device) for k, n in (("pos", 3), ("quat", 4), ("lin", 3), ("ang", 3), ("q", rob.data.joint_pos.shape[1]), ("qd", rob.data.joint_vel.shape[1]))}
    for k, v in (("pos", rob.data.root_link_pos_w - env.scene.env_origins), ("quat", rob.data.root_link_quat_w), ("lin", rob.data.root_link_lin_vel_w), ("ang", rob.data.root_link_ang_vel_w), ("q", rob.data.joint_pos), ("qd", rob.data.joint_vel)):
      env._ys_bank[k][ids] = v[ids]
    for key in TRACK:  # the camera trackers' memory too: they were looking at the rope, not at the side
      for f, v in env.__dict__.get(key, {}).items():
        if torch.is_tensor(v) and v.shape[:1] == (env.num_envs,):
          env._ys_bank.setdefault(f"{key}.{f}", torch.zeros_like(v))[ids] = v[ids]
  if len(ids) and not INPLACE and not WALKIN:
    if BANK_IN:
      bk = env._ys_bank
      so = env.__dict__.get("_ys_spot_rel")  # walk-out: J's spot from where L and F actually are (xy offset, yaw)
      pos, quat, lin, ang = bk["pos"][ids].clone(), bk["quat"][ids].clone(), bk["lin"][ids].clone(), bk["ang"][ids].clone()
      if so is not None:
        from mjlab.utils.lab_api.math import quat_mul, quat_apply
        th = so[ids, 2]; qz = torch.stack([torch.cos(th / 2), torch.zeros_like(th), torch.zeros_like(th), torch.sin(th / 2)], 1)
        pos = quat_apply(qz, pos); pos[:, :2] += so[ids, :2]; quat = quat_mul(qz, quat); lin = quat_apply(qz, lin); ang = quat_apply(qz, ang)
      rob.write_root_state_to_sim(torch.cat([pos + env.scene.env_origins[ids], quat, lin, ang], 1), ids)
      rob.write_joint_state_to_sim(bk["q"][ids].clone(), bk["qd"][ids].clone(), env_ids=ids)
      for kf, v in bk.items():
        key, _, f = kf.partition(".")
        if f and key in env.__dict__ and f in env.__dict__[key]: env.__dict__[key][f][ids] = v[ids].to(env.__dict__[key][f].dtype)
    else:
      rob.write_root_state_to_sim(torch.cat([env._ys_spot[ids], torch.zeros(len(ids), 6, device=env.device)], 1), ids)
      rob.write_joint_state_to_sim(rob.data.default_joint_pos[ids].clone(), torch.zeros_like(rob.data.default_joint_pos[ids]), env_ids=ids)
  if len(ids):
    done[ids] = True

if __name__ == "__main__":
  patch_cfg()
  import mjlab_microduck.tasks
  import mjlab_microduck.tasks.ys_trio_env_cfg as T
  import mjlab_microduck.tasks.ys_beat as B
  from ys_eval import make_env, make_policy
  for k, a, f in (("BEAT_MISS", "MISS", float), ("BEAT_FALSE", "FALSE", float), ("BEAT_CALERR", "CAL_ERR", float), ("BEAT_AFTER", "AFTER_LOCK", str), ("BEAT_CORRECT", "CORRECT", lambda v: v == "1")):
    if os.environ.get(k): setattr(B, a, f(os.environ[k]))
  TASK = "Mjlab-YsTrioJumpBeatPick-Flat-MicroDuck" if PICK else "Mjlab-YsTrioJumpBeat-Flat-MicroDuck"; N = int(os.environ.get("BEAT_N", "128")); SECS = EPISODE - 0.1; CURV = {}
  STARTS = ([] if os.environ.get("BEAT_NOREF") else ["0"]) + (sys.argv[2] if len(sys.argv) > 2 else "60,120,180").split(",")  # deg, or "ear": start sound
  if os.environ.get("BEAT_START_BIAS"): B.START_BIAS = float(os.environ["BEAT_START_BIAS"])
  DRIFT = math.radians(float(os.environ.get("BEAT_DRIFT", "0")))  # rad/s the follower's clock walks off (random sign)
  SLIP = float(os.environ.get("BEAT_SLIP", "0"))  # per s: a skipped control tick, the follower's clock 9 deg behind
  env, venv = make_env(TASK, (0., 0., 0.), N, drop=("sample_delta", "sample_offset"), render=os.environ.get("BEAT_RENDER") == "1")
  pol = make_policy(TASK, sys.argv[1], venv); cmd = env.command_manager.get_term("head_pose"); dt = env.step_dt; dev = env.device
  rob, tu, pa = (env.scene[k] for k in ("robot", "turner", "partner")); act = env.action_manager.get_term("partner_joint_pos")
  si, sj = tu.find_sites("stem_end")[0][0], pa.find_sites("stem_end")[0][0]
  cam = "leader_ball" in env.cfg.observations["partner"].terms
  print(f"follower {act.cfg.checkpoint.split('/')[-2]} camera {cam} after lock {B.AFTER_LOCK}  MISS {B.MISS} FALSE {B.FALSE}  enter from {T_ENTER} s")
  import mjlab_microduck.tasks.ys_jump_env_cfg as J
  if PICK:  # calibration: bow once on the standing ONNX, where is each stem end in its turner's frame?
    from mjlab.utils.lab_api.math import quat_apply, quat_inv
    env._ys_pick_onnx = PICK; venv.reset(); away(env); obs = venv.get_observations(); REL = {}
    with torch.inference_mode():  # the rope ends first in holders at the stem ends as they are (else the rope blows up)
      cal = torch.arange(N, device=dev)
      REL0 = {w: quat_apply(quat_inv(env.scene[w].data.root_link_quat_w), env.scene[w].data.site_pos_w[:, s_] - env.scene[w].data.root_link_pos_w).median(0).values for w, s_ in (("turner", si), ("partner", sj))}
      tmp0 = [env.scene[w].data.root_link_pos_w + quat_apply(env.scene[w].data.root_link_quat_w, REL0[w].expand(N, 3)) for w in ("partner", "turner")]
      place_holders(env, cal, REL0, float((tmp0[1] - tmp0[0]).norm(dim=-1).median()))
      for k in range(0 if WALKOUT else int(P_BOW1 / dt) + 10):
        enter(env); te = env.episode_length_buf * dt; env._ys_walk = {g: bow_cmd(1.1 * ((te - 0.3) / (P_BOW1 - 0.3 - 0.2)).clamp(0, 1)) for g in ("turner", "partner")}
        obs, *_ = venv.step(pol(obs)); assert not (env.episode_length_buf[1:] == 0).any() or k == 0, f"calibration reset at step {k}"
      for who, sid_ in (("turner", si), ("partner", sj)):
        e = env.scene[who]; q = e.data.root_link_quat_w
        REL[who] = (REL0[who] if WALKOUT else quat_apply(quat_inv(q), e.data.site_pos_w[:, sid_] - e.data.root_link_pos_w).median(0).values)
    print(f"pick: stem end at the full bow, own frame: turner {1000 * REL['turner'].cpu().numpy().round(4)} mm, partner {1000 * REL['partner'].cpu().numpy().round(4)} mm")
  ref = None; PHS = [math.radians(float(v)) for v in os.environ["BEAT_ENTER_PH"].split(",")] if os.environ.get("BEAT_ENTER_PH") else [ENTER_PH]
  RUNS = [(ph, X) for ph in PHS for X in STARTS]
  for ENTER_PH, X in RUNS:
    globals()["ENTER_PH"] = ENTER_PH; ins, fj, rs, sole, leads, taus, stops, gaps = [], [], [], [], [], [], [], []
    fell = torch.zeros(N, 3, dtype=torch.bool, device=dev); ce, gap, pen, vis = [], [], [], []
    with torch.inference_mode():
      if os.environ.get("SEED"): torch.manual_seed(int(os.environ["SEED"])); torch.cuda.manual_seed_all(int(os.environ["SEED"]))
      venv.reset(); away(env); obs = venv.get_observations()
      if PICK:
        allids = torch.arange(N, device=dev)
        tmp = [env.scene[w].data.root_link_pos_w + quat_apply(env.scene[w].data.root_link_quat_w, REL[w].expand(N, 3)) for w in ("partner", "turner")]
        PSPAN = float((tmp[1] - tmp[0]).norm(dim=-1).median())
        if WALKOUT:
          zN = lambda: torch.zeros(N, device=dev)
          WS = new_ws(env, N)
          pf_walk = torch.zeros(N, 2, dtype=torch.bool, device=dev); pf_turn = torch.zeros(N, 2, dtype=torch.bool, device=dev); TF = torch.full((N, 2), math.inf, device=dev); TJ = torch.full((N,), math.inf, device=dev); ST = torch.full((N, 2, 3), math.nan, device=dev); SPN = torch.full((N,), math.nan, device=dev)
          EQI = setup_walkout(env, allids, REL, WS); t_start = torch.full((N,), torch.inf, device=dev); rv_sum = zN(); rv_n = zN()
        else:
          EQI = place_holders(env, allids, REL, PSPAN)
        got = torch.zeros(N, 2, dtype=torch.bool, device=dev); gerr = torch.full((N, 2), math.nan, device=dev); pfell = torch.zeros(N, 2, dtype=torch.bool, device=dev); rv_turn = []
      ear = X == "ear"
      env._ys_off = B.start_offset(N, dev) if ear else torch.where(torch.rand(N, device=dev) < 0.5, -1.0, 1.0) * math.radians(float(X)); c = None
      off0 = env._ys_off.clone(); dsg = torch.where(torch.rand(N, device=dev) < 0.5, -1.0, 1.0)
      for k in range(int(SECS / dt)):
        enter(env)
        if PICK:
          te = env.episode_length_buf * dt
          if WALKOUT:  # L backs out first, F after it is there, the start call 0.5 s after both
            t_start = walkout_pre(env, WS)
          else:
            env._ys_amp = ((te - P_START) / 1.0).clamp(0, 1)
            env._ys_walk = {g: bow_cmd(pick_bow(te)) for g in ("turner", "partner")}
        if STOP:  # the turners stop: J's first hit (+ its call's delay) or the forced time
          te = env.episode_length_buf * dt
          if k == 0: t_stop = torch.full((N,), STOP[2], device=dev); hit_t = torch.full((N,), math.inf, device=dev)
          t_stop = torch.where(env.episode_length_buf == 0, STOP[2], t_stop); hit_t = torch.where(env.episode_length_buf == 0, math.inf, hit_t)
          sa = (1 - (te - t_stop) / max(STOP[1], dt)).clamp(0, 1)  # the stop ramp; with the walk-out, also its start ramp
          env._ys_amp = torch.minimum(env._ys_amp, sa) if (WALKOUT and torch.is_tensor(env.__dict__.get("_ys_amp"))) else sa
        act_j = pol(obs)
        if REFA:
          if "ref" not in globals():
            from mjlab_microduck.tasks.ys_rope_duo_env_cfg import load_frozen_actor; ref = load_frozen_actor(REFA, dev)
          gaps.append(((act_j - ref(obs["actor"])) ** 2).mean(-1).sqrt() * (env._ys_in & (env.episode_length_buf * dt > env._ys_tin + 1.0)))
        if WALKIN:  # still waiting: standing on its own beside the rope
          m = (~env._ys_in).nonzero()[:, 0]
          if len(m): act_j[m] = jstand(obs["actor"][m])
        if STOP and JSTAND:  # J stops too: from the stop, its actions come from a standing policy (zero command)
          m = (env.episode_length_buf * dt >= t_stop).nonzero()[:, 0]
          if len(m): act_j[m] = jstand(obs["actor"][m])
        cmd.amp[:] = 1.0; cmd.freq[:] = 1.25; cmd._write(); obs, *_ = venv.step(act_j)
        if STOP:
          te = env.episode_length_buf * dt; ov_now = (T.rope_overlap(env) > 0) | (rob.data.projected_gravity_b[:, 2] > -0.5)
          new = env._ys_in & ov_now & hit_t.isinf() & (te > env._ys_tin + 0.04)
          hit_t = torch.where(new, te, hit_t); t_stop = torch.minimum(t_stop, hit_t + STOP[0])
          rope_e = env.scene["rope"]; rv = rope_e.data.body_link_lin_vel_w[:, rope_e.data.body_link_lin_vel_w.shape[1] // 2].norm(dim=-1)
          stops.append(torch.stack([te - t_stop, rv, (tu.data.projected_gravity_b[:, 2] > -0.5).float(), (pa.data.projected_gravity_b[:, 2] > -0.5).float(),
                                    env.scene["robot"].data.projected_gravity_b[:, 2].clone(), (env.episode_length_buf == 0).float(),
                                    *(J.foot_clearance(rob, cmd._sole) + J.FOOT_Z0).T, *tu.data.site_pos_w[:, si].T, *pa.data.site_pos_w[:, sj].T], 1))
        re = env.episode_length_buf == 0  # restarted: a new start (sound: a new draw; fixed X: the same offset)
        if PICK:
          te = env.episode_length_buf * dt
          if re.any():
            got[re] = False
            if not WALKOUT: place_holders(env, re.nonzero()[:, 0], REL, PSPAN)
          if WALKOUT: walkout_post(env, WS, EQI, REL, re, got, gerr)
          if WALKOUT:  # J lines up on the rope it sees (kept once it is in)
            prev_ = env.__dict__.get("_ys_spot_rel"); new_ = spot_from_turners(env, REL)
            if prev_ is not None: env._ys_spot_rel = torch.where(env._ys_in[:, None], prev_, new_)
          else:
            bite = (te >= P_GRASP - dt / 2) & (te < P_GRASP + dt / 2)
            if bite.any(): do_bite(env, bite, EQI, got, gerr)
          tst = t_start if WALKOUT else torch.full_like(te, P_START)
          if WALKOUT and (at_s := ((te - dt >= t_start - dt / 2) & (te - dt < t_start + dt / 2))).any():  # the state at the start call
            for c_, (who, e_) in enumerate((("partner", pa), ("turner", tu))):
              w_, x_, y_, z_ = e_.data.root_link_quat_w.unbind(-1); yw = torch.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
              ST[at_s, c_, 0] = torch.rad2deg(torch.atan2(torch.sin(yw - WS["yaw0"][who]), torch.cos(yw - WS["yaw0"][who])))[at_s].abs()
              ST[at_s, c_, 1] = (1000 * (e_.data.root_link_pos_w[:, :2] - WS["spot"][who]).norm(dim=-1))[at_s]
              fz = e_.data.site_pos_w[:, e_.find_sites(["left_foot", "right_foot"], preserve_order=True)[0]]
              q_ = e_.data.root_link_quat_w; from mjlab.utils.lab_api.math import quat_apply_inverse as qai
              fl = qai(q_, fz[:, 0] - e_.data.root_link_pos_w); fr = qai(q_, fz[:, 1] - e_.data.root_link_pos_w)
              ST[at_s, c_, 2] = (1000 * (fl[:, 0] - fr[:, 0]).abs())[at_s]  # stance: how far one foot is ahead of the other
            SPN[at_s] = (1000 * ((tu.data.site_pos_w[:, si] - pa.data.site_pos_w[:, sj]).norm(dim=-1) - 0.60))[at_s]  # stem ends apart vs rope14's 0.60 m
          if WALKOUT and W_RELAY and (rl := ((te >= t_start - dt / 2) & (te < t_start + dt / 2))).any():
            ids_ = rl.nonzero()[:, 0]; rope_e = env.scene["rope"]
            pp, pt = pa.data.site_pos_w[ids_, sj], tu.data.site_pos_w[ids_, si]
            flip_ = torch.tensor((0.0, 0.0, 0.0, 1.0), device=dev).expand(len(ids_), 4)
            rope_e.write_root_state_to_sim(torch.cat([pp, flip_, torch.zeros(len(ids_), 6, device=dev)], 1), ids_)
            import mjlab_microduck.tasks.ys_rope_env_cfg as RE_; RE_.reset_rope_joints(env, ids_, float((pt - pp).norm(dim=-1).median()))
          for c_, e_ in enumerate((pa, tu)):
            dn = e_.data.projected_gravity_b[:, 2] > -0.5; pfell[:, c_] |= dn & (te < tst + 4.0)
            if WALKOUT: pf_walk[:, c_] |= dn & (te < tst); pf_turn[:, c_] |= dn & (te >= tst) & (te < tst + 4.0)
            if WALKOUT: TF[:, c_] = torch.where(dn & TF[:, c_].isinf(), te - tst, TF[:, c_])  # first fall, s after the start call
          if WALKOUT:  # J's first hit (as the stop uses it), s after the start call
            jh = env._ys_in & ((T.rope_overlap(env) > 0) | (rob.data.projected_gravity_b[:, 2] > -0.5)) & TJ.isinf() & (te > env._ys_tin + 0.04) & TF.isinf().all(1)
            TJ = torch.where(jh, te - tst, TJ)
          rope_e = env.scene["rope"]; rv = rope_e.data.body_link_lin_vel_w[:, rope_e.data.body_link_lin_vel_w.shape[1] // 2].norm(dim=-1)
          if WALKOUT:
            inw = (te >= t_start + 3.0) & (te < t_start + 4.0); rv_sum += rv * inw; rv_n += inw.float()
          elif k * dt >= P_START + 3.0 and k * dt < P_START + 4.0: rv_turn.append(rv)
        env._ys_off[re] = (B.start_offset(int(re.sum()), dev) if ear else off0[re])
        env._ys_off += dsg * DRIFT * dt - (torch.rand(N, device=dev) < SLIP * dt).float() * B.OMEGA * dt
        for i, e in enumerate((rob, tu, pa)):  # the jumper from 0.5 s after it came in
          fell[:, i] |= (e.data.projected_gravity_b[:, 2] > -0.5) & ((i > 0) | (env.episode_length_buf * dt > env._ys_tin + 0.5))
        p = torch.stack([tu.data.site_pos_w[:, si, 1:], pa.data.site_pos_w[:, sj, 1:]], 1)
        c = p.clone() if c is None else c + (p - c) * dt / 0.5
        a = torch.atan2(p[..., 1] - c[..., 1], p[..., 0] - c[..., 0]); gap.append(B.wrap(a[:, 0] - a[:, 1]))
        ce.append(B.wrap(env._ys_off + act._beat.f.c * B.CORRECT).abs()); pen.append(T.rope_overlap(env))
        ins.append(env._ys_in.clone()); fj.append(rob.data.projected_gravity_b[:, 2] > -0.5); rs.append(re.clone())
        sole.append(J.foot_clearance(rob, cmd._sole) + J.FOOT_Z0); leads.append(cmd.phase - getattr(env, "_ys_delta", math.radians(60)))
        if LOAD: taus.append(torch.stack([e.data.qfrc_actuator for e in (rob, tu, pa)], 1))  # (N, 3, nj) N m
        vis.append(env._ys_trio_fol["cur"][:, 2].clone() if cam else torch.zeros(N, device=dev))
    ok = ~fell.any(1).cpu().numpy(); t = np.arange(len(ce)) * dt
    print(f"  fell anywhere J/L/F {(100 * fell.float().mean(0)).round().tolist()}%"); ok = ok if ok.any() else np.ones_like(ok)  # all fell: stats over all
    G = torch.stack(gap).cpu().numpy()[:, ok]; CE = np.degrees(torch.stack(ce).cpu().numpy())[:, ok]; V = torch.stack(vis).cpu().numpy()[:, ok]
    if ref is None: ref = np.angle(np.exp(1j * G[t >= 10]).mean()); print(f"ref gap {math.degrees(ref):.1f} deg (start 0, after entry)")
    ME = np.degrees(np.abs(np.angle(np.exp(1j * (G - ref)))))
    held = int(2 / dt); a0 = int(T_ENTER / dt); conv = []
    for col in CE[a0:].T:
      h = np.convolve(col < 15, np.ones(held), "valid") == held; conv.append(np.argmax(h) * dt if h.any() else np.inf)
    conv = np.minimum(np.array(conv), 99.0); P = torch.stack(pen).cpu().numpy()[int(16 / dt):] * 10; per = 40
    cyc = P[: len(P) // per * per].reshape(-1, per, N).max(1)[:, ok]
    b, aft = (t >= 4) & (t < T_ENTER), t >= 16
    CURV[f"s{X}"] = np.percentile(ME, [10, 50, 90], axis=1)
    PA = torch.stack(pen).cpu().numpy() * 10; win = []
    for t0, t1 in ((6, 9), (9, 12), (12, 16), (16, 20), (20, 30)):
      q = PA[int(t0 / dt): int(t1 / dt)]; q = q[: len(q) // 40 * 40].reshape(-1, 40, N).max(1)[:, ok]; win.append(f"{t0}-{t1}s {100 * (q == 0).mean():.0f}%")
    if N == 1: print("per-cycle max overlap mm:", " ".join(f"{i * 0.8:.1f}:{q:.0f}" for i, q in enumerate(PA[: len(PA) // 40 * 40].reshape(-1, 40, N).max(1)[:, 0]) if 6 <= i * 0.8 < 22))
    IN, FJ, RS, PN = (torch.stack(v).cpu().numpy() for v in (ins, fj, rs, pen)); TT = len(PN); PN = PN * 10  # mm
    def streaks(th):  # passes before the first overlap > th mm (or a jumper fall); a turner restart censors
      stk, hit = [], []
      for n in range(N):
        if not IN[:, n].any(): stk.append(0); hit.append(False); continue
        i0, k, h = int(np.argmax(IN[:, n])), 0, False
        while i0 + 40 * (k + 1) <= TT:
          w = slice(i0 + 40 * k, i0 + 40 * (k + 1))
          if RS[w, n].any(): break
          if (PN[w, n] > th).any() or FJ[w, n].any(): h = True; break
          k += 1
        stk.append(k); hit.append(h)
      return np.array(stk), np.array(hit)
    i0s = np.argmax(IN, 0); first = np.array([PN[i: i + 40, n].max() for n, i in enumerate(i0s)])
    print(f"  pass 1 max overlap mm p25/50/75/90 {np.percentile(first, [25, 50, 75, 90]).round(2)}  J fell in pass 1 {100 * np.mean([FJ[i: i + 40, n].any() for n, i in enumerate(i0s)]):.0f}%")
    for th in (1.0, 2.0, 5.0):
      sk, hk = streaks(th)
      print(f"  streak if a hit needs > {th:.0f} mm: median {np.median(sk):4.1f}  >=10 {100 * (sk >= 10).mean():3.0f}%  >=20 {100 * (sk >= 20).mean():3.0f}%  1st pass {100 * (hk & (sk == 0)).mean():3.0f}%")
    stk, hit = streaks(0.0); full = (TT - int(T_ENTER / dt)) // 40 - 2
    if os.environ.get("BEAT_STK_NPZ"): np.savez(os.environ["BEAT_STK_NPZ"].replace(".npz", f"_{math.degrees(ENTER_PH):.0f}_{X}.npz"), stk=stk, hit=hit, first=first, full=full)
    SO, LD = torch.stack(sole).cpu().numpy() * 1e3, torch.stack(leads).cpu().numpy()
    def ev(x, up):  # leader phase (deg) where a sole crosses 5 mm, from entry + 3 s
      s_ = x > 5; e = (s_[1:] & ~s_[:-1]) if up else (~s_[1:] & s_[:-1]); m_ = e & (np.arange(1, TT)[:, None] > np.argmax(IN, 0)[None] + 150) & IN[1:]
      return np.degrees(np.angle(np.exp(1j * LD[1:][m_])))
    for nm, x, up in (("R up", SO[:, :, 1], 1), ("L down", SO[:, :, 0], 0), ("L up", SO[:, :, 0], 1)):
      v = ev(x, up); mu = np.degrees(np.angle(np.exp(1j * np.radians(v)).mean())); sd = np.degrees(np.std(np.angle(np.exp(1j * np.radians(v - mu)))))
      print(f"  leader phase at {nm}: mean {mu:6.1f} deg  sd {sd:4.1f}  (n {len(v)})")
    if PICK:
      rvt = (rv_sum / rv_n.clamp(min=1)) if WALKOUT else (torch.stack(rv_turn).mean(0) if rv_turn else torch.zeros(N, device=dev))
      if WALKOUT:
        for who in ("turner", "partner"):
          e = env.scene[who]; fin = (e.data.root_link_pos_w[:, :2] - WS["spot"][who]).norm(dim=-1)
          print(f"  WALKOUT {who}: arrived {100 * WS['arr'][who].float().mean():.0f}% at t p50 {float(WS['t_arr'][who][WS['arr'][who]].median()) if WS['arr'][who].any() else float('nan'):.2f} s, place error now p50 {1000 * float(fin.median()):.0f} mm")
        print(f"  fell while walking out F {100 * pf_walk[:, 0].float().mean():.0f}% L {100 * pf_walk[:, 1].float().mean():.0f}%, in the first 4 s of turning F {100 * pf_turn[:, 0].float().mean():.0f}% L {100 * pf_turn[:, 1].float().mean():.0f}%")
        for c_, w_ in enumerate(("F", "L")):
          f_ = pf_turn[:, c_].cpu().numpy(); st_ = ST[:, c_].cpu().numpy()
          for nm, j in (("yaw err deg", 0), ("place err mm", 1), ("feet fore-aft gap mm", 2)):
            print(f"    {w_} at the start call {nm}: fell later p50 {np.nanmedian(st_[f_, j]) if f_.any() else float('nan'):.1f} (n {f_.sum()}) vs not p50 {np.nanmedian(st_[~f_, j]):.1f}")
        for c_, w_ in enumerate(("F", "L")):  # when the falls come: walking / first 4 s / before J's first hit / after it (a real game is over then)
          f_ = TF[:, c_]; bins = [(f_ < 0), (f_ >= 0) & (f_ < 4), (f_ >= 4) & (f_ < TJ), f_.isfinite() & (f_ >= TJ)]
          print(f"    {w_} first fall: walking {100 * bins[0].float().mean():.0f}%  start..+4 s {100 * bins[1].float().mean():.0f}%  later, J not hit yet {100 * bins[2].float().mean():.0f}%  after J's hit {100 * bins[3].float().mean():.0f}%"
                + (f"  (after the hit p50 {float((f_ - TJ)[bins[3]].median()):.1f} s later)" if bins[3].any() else ""))
        sp_ = SPN.cpu().numpy(); fa_ = pf_turn.any(1).cpu().numpy()
        print(f"  stem ends apart at the start call minus 0.60 m: p50 {np.nanmedian(sp_):+.0f} mm, |.| p90 {np.nanpercentile(np.abs(sp_), 90):.0f} mm; fell later p50 {np.nanmedian(sp_[fa_]) if fa_.any() else float('nan'):+.0f} vs not {np.nanmedian(sp_[~fa_]):+.0f}"
              + (f"; ball size targets L {float(WS['r_t']['turner'].median()):.1f} px, F {float(WS['r_t']['partner'].median()):.1f} px" if W_CAM else ""))
        print(f"  start call at t p50 {float(t_start[t_start.isfinite()].median()) if t_start.isfinite().any() else float('nan'):.2f} s ({100 * t_start.isfinite().float().mean():.0f}% got there); J entered at {T_ENTER} s")
      print(f"  PICK: bit the handle F {100 * got[:, 0].float().mean():.0f}% L {100 * got[:, 1].float().mean():.0f}%  (stem end to holder at the bite p50 F {np.nanmedian(gerr[:, 0].cpu()):.1f} L {np.nanmedian(gerr[:, 1].cpu()):.1f} mm, max {np.nanmax(gerr.cpu()):.1f})"
            f"  fell up to start+4 s F {100 * pfell[:, 0].float().mean():.0f}% L {100 * pfell[:, 1].float().mean():.0f}%  rope middle speed at start+3-4 s p50 {float(rvt.median()):.2f} m/s (turning ~2)")
    if REFA and gaps:
      g_ = torch.stack(gaps); m_ = g_ > 0; print(f"  action gap to the reference actor while jumping in place: RMS p50 {float(g_[m_].median()):.3f} rad, p90 {float(g_[m_].quantile(0.9)):.3f} rad")
    if LOAD: print_load(torch.stack(taus), torch.stack(ins), rob.joint_names, int(0.5 / dt))
    if STOP:
      SS = torch.stack(stops).cpu().numpy(); print_stop(SS, dt)
      if os.environ.get("BEAT_STOP_NPZ"): np.savez(os.environ["BEAT_STOP_NPZ"], S=SS, dt=dt)
    ph_s = "MU_LAND" if ENTER_PH is None else f"{math.degrees(ENTER_PH):.0f}"
    print(f"enter {ph_s} start {X}: STREAK (hit=end) median {np.median(stk):4.1f}  mean {stk.mean():4.1f}  >=10 {100 * (stk >= 10).mean():3.0f}%  >=20 {100 * (stk >= 20).mean():3.0f}%"
          f"  hit in passes 1-3 {100 * (hit & (stk < 3)).mean():3.0f}%  1st pass {100 * (hit & (stk == 0)).mean():3.0f}%  no hit to end {100 * (~hit).mean():3.0f}% (max ~{full})", flush=True)
    if BANK_OUT: torch.save(env._ys_bank, BANK_OUT); print(f"  saved the jumper's entry states to {BANK_OUT}")
    print(f"start {X}: clean by window (J enters ~6-7 s): " + "  ".join(win))
    print(f"start {X:>3s}: end |clock err| p50 {np.percentile(CE[-1], 50):4.0f} p90 {np.percentile(CE[-1], 90):4.0f}  BEFORE motion err p50 {np.percentile(ME[b], 50):4.0f} p90 {np.percentile(ME[b], 90):4.0f} deg, cam valid {100 * V[b].mean():3.0f}%"
          f" | AFTER clock conv p50 {np.percentile(conv, 50):4.1f} p90 {np.percentile(conv, 90):4.1f} s, motion err p50 {np.percentile(ME[aft], 50):4.0f} p90 {np.percentile(ME[aft], 90):4.0f} deg,"
          f" cam valid {100 * V[t >= T_ENTER + 1].mean():3.0f}%, clean t>=16 {100 * (cyc == 0).mean():3.0f}%, fell J/L/F {(100 * fell.float().mean(0)).round().tolist()}%", flush=True)
  if os.environ.get("NPZ"): np.savez(os.environ["NPZ"], dt=dt, **CURV)
