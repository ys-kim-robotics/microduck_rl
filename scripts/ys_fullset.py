"""ys_fullset.py (2026-11-08, user: rank the entry jumpers on the FULL game, "L과 F가 핸들을 물고 자리에 서는것부터 ... J가 소리를
내는것도 전부", with joint torques, 5 unpicked games each). 7-8's game (ys_trio_scn BEAT_WALKOUT + BEAT_WALKOUT_CAM, velstand)
with the 7-12 entry jumper in place of the stand-in put-in:
  L and F bite the handles at the holder outside (y -0.5 m), walk out (L by odometry, F on L's ball size), stand still ->
  J's start call (L and F start their clocks on it, F's start error by ear; the circle ramps up over 1 s) -> J, standing on
  velstand from the reset at its waiting place 12-20 cm to its left (+y), takes over with its own policy at its call and sets
  off >= J_AFTER s later in its leader-phase window (its training's enter_hook, in the frame of the spot the turners make,
  kept from the set-off) -> pecks on its clock once at the spot -> its first hit (user's rule, or down) or 20 passes (16 s)
  after the set-off: J's end call -> 0.1 s -> L and F's circle to 0 over 0.4 s, J stands (velstand).
No random push on J (user 2026-11-08: the base score); PUSH=mild: 0.1 m/s every 2-4 s. Streak = 0.8 s windows from the
set-off before the hit's (max 20; >= 10 = success). Torque |qfrc_actuator| per joint: J waiting / running in / jumping
(arrival + 0.5 s to its end call), L and F turning (call + 1 s to the stop).
  ys_fullset.py stats <ckpt> <N> <out.json> [handover ckpt]    handover: at the spot facing L (|yaw| < 0.3), its clock slots 0
  ys_fullset.py video <ckpt> <stem> <label> <rate text> <p> [games 5]
Env: YS_ENTER_MODE, YS_ENTER_PH (the jumper's window), YS_ENTER_SIDE=+, YS_HIT_RULE=caught, YS_J_CLOCK_IN=1, SEED, PUSH."""
import sys, math, os, json, wave
VEL = os.environ.get("VEL", "../policies/velstand.onnx")  # velstand (J standing, L / F walk-out); run from the repo root
for k_, v_ in (("BEAT_PICK_ONNX", VEL), ("BEAT_WALKOUT", "1"), ("BEAT_WALKOUT_CAM", "1"), ("BEAT_EPISODE", "45")): os.environ[k_] = v_
sys.path.insert(0, "scripts")
import numpy as np, torch, onnxruntime as ort
import ys_trio_scn as SC; SC.patch_cfg()  # J's fall terminations off, 45 s episodes
import ys_eval, mjlab_microduck.tasks  # noqa
import mjlab_microduck.tasks.ys_trio_env_cfg as T
import mjlab_microduck.tasks.ys_jump_env_cfg as J
import mjlab_microduck.tasks.ys_beat as B
import ys_camfix  # noqa: E402 (C_NOM for WAITLOOK's aim; CamFix for CAM_FIX)
from mjlab.utils.lab_api.math import quat_apply, quat_inv

NOWALK = os.environ.get("NOWALK") == "1"  # control: no handle walk-out (L and F turn from their places, the rope on the stems
# from the reset, YsTrioJumpBeat); the rest as the full game: J's start call at T_CALL0, the 1 s ramp, J standing until it
T_CALL0, RAMP0 = float(os.environ.get("CALL0", "1.0")), os.environ.get("RAMP0", "1") == "1"  # NOWALK: the call time; RAMP0=0: full circle at once
PLACE_FROM = os.environ.get("PLACE_FROM")  # 7-26 (with NOWALK): L / F / the rope put where recorded walk-outs left them (T.place_reset, one
# drawn row per game); <npz> p (M, 6): spot x, y (m), rope-axis heading (deg), stem ends further apart (m), L's / F's yaw error (deg)
JPOL0, NOEAR, KEEP_DELTA = (os.environ.get(k) == "1" for k in ("JPOL0", "NOEAR", "DELTA"))  # ablations toward the entry env: J's own policy from
# the reset (no standing on velstand first), F's start error 0, J's clock offset drawn +-20 deg per episode as in the entry env
NOCAM = os.environ.get("NOCAM") == "1"  # 7-13 real-robot checks: J's camera-tracker inputs (actor 61:) zeroed (no tracker process);
NOCENTER = os.environ.get("NOCENTER") == "1"  # no position correction once J is at its spot (vx, vy 0; the turn to L kept: IMU yaw);
DRIFT = float(os.environ.get("DRIFT", "0"))  # m/sqrt(s): random walk on J's position estimate from its set-off (odometry drift)
ERRK = float(os.environ.get("ERRK", "1"))  # 7-26 (1-6, with ODO): an oracle, J's position-estimate error scaled by this (0 = it knows the truth, 1 = as it is)
CORR, EXPL, CORR_TAU = os.environ.get("CORR"), float(os.environ.get("CORR_EXPLORE", "0")), float(os.environ.get("CORR_TAU", "0.2"))  # 7-27 (DEPLOY 1-8, with ODO):
# CORR=<model.pt> (ys_memprobe.py FIT): a small net reads J's last 50 observations + the correction in use and says what is left of its position-estimate
# error; the correction (added to the estimate) integrates that over CORR_TAU s, from 0.5 s after J's arrival to its end call. The policy is untouched.
# CORR_EXPLORE=<cm>: no net; the correction wanders (redrawn every 2 s, sd cm around EXPL_MU) to record what a correction does to what J sees (the fit's data)
EXPL_MU = (-1.2, -0.9)  # cm: the mean error of the estimate as it is (7-26)
CORR_GO = os.environ.get("CORR_FROM") == "go"  # the correction starts at J's set-off instead (the error grows as J walks in and is largest, 3 cm, in the first
# second after its arrival, when a third of the games are lost). A CORR model brings its own (FIT with CORR_FROM=go); the knob is for CORR_EXPLORE
# measured (7-27, realistic condition, 10+): as it is 52%; from arrival, CORR_TAU 0.4: 57%; from set-off, 0.4 / 0.2 / 0.1: 59 / 68 (4 seeds) / 70%
CLOCKOFF = os.environ.get("CLOCKOFF")  # 7-13 check 3, deg: L / F turn the rope this much behind J's clock (late start by ear); J's
# set-off still reads its own clock (it does not know the offset)
HOLD = [math.radians(float(v)) for v in os.environ["ODO_HOLD"].split(",")] if os.environ.get("ODO_HOLD") else None  # 7-15 (1-2): "c,w" deg: while J's
# clock phase is within c +- w (its flight, s16/odo/clock2.py), the odometry output keeps its last velocity (runtime untouched)
WAITLOOK = os.environ.get("WAITLOOK")  # 7-15 (1-4 + user): J waits on velstand until HANDOVER s before its earliest set-off: "stand"
# head straight, "L" head turned to L's ball, "LF" to F's ball then L's (needs CAM_FIX: the fix then uses both while J waits)
HANDOVER, LOOK0, LOOKF, RAMPED = 1.0, 0.3, float(os.environ.get("LOOK_SPLIT", "1.8")), float(os.environ.get("LOOK_FROM", "1.1"))  # s: own policy back before the earliest
# set-off; after the call: head turns, L -> F (LF), frames used (the circle is full from about 1 s: s16/wait, 117 / 103 mm at 1.0-1.8 s)
CAMFIX = os.environ.get("CAM_FIX") == "1"  # 7-15 (1-3): + J's camera fix of the odometry position (scripts/ys_camfix.py; needs ODO=1)
ODO = os.environ.get("ODO") == "1"  # 7-14: J's position estimate = the runtime's contact odometry (scripts/ys_odo_replay, unchanged runtime
# code) on the encoders (bias included) and the trunk quaternion, live from J's start call, from where J truly stood then; its
# command and its arrival test use that estimate (with NOCENTER=1: no position command once it believes it has arrived)
P0ERR = float(os.environ.get("P0ERR", "0"))  # 7-29 (RUNTIME-ys.md, with ODO): J's position estimate starts this many cm off where J stands (one
# direction per env), as a floor mark would leave it; the camera fix can take it out once J sees L
BELIEF = [float(v) for v in os.environ.get("BELIEF", "0,0").split(",")]  # 7-13 check 4: "cm,deg": J believes its spot is cm off (random
# direction) and the rope axis deg off (random sign), from the start: placed by hand; its command uses that belief
TASK, MODE, CK = ("Mjlab-YsTrioJumpBeat-Flat-MicroDuck" if NOWALK else "Mjlab-YsTrioJumpBeatPick-Flat-MicroDuck"), sys.argv[1], sys.argv[2]
VIDEO = MODE == "video"; N = 1 if VIDEO else int(sys.argv[3]); SEED = int(os.environ.get("SEED", 1)); PUSH = os.environ.get("PUSH", "none")
J_AFTER, STOP_DELAY, STOP_RAMP, AFTER, WIN, MAXP = float(os.environ.get("JAFTER", "4.0")), 0.1, 0.4, 3.0, 40, 20  # s call -> earliest set-off (7-8), s, s, s shown after the stop, steps/pass, passes
cfg = ys_eval.load_env_cfg(TASK, play=True); push = cfg.events["push_robot"]
for k_ in (("sample_offset",) if KEEP_DELTA else ("sample_delta", "sample_offset")): cfg.events.pop(k_)  # one clock for all three (v2); F's start error by ear below
ys_eval.fix_command(cfg, (0.0, 0.0, 0.0))  # also drops the random push and the curricula
if PUSH == "mild": push.interval_range_s = (2.0, 4.0); push.params["velocity_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}; cfg.events["push_robot"] = push
if PUSH == "medium": push.interval_range_s = (2.0, 4.0); push.params["velocity_range"] = {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}; cfg.events["push_robot"] = push  # full check: mild's timing, the walking recipe's size
# full check (2026-11-15): J's body at one end of its trained randomization, J only (L / F keep theirs): VINJ volts (BAM vin_range 6.5-8.2),
# JFRIC foot friction (0.7-1.3), JMASS mass and inertia scale (0.95-1.05)
if os.environ.get("VINJ"):
  from dataclasses import replace as _rp; _e = cfg.scene.entities["robot"]
  cfg.scene.entities["robot"] = _rp(_e, articulation=_rp(_e.articulation, actuators=tuple(_rp(a_, vin_range=(float(os.environ["VINJ"]),) * 2) for a_ in _e.articulation.actuators)))
if os.environ.get("JFRIC"): cfg.events["foot_friction"].params["ranges"] = (float(os.environ["JFRIC"]),) * 2
if os.environ.get("JMASS"): cfg.events["randomize_mass_inertia"].params["alpha_range"] = (math.log(float(os.environ["JMASS"])) / 2,) * 2
cfg.scene.num_envs, cfg.seed = N, SEED
if os.environ.get("VIN"):  # battery voltage for all three (BAM vin_range, drawn once per env; the load drop stays): VIN=6.5 ~ a flat 2S pack
  for w_ in ("robot", "turner", "partner"):
    for a_ in cfg.scene.entities[w_].articulation.actuators: a_.vin_range = (float(os.environ["VIN"]),) * 2
if os.environ.get("DELAY"):  # 7-13 check 7: J's command delay, physics steps of 5 ms "min,max" (trained 3,6). J ONLY: the actuator cfg
  # object is shared by all three ducks (microduck_constants.actuators), so J gets its own copies (the first run mutated it: L / F too)
  from dataclasses import replace as _rp
  _lo, _hi = (int(v) for v in os.environ["DELAY"].split(",")); _e = cfg.scene.entities["robot"]
  cfg.scene.entities["robot"] = _rp(_e, articulation=_rp(_e.articulation, actuators=tuple(_rp(a_, delay_min_lag=_lo, delay_max_lag=_hi) for a_ in _e.articulation.actuators)))
if VIDEO: cfg.viewer.height, cfg.viewer.width = 720, 1280
env = ys_eval.ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array" if VIDEO else None)
if os.environ.get("DELAY"): print("DELAY max lag J", env.scene["robot"].actuators[0].cfg.delay_max_lag, "L", env.scene["turner"].actuators[0].cfg.delay_max_lag, flush=True)
if os.environ.get("VINJ"): print("VINJ J", env.scene["robot"].actuators[0].cfg.vin_range, "L", env.scene["turner"].actuators[0].cfg.vin_range, flush=True)
if os.environ.get("IMAX"):  # 7-13 check 7: J's firmware current limit (A; 1.75 trained) on its BAM motor model
  for a_ in env.scene["robot"].actuators: a_._bam_model.actuator.max_current = float(os.environ["IMAX"])
  print("IMAX", os.environ["IMAX"], "J motor model shared with L:", env.scene["robot"].actuators[0]._bam_model is env.scene["turner"].actuators[0]._bam_model,
        "L max_current", env.scene["turner"].actuators[0]._bam_model.actuator.max_current, flush=True)
venv = ys_eval.RslRlVecEnvWrapper(env, clip_actions=ys_eval.load_rl_cfg(TASK).clip_actions)
pol = ys_eval.make_policy(TASK, CK, venv); HO = os.environ.get("HANDOVER") or (sys.argv[5] if MODE == "stats" and len(sys.argv) > 5 else None)
if os.environ.get("ONNX"):  # 7-13 check 2: J's actions from an exported ONNX (onnxruntime CPU, batch 1 per env, its own input width);
  # PARITY: the torch checkpoint on the same obs (dims past the ONNX width zeroed) alongside, max |difference| kept in PAR
  _sess = ort.InferenceSession(os.environ["ONNX"]); _din = _sess.get_inputs()[0].shape[1]; _tp = pol; PAR = [0.0]
  def pol(o):
    x = o["actor"][:, :_din].float().cpu().numpy(); a_ = torch.from_numpy(np.concatenate([_sess.run(None, {"obs": x[i: i + 1]})[0] for i in range(len(x))])).to(o["actor"].device)
    if os.environ.get("PARITY") == "1":
      oz = {k_: v_.clone() for k_, v_ in o.items()}; oz["actor"][:, _din:] = 0.0; PAR[0] = max(PAR[0], float((a_ - _tp(oz)).abs().max()))
    return a_
pol2 = ys_eval.make_policy(TASK, HO, venv) if HO else None
dev, dt = env.device, env.step_dt; rob, tu, pa = (env.scene[k] for k in ("robot", "turner", "partner")); cmd = env.command_manager.get_term("head_pose")
env._ys_pick_onnx = VEL; vs = ort.InferenceSession(VEL); NS = int(44.8 / dt); ALL = torch.arange(N, device=dev); E = None
CLIP = os.environ.get("CLIP", ""); WAITM = torch.zeros(N, dtype=torch.bool, device=dev)  # WAITM: J waiting (its call .. its set-off)
if CLIP:  # deploy-time fix (no training): J's PD targets kept inside its joint range (the A jumpers push head_roll
  # and left_hip_yaw past their stops while waiting: target +3.2 / -1.4 rad vs +-0.44, the motor at the current cap). The policy's
  # own last-action observation is untouched (raw actions); only the targets sent to the motors are clamped.
  # CLIP=1: every joint, always; wait: every joint while J waits only; head / yaw: head_roll / left_hip_yaw only, always
  _jt = env.action_manager.get_term("joint_pos"); _lim = rob.data.joint_pos_limits[:, :, :].clone(); _proc = _jt.process_actions
  _sel = torch.full((_lim.shape[1],), CLIP in ("1", "wait"), dtype=torch.bool, device=dev)
  for c_ in CLIP.split("+"):  # head / yaw shorthands, or any joint name (7-13 check 6)
    nm_ = {"head": "head_roll", "yaw": "left_hip_yaw"}.get(c_, c_)
    if nm_ in rob.joint_names: _sel[rob.joint_names.index(nm_)] = True
  for it_ in filter(None, os.environ.get("CLIPLIM", "").split(",")):  # 7-13 check 5: "joint=lo:hi" deg, the clip range in place of the model's
    nm_, rg_ = it_.split("="); _lim[:, rob.joint_names.index(nm_)] = torch.tensor([math.radians(float(v)) for v in rg_.split(":")], device=dev)
  def _clipped(actions):
    _proc(actions); p = _jt._processed_actions; c = torch.max(torch.min(p, _lim[..., 1]), _lim[..., 0])
    _jt._processed_actions[:] = torch.where((_sel[None] & WAITM[:, None]) if CLIP == "wait" else _sel[None], c, p)
  _jt.process_actions = _clipped
SI, SJ = tu.find_sites("stem_end")[0][0], pa.find_sites("stem_end")[0][0]
KSTOP = int(round(STOP_DELAY / dt)); DIAG = os.environ.get("DIAG"); TR = []; OB = [] if os.environ.get("OBSLOG") == "1" else None  # OBSLOG=1 (with DIAG, 7-26 memory probe): + ob, J's
# actor observation after every step (steps x N x 73, what its policy is given next). DIAG=<npz>: per-step J touch / down / xy / speed, walk phase, amp, z, sole
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
with torch.inference_mode():
  venv.reset()  # the stem end in each turner's own frame (ys_trio_scn's REL for the walk-out)
  REL = {w: quat_apply(quat_inv(env.scene[w].data.root_link_quat_w), env.scene[w].data.site_pos_w[:, env.scene[w].find_sites("stem_end")[0][0]] - env.scene[w].data.root_link_pos_w).median(0).values for w in ("turner", "partner")}
  if "IMU_SCALE" in os.environ:  # 7-28; the realistic bundle sets 0.3333 (0~2 deg) since 7-29 (user, WORKFLOW-ys.md): the IMU mounting tilt each env's three ducks read gravity and turn rate through (mdp's draw: up to 6 deg
    # about a random axis, one per env, shared by L, F and J) scaled; 0 = levelled, as after a level calibration on the real ducks. Seed 3's first env
    # draws 5.6 deg, an upright duck there reads -5 deg of pitch: L stands leaning forward and steps as it bites (video game 3). Of L·F's 41 failures in
    # 768 games, 35 are in the 17% of games where an upright duck reads >= +2 or < -4 deg of pitch
    _q = env._imu_misalign_quat; _h = float(os.environ["IMU_SCALE"]) * torch.acos(_q[:, :1].clamp(-1, 1))
    env._imu_misalign_quat = torch.cat([_h.cos(), torch.nn.functional.normalize(_q[:, 1:], dim=1) * _h.sin()], 1)


def jstand(x, head=None):  # J obs -> velstand actions (61-D, command block 0: stand still; head: (n, 4) its head command), one env
  z = x[:, :61].clone(); z[:, 48:61] = 0.0  # at a time (the ONNX is batch 1)
  if head is not None: z[:, 51:55] = head
  z = z.cpu().numpy().astype(np.float32)
  return torch.from_numpy(np.concatenate([vs.run(None, {"obs": z[i: i + 1]})[0] for i in range(len(z))])).to(x.device)


def look(te):
  """WAITLOOK: (N, 4) velstand head command [neck_pitch, head_pitch, head_yaw, head_roll] turning J's head to the ball the procedure
  expects (L's / F's circle centre: its spot -+ the nominal half axis, ys_camfix C_NOM), from J's estimated place and IMU yaw."""
  sr = env.__dict__.get("_ys_spot_rel")
  if sr is None: return torch.zeros(N, 4, device=dev)  # before the turners' spot exists (start of the game)
  h = sr[:, 2]; ax, nr = torch.stack([h.cos(), h.sin()], 1), torch.stack([-h.sin(), h.cos()], 1)
  c0, c1 = ys_camfix.C_NOM; rel = te - (E["t"] - J_AFTER); end = J_AFTER - HANDOVER - 0.3
  tL = (rel >= LOOK0) & (rel < (LOOKF if WAITLOOK == "LF" else end)) & (WAITLOOK != "stand"); tF = (rel >= LOOKF) & (rel < end) & (WAITLOOK == "LF")
  tgt = torch.where(tF[:, None], sr[:, :2] - (-c0 * ax + c1 * nr), sr[:, :2] - (c0 * ax + c1 * nr))
  me = rob.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2] - (E["drift"] if ODO else 0.0)
  b = T._wrap(torch.atan2(tgt[:, 1] - me[:, 1], tgt[:, 0] - me[:, 0]) - T._yaw(rob))
  out = torch.zeros(N, 4, device=dev); on_ = tF | tL
  out[:, 2] = torch.where(on_, b.clamp(-LOOK_YAW, LOOK_YAW), 0.0); out[:, 1] = torch.where(on_, LOOK_PITCH, 0.0)
  return out


LOOK_YAW, LOOK_PITCH = 1.35, float(os.environ.get("LOOK_PITCH", "-0.3"))  # rad: velstand's head commands were trained to +-1.40 / +-1.10;
# pitch -0.3 (head up): L's ball in J's tracker 86% of the wait vs 78% at 0, 65% at +0.3 (16 games, s16/wait/looktest.py)


def noclock(o):  # the handover jumper (E5 line) was trained clock-free: its head-command slots read 0
  o = {k: v.clone() for k, v in o.items()}; o["actor"][:, 51:55] = 0.0; return o


def hook(c):
  """enter_hook (the jumper's training command: wait at y0, set off in the window, run in at gain 6 / 0.8 m/s, A turns over
  the last 12 cm) in the frame of the spot the turners make (ys_trio_scn.spot_from_turners: xy, heading F->L)."""
  e, t = E, env.episode_length_buf * dt
  sr = env.__dict__.get("_ys_spot_rel"); sr = torch.zeros(N, 3, device=dev) if sr is None else sr
  sr = sr + e["belief"]  # BELIEF: the spot J thinks the turners make (0 unless set)
  th = sr[:, 2]; cs, sn = th.cos(), th.sin()
  dl = T.DELTA0 if CLOCKOFF else env.__dict__.get("_ys_delta", T.DELTA0)  # CLOCKOFF: J sets off on its own clock
  e["go"] |= (t >= e["t"]) & (torch.remainder(c.phase - dl - e["ph"], 2 * math.pi) < 2 * math.pi * c.freq.abs() * dt)
  y = torch.where(e["go"], 0.0, e["y0"])
  d = (sr[:, :2] - (rob.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2] - (e["drift"] if ODO else 0.0))).norm(dim=-1); e["d"] = d
  e["near"] |= e["go"] & (d < T.ENTER_NEAR); walk = e["go"] & ~e["near"]
  g = T.ENTER_DASH / d.clamp(min=T.ENTER_NEAR) if T.ENTER_DASH else T.ENTER_GAIN
  fwA, h = e["fw"] & (e["mode"] == 1), -0.5 * math.pi * e["side"]
  dy = T._wrap(th + torch.where(fwA & ~e["near"], h * ((d - T.ENTER_NEAR) / T.ENTER_TURN_D).clamp(0, 1), 0.0) - T._yaw(rob))
  wz = torch.where(e["fw"] & e["go"], (T.TURN_GAIN * dy).clamp(-T.TURN_WMAX, T.TURN_WMAX), (J.CENTER_GAIN * dy).clamp(-0.5, 0.5))
  tgt = sr[:, :2] + torch.stack([-sn * y, cs * y], 1) + e["drift"]  # DRIFT: the command steers by a drifting position estimate
  gc = torch.where(e["near"] & torch.tensor(NOCENTER, device=dev), 0.0, J.CENTER_GAIN)
  return tgt, torch.where(walk, g, gc), torch.where(walk, max(T.ENTER_VMAX, T.ENTER_DASH), 0.3), wz


if ODO:
  import subprocess
  _ODO = subprocess.Popen([os.path.join(os.path.dirname(os.path.abspath(__file__)), "ys_odo_replay/target/release/odo_replay")],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
  _JN = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
         "mouth", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]  # duck_ipc_proto::JOINT_NAMES
  _COL = torch.tensor([rob.joint_names.index(n) if n != "mouth" else 0 for n in _JN], device=dev)
  _P0, _O0 = torch.full((N, 2), math.nan, device=dev), torch.full((N, 2), math.nan, device=dev)  # true / odometry xy at the first tick
  _a = torch.tensor(np.random.default_rng(7).uniform(0, 2 * math.pi, N), dtype=torch.float32, device=dev)  # its own stream: the games stay the reference run's
  _P0E = P0ERR / 100 * torch.stack([_a.cos(), _a.sin()], 1)
  _OFF, _H = torch.zeros(N, 2, device=dev), torch.full((N, 2, 2), math.nan, device=dev)  # HOLD: offset on the output, last two outputs
  if CAMFIX: import ys_camfix; _CF = ys_camfix.CamFix(N, dt)
  if CORR:
    import ys_memprobe; _cm = torch.load(CORR, weights_only=False); _CN = ys_memprobe.make_net(len(_cm["mu"]) + 2).to(dev); _CN.load_state_dict(_cm["net"])
    _CMU, _CSD, _CYM = (torch.as_tensor(_cm[k_], dtype=torch.float32, device=dev) for k_ in ("mu", "sd", "ym")); CORR_GO = _cm.get("frm") == "go"


LFODO = os.environ.get("LFODO") == "1"  # 4-10 (2026-11-15): L / F walk out on the runtime odometry from the holder, not their true xy
if LFODO:
  assert ODO, "LFODO goes with ODO=1"
  _LO = subprocess.Popen(_ODO.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)  # its own odometry per duck: L = n, F = N + n
  _LW = ("turner", "partner")
  _LC = {w: torch.tensor([env.scene[w].joint_names.index(n) if n != "mouth" else 0 for n in _JN], device=dev) for w in _LW}
  _LS = {k_: {w: torch.full((N, 2), math.nan, device=dev) for w in _LW} for k_ in ("p0", "o0", "xy")}
  SC.XY = lambda env_, w: torch.where(_LS["xy"][w].isnan(), env_.scene[w].data.root_link_pos_w[:, :2], _LS["xy"][w])


def lf_odo(k, WS):
  """LFODO: one tick of L's and F's odometry while they walk out (before the turning starts); the estimate is anchored at the holder."""
  on = (WS["ph"] < 8).nonzero().flatten()
  if len(on) == 0: return
  ln = []
  for i, w in enumerate(_LW):
    e = env.scene[w]; j = e.data.joint_pos_biased[on][:, _LC[w]].clone(); j[:, _JN.index("mouth")] = 0.0
    x = torch.cat([j, e.data.root_link_quat_w[on]], 1).cpu().numpy()
    ln += [f"{n + i * N} {k} " + " ".join(f"{v:.7f}" for v in r) + "\n" for n, r in zip(on.tolist(), x)]
  _LO.stdin.write("".join(ln)); _LO.stdin.flush()
  est = torch.tensor([[float(v) for v in _LO.stdout.readline().split()[2:4]] for _ in ln], device=dev)
  for i, w in enumerate(_LW):
    es, pos, S_ = est[i * len(on):(i + 1) * len(on)], env.scene[w].data.root_link_pos_w[on, :2], _LS
    S_["p0"][w][on] = torch.where(S_["p0"][w][on].isnan(), pos, S_["p0"][w][on]); S_["o0"][w][on] = torch.where(S_["o0"][w][on].isnan(), es, S_["o0"][w][on])
    S_["xy"][w][on] = S_["p0"][w][on] + es - S_["o0"][w][on]


def odo_step(k, M):
  """ODO: one 20 ms tick of the runtime odometry for every J past its start call; E["drift"] = true - estimate."""
  on = ((M["dead"] < 0) & (M["start"] >= 0)).nonzero().flatten()
  if len(on) == 0: return
  pos = rob.data.root_link_pos_w[on, :2]
  _P0[on] = torch.where(_P0[on].isnan(), pos + _P0E[on], _P0[on])
  j = rob.data.joint_pos_biased[on][:, _COL].clone(); j[:, _JN.index("mouth")] = 0.0
  x = torch.cat([j, rob.data.root_link_quat_w[on]], 1).cpu().numpy(); ids = on.cpu().numpy()
  _ODO.stdin.write("".join(f"{n} {k} " + " ".join(f"{v:.7f}" for v in r) + "\n" for n, r in zip(ids, x))); _ODO.stdin.flush()
  est3 = torch.tensor([[float(v) for v in _ODO.stdout.readline().split()[2:5]] for _ in ids], device=dev); est = est3[:, :2]
  if HOLD:
    est = est + _OFF[on]; h0, h1 = _H[on, 0], _H[on, 1]
    w_ = (T._wrap(cmd.phase[on] - HOLD[0]).abs() < HOLD[1]) & ~h0.isnan().any(1)
    want = 2 * h1 - h0; _OFF[on] += torch.where(w_[:, None], want - est, 0.0); est = torch.where(w_[:, None], want, est); _H[on, 0], _H[on, 1] = h1, est
  _O0[on] = torch.where(_O0[on].isnan(), est, _O0[on])  # the odometry keeps running across games (video): from this game's first tick
  world = _P0[on] + est - _O0[on]
  sp = env.__dict__.get("_ys_spot_rel"); cb = env.__dict__.get("_ys_cam_ball")
  if CAMFIX and sp is not None and cb is not None:
    spw = torch.cat([sp[:, :2] + env.scene.env_origins[:, :2], sp[:, 2:]], 1).cpu().numpy(); cur, curF, wt = cb["cur"], None, None
    if WAITLOOK:  # frames only once the circle is full (it ramps up over 1 s after the call); F's ball tracked too (LF)
      ramp = (env.episode_length_buf * dt - (E["t"] - J_AFTER) < RAMPED)[:, None]; cur = torch.where(ramp, torch.zeros_like(cur), cur)
      if WAITLOOK == "LF":
        curF = T._cam_track(env, "_ys_cam_fball", "stem_end", True, true_fn=lambda e: T._cam_true(e, "stem_end", True, who="partner"))[:, :3]
        curF = torch.where(ramp, torch.zeros_like(curF), curF).cpu().numpy()
      wt = (~E["go"][on]).cpu().numpy()
    corr = _CF.step(k, ids, x[:, :15], x[:, 15:], torch.cat([world, est3[:, 2:]], 1).cpu().numpy(), cmd.phase[on].cpu().numpy(),
                    cur.cpu().numpy(), env._ys_cam_clock["lag"].cpu().numpy(), spw, curF, wt)
    world = world + torch.as_tensor(corr, dtype=world.dtype, device=dev)
  if CORR or EXPL:  # c0: the correction the latest observation was made with
    c0 = E["corr"][on]; ref = (M["go"] if CORR_GO else M["arr"])[on]; act = ((ref >= 0) & (k >= ref + (0 if CORR_GO else 25)) & (M["call"][on] < 0))[:, None]
    if CORR: left = (_CN(torch.cat([((E["hb"][on].flatten(1) - _CMU) / _CSD).clamp(-5, 5), 100 * c0], 1)) + _CYM) / 100
    else:
      if k % 100 == 0 or "xt" not in E: E["xt"] = (torch.tensor(EXPL_MU, device=dev) + EXPL * torch.randn(N, 2, device=dev)) / 100
      left = E["xt"][on] - c0
    E["corr"][on] = torch.where(act, (c0 + dt / CORR_TAU * left).clamp(-0.06, 0.06), c0)
  E["drift"][on] = ERRK * (pos - world - E["corr"][on])


def episode(on_step=None):
  """One full game for all N envs. on_step(k, M) after each step (video), returns True to end early. -> M (marks)."""
  global E
  venv.reset()
  pl_ = None
  if PLACE_FROM:
    p_ = np.load(PLACE_FROM)["p"]; pl_ = p_[torch.randint(len(p_), (N,)).numpy()]
    for n, row in enumerate(pl_):
      T.PLACE_RANGE = dict(zip(("sx", "sy", "th", "dd", "yl", "yf"), ((float(v), float(v)) for v in row))); T.place_reset(env, torch.tensor([n], device=dev))
  env.__dict__.pop("_ys_spot_rel", None)
  if ODO: _P0[:] = math.nan; _O0[:] = math.nan; _OFF[:] = 0.0; _H[:] = math.nan
  if LFODO: [t_.fill_(math.nan) for d_ in _LS.values() for t_ in d_.values()]
  if CAMFIX: _CF.reset()
  env._ys_off = torch.zeros(N, device=dev) if NOEAR else B.start_offset(N, dev)  # F's start error by ear (ys_trio_scn "ear")
  WS = SC.new_ws(env, N); EQI = None if NOWALK else SC.setup_walkout(env, ALL, REL, WS)
  got, gerr = torch.zeros(N, 2, dtype=torch.bool, device=dev), torch.full((N, 2), math.nan, device=dev)
  z = lambda: torch.zeros(N, device=dev)
  E = {"y0": z(), "ph": z(), "side": z(), "d_prev": z(), "d": z(), "mode": z().long(), "fw": z().bool(), "go": z().bool(), "near": z().bool(), "t": torch.full((N,), math.inf, device=dev),
       "drift": torch.zeros(N, 2, device=dev), "belief": torch.zeros(N, 3, device=dev), "corr": torch.zeros(N, 2, device=dev), "hb": torch.zeros(N, 50, 73, device=dev)}
  if BELIEF[0] or BELIEF[1]:  # check 4: a fixed belief error per game (position: random direction; heading: random sign)
    a_ = torch.rand(N, device=dev) * 2 * math.pi; sg = torch.where(torch.rand(N, device=dev) < 0.5, -1.0, 1.0)
    E["belief"] = torch.stack([BELIEF[0] / 100 * a_.cos(), BELIEF[0] / 100 * a_.sin(), sg * math.radians(BELIEF[1])], 1)
  if CLOCKOFF: env._ys_delta = torch.full((N,), T.DELTA0 + math.radians(float(CLOCKOFF)), device=dev)  # check 3: the rope runs late
  T._draw(E, ALL, torch.zeros(N, dtype=torch.bool, device=dev), T.ENTER_Y, dev)  # y0 12-20 cm (+y), the window, the scenario
  pr = env.event_manager.get_term_cfg("reset_base").params["pose_range"]  # as enter_reset: the default root (z 0) + reset_base's x / y / z draw
  st = rob.data.default_root_state.clone(); st[:, 7:] = 0.0; st[:, 1] += E["y0"]
  st[:, :3] += env.scene.env_origins + torch.stack([torch.empty(N, device=dev).uniform_(*pr.get(k, (0.0, 0.0))) for k in ("x", "y", "z")], 1)
  T._face(st, -0.5 * math.pi * E["side"] * E["fw"].float())  # A faces the spot (-y), C faces L
  rob.write_root_state_to_sim(st); rob.write_joint_state_to_sim(rob.data.default_joint_pos.clone(), torch.zeros_like(rob.data.default_joint_pos))
  env._ys_in = torch.zeros(N, dtype=torch.bool, device=dev)  # ys_beat reads it (J's beeps once in): its spot
  M = {k: torch.full((N,), -1, dtype=torch.long, device=dev) for k in ("start", "go", "arr", "hit", "call", "dead", "jdown", "jdph")}
  M["cause"] = torch.zeros(N, dtype=torch.long, device=dev); M["spot"] = torch.full((N, 3), math.nan, device=dev); M["werr"] = torch.full((N, 2), math.nan, device=dev)
  M["geo"] = torch.full((N, 3), math.nan, device=dev); M["bite"] = torch.zeros(N, dtype=torch.bool, device=dev); M["gerr"] = gerr
  M["tq"] = {k: [] for k in ("wait", "run", "jump", "L", "F")}; M["ws"] = WS; sw = torch.zeros(N, dtype=torch.bool, device=dev)
  if pl_ is not None: M["place"] = torch.as_tensor(pl_)  # each game's drawn row, into its record
  obs = venv.get_observations()
  for k in range(NS):
    if NOWALK:  # the start call at T_CALL0, the circle's 1 s ramp (as walkout_pre's after the call)
      te_ = env.episode_length_buf * dt; WS["t_start"] = torch.where(WS["t_start"].isinf() & (te_ >= T_CALL0), te_, WS["t_start"]); WS["ph"][:] = 8
      env._ys_amp = ((te_ - WS["t_start"]) / SC.W_RAMP).clamp(0, 1).nan_to_num(0.0) if RAMP0 else WS["t_start"].isfinite().float()
    else: SC.walkout_pre(env, WS)  # L / F commands, the start call (WS t_start), the circle's start ramp
    M["start"] = torch.where((M["start"] < 0) & WS["t_start"].isfinite(), k, M["start"]); E["t"] = WS["t_start"] + J_AFTER
    kstop = torch.where(M["call"] >= 0, M["call"] + KSTOP, 10 ** 9)  # the turners hear J's end call
    env._ys_amp = torch.minimum(env._ys_amp, (1 - (k - kstop).float() * dt / STOP_RAMP).clamp(0, 1))
    if NOCAM: obs = {k_: v_.clone() for k_, v_ in obs.items()}; obs["actor"][:, 61:] = 0.0  # tracker never sees anything
    if DRIFT: E["drift"] += DRIFT * math.sqrt(dt) * torch.randn(N, 2, device=dev) * E["go"][:, None]
    a = pol(obs)
    if pol2 is not None: a = torch.where(sw[:, None], pol2(noclock(obs)), a)
    vel = ((M["start"] < 0) & ~torch.tensor(JPOL0, device=dev)) | (k >= kstop)  # J stands: before its call (JPOL0: its own policy), after the stop
    if WAITLOOK:  # 7-15: J stands (and looks) while it waits, its own policy HANDOVER s before the earliest set-off
      te_ = env.episode_length_buf * dt; vel = vel | ((M["start"] >= 0) & ~E["go"] & (te_ < E["t"] - HANDOVER))
    if vel.any(): a[vel] = jstand(obs["actor"][vel], look(env.episode_length_buf * dt)[vel] if WAITLOOK else None)
    WAITM[:] = (M["start"] >= 0) & ~E["go"]  # CLIP=wait's mask
    cmd.amp[:] = 1.0; cmd.freq[:] = B.BEAT_HZ; cmd._write(); obs, *_ = venv.step(a)
    if CORR or EXPL: E["hb"] = torch.roll(E["hb"], -1, 1); E["hb"][:, -1] = obs["actor"]  # J's last 50 observations
    if ODO: odo_step(k, M)
    if LFODO and not NOWALK: lf_odo(k, WS)
    re = env.episode_length_buf == 0; M["dead"] = torch.where(re & (M["dead"] < 0), k, M["dead"]); alive = M["dead"] < 0  # a turner fell: restarted
    if not NOWALK:
      SC.walkout_post(env, WS, EQI, REL, re, got, gerr)
      bf = alive & ~gerr.isnan().any(1) & ~got.all(1)  # 7-15 bug: a beak missed its handle (the rope stays on the holder): L·F's failure, as a fall
      M["dead"][bf] = k; M["bite"] |= bf; alive &= ~bf
    prev = env.__dict__.get("_ys_spot_rel"); new = SC.spot_from_turners(env, REL)  # J lines up on the rope it sees, kept from its set-off
    if prev is not None: env._ys_spot_rel = torch.where(E["go"][:, None], prev, new)
    env._ys_center_hook = hook
    sr = env._ys_spot_rel
    m = alive & (M["start"] == k); M["spot"] = torch.where(m[:, None], torch.stack([1000 * sr[:, 0], 1000 * sr[:, 1], torch.rad2deg(sr[:, 2])], 1), M["spot"])
    if m.any():  # the turners at the call: stem ends apart minus rope14's 0.60 m (mm), L's and F's yaw vs facing each other along F->L (deg)
      ax_ = sr[:, 2]; yl = torch.rad2deg(T._wrap(T._yaw(tu) - ax_ - math.pi)); yf = torch.rad2deg(T._wrap(T._yaw(pa) - ax_))
      M["geo"] = torch.where(m[:, None], torch.stack([1000 * ((tu.data.site_pos_w[:, SI] - pa.data.site_pos_w[:, SJ]).norm(dim=-1) - 0.60), yl, yf], 1), M["geo"])
    m = alive & (M["go"] < 0) & E["go"]; M["go"] = torch.where(m, k, M["go"])
    if m.any():  # J's place at the set-off vs its waiting place, in the spot frame (mm): along F->L, out to its left
      d_ = rob.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2] - sr[:, :2]; c_, s_ = sr[:, 2].cos(), sr[:, 2].sin()
      M["werr"] = torch.where(m[:, None], 1000 * torch.stack([d_[:, 0] * c_ + d_[:, 1] * s_, -d_[:, 0] * s_ + d_[:, 1] * c_ - E["y0"]], 1), M["werr"])
    M["arr"] = torch.where(alive & (M["arr"] < 0) & E["near"], k, M["arr"]); env._ys_in = M["arr"] >= 0
    down = rob.data.projected_gravity_b[:, 2] > -0.5; rs = T.rope_state(env)
    jd = alive & (M["jdown"] < 0) & down; M["jdown"] = torch.where(jd, k, M["jdown"]); M["jdph"] = torch.where(jd, WS["ph"], M["jdph"])  # J's first fall, the walk-out phase then
    bad = alive & (M["go"] >= 0) & (M["call"] < 0) & (T.rope_bad(env) | down)
    M["hit"] = torch.where(bad, k, M["hit"]); M["call"] = torch.where(bad, k, M["call"])
    M["cause"] = torch.where(bad, torch.where(down, 3, torch.where(rs["run"] >= T.CAUGHT_STEPS, 1, 2)), M["cause"])  # 1 touch, 2 lag, 3 down
    fin = alive & (M["go"] >= 0) & (M["call"] < 0) & (k >= M["go"] + WIN * MAXP); M["call"] = torch.where(fin, k, M["call"])  # 20 passes: J calls the end
    if pol2 is not None: sw |= E["near"] & (T._wrap(T._yaw(rob) - sr[:, 2]).abs() < 0.3)
    tq = torch.stack([rob.data.qfrc_actuator, tu.data.qfrc_actuator, pa.data.qfrc_actuator], 1).abs()  # (N, 3, nj) N m
    ended = (M["call"] >= 0) & (M["call"] <= k); kst = torch.where(M["call"] >= 0, M["call"] + KSTOP, 10 ** 9)
    for nm, m in (("wait", alive & (M["start"] >= 0) & (M["go"] < 0)), ("run", alive & (M["go"] >= 0) & (M["arr"] < 0) & ~ended),
                  ("jump", alive & (M["arr"] >= 0) & (k >= M["arr"] + 25) & ~ended)):
      M["tq"][nm].append(tq[m, 0])
    m = alive & (M["start"] >= 0) & (k >= M["start"] + 50) & (k < kst); M["tq"]["L"].append(tq[m, 1]); M["tq"]["F"].append(tq[m, 2])
    if DIAG: TR.append(torch.stack([rs["touch"].float(), down.float(), *(rob.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).T, rob.data.root_link_lin_vel_w[:, :2].norm(dim=-1),
                                    WS["ph"].float(), env._ys_amp, rob.data.root_link_pos_w[:, 2], (J.foot_clearance(rob, cmd._sole) + J.FOOT_Z0).amin(-1),
                                    torch.atan2(*T.rope_phase_obs(env).unbind(-1)[::-1]), T._wrap(cmd.phase - env.__dict__.get("_ys_delta", T.DELTA0)), *E["drift"].T,
                                    *(T.rope_section(env.scene["rope"].data.body_link_pos_w, rob.data.root_link_pos_w[:, 0]) - torch.stack([rob.data.root_link_pos_w[:, 1], torch.zeros_like(rob.data.root_link_pos_w[:, 1])], 1)).T, *E["corr"].T], 1))  # + rope angle - clock, leader phase, 7-15: odometry error xy (true - estimate), 7-27: the correction in use (cols 15:17)
    if DIAG and OB is not None: OB.append(obs["actor"].cpu().numpy())
    if on_step is not None and on_step(k, M): break
    if on_step is None and (~alive | ((M["call"] >= 0) & (k > M["call"] + KSTOP + int(AFTER / dt)))).all(): break
  M["k_end"] = k
  if DIAG: np.savez(DIAG, tr=torch.stack(TR).cpu().numpy(), jdown=M["jdown"].cpu().numpy(), start=M["start"].cpu().numpy(), go=M["go"].cpu().numpy(), y0=E["y0"].cpu().numpy(),
                   arr=M["arr"].cpu().numpy(), call=M["call"].cpu().numpy(), spot=env._ys_spot_rel.cpu().numpy(), **({"ob": np.stack(OB)} if OB else {}))  # 7-14: + arrival, end call, the spot (m, rad)
  return M


def records(M):
  """per-env outcome. kind: game (J's game played out), lf (L or F down before J's game was over), nocall (no start call)."""
  out = []; C = {0: None, 1: "touch", 2: "lag", 3: "down"}
  g = {k: v.cpu().numpy() for k, v in M.items() if torch.is_tensor(v)}
  for n in range(N):
    ks, kg, ka, kh, kc, kd = (int(g[x][n]) for x in ("start", "go", "arr", "hit", "call", "dead"))
    r = dict(y0=round(100 * float(E["y0"][n]), 1), ph=round(math.degrees(float(E["ph"][n])), 1), t_call=None if ks < 0 else round(ks * dt, 2),
             spot=None if ks < 0 else [round(float(v), 1) for v in g["spot"][n]], werr=None if kg < 0 else [round(float(v), 1) for v in g["werr"][n]],
             geo=None if ks < 0 else [round(float(v), 1) for v in g["geo"][n]], place=[round(float(v), 4) for v in g["place"][n]] if "place" in g else None, bite_mm=[None if math.isnan(v) else round(float(v), 1) for v in g["gerr"][n]],
             jdown=None if g["jdown"][n] < 0 else dict(t=round(int(g["jdown"][n]) * dt, 2), walk_ph=int(g["jdph"][n]), vs_call=None if ks < 0 else round((int(g["jdown"][n]) - ks) * dt, 2),
                                                        vs_go=None if kg < 0 else round((int(g["jdown"][n]) - kg) * dt, 2)))
    if kd >= 0 and (kc < 0 or kd < kc):  # L or F down (its episode restarted) before J's game was over
      r.update(kind="lf", lf_when="bite" if g["bite"][n] else "walk-out" if (ks < 0 or kd < ks) else ("before set-off" if (kg < 0 or kd < kg) else "J in play"), lf_down_s=round(kd * dt, 2),
               streak_then=None if (kg < 0 or kd < kg) else (kd - kg) // WIN); out.append(r); continue
    if ks < 0 or kg < 0: r["kind"] = "nocall" if ks < 0 else "nogo"; out.append(r); continue
    hit = kh >= 0
    r.update(kind="game", t_go=round(kg * dt, 2), wait_s=round((kg - ks) * dt, 2), arr_s=None if ka < 0 else round((ka - kg) * dt, 2), hit=hit, cause=C[int(g["cause"][n])] if hit else None,
             streak=min((kh - kg) // WIN, MAXP) if hit else MAXP, pre=bool(hit and (ka < 0 or kh < ka)), after_arr=None if (not hit or ka < 0 or kh < ka) else round((kh - ka) * dt / 0.8, 2),
             lf_down_after_end=bool(kd >= 0))
    out.append(r)
  return out


KT, R_OHM, CAP = SC.KT, SC.R_OHM, SC.KT * SC.I_MAX  # xl330 (bam m6): N m/A, ohm; the firmware current limit cap 0.64 N m


def torque(rows):
  x = torch.cat(rows) if rows else torch.zeros(0, len(rob.joint_names), device=dev)
  if not len(x): return None
  keep = [j for j, nm in enumerate(rob.joint_names) if not nm.startswith("passive_")]; x = x[:, keep]
  q = lambda p: [round(float(v), 4) for v in torch.quantile(x, p, dim=0)]
  heat = ((x / KT) ** 2 * R_OHM).mean(0)  # W of Joule heat per motor
  return dict(n=len(x), joints=[rob.joint_names[j] for j in keep], p99=q(0.99), p50=q(0.5), max=[round(float(v), 4) for v in x.max(0).values],
              rms=[round(float(v), 4) for v in x.pow(2).mean(0).sqrt()], sat=[round(100 * float(v), 2) for v in (x >= 0.98 * CAP).float().mean(0)],
              heat=[round(float(v), 4) for v in heat], heat_sum=round(float(heat.sum()), 3))


def summary(R):
  G = [r for r in R if r["kind"] == "game"]; s = np.array([r["streak"] for r in G]) if G else np.zeros(0)
  pc = lambda m, n: round(100 * m / max(n, 1), 1)
  hits = [r for r in G if r["hit"]]
  return dict(n_all=len(R), n_game=len(G), ok10_game=pc((s >= 10).sum(), len(G)), ok10_all=pc((s >= 10).sum(), len(R)), ok15_game=pc((s >= 15).sum(), len(G)),
              median=float(np.median(s)) if len(s) else None, j_down=pc(sum(r["cause"] == "down" for r in hits), len(G)), arrived=pc(sum(r["arr_s"] is not None for r in G), len(G)),
              arr_p50=float(np.median([r["arr_s"] for r in G if r["arr_s"] is not None])) if G else None, pre_hit=pc(sum(r["pre"] for r in hits), len(G)),
              first_pass=pc(sum(r["streak"] == 0 for r in hits), len(G)), early=pc(sum((not r["pre"]) and r["after_arr"] is not None and r["after_arr"] < 4 for r in hits), len(G)),
              later=pc(sum((not r["pre"]) and r["after_arr"] is not None and r["after_arr"] >= 4 for r in hits), len(G)),
              lf=pc(sum(r["kind"] == "lf" for r in R), len(R)), lf_when={w: sum(r.get("lf_when") == w for r in R) for w in ("bite", "walk-out", "before set-off", "J in play")},
              nocall=pc(sum(r["kind"] in ("nocall", "nogo") for r in R), len(R)),
              call_p50=float(np.median([r["t_call"] for r in R if r["t_call"] is not None])) if any(r["t_call"] is not None for r in R) else None,
              causes={c: sum(r["cause"] == c for r in hits) for c in ("touch", "lag", "down")})


if MODE == "stats":
  with torch.inference_mode(): M = episode()
  R = records(M); S = summary(R); TQ = {k: torque(v) for k, v in M["tq"].items()}
  json.dump(dict(ckpt=CK, handover=HO, seed=SEED, push=PUSH, mode=T.ENTER_MODE, window=[round(math.degrees(v)) for v in T.ENTER_PH], summary=S, torque=TQ, games=R),
            open(sys.argv[4], "w"), indent=1)
  print("SUMMARY", sys.argv[4].split("/")[-1], json.dumps(S), flush=True)
  if os.environ.get("PARITY") == "1": print("PARITY max |onnx - torch| over the game:", PAR[0], flush=True)
  for k, v in TQ.items():
    if v: print(f"  torque {k:4s} n {v['n']:7d}  worst p99 {max(v['p99']):.3f} N m ({v['joints'][int(np.argmax(v['p99']))]})  at the cap {max(v['sat']):.1f}%  heat {v['heat_sum']:.2f} W")
  sys.exit()

# ---------------- video: the first NG games, not picked ----------------
import mujoco, imageio, imageio_ffmpeg, subprocess
from collections import deque
from PIL import Image, ImageDraw, ImageFont
STEM, LAB, RATE, P = sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]); NG = int(sys.argv[7]) if len(sys.argv) > 7 else 5
FACE = os.environ.get("FACE") == "1"
GI0, NTOT, NOSUM = int(os.environ.get("GI0", "0")), int(os.environ.get("NTOT", str(NG))), os.environ.get("NOSUMMARY") == "1"  # one game per process
# (a new robot each: the startup DR - foot friction, encoder bias, COM, mass - is drawn once per process), numbered GI0 + 1 of NTOT
W_, H_, FPS = 1280, 720, 50
F = lambda s: ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", s); FB, FM, FS, FX = F(32), F(24), F(20), F(64)
r_ = env._offline_renderer; mm = env.sim.mj_model; o = env.scene.env_origins[0].cpu().numpy(); fov = mm.vis.global_.fovy
WIDE, GAME = ((0.0, -0.18, 0.20), 70.0, -32.0, 1.85), ((0.0, 0.08, 0.05), 75.0, -28.0, 1.5)  # lookat, azimuth, elevation, distance
HEADS = {k: env.scene[k].find_sites("head_imu")[0][0] for k in ("robot", "turner", "partner")}
VOICE = "scripts/ys_voice"; WAVS = {}
for nm in ("greet_0", "peck_2", "alarm_0"):
  with wave.open(f"{VOICE}/{nm}.wav") as w: WAVS[nm] = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
WSTEP = ["① 바깥 거치대에서 L·F가 손잡이를 묾", "② 조금 뒤로", "③ 작업 위치 쪽으로 돌기", "④ 앞으로 걸어 작업 위치로", "⑤ 다시 마주 보기",
         "⑥ L 뒷걸음 (거리: odometry)", "⑦ F 뒷걸음 (L의 공 크기로 간격 맞춤)", "⑧ 둘 다 멈출 때까지 대기"]
CAUSE = {"touch": "발·다리가 줄에 60 ms 넘게 닿음", "lag": "J가 줄을 붙잡아 90° 넘게 늦춤", "down": "J 넘어짐"}


def cam_at(k, ks):  # wide for the walk-out, then the jump view (1.5 s blend from the start call)
  a = 0.0 if ks < 0 else min(1.0, max(0.0, (k - ks) * dt / 1.5)); a = a * a * (3 - 2 * a)
  return tuple(np.array(w) * (1 - a) + np.array(g) * a for w, g in zip(WIDE, GAME))


def render(view):
  c = r_._cam; c.type = mujoco.mjtCamera.mjCAMERA_FREE; la, az, el, di = view
  c.lookat[:] = o + np.asarray(la); c.azimuth, c.elevation, c.distance = float(az), float(el), float(di)
  r_.update(env.sim.data); return r_.render().copy()


def project(p, view):
  la, az, el, di = view; az, el = math.radians(az), math.radians(el); L_ = o + np.asarray(la)
  f = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
  up = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)]); rt = np.cross(f, up)
  q = p - (L_ - di * f); zz = q @ f; th = math.tan(math.radians(fov) / 2)
  return W_ / 2 * (1 + (q @ rt) / zz / (th * W_ / H_)), H_ / 2 * (1 - (q @ up) / zz / th)


def box(dr, xy, s_, font, fill, col=(255, 255, 255)):
  bw = dr.textbbox((0, 0), s_, font=font)[2]; x, y = xy; h = font.size + 14
  dr.rounded_rectangle((x, y, x + bw + 24, y + h), 9, fill=fill); dr.text((x + 12, y + 6), s_, font=font, fill=col)


def tag(dr, xy, s_, font, col):
  b = dr.textbbox((0, 0), s_, font=font); w, h = b[2] - b[0], b[3] - b[1]; x, y = xy
  dr.rounded_rectangle((x - w / 2 - 8, y - h - 12, x + w / 2 + 8, y - 2), 7, fill=(0, 0, 0, 170)); dr.text((x - w / 2, y - h - 8), s_, font=font, fill=col)


def result(rec):
  if rec["kind"] == "lf": return "결과: " + ("L·F가 손잡이를 못 물었음" if rec["lf_when"] == "bite" else "줄 돌리는 오리가 넘어짐") + " (J 판정 없음)"
  if rec["kind"] != "game": return "결과: 시작 소리까지 못 감 (L·F 자리 서기 실패)"
  if not rec["hit"]: return f"결과: 20줄 끝까지 안 걸림 → J의 끝 소리"
  w = "자리 도착 전" if rec["pre"] else f"도착 후 {rec['after_arr']:.1f}줄째"
  return f"결과: 넘은 줄 {rec['streak']} → 걸림 ({w}, {CAUSE[rec['cause']]})"


wr = imageio.get_writer(STEM + "_noaudio.mp4", fps=FPS, codec="libx264", quality=8, macro_block_size=8); NF = [0]; SND = []
def put(img): wr.append_data(img); NF[0] += 1


def card(img, lines, big):
  im = Image.fromarray(img).convert("RGBA"); ov = Image.new("RGBA", im.size, (0, 0, 0, 235)); dr = ImageDraw.Draw(ov)
  dr.text((60, 60), big, font=FX, fill=(255, 255, 255))
  for i, (s_, c) in enumerate(lines): dr.text((60, 170 + 40 * i), s_, font=FM, fill=c)
  return np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


recs, sigs = [], []
for gi in range(1, NG + 1):
  st = dict(close=deque(maxlen=int(1.2 / dt)), post=[], peck=-9.0, last=None, hitk=None); sig = {k: [] for k in ("t", "lead", "soleL", "soleR", "rope_y", "rope_z", "d", "yaw", "touch", "run", "lag")}
  first = [True]

  def on_step(k, M):
    ks, kg, ka, kc, kd = (int(M[x][0]) for x in ("start", "go", "arr", "call", "dead")); te = (k + 1) * dt
    if kd >= 0: return True  # 7-15 bug: a fall restarts the env in this step: no frames of the restarted scene (the result card follows)
    view = cam_at(k, ks); img = render(view)
    if FACE:  # 7-24: J's face from the front, inset bottom-right (head_roll lean visible)
      hp = env.scene["robot"].data.site_pos_w[0, HEADS["robot"]].cpu().numpy()
      fc = Image.fromarray(render((hp - o + np.array([0, 0, -0.03]), math.degrees(float(T._yaw(rob)[0])) + 180.0, -15.0, 0.38))).resize((576, 324))
      ImageDraw.Draw(fc).text((12, 8), "J 얼굴 정면", font=FB, fill=(255, 255, 255)); img[H_ - 324 - 70: H_ - 70, W_ - 576 - 12: W_ - 12] = np.asarray(fc)
    if first[0]:
      for _ in range(40): put(card(img, [(LAB, (255, 255, 255)), ("풀세트: 손잡이 물기 → 자리 서기 → J 시작 소리 → 입장 → 줄넘기 → 끝 소리 → 멈춤", (220, 220, 220)),
                                         (f"로봇 {gi + GI0}호 (판마다 다른 로봇: 마찰·엔코더 오차·무게중심·질량 무작위)", (200, 200, 200))], f"{gi + GI0} / {NTOT}판"))
      first[0] = False
    if ks == k: SND.append((NF[0] / FPS, "greet_0"))
    if kc == k: SND.append((NF[0] / FPS, "alarm_0")); st["hitk"] = k if int(M["hit"][0]) == k else None
    lead = float(T._wrap(cmd.phase - T.DELTA0)[0]); past = (lead - B.BEEP_PH) % (2 * math.pi)
    if ka >= 0 and (kc < 0 or k < kc) and past < B.OMEGA * dt: SND.append((NF[0] / FPS - past / B.OMEGA, "peck_2")); st["peck"] = te
    if kg >= 0:  # signals for the hit review + the close view for the replay
      so = (J.foot_clearance(rob, cmd._sole) + J.FOOT_Z0)[0]; sec = T.rope_section(env.scene["rope"].data.body_link_pos_w, rob.data.root_link_pos_w[:, 0])[0]; rs = T.rope_state(env)
      for kk, v in zip(sig, (te, math.degrees(lead), float(so[0]), float(so[1]), float(sec[0]), float(sec[1]), float(E["d"][0]), math.degrees(float(T._wrap(T._yaw(rob) - env._ys_spot_rel[:, 2])[0])),
                             bool(rs["touch"][0]), int(rs["run"][0]), float(rs["lag"][0]))): sig[kk].append(v)
      p = rob.data.root_link_pos_w[0].cpu().numpy(); cv = (p - o + np.array([0, 0, 0.05 - p[2]]), 75.0, -12.0, 0.75)
      if st["hitk"] is None and kc < 0: st["close"].append((render(cv), k))
      elif st["hitk"] is not None and k <= st["hitk"] + int(0.4 / dt): st["post"].append((render(cv), k))
    im = Image.fromarray(img).convert("RGBA"); ov = Image.new("RGBA", im.size); dr = ImageDraw.Draw(ov)
    box(dr, (12, 10), LAB, FB, (30, 60, 140, 215))
    box(dr, (12, 62), f"고르지 않은 연속 {gi + GI0}/{NTOT}판 (로봇 {gi + GI0}호) · 풀세트 (손잡이 물기부터) · 무작위 밀기 " + ("약하게 (2~4 s마다 0.1 m/s)" if PUSH == "mild" else "없음"), FS, (0, 0, 0, 160))
    box(dr, (12, 100), RATE, FS, (0, 0, 0, 160), (255, 230, 150))
    n = 0 if kg < 0 else max(0, (min(k, kc) if kc >= 0 else k) - kg) // WIN
    if kc >= 0 and k >= kc:
      s_ = ("⑬ 걸림! J의 끝 소리 → L·F 멈춤, J 서기" if st["hitk"] is not None else "⑬ 20줄 → J의 끝 소리 → L·F 멈춤, J 서기"); col = (190, 40, 40) if st["hitk"] is not None else (20, 110, 40)
    elif ka >= 0: s_, col = "⑫ 자리 도착 → 줄넘기 (오른발 들 때마다 '톡')", (20, 110, 40)
    elif kg >= 0: s_, col = "⑪ J 출발: 잽싸게 들어가기", (0, 0, 0)
    elif ks >= 0 and k < ks + 50: s_, col = "⑨ J의 시작 소리 → L·F가 같은 시계로 줄 돌리기 시작", (120, 80, 0)
    elif ks >= 0: s_, col = f"⑩ J 대기: 줄 옆 {100 * float(E['y0'][0]):.0f} cm, 출발 위상 {math.degrees(float(E['ph'][0])):.0f}° (시작 소리 4 s 뒤부터)", (0, 0, 0)
    else: s_, col = ("⑦-2 멈춘 뒤 다시 마주 보게 고쳐 서기 (새 단계)" if int(M["ws"]["ph"][0]) == 10 else WSTEP[min(int(M["ws"]["ph"][0]), 7)]) + " · J는 옆에서 서서 대기", (0, 0, 0)
    box(dr, (12, 146), s_, FM, (*col, 190))
    if kg >= 0:
      dr.rounded_rectangle((W_ - 250, 10, W_ - 12, 128), 14, fill=(20, 110, 40, 220) if n >= 10 else (0, 0, 0, 170))
      dr.text((W_ - 238, 16), "넘은 줄", font=FM, fill=(255, 255, 255)); dr.text((W_ - 238, 46), f"{n}", font=FX, fill=(255, 255, 255)); dr.text((W_ - 140, 92), "10 = 성공", font=FS, fill=(220, 220, 220))
    for who, nm_, c_ in (("turner", "L (리더)", (255, 110, 110)), ("partner", "F (팔로워)", (120, 255, 140)), ("robot", "J (점퍼)", (255, 255, 255))):
      xy = project(env.scene[who].data.site_pos_w[0, HEADS[who]].cpu().numpy() + np.array([0, 0, 0.07]), view)
      if who == "robot":
        if ks >= 0 and 0 <= k - ks < 30: nm_, c_ = nm_ + "   시작 소리!", (255, 220, 60)
        elif kc >= 0 and 0 <= k - kc < 30: nm_, c_ = nm_ + "   끝 소리!", (255, 120, 60)
        elif te - st["peck"] < 0.25: nm_, c_ = nm_ + "   톡!", (255, 220, 60)
      tag(dr, xy, nm_, FS, c_)
    info = f"t {te:4.1f} s" + (f"   시작 소리 후 {(k - ks) * dt:4.1f} s" if ks >= 0 else "") + (f"   출발 후 {(k - kg) * dt:4.2f} s   자리까지 {100 * float(E['d'][0]):4.1f} cm" if kg >= 0 else "") \
      + (f"   J 위치 추정 오차 {100 * float(E['drift'][0].norm()):4.1f} cm" if ODO and ks >= 0 else "")  # ODO: 자리까지 = what J believes
    box(dr, (12, H_ - 54), info, FS, (0, 0, 0, 150), (220, 235, 255))
    fr = np.asarray(Image.alpha_composite(im, ov).convert("RGB")); put(fr); st["last"] = fr
    return (kc >= 0 and k > kc + KSTOP + int(AFTER / dt))

  with torch.inference_mode(): M = episode(on_step)
  rec = records(M)[0]; rec["game"] = gi; recs.append(rec); print(f"game {gi}:", json.dumps(rec), flush=True)
  np.savez(f"{STEM}_g{gi}.npz", **{k: np.array(v) for k, v in sig.items()})
  im = Image.fromarray(st["last"]).convert("RGBA"); ov = Image.new("RGBA", im.size); dr = ImageDraw.Draw(ov)
  good = rec["kind"] == "game" and rec["streak"] >= 10; box(dr, (12, H_ - 112), result(rec), FM, (20, 110, 40, 225) if good else (190, 40, 40, 225))
  last = np.asarray(Image.alpha_composite(im, ov).convert("RGB"))
  for _ in range(75): put(last)
  if st["hitk"] is not None:  # 0.5x close replay of the hit
    cl = list(st["close"]) + st["post"]
    for img, kk in cl:
      im = Image.fromarray(img).convert("RGBA"); ov = Image.new("RGBA", im.size); dr = ImageDraw.Draw(ov)
      box(dr, (12, 10), LAB, FB, (30, 60, 140, 215)); box(dr, (12, 62), f"{gi + GI0}/{NTOT}판 · 0.5배속 가까이: 걸린 순간 " + ("전" if kk < st["hitk"] else "후"), FM, (200, 40, 40, 230))
      box(dr, (12, H_ - 54), f"걸린 순간까지 {(kk - st['hitk']) * dt:+.2f} s", FS, (0, 0, 0, 150), (220, 235, 255))
      x = np.asarray(Image.alpha_composite(im, ov).convert("RGB")); put(x); put(x)
    mi = min(range(len(cl)), key=lambda i: abs(cl[i][1] - st["hitk"])); Image.fromarray(cl[mi][0]).save(f"{STEM}_hit{gi}.png")
  del st

G = [x for x in recs if x["kind"] == "game"]; ok = sum(x["streak"] >= 10 for x in G)
pk = [math.comb(len(G), j) * P ** j * (1 - P) ** (len(G) - j) for j in range(len(G) + 1)]; cdf = np.cumsum(pk)
lo, hi = (int(np.argmax(cdf >= 0.05)), int(np.argmax(cdf >= 0.95))) if G else (0, 0)
summ = dict(ckpt=CK, p=P, n=len(G), ok10=ok, range90=[lo, hi], median_streak=float(np.median([x["streak"] for x in G])) if G else None, games=recs)
json.dump(summ, open(STEM + ".json", "w"), indent=1)
lines = [(f"판 {x['game']}: " + (f"넘은 줄 {x['streak']:2d}  " + ("성공 (10줄 이상)" if x["streak"] >= 10 else result(x)[4:]) if x["kind"] == "game" else result(x)[4:]),
          (150, 255, 150) if x["kind"] == "game" and x["streak"] >= 10 else (255, 200, 200)) for x in recs]
lines.append((f"10줄 이상 {ok}/{len(G)}판 · 통계 {100 * P:.0f}%면 {len(G)}판 중 {lo}~{hi}판이 보통 (90% 범위)", (255, 230, 150)))
for _ in range(0 if NOSUM else 250): put(card(last, lines, f"{NG}판 요약"))
wr.close()
sr = 48000; track = np.zeros(int((NF[0] / FPS + 1.0) * sr), np.float32)
for at, nm in SND:
  j0 = max(0, int(at * sr)); s_ = WAVS[nm] * (0.7 if nm == "peck_2" else 0.8); track[j0: j0 + len(s_)] += s_[: max(0, len(track) - j0)]
with wave.open(STEM + ".wav", "wb") as w:
  w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", STEM + "_noaudio.mp4", "-i", STEM + ".wav", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", STEM + ".mp4"], check=True)
os.remove(STEM + "_noaudio.mp4"); os.remove(STEM + ".wav")
print("SUMMARY", json.dumps({k: v for k, v in summ.items() if k != "games"}), f"frames {NF[0]} sounds {len(SND)}", flush=True)
