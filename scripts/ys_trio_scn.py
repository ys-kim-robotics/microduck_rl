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
BEAT_LOAD=1: per-joint actuator torque of J / L / F from J's entry + 0.5 s (p99, max, rms vs the BAM xl330 stall).
Also importable: away(env), enter(env) for videos."""
import sys, math, os; sys.path.insert(0, "scripts")
import numpy as np, torch
T_ENTER, AWAY, EPISODE = 6.0, (0.0, 0.8, 0.0), 30.0  # s, m, s (trained at 20 s: longer to see after convergence)
OMEGA_DT = 2 * math.pi * 1.25 * 0.02  # rad of rope per step
ENTER_PH = math.radians(225)  # rad, leader phase at the put-in: the rope just passed (scn_v2_sweep.eval: standing
# put-in, 1st pass hit 63% here vs 89-100% elsewhere; was 154, the left landing: 100%)
TRACK = ("_ys_cam", "_ys_cam_ball")  # ys_trio_env_cfg._cam_track state of the jumper's head / ball trackers
INPLACE = os.environ.get("BEAT_INPLACE") == "1"  # the jumper jumps in its spot from t 0 (training-like): the 'entry' only starts the count
BANK_OUT, BANK_IN = os.environ.get("BEAT_BANK_OUT"), os.environ.get("BEAT_BANK_IN")  # INPLACE: save its state at the
# entry; else: put it in with that state (already jumping: a stand-in for walking in) instead of standing

LOAD = os.environ.get("BEAT_LOAD") == "1"
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
  if len(out) and not INPLACE:
    st = torch.cat([env._ys_spot[out], torch.zeros(len(out), 6, device=env.device)], 1); st[:, :3] += torch.tensor(AWAY, device=env.device)
    rob.write_root_state_to_sim(st, out)
    rob.write_joint_state_to_sim(rob.data.default_joint_pos[out].clone(), torch.zeros_like(rob.data.default_joint_pos[out]), env_ids=out)
  lead = env.command_manager.get_term("head_pose").phase - getattr(env, "_ys_delta", math.radians(60))
  ids = (~done & (te >= T_ENTER) & (torch.remainder(lead - ENTER_PH, 2 * math.pi) < OMEGA_DT)).nonzero()[:, 0]
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
  if len(ids) and not INPLACE:
    if BANK_IN:
      bk = env._ys_bank
      rob.write_root_state_to_sim(torch.cat([bk["pos"][ids] + env.scene.env_origins[ids], bk["quat"][ids], bk["lin"][ids], bk["ang"][ids]], 1), ids)
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
  TASK = "Mjlab-YsTrioJumpBeat-Flat-MicroDuck"; N = int(os.environ.get("BEAT_N", "128")); SECS = EPISODE - 0.1; CURV = {}
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
  ref = None; PHS = [math.radians(float(v)) for v in os.environ["BEAT_ENTER_PH"].split(",")] if os.environ.get("BEAT_ENTER_PH") else [ENTER_PH]
  RUNS = [(ph, X) for ph in PHS for X in STARTS]
  for ENTER_PH, X in RUNS:
    globals()["ENTER_PH"] = ENTER_PH; ins, fj, rs, sole, leads, taus = [], [], [], [], [], []
    fell = torch.zeros(N, 3, dtype=torch.bool, device=dev); ce, gap, pen, vis = [], [], [], []
    with torch.inference_mode():
      if os.environ.get("SEED"): torch.manual_seed(int(os.environ["SEED"])); torch.cuda.manual_seed_all(int(os.environ["SEED"]))
      venv.reset(); away(env); obs = venv.get_observations()
      ear = X == "ear"
      env._ys_off = B.start_offset(N, dev) if ear else torch.where(torch.rand(N, device=dev) < 0.5, -1.0, 1.0) * math.radians(float(X)); c = None
      off0 = env._ys_off.clone(); dsg = torch.where(torch.rand(N, device=dev) < 0.5, -1.0, 1.0)
      for k in range(int(SECS / dt)):
        enter(env)
        cmd.amp[:] = 1.0; cmd.freq[:] = 1.25; cmd._write(); obs, *_ = venv.step(pol(obs))
        re = env.episode_length_buf == 0  # restarted: a new start (sound: a new draw; fixed X: the same offset)
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
    if LOAD: print_load(torch.stack(taus), torch.stack(ins), rob.joint_names, int(0.5 / dt))
    ph_s = "MU_LAND" if ENTER_PH is None else f"{math.degrees(ENTER_PH):.0f}"
    print(f"enter {ph_s} start {X}: STREAK (hit=end) median {np.median(stk):4.1f}  mean {stk.mean():4.1f}  >=10 {100 * (stk >= 10).mean():3.0f}%  >=20 {100 * (stk >= 20).mean():3.0f}%"
          f"  hit in passes 1-3 {100 * (hit & (stk < 3)).mean():3.0f}%  1st pass {100 * (hit & (stk == 0)).mean():3.0f}%  no hit to end {100 * (~hit).mean():3.0f}% (max ~{full})", flush=True)
    if BANK_OUT: torch.save(env._ys_bank, BANK_OUT); print(f"  saved the jumper's entry states to {BANK_OUT}")
    print(f"start {X}: clean by window (J enters ~6-7 s): " + "  ".join(win))
    print(f"start {X:>3s}: end |clock err| p50 {np.percentile(CE[-1], 50):4.0f} p90 {np.percentile(CE[-1], 90):4.0f}  BEFORE motion err p50 {np.percentile(ME[b], 50):4.0f} p90 {np.percentile(ME[b], 90):4.0f} deg, cam valid {100 * V[b].mean():3.0f}%"
          f" | AFTER clock conv p50 {np.percentile(conv, 50):4.1f} p90 {np.percentile(conv, 90):4.1f} s, motion err p50 {np.percentile(ME[aft], 50):4.0f} p90 {np.percentile(ME[aft], 90):4.0f} deg,"
          f" cam valid {100 * V[t >= T_ENTER + 1].mean():3.0f}%, clean t>=16 {100 * (cyc == 0).mean():3.0f}%, fell J/L/F {(100 * fell.float().mean(0)).round().tolist()}%", flush=True)
  if os.environ.get("NPZ"): np.savez(os.environ["NPZ"], dt=dt, **CURV)
