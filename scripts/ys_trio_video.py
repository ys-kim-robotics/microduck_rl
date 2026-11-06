"""ys_trio_video.py: the J/F/L baseline video (stage 7, v2 (user 2026-11-06). One clock for all three (J's start call starts L
and F; F uncorrected), J's real voice (runtime/sounds seed 1: greet = start call, peck = each pass of its clock
over BEEP_PH, its right-foot lift), J put in right after the rope passed, a hit ends the run. No title card.
Slow part: feet/rope, clearance per jump (A), the rope's circle at J (C), three clocks (D); E (L's head in J's eye
with the gaze box) is drawn on the J-eye inset in both parts.
  record <jumper ckpt> <dir> [secs 22]   dir/raw.mp4 (main view + J-eye inset, no text), dir/data.npz
  compose <dir> <ko|en> <out.mp4>        both languages share one run (GPU runs differ)
Env (record): SEED, ENTER_PH (deg, leader phase at the put-in; default ys_trio_scn's), BEAT_BANK_IN (running start).
Run from the repo root (uv run python scripts/ys_trio_video.py ...)."""
import sys, math, os, wave, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
NAN_R = "/usr/share/fonts/truetype/nanum/NanumSquare_acR.ttf"; NAN_B = "/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf"
W_, H_, FPS = 1280, 720, 50
AZ, EL, DIST, DY, DZ = 62.0, -32.0, 1.85, 0.28, 0.05
INSET = (W_ - 384 - 16, H_ - 216 - 16, 384, 216)  # x, y, w, h of J's eye


def record(ckpt, d, secs):
  sys.path.insert(0, "scripts")
  import torch, mujoco, imageio
  import ys_trio_scn as beat_scn; beat_scn.patch_cfg()
  import ys_eval
  import mjlab_microduck.tasks
  import mjlab_microduck.tasks.ys_trio_env_cfg as T
  import mjlab_microduck.tasks.ys_jump_env_cfg as J
  import mjlab_microduck.tasks.ys_beat as B
  from mjlab_microduck.tasks.ys_rope import ROPE_RADIUS
  from mjlab.utils.lab_api.math import matrix_from_quat
  WALK = beat_scn.WALKOUT  # BEAT_PICK_ONNX + BEAT_WALKOUT=1: L and F take the handles at the middle holder and back out
  TASK = "Mjlab-YsTrioJumpBeatPick-Flat-MicroDuck" if WALK else "Mjlab-YsTrioJumpBeat-Flat-MicroDuck"
  cfg = ys_eval.load_env_cfg(TASK, play=True)
  for k in ("sample_delta", "sample_offset"): cfg.events.pop(k)
  cfg.scene.num_envs, cfg.seed = 1, int(os.environ.get("SEED", 0))  # startup DR (battery, friction, encoder bias, mass) varies with SEED too
  ys_eval.fix_command(cfg, (0., 0., 0.)); cfg.viewer.height, cfg.viewer.width = H_, W_
  env = ys_eval.ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode="rgb_array")
  venv = ys_eval.RslRlVecEnvWrapper(env, clip_actions=ys_eval.load_rl_cfg(TASK).clip_actions)
  pol = ys_eval.make_policy(TASK, ckpt, venv)
  if os.environ.get("SEED"): torch.manual_seed(int(os.environ["SEED"])); torch.cuda.manual_seed_all(int(os.environ["SEED"])); venv.reset()
  B.CORRECT = False  # v2: one clock for all three, F uncorrected
  if os.environ.get("ENTER_PH"): beat_scn.ENTER_PH = math.radians(float(os.environ["ENTER_PH"]))
  env._ys_off = B.start_offset(1, env.device); off0 = math.degrees(float(env._ys_off[0]))
  beat_scn.away(env)
  if WALK:
    from mjlab.utils.lab_api.math import quat_apply, quat_inv
    env._ys_pick_onnx = beat_scn.PICK; ids1 = torch.arange(1, device=env.device)
    REL0 = {w: quat_apply(quat_inv(env.scene[w].data.root_link_quat_w), env.scene[w].data.site_pos_w[:, env.scene[w].find_sites("stem_end")[0][0]] - env.scene[w].data.root_link_pos_w)[0] for w in ("turner", "partner")}
    WS = beat_scn.new_ws(env, 1)
    EQI = beat_scn.setup_walkout(env, ids1, REL0, WS); got, gerr = torch.zeros(1, 2, dtype=torch.bool, device=env.device), torch.full((1, 2), math.nan, device=env.device)
  cmd = env.command_manager.get_term("head_pose"); dt = env.step_dt; act = env.action_manager.get_term("partner_joint_pos")
  rob, rope = env.scene["robot"], env.scene["rope"]
  r = env._offline_renderer; m = env.sim.mj_model; o = env.scene.env_origins[0].cpu().numpy()
  fov0 = m.vis.global_.fovy; fovc = 2 * math.degrees(math.atan(math.tan(math.radians(31)) * H_ / W_))
  heads = {k: env.scene[k].find_sites("head_imu")[0][0] for k in ("robot", "turner", "partner")}

  def main_view():
    r._cam.type = mujoco.mjtCamera.mjCAMERA_FREE; m.vis.global_.fovy = fov0
    r._cam.lookat[:] = o + np.array([0, DY, DZ]); r._cam.azimuth, r._cam.elevation, r._cam.distance = AZ, EL, DIST
    r.update(env.sim.data); return r.render().copy()

  def project(p):
    az, el = math.radians(AZ), math.radians(EL)
    f = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    up = np.array([-math.sin(el) * math.cos(az), -math.sin(el) * math.sin(az), math.cos(el)]); rt = np.cross(f, up)
    q = p - (o + np.array([0, DY, DZ]) - DIST * f); z = q @ f; th = math.tan(math.radians(fov0) / 2)
    return W_ / 2 * (1 + (q @ rt) / z / (th * W_ / H_)), H_ / 2 * (1 - (q @ up) / z / th)

  def j_eye():
    i = rob.find_sites("head_camera")[0][0]
    p = rob.data.site_pos_w[0, i].cpu().numpy(); R = matrix_from_quat(rob.data.site_quat_w[0, i]).cpu().numpy()
    r._cam.type = mujoco.mjtCamera.mjCAMERA_FREE; m.vis.global_.fovy = fovc
    r._cam.lookat[:] = p + R[:, 0] * 0.52; r._cam.distance = 0.5
    r.update(env.sim.data)
    for c in r._renderer.scene.camera:
      c.pos[:] = p + R[:, 0] * 0.02; c.forward[:] = R[:, 0]; c.up[:] = R[:, 2]
    return r.render().copy()

  def feet():  # sole heights, rope centre z and lateral place at J's x section, overlap per foot (trio_video.py)
    sec = T.rope_section(rope.data.body_link_pos_w, rob.data.root_link_pos_w[:, 0])
    w_, x_, y_, z_ = rob.data.root_link_quat_w.unbind(-1)
    yaw = torch.atan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
    yj = yaw.cos() * (sec[:, 0] - rob.data.root_link_pos_w[:, 1])
    soles = J.foot_clearance(rob, cmd._sole) + J.FOOT_Z0
    clr = []  # signed sole - rope top (m) while the rope is under that foot, else nan (<= 0: a hit)
    for j, (lo, hi) in enumerate(T.FOOT_BANDS):
      inb = (yj > lo) & (yj < hi) & (sec[:, 1] - ROPE_RADIUS < T.HEAD_TOP)
      clr.append(float(torch.where(inb, soles[:, j] - (sec[:, 1] + ROPE_RADIUS), math.nan)[0]))
    return [float(soles[0, 0]), float(soles[0, 1]), float(sec[0, 1]), float(yj[0])] + clr

  os.makedirs(d, exist_ok=True)
  out = imageio.get_writer(f"{d}/raw.mp4", fps=FPS, codec="libx264", quality=9, macro_block_size=8)
  K = ("t", "wphase", "tstart", "Lfell", "Ffell", "amp", "lead", "err", "soleL", "soleR", "rope_z", "rope_y", "clrL", "clrR", "inn", "emit", "jfell", "restart", "lab", "el", "az")
  rec = {k: [] for k in K}; obs = venv.get_observations(); delta = getattr(env, "_ys_delta", math.radians(60))
  with torch.inference_mode():
    STOP = tuple(float(v) for v in os.environ["STOP"].split(",")) if os.environ.get("STOP") else None  # delay, ramp, force (s)
    t_stop, t_hit = (STOP[2] if STOP else math.inf), math.inf
    for k in range(int(secs / dt)):
      beat_scn.enter(env)
      if STOP:  # J's end call at its first hit -> both turners' circle ramps to 0 (ys_trio_scn BEAT_STOP)
        env._ys_amp = torch.full((1,), min(1.0, max(0.0, 1 - (k * dt - t_stop) / max(STOP[1], dt))), device=env.device)
      if WALK: t_start = beat_scn.walkout_pre(env, WS)
      act_j = pol(obs)
      if STOP and beat_scn.JSTAND and k * dt >= t_stop: act_j = beat_scn.jstand(obs["actor"])  # J stands (BEAT_JSTAND onnx)
      cmd.amp[:] = 1.0; cmd.freq[:] = 1.25; cmd._write(); obs, *_ = venv.step(act_j)
      if STOP and math.isinf(t_hit) and bool(env._ys_in[0]) and k * dt > float(env._ys_tin[0]) + 0.04:
        if float(T.rope_overlap(env)[0]) > 0 or float(rob.data.projected_gravity_b[0, 2]) > -0.5:
          t_hit = k * dt; t_stop = min(t_stop, t_hit + STOP[0])
      if WALK:
        beat_scn.walkout_post(env, WS, EQI, REL0, env.episode_length_buf == 0, got, gerr); te_ = k * dt; ts_ = float(t_start[0])
        if abs(te_ - (beat_scn.T_ENTER - 0.5)) < dt / 2: beat_scn.spot_from_turners(env, REL0)  # J lines up on the rope
        rec["wphase"].append(0 if te_ < beat_scn.W_BITE + 0.2 else 1 if not bool(WS["arr"]["turner"][0]) else 2 if te_ < ts_ else 3); rec["tstart"].append(ts_)
      else:
        rec["wphase"].append(3); rec["tstart"].append(0.0)
      rec["Lfell"].append(bool(env.scene["turner"].data.projected_gravity_b[0, 2] > -0.5)); rec["Ffell"].append(bool(env.scene["partner"].data.projected_gravity_b[0, 2] > -0.5))
      rec["amp"].append(float(env._ys_amp[0]) if (STOP or WALK) else 1.0)
      t_, inn = k * dt, bool(env._ys_in[0]); lead = float(B.wrap(cmd.phase - delta)[0])
      past = (lead - B.BEEP_PH) % (2 * math.pi)  # J calls on its own clock as it passes BEEP_PH (its right-foot lift)
      rec["emit"].append(t_ - past / B.OMEGA if inn and past < B.OMEGA * dt else math.nan)
      rec["t"].append(t_); rec["lead"].append(math.degrees(lead)); rec["inn"].append(inn)
      rec["err"].append(math.degrees(float(B.wrap(env._ys_off + (act._beat.f.c if B.CORRECT else 0.0))[0])))
      for key, v in zip(("soleL", "soleR", "rope_z", "rope_y", "clrL", "clrR"), feet()): rec[key].append(v)
      rec["jfell"].append(bool(rob.data.projected_gravity_b[0, 2] > -0.5)); rec["restart"].append(int(env.episode_length_buf[0]) == 0)
      dv = T._cam_vec(env, "head_imu")[0][0]
      rec["el"].append(math.degrees(math.atan2(float(dv[2]), float(dv[0])))); rec["az"].append(math.degrees(math.atan2(-float(dv[1]), float(dv[0]))))
      rec["lab"].append([project(env.scene[n].data.site_pos_w[0, heads[n]].cpu().numpy() + np.array([0, 0, 0.07])) for n in ("robot", "turner", "partner")])
      fr = Image.fromarray(main_view()); x, y, w, hh = INSET
      fr.paste(Image.fromarray(j_eye()).resize((w, hh)), (x, y)); ImageDraw.Draw(fr).rectangle((x - 1, y - 1, x + w, y + hh), outline=(255, 255, 255), width=2)
      out.append_data(np.asarray(fr))
  out.close()
  np.savez(f"{d}/data.npz", off0=off0, dt=dt, beep_ph=math.degrees(B.BEEP_PH), gaze=(math.degrees(T.GAZE_V), math.degrees(T.GAZE_H)), **{k: np.array(v) for k, v in rec.items()})
  D = np.load(f"{d}/data.npz"); i0, hit, stk = run_end(D)
  print(f"recorded {d}: start {off0:+.1f} deg, J in {D['t'][i0]:.2f} s, restarts {int(D['restart'].sum())}, streak {stk}" + (f", HIT at {D['t'][hit]:.2f} s" if hit is not None else ", no hit"))


def run_end(D):
  """(entry step, first hit/fall step or None, rope passes cleared before it): a hit ends the run (user)."""
  inn = D["inn"]; i0 = int(np.argmax(inn)); ov = (np.nan_to_num(D["clrL"], nan=1) <= 0) | (np.nan_to_num(D["clrR"], nan=1) <= 0) | D["jfell"]
  bad = np.flatnonzero(ov & inn & (np.arange(len(inn)) >= i0)); hit = int(bad[0]) if len(bad) else None
  return i0, hit, ((hit if hit is not None else len(inn)) - i0) // 40


VOICE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ys_voice")  # runtime/sounds: sounds render <tag> <out.wav> --seed 1 --variant v
START_WAV, PECK_WAV, END_WAV = f"{VOICE}/greet_0.wav", f"{VOICE}/peck_2.wav", f"{VOICE}/alarm_0.wav"
TXT = {
  "ko": dict(steps=["① J의 시작 소리 → 셋이 같은 시계로 출발", "② L·F 줄 돌리기 (J는 옆에서 대기)", "③ J 입장 (줄이 막 지나간 순간)", "④ J는 오른발 들 때마다 '톡' · 걸리면 끝"],
             wsteps=["① 가운데 거치대에서 L·F가 손잡이를 묾", "② L이 뒷걸음으로 제자리로", "③ F가 뒷걸음으로 제자리로"], tfell="줄 돌리는 오리가 넘어짐 (실패)",
             hit_step="걸림 → 끝", stop_step="⑤ 걸림! J의 끝 소리 → L·F 멈춤", L="L (리더)", F="F (팔로워)", J="J (점퍼)", call="시작 소리!", peck="톡!",
             info="t {t:4.1f} s    연속 {k}번    F 시계 (L 대비) {e:+.0f}°", info0="t {t:4.1f} s    F 시계 (L 대비) {e:+.0f}°", eye="J의 눈: L 머리(●)는 상자 안에", card="0.5배속 슬로모션 + 실시간 그래프", slow="0.5배속",
             gF="J의 발과 줄 (J 위치)", gF_l=("왼발 바닥", "오른발 바닥", "줄 윗면 (발 아래)", "걸림"), gF_y="높이 (mm)", wait="J 대기 중",
             gA="점프마다 발-줄 여유 (mm)", gA_l=("왼발", "오른발"), gA_x="점프 (입장 후)", gA_0="0 = 닿음",
             gC="J 위치의 줄 단면 (L 쪽에서 봄)", gC_x="옆 (mm)", gC_y="높이 (mm)", gC_feet="J 발", gC_head="J 머리 높이",
             gD="세 시계", gD_l=("L 리더", "F 팔로워", "J 점퍼"), gD_v=("{:.0f}°", "L {:+.0f}°", "L +0°"), gD_b="▼ J 소리 위치", font_mpl=NAN_R),
  "en": dict(steps=["① J's start call → all three start one clock", "② L and F turn the rope (J waits aside)", "③ J steps in (right after the rope passes)", "④ J pecks on every right-foot lift · a hit ends it"],
             wsteps=["① L and F bite their handles at the middle holder", "② L backs out to its place", "③ F backs out to its place"], tfell="a turning duck fell (failure)",
             hit_step="hit → over", stop_step="⑤ hit! J's end call → L and F stop", L="L (leader)", F="F (follower)", J="J (jumper)", call="start call!", peck="peck!",
             info="t {t:4.1f} s    streak {k}    F clock vs L {e:+.0f}°", info0="t {t:4.1f} s    F clock vs L {e:+.0f}°", eye="J's eye: L's head (●) inside the box", card="0.5x slow motion + live graphs", slow="0.5x",
             gF="J's feet and the rope (at J)", gF_l=("left sole", "right sole", "rope top (under the feet)", "hit"), gF_y="height (mm)", wait="J waiting",
             gA="clearance per jump (mm)", gA_l=("left foot", "right foot"), gA_x="jump (after entry)", gA_0="0 = touch",
             gC="the rope at J, cross-section (from L)", gC_x="side (mm)", gC_y="height (mm)", gC_feet="J's feet", gC_head="J's head top",
             gD="three clocks", gD_l=("L leader", "F follower", "J jumper"), gD_v=("{:.0f}°", "L {:+.0f}°", "L +0°"), gD_b="▼ J's call", font_mpl=None),
}


def wav48(p):
  with wave.open(p) as w:
    assert w.getframerate() == 48000 and w.getnchannels() == 1, p
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
  return x


def compose(d, lang, outp):
  import imageio, imageio_ffmpeg
  import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
  from matplotlib import font_manager as fm
  sys.path.insert(0, "src"); import mjlab_microduck.tasks.ys_trio_env_cfg as T
  X = TXT[lang]; D = np.load(f"{d}/data.npz"); t = D["t"]; dt = float(D["dt"])
  if X["font_mpl"]:
    fm.fontManager.addfont(X["font_mpl"]); plt.rcParams["font.family"] = fm.FontProperties(fname=X["font_mpl"]).get_name()
  plt.rcParams["axes.unicode_minus"] = False
  fB, fM, fS = ImageFont.truetype(NAN_B, 34), ImageFont.truetype(NAN_B, 24), ImageFont.truetype(NAN_R, 17)
  i0, hit, _ = run_end(D); stopm = "amp" in D.files and D["amp"].min() < 1.0  # stop mode: show the turners stopping
  n = len(t) if hit is None else min(len(t), hit + int((5.0 if stopm else 1.0) / dt))  # a hit: show 1 s (stop: 5 s) more
  t_in = t[i0]; emit = D["emit"][~np.isnan(D["emit"])]; emit = emit[emit < t[n - 1]]
  streak = lambda i: max(0, min(i, hit if hit is not None else i) - i0) // 40 if i >= i0 else 0
  clr = np.stack([D["clrL"], D["clrR"]]) * 1000; npass = (len(t) - i0) // 40  # pass k = steps i0+40k .. i0+40(k+1)
  pmin = np.array([[np.nanmin(c[i0 + 40 * k: i0 + 40 * (k + 1)]) if np.isfinite(c[i0 + 40 * k: i0 + 40 * (k + 1)]).any() else np.nan for k in range(npass)] for c in clr])
  gv, gh = (float(v) for v in D["gaze"]); VH = math.degrees(math.atan(math.tan(math.radians(31)) * 9 / 16))

  def tag(dr, xy, text, fill, font, s=1.0):
    x, y = xy[0] * s, xy[1] * s; b = dr.textbbox((0, 0), text, font=font); w, h = b[2] - b[0], b[3] - b[1]
    dr.rounded_rectangle((x - w / 2 - 8, y - h - 12, x + w / 2 + 8, y - 2), 7, fill=(0, 0, 0, 170)); dr.text((x - w / 2, y - h - 8), text, font=font, fill=fill)

  def eye_px(el, az, s):  # E: a direction in J's camera -> inset pixel (62 deg wide, 16:9)
    x, y, w, h = (v * s for v in INSET)
    return x + w / 2 * (1 + math.tan(math.radians(az)) / math.tan(math.radians(31))), y + h / 2 * (1 - math.tan(math.radians(el)) / math.tan(math.radians(VH)))

  def overlay(img, i, s=1.0, info=True, fbig=fB, fmid=fM):
    ti = t[i]; ov = Image.new("RGBA", img.size); dr = ImageDraw.Draw(ov)
    se = ti - emit[emit <= ti].max() if (emit <= ti).any() else 9.0; lj, ll, lf = D["lab"][i]
    tsc = float(D["tstart"][i]) if "tstart" in D.files else 0.0; called = 0 <= ti - tsc < 0.6
    jt = X["J"] + (f"   {X['call']}" if called else f"   {X['peck']}" if se < 0.25 else "")
    tag(dr, lj, jt, (255, 220, 60) if (called or se < 0.25) else (255, 255, 255), fmid, s)
    tag(dr, ll, X["L"], (255, 110, 110), fmid, s); tag(dr, lf, X["F"], (120, 255, 140), fmid, s)
    wph = int(D["wphase"][i]) if "wphase" in D.files else 3; tst = float(D["tstart"][i]) if "tstart" in D.files else 0.0
    tfell = ("Lfell" in D.files) and (D["Lfell"][: i + 1].any() or D["Ffell"][: i + 1].any())
    over = hit is not None and i >= hit
    if tfell: step = X["tfell"]
    elif over: step = X["stop_step"] if stopm else X["hit_step"]
    elif wph < 3: step = X["wsteps"][wph]
    else: step = X["steps"][0 if ti < tst + 0.8 else 1 if not D["inn"][i] else 2 if ti < t_in + 1.6 else 3]
    bw = dr.textbbox((0, 0), step, font=fbig)[2]; dr.rounded_rectangle((12, 12, 12 + bw + 24, 12 + fbig.size + 22), 10, fill=(170, 30, 30, 200) if over else (0, 0, 0, 160)); dr.text((24, 18), step, font=fbig, fill=(255, 255, 255))
    if info:
      s_ = X["info"].format(t=ti, k=streak(i), e=D["err"][i]) if D["inn"][i] else X["info0"].format(t=ti, e=D["err"][i]); bw = dr.textbbox((0, 0), s_, font=fmid)[2]
      dr.rounded_rectangle((12, img.size[1] - 52, 12 + bw + 24, img.size[1] - 12), 8, fill=(0, 0, 0, 150)); dr.text((24, img.size[1] - 48), s_, font=fmid, fill=(220, 235, 255))
    x, y, w, h = (v * s for v in INSET)  # E: gaze box (+-gh x +-gv deg) and L's head with a 0.5 s trail
    bx, by = w / 2 * math.tan(math.radians(gh)) / math.tan(math.radians(31)), h / 2 * math.tan(math.radians(gv)) / math.tan(math.radians(VH))
    dr.rectangle((x + w / 2 - bx, y + h / 2 - by, x + w / 2 + bx, y + h / 2 + by), outline=(255, 220, 60, 230), width=max(1, round(2 * s)))
    seen = lambda q: x < q[0] < x + w and y + 26 * s < q[1] < y + h  # in J's picture (L in front of it)
    pts = [q for j in range(max(0, i - 25), i + 1) if D["az"][j] ** 2 < 80 ** 2 and seen(q := eye_px(D["el"][j], D["az"][j], s))]
    if len(pts) > 1: dr.line(pts, fill=(255, 90, 90, 150), width=max(1, round(2 * s)))
    if pts and pts[-1] == eye_px(D["el"][i], D["az"][i], s):
      px, py = pts[-1]; r_ = 7 * s; inside = abs(D["el"][i]) <= gv and abs(D["az"][i]) <= gh
      dr.ellipse((px - r_, py - r_, px + r_, py + r_), outline=(255, 60, 60, 255), width=max(2, round(3 * s)), fill=None if inside else (255, 60, 60, 120))
    dr.rectangle((x, y, x + w, y + 26 * s), fill=(0, 0, 0, 200)); dr.text((x + 8, y + 4), X["eye"], font=fS if s == 1 else ImageFont.truetype(NAN_R, 12), fill=(255, 255, 255))
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")

  def card(lines):
    im = Image.new("RGB", (W_, H_), (12, 16, 24)); dr = ImageDraw.Draw(im); y = H_ / 2 - 40 * len(lines)
    for txt, f, c in lines:
      w = dr.textbbox((0, 0), txt, font=f)[2]; dr.text(((W_ - w) / 2, y), txt, font=f, fill=c); y += f.size + 30
    return np.asarray(im)

  # slow-part graphs: D (448x234) and C (448x234) right of the main view; feet/rope (768x252) and A (512x252) below
  dpi, BG, AX, TC = 100, "#10141c", "#18202c", "#c8d0dc"
  figD = plt.figure(figsize=(4.48, 2.34), dpi=dpi); figC, axC = plt.subplots(figsize=(4.48, 2.34), dpi=dpi)
  figF, axF = plt.subplots(figsize=(7.68, 2.52), dpi=dpi); figA, axA = plt.subplots(figsize=(5.12, 2.52), dpi=dpi)
  for f_ in (figD, figC, figF, figA): f_.patch.set_facecolor(BG)
  for a in (axC, axF, axA):
    a.set_facecolor(AX); a.tick_params(colors=TC, labelsize=8); [sp.set_color("#5a6578") for sp in a.spines.values()]; a.grid(alpha=.25)
  leg = dict(fontsize=8, facecolor=BG, labelcolor="#e0e6ee", framealpha=.8)
  # D: three clocks on one dial (L's clock frame), J's call marker
  axD = figD.add_axes([0.05, 0.1, 0.44, 0.8], projection="polar"); axD.set_facecolor(AX)
  axD.set_theta_zero_location("N"); axD.set_theta_direction(-1); axD.set_rticks([]); axD.set_ylim(0, 1.12)
  axD.set_xticks(np.radians([0, 90, 180, 270])); axD.set_xticklabels(["0°", "90°", "180°", "270°"], color=TC, fontsize=7); axD.tick_params(pad=-1); axD.spines["polar"].set_color("#5a6578")
  CL = ("#ff6e6e", "#78ff8c", "#ffdc3c"); bp = math.radians(float(D["beep_ph"]))
  hL, = axD.plot([], [], color=CL[0], lw=4, solid_capstyle="round"); hF, = axD.plot([], [], color=CL[1], lw=2.5); hJ, = axD.plot([], [], color=CL[2], lw=1.5, ls="--")
  axD.plot([bp], [1.06], "v", color="#ffdc3c", ms=8); glow, = axD.plot([bp], [1.06], "o", color="#ffdc3c", ms=16, alpha=.0)
  figD.text(0.56, 0.86, X["gD"], color="white", fontsize=12)
  vt = [figD.text(0.56, 0.66 - 0.17 * k, "", color=CL[k], fontsize=10) for k in range(3)]; figD.text(0.56, 0.12, X["gD_b"], color="#ffdc3c", fontsize=9)
  # C: the rope's cross-section at J over the last cycle, J's feet live
  lC, = axC.plot([], [], color="#e0e6ee", lw=1.5, alpha=.6); dC, = axC.plot([], [], "o", color="white", ms=7)
  fL, = axC.plot([], [], color="#4aa3ff", lw=5); fR, = axC.plot([], [], color="#ff9f40", lw=5)
  axC.axhline(0, color="#8a96a8", lw=1); axC.plot([-70, 70], [T.HEAD_TOP * 1000] * 2, color="#8a96a8", ls=":", lw=1); axC.text(75, T.HEAD_TOP * 1000 - 8, X["gC_head"], color="#8a96a8", fontsize=7)
  axC.set_xlim(-330, 330); axC.set_ylim(-20, 520); axC.set_title(X["gC"], color="white", fontsize=10, pad=3); axC.set_xlabel(X["gC_x"], color=TC, fontsize=8, labelpad=0); axC.set_ylabel(X["gC_y"], color=TC, fontsize=8)
  axC.invert_xaxis()  # seen from L, facing J: J's left (+y) on the viewer's right
  # feet/rope over a 2.4 s window (as v1 graph A)
  sole = np.stack([D["soleL"], D["soleR"]]) * 1000; top = (D["rope_z"] + 0.003) * 1000  # + ROPE_RADIUS
  under = (D["rope_y"] > T_LO) & (D["rope_y"] < T_HI) & D["inn"]; rope_show = np.where(under, top, np.nan)
  bad = np.zeros(len(t), bool)
  if hit is not None: bad[hit] = True
  lL, = axF.plot([], [], color="#4aa3ff", lw=2, label=X["gF_l"][0]); lR, = axF.plot([], [], color="#ff9f40", lw=2, label=X["gF_l"][1])
  lP, = axF.plot([], [], color="#f0f0f0", lw=3, label=X["gF_l"][2]); lH, = axF.plot([], [], "x", color="#ff4040", ms=10, mew=3, label=X["gF_l"][3])
  axF.set_ylim(-5, 75); axF.set_ylabel(X["gF_y"], color=TC, fontsize=9); axF.set_title(X["gF"], color="white", fontsize=11, pad=3)
  axF.legend(loc="upper right", ncol=4, **leg); wF = axF.text(0.5, 0.45, X["wait"], transform=axF.transAxes, ha="center", color="#8899aa", fontsize=14)
  # A: the closest each foot came to the rope, per jump
  axA.axhline(0, color="#ff4040", lw=1.2); axA.text(0.99, 0.03, X["gA_0"], transform=axA.transAxes, ha="right", color="#ff8080", fontsize=8)
  axA.set_ylim(-8, 62); axA.set_title(X["gA"], color="white", fontsize=11, pad=3); axA.set_xlabel(X["gA_x"], color=TC, fontsize=8, labelpad=0)
  bars = {}; wA = axA.text(0.5, 0.45, X["wait"], transform=axA.transAxes, ha="center", color="#8899aa", fontsize=14)
  from matplotlib.patches import Patch
  axA.legend(handles=[Patch(color="#4aa3ff", label=X["gA_l"][0]), Patch(color="#ff9f40", label=X["gA_l"][1])], loc="upper right", ncol=2, **leg)
  for f_ in (figC, figF, figA): f_.tight_layout(pad=0.6)

  def graphs(i):
    ti = t[i]; th = math.radians(D["lead"][i])
    hL.set_data([th, th], [0, 1.0]); hF.set_data([th + math.radians(D["err"][i])] * 2, [0, 0.88]); hJ.set_data([th, th], [0, 0.78])
    se = ti - emit[emit <= ti].max() if (emit <= ti).any() else 9.0; glow.set_alpha(0.7 if se < 0.15 else 0.0)
    for k, txt in enumerate((X["gD_v"][0].format(D["lead"][i] % 360), X["gD_v"][1].format(D["err"][i]), X["gD_v"][2])): vt[k].set_text(f"{X['gD_l'][k]}  {txt}")
    s0 = max(0, i - 40); lC.set_data(D["rope_y"][s0: i + 1] * 1000, D["rope_z"][s0: i + 1] * 1000); dC.set_data([D["rope_y"][i] * 1000], [D["rope_z"][i] * 1000])
    for ln, (lo, hi), z in ((fL, T.FOOT_BANDS[0], sole[0, i]), (fR, T.FOOT_BANDS[1], sole[1, i])): ln.set_data([lo * 1000, hi * 1000], [max(z, 0), max(z, 0)])
    s0 = max(0, i - int(2.4 / dt)); ts = t[s0: i + 1]
    lL.set_data(ts, sole[0, s0: i + 1]); lR.set_data(ts, sole[1, s0: i + 1]); lP.set_data(ts, rope_show[s0: i + 1])
    hh = bad[s0: i + 1]; lH.set_data(ts[hh], np.full(hh.sum(), 3.0)); axF.set_xlim(ti - 2.4, ti + 0.05); wF.set_visible(not D["inn"][i]); wA.set_visible(not D["inn"][i])
    done = 0 if i < i0 else min(npass, (i - i0) // 40 + (1 if hit is not None and i >= hit else 0))
    if hit is not None: done = min(done, (hit - i0) // 40 + 1)  # hit = end: no bars for the passes after it
    for k in range(done):
      if k in bars: continue
      for j, dx, col in ((0, -0.2, "#4aa3ff"), (1, 0.2, "#ff9f40")):
        v = pmin[j, k]
        if np.isfinite(v): bars[(k, j)] = axA.bar(k + 1 + dx, v if abs(v) > 0.6 else 0.6 * np.sign(v or 1), 0.36, color="#ff4040" if v <= 0 else col)
      bars[k] = True
    axA.set_xlim(0.3, max(12, done + 0.7) + 0.0)
    out = []
    for f_ in (figD, figC, figF, figA):
      f_.canvas.draw(); out.append(np.asarray(f_.canvas.buffer_rgba())[..., :3].copy())
    return out

  CARD2 = 1.2; tmp = outp.replace(".mp4", "_noaudio.mp4")
  wr = imageio.get_writer(tmp, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
  for i, fr in enumerate(imageio.get_reader(f"{d}/raw.mp4")):
    if i >= n: break
    wr.append_data(np.asarray(overlay(Image.fromarray(fr), i)))
  c2 = card([(X["card"], ImageFont.truetype(NAN_B, 50), (255, 255, 255))])
  for _ in range(int(CARD2 * FPS)): wr.append_data(c2)
  sm = 832 / W_; fbs, fms = ImageFont.truetype(NAN_B, 22), ImageFont.truetype(NAN_B, 16)
  for i, fr in enumerate(imageio.get_reader(f"{d}/raw.mp4")):
    if i >= n: break
    canvas = Image.new("RGB", (W_, H_), (16, 20, 28)); mv = overlay(Image.fromarray(fr).resize((832, 468)), i, s=sm, info=False, fbig=fbs, fmid=fms)
    dr = ImageDraw.Draw(mv); bw = dr.textbbox((0, 0), X["slow"], font=fms)[2]; dr.rounded_rectangle((832 - bw - 30, 10, 822, 40), 8, fill=(200, 40, 40)); dr.text((832 - bw - 20, 14), X["slow"], font=fms, fill=(255, 255, 255))
    gD, gC, gF, gA = graphs(i)
    canvas.paste(mv, (0, 0)); canvas.paste(Image.fromarray(gD), (832, 0)); canvas.paste(Image.fromarray(gC), (832, 234)); canvas.paste(Image.fromarray(gF), (0, 468)); canvas.paste(Image.fromarray(gA), (768, 468))
    a = np.asarray(canvas); wr.append_data(a); wr.append_data(a)
  wr.close()

  # J's voice: the start call at each part's t=0, a peck at every call (same pitch in slow motion)
  sr = 48000; total = n * dt + CARD2 + 2 * n * dt + 0.5; track = np.zeros(int(total * sr), np.float32)
  call, peck = 0.8 * wav48(START_WAV), 0.7 * wav48(PECK_WAV)
  def put(at, s_):
    j0 = int(at * sr); track[j0: j0 + len(s_)] += s_[: max(0, len(track) - j0)]
  t_call = float(D["tstart"][np.isfinite(D["tstart"])].max()) if "tstart" in D.files and np.isfinite(D["tstart"]).any() else 0.0
  for base, k in ((0.0, 1.0), (n * dt + CARD2, 2.0)):
    put(base + k * t_call, call)
    if stopm and hit is not None: put(base + k * t[hit], 0.8 * wav48(END_WAV))
    for ts in emit: put(base + k * ts, peck)
  wav = outp.replace(".mp4", ".wav")
  with wave.open(wav, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((np.clip(track, -1, 1) * 32767).astype(np.int16).tobytes())
  subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", tmp, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", outp], check=True)
  os.remove(tmp); os.remove(wav)
  print(f"composed {outp}: {total:.1f} s, {len(emit)} pecks, streak {streak(n - 1)}" + (f", hit at {t[hit]:.2f} s" if hit is not None else ""))


T_LO, T_HI = -0.065, 0.064  # m: J's feet across the rope line (FOOT_BANDS outer edges)

if __name__ == "__main__":
  if sys.argv[1] == "record":
    record(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 22)
  else:
    compose(sys.argv[2], sys.argv[3], sys.argv[4])
