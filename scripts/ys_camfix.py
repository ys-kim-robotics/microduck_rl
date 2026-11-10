"""7-15 (1-3 step 3, DEPLOY-ys.md): J's position from its own camera, fused with the odometry, for scripts/ys_fullset.py (CAM_FIX=1,
needs ODO=1). Only what the robot would have: the ball tracker's delivered frames (u, v, valid; captured `lag` ticks before delivery),
the encoder angles (FK to the head camera), the trunk quaternion, J's clock phase and the odometry position.

L's red ball runs an ellipse (radii RL lateral, RZ vertical: L's measured motion) about its centre A across the rope axis, at the
angle -clock - DELTA. In a W-second window the camera rays fix A in the odometry frame (least squares, 3 unknowns), so J - A =
odometry - A and the odometry drift cancels. J's spot = A + C_NOM (axis frame: the sim's median, s16/cam). The correction
(spot - A - C_NOM) is blended into the odometry position with time constant TAU; without enough frames it holds.
Constants from the 재학습 7 recording (s16/cam/campos.py): the radius must be fixed (free, a small near circle and a big far
one look the same); every game fitted DELTA 255-265 deg (sim: the rope's offset from J's clock is the same in every game; the robot
would fit it online).
7-15 (1-4 + user): J may look at F's ball too while it waits (ys_fullset WAITLOOK=LF; curF = J's tracker on F's ball). F's ball runs
its own ellipse (RLF, RZF, DELTAF); while J waits, the two centres give J's spot directly (their middle + CMID in the axis frame),
which replaces C_NOM for that game by J's own measured offset from L's ball (F's placement then no longer an error).
"""
import math
import os

import mujoco
import numpy as np

RL, RZ, DELTA, C_NOM, W = 0.117, 0.108, math.radians(260.0), (-0.3928, 0.0610), 1.0
RLF, RZF, DELTAF, CMID = 0.120, 0.109, math.radians(259.3), (-0.0009, 0.0588)  # F's ball, spot - centres' middle: s16/wait/fball.py
# (재학습 7, 60 games; L refitted the same way: 119 / 109 mm, 258 deg. J's spot vs L's centre spreads 0.7 / 1.6 cm, vs the middle 0.5 / 1.0)
TAU, NMIN = float(os.environ.get("CAMFIX_TAU", "0.5")), int(os.environ.get("CAMFIX_NMIN", "6"))
W_WAIT = float(os.environ.get("CAMFIX_WWAIT", str(W)))
# 7-15: the balls also move along the axis in step with the circle (the turners lean as they turn: L 40 mm, F 38 mm, the other way;
# s16/wait). Along-axis offset = a1 cos(angle) + a2 sin(angle) (m, median of 60 games). Without it, J standing and looking along the
# axis put L's centre 1.6 cm off (exact frames: 1.3 cm); with it 0.7 cm. But on the full set it gained nothing (4 same-condition pairs,
# 2 seeds each: -4%p on average, within the noise; the error after arrival stays ~2.3 cm): off by default, CAMFIX_ALONG=1 turns it on.
ALONG = os.environ.get("CAMFIX_ALONG", "0") == "1"
AL_L, AL_F = ((0.0085, 0.0377), (0.0042, -0.0360)) if ALONG else ((0.0, 0.0), (0.0, 0.0))  # s: the frame window while J stands and waits (no drift to outrun)  # blend time constant (s), frames for a fix
TH = math.tan(math.radians(31.0)); TV = TH * 9 / 16  # the sim camera's half-FOV (ys_trio_env_cfg CAM_TAN_H/V)
JN = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
      "mouth", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]  # duck_ipc_proto::JOINT_NAMES


class CamFix:
  def __init__(self, n, dt, xml="src/mjlab_microduck/robot/microduck/scene.xml"):
    self.m = mujoco.MjModel.from_xml_path(xml); self.d = mujoco.MjData(self.m)
    self.qadr = [(i, self.m.jnt_qposadr[self.m.joint(j).id]) for i, j in enumerate(JN) if j != "mouth"]
    self.cs = self.m.site("head_camera").id; self.n, self.dt = n, dt; self.reset()

  def reset(self):
    self.hist = [{} for _ in range(self.n)]  # tick -> (encoders, quat, odometry xyz, clock phase)
    self.frames = [[] for _ in range(self.n)]  # L's ball: (capture tick, P = I - d d^T, P (camera centre - ellipse offset))
    self.framesF = [[] for _ in range(self.n)]  # F's ball, same
    self.last = self.lastF = None; self.corr = np.zeros((self.n, 2)); self.seen = np.zeros(self.n, bool)
    self.C = np.tile(np.array(C_NOM), (self.n, 1)); self.AF, self.AL = [None] * self.n, [None] * self.n  # per game: spot - L's centre
    # (axis frame); F's and L's last centres while J waits

  def _ray(self, enc, quat):  # camera centre (trunk frame) and world ray rotation from the encoders and the IMU
    self.d.qpos[:] = 0; self.d.qpos[3] = 1
    for i, q in self.qadr: self.d.qpos[q] = enc[i]
    mujoco.mj_kinematics(self.m, self.d); R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); Rt = R.reshape(3, 3)
    return Rt @ self.d.site_xpos[self.cs], Rt @ self.d.site_xmat[self.cs].reshape(3, 3)

  def _add(self, F, H, c, last, kc, h, rl, rz, delta, al=(0.0, 0.0)):  # one new delivered frame of a ball -> its ray constraint on
    # that ball's centre
    if last is None or c[2] <= 0 or not np.any(c != last) or kc not in H: return
    en, qu, od, p = H[kc]; pc, Rc = self._ray(en, qu); dd = Rc @ np.array([1.0, -c[0] * TH, c[1] * TV]); dd /= np.linalg.norm(dd)
    e1, e2, e0 = np.array([-math.sin(h), math.cos(h), 0.0]), np.array([0.0, 0.0, 1.0]), np.array([math.cos(h), math.sin(h), 0.0]); a = -p - delta
    w = rl * math.cos(a) * e1 + rz * math.sin(a) * e2 + (al[0] * math.cos(a) + al[1] * math.sin(a)) * e0
    P = np.eye(3) - np.outer(dd, dd); F.append((kc, P, P @ (od + pc - w)))

  @staticmethod
  def _centre(F):  # least-squares centre of the window's rays, None when the rays are all alike
    M = sum(f[1] for f in F)
    return None if np.linalg.cond(M) > 1e6 else np.linalg.solve(M, sum(f[2] for f in F))

  def step(self, k, ids, enc, quat, odo, ph, cur, lag, spot, curF=None, wait=None):
    """ids: the envs J is on; enc (n, 15), quat (n, 4), odo (n, 3) world odometry position, ph (n,): theirs; cur (N, 3) the ball
    tracker's last frame, lag (N,), spot (N, 3) world xy + heading of J's spot; curF (N, 3) the tracker on F's ball and wait (n,)
    J not yet set off (both optional). -> (n, 2) correction to add to the odometry xy."""
    for j, e in enumerate(ids):
      H = self.hist[e]; H[k] = (enc[j], quat[j], odo[j], ph[j])
      for t in [t for t in H if t < k - 10]: del H[t]
      kc, h, wj = k - int(lag[e]), spot[e, 2], wait is not None and bool(wait[j]); win = (W_WAIT if wj else W) / self.dt
      self._add(self.frames[e], H, cur[e], None if self.last is None else self.last[e], kc, h, RL, RZ, DELTA, AL_L)
      if curF is not None:
        self._add(self.framesF[e], H, curF[e], None if self.lastF is None else self.lastF[e], kc, h, RLF, RZF, DELTAF, AL_F)
        FF = self.framesF[e] = [f for f in self.framesF[e] if f[0] > k - win]
        if wj and len(FF) >= NMIN:
          AF = self._centre(FF)
          if AF is not None: self.AF[e] = AF[:2]
      F = self.frames[e] = [f for f in self.frames[e] if f[0] > k - win]
      A = None
      if len(F) >= NMIN:
        A = self._centre(F)
        if A is None: continue  # rays all alike: no fix this tick
        A = A[:2]
        if wj: self.AL[e] = A
      elif wj: A = self.AL[e]  # J stands and looks elsewhere (at F): L's centre from its look still holds
      if A is None: continue
      Rh = np.array([[math.cos(h), -math.sin(h)], [math.sin(h), math.cos(h)]])
      if curF is not None and wj and self.AL[e] is not None and self.AF[e] is not None:  # both centres while waiting: J's own spot
        self.C[e] = Rh.T @ ((self.AL[e] + self.AF[e]) / 2 + Rh @ np.array(CMID) - self.AL[e])  # offset from L's (F's place included)
      self.corr[e] += self.dt / TAU * ((spot[e, :2] - A - Rh @ self.C[e]) - self.corr[e]); self.seen[e] = True
    self.last = cur.copy()
    if curF is not None: self.lastF = curF.copy()
    return self.corr[ids]


if __name__ == "__main__":  # check: a ball on the ellipse seen from a still camera puts the spot where it is
  import types
  for off in ((0.0, 0.0), (0.03, -0.02)):  # the odometry right, then 3 cm / -2 cm off: odometry + correction must land on the truth (0, 0)
    cf = CamFix(1, 0.02); m, d = cf.m, cf.d
    d.qpos[:] = 0; d.qpos[3] = 1; mujoco.mj_kinematics(m, d); pc0 = d.site_xpos[cf.cs].copy(); Rc0 = d.site_xmat[cf.cs].reshape(3, 3)
    odo = np.array([off[0], off[1], 0.0]); h = 0.0  # J truly at the origin, facing +x; the axis along +x
    A = pc0 + Rc0 @ np.array([0.30, 0.0, 0.0])  # the ball centre 30 cm ahead of the camera
    spot = np.array([[A[0] + C_NOM[0], A[1] + C_NOM[1], h]]); quat = np.array([1.0, 0.0, 0.0, 0.0]); enc = np.zeros(15)
    for k in range(150):
      p = 2 * math.pi * 1.25 * k * 0.02; a = -p - DELTA
      B = A + RL * math.cos(a) * np.array([0.0, 1.0, 0.0]) + RZ * math.sin(a) * np.array([0.0, 0.0, 1.0]) + (AL_L[0] * math.cos(a) + AL_L[1] * math.sin(a)) * np.array([1.0, 0.0, 0.0])
      dc = Rc0.T @ (B - pc0); u, v = -dc[1] / dc[0] / TH, dc[2] / dc[0] / TV
      corr = cf.step(k, [0], enc[None], quat[None], odo[None], np.array([p]), np.array([[u, v, 1.0 if abs(u) <= 1 and abs(v) <= 1 else 0.0]]),
                     np.array([0]), spot)
    assert np.allclose(odo[:2] + corr[0], 0.0, atol=1e-3), (off, corr)
  print("camfix self-check ok", corr[0])
  # F too (WAITLOOK=LF): L's ball first, then F's, while waiting; the spot sits at the centres' middle + CMID, not at A + C_NOM
  cf = CamFix(1, 0.02); odo = np.array([0.03, -0.02, 0.0]); B = A + np.array([0.0, -0.08, 0.0])
  spot = np.array([[(A[0] + B[0]) / 2 + CMID[0], (A[1] + B[1]) / 2 + CMID[1], 0.0]])
  def uv(P_): dc = Rc0.T @ (P_ - pc0); return np.array([[-dc[1] / dc[0] / TH, dc[2] / dc[0] / TV, 1.0]])
  for k in range(250):  # the synthetic spot is 35 cm off L's C_NOM: the blend needs ~4 s from the L-only target
    p = 2 * math.pi * 1.25 * k * 0.02; e12 = lambda r1, r2, d: r1 * math.cos(-p - d) * np.array([0.0, 1.0, 0.0]) + r2 * math.sin(-p - d) * np.array([0.0, 0.0, 1.0])
    ea = lambda al, dd: (al[0] * math.cos(-p - dd) + al[1] * math.sin(-p - dd)) * np.array([1.0, 0.0, 0.0])
    cL, cF = uv(A + e12(RL, RZ, DELTA) + ea(AL_L, DELTA)), uv(B + e12(RLF, RZF, DELTAF) + ea(AL_F, DELTAF))
    if k < 60: cF[0, 2] = 0.0
    else: cL[0, 2] = 0.0
    corr = cf.step(k, [0], enc[None], quat[None], odo[None], np.array([p]), cL, np.array([0]), spot, curF=cF, wait=np.array([True]))
  assert np.allclose(odo[:2] + corr[0], 0.0, atol=1e-3) and np.allclose(cf.C[0], spot[0, :2] - A[:2], atol=1e-3), (corr, cf.C)
  print("camfix self-check (L and F) ok", corr[0], cf.C[0])
