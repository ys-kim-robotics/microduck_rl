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
"""
import math
import os

import mujoco
import numpy as np

RL, RZ, DELTA, C_NOM, W = 0.117, 0.108, math.radians(260.0), (-0.3928, 0.0610), 1.0
TAU, NMIN = float(os.environ.get("CAMFIX_TAU", "0.5")), int(os.environ.get("CAMFIX_NMIN", "6"))  # blend time constant (s), frames for a fix
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
    self.frames = [[] for _ in range(self.n)]  # (capture tick, P = I - d d^T, P (camera centre - ellipse offset))
    self.last = None; self.corr = np.zeros((self.n, 2)); self.seen = np.zeros(self.n, bool)

  def _ray(self, enc, quat):  # camera centre (trunk frame) and world ray rotation from the encoders and the IMU
    self.d.qpos[:] = 0; self.d.qpos[3] = 1
    for i, q in self.qadr: self.d.qpos[q] = enc[i]
    mujoco.mj_kinematics(self.m, self.d); R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); Rt = R.reshape(3, 3)
    return Rt @ self.d.site_xpos[self.cs], Rt @ self.d.site_xmat[self.cs].reshape(3, 3)

  def step(self, k, ids, enc, quat, odo, ph, cur, lag, spot):
    """ids: the envs J is on; enc (n, 15), quat (n, 4), odo (n, 3) world odometry position, ph (n,): theirs; cur (N, 3) the ball
    tracker's last frame, lag (N,), spot (N, 3) world xy + heading of J's spot. -> (n, 2) correction to add to the odometry xy."""
    for j, e in enumerate(ids):
      H = self.hist[e]; H[k] = (enc[j], quat[j], odo[j], ph[j])
      for t in [t for t in H if t < k - 10]: del H[t]
      c = cur[e]; kc = k - int(lag[e])
      if self.last is not None and c[2] > 0 and np.any(c != self.last[e]) and kc in H:
        en, qu, od, p = H[kc]; pc, Rc = self._ray(en, qu); dd = Rc @ np.array([1.0, -c[0] * TH, c[1] * TV]); dd /= np.linalg.norm(dd)
        h = spot[e, 2]; e1, e2 = np.array([-math.sin(h), math.cos(h), 0.0]), np.array([0.0, 0.0, 1.0]); a = -p - DELTA
        P = np.eye(3) - np.outer(dd, dd); self.frames[e].append((kc, P, P @ (od + pc - RL * math.cos(a) * e1 - RZ * math.sin(a) * e2)))
      F = self.frames[e] = [f for f in self.frames[e] if f[0] > k - W / self.dt]
      if len(F) >= NMIN:
        M = sum(f[1] for f in F)
        if np.linalg.cond(M) > 1e6: continue  # rays all alike: no fix this tick
        A = np.linalg.solve(M, sum(f[2] for f in F)); h = spot[e, 2]
        cw = C_NOM[0] * np.array([math.cos(h), math.sin(h)]) + C_NOM[1] * np.array([-math.sin(h), math.cos(h)])
        self.corr[e] += self.dt / TAU * ((spot[e, :2] - A[:2] - cw) - self.corr[e]); self.seen[e] = True
    self.last = cur.copy()
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
      B = A + RL * math.cos(a) * np.array([0.0, 1.0, 0.0]) + RZ * math.sin(a) * np.array([0.0, 0.0, 1.0])
      dc = Rc0.T @ (B - pc0); u, v = -dc[1] / dc[0] / TH, dc[2] / dc[0] / TV
      corr = cf.step(k, [0], enc[None], quat[None], odo[None], np.array([p]), np.array([[u, v, 1.0 if abs(u) <= 1 and abs(v) <= 1 else 0.0]]),
                     np.array([0]), spot)
    assert np.allclose(odo[:2] + corr[0], 0.0, atol=1e-3), (off, corr)
  print("camfix self-check ok", corr[0])
