"""Stage 7-6 (ys): the back turner keeps time by ear.

The follower (back turner, its own clock off by an unknown offset) cannot see the leader past the
jumper (trio_cams.py: all 68 'visible' frames of the leader's ball went through the jumper's body). The
jumper beeps when its LEFT foot lands, and its landings are locked to the leader, whose red ball it
watches: leader phase 154 deg at the left touchdown, per-env std ~4 deg, unchanged with the follower
60 deg off (beep_probe.py, G1@12700). The leader never beeps.

The follower turns heard beeps into a clock correction: a port of runtime sounds/chorale/beat.rs's
Follower (tempo held, phase from a window of beats, outliers dropped, a slow drag, hold when silent) with
two changes: a beep carries no beat number (the nearest beat is assumed), and a circular median replaces
the mean (one false beep moved an 8-beat mean ~11 deg; beat_mc.py: steady p90 9.6 -> 4.5 deg).

Real counterparts: the jumper reads its left landing off the left leg's present current (read every
tick, duck-control bus.rs); the beep goes out through plughw:aic3104; the follower detects it in the
pet-detect mic worker (16 kHz, a Goertzel tone detector) and knows its own speaker + mic latency from
hearing its own beep (CAL_SD).
"""

import math

import torch

OMEGA = 2 * math.pi * 1.25  # rad/s: the rope tempo, held (both turners read it off the same "score")
MU_LAND = math.radians(154)  # leader phase at the jumper's left touchdown (G1@12700, beep_probe.py)
DET = 0.02  # s: the landing shows in the leg current at most one control tick late
LAT_SPK, LAT_MIC = (0.02, 0.04), (0.01, 0.03)  # s, constant per duck: unmeasured, randomized per episode
AIR, JIT = 0.0025, 0.002  # s: 0.87 m of air; detector timing noise
MISS, FALSE = 0.10, 0.3  # beeps lost; false beeps per s (rope slap, servos)
CAL_SD, CAL_ERR = 0.003, 0.0  # s: self-loopback latency measurement noise; a deliberate bias (robustness)
W, GATE0, GATE, NREJ = 8, math.radians(90), math.radians(45), 4  # window; outlier gates (filling / full)
LOCK_N, LOCK_OF, LOCK_TOL = 4, 5, math.radians(15)  # 4 of the newest 5 samples within 15 deg of their median =
# a beat: the jumper is in. Before that the correction holds (the follower keeps its camera logic): beat_scn.py,
# lockless, false beeps before the jumper came in dragged the clock at random (start 0: motion error p50 30-51 deg);
# 4 in a row: a false beep among them most of the time (lock p50 6-9 s after entry). ponytail: noise alone locks
# ~0.3% per false beep (~1 per 19 min at 0.3/s); the real beat then unlocks it (NREJ)
AFTER_LOCK = "switch"  # follower network once locked: switch (rope14: clock only), keep (F1 + its blocked
# camera), mask (camera input 0). beat_scn.py, motion error p50/p90 after entry: switch 6/14, keep 12/31, mask 12/30
K, RMAX = 2.0, math.radians(90)  # 1/s, rad/s: the drag on the clock (<= 20% tempo). beat_s0.py A: rope14
# takes even a clock jump (start 120 deg, drag 22.5/45/90 deg/s or a jump: clean 71/73/72/76%); what costs falls
# is time spent far off (start 180: follower 15/9/11/6%), so the drag is fast
LISTEN = False  # start: hold the handle still until 3 beeps are in, start at that phase (amp 0 -> 1 in 1 s).
# Off: rope14 never learned to hold still (beat_s0.py B, 3 s still: follower 28%, leader 12% fall)
START_BIAS = 0.0  # s: the follower's latency calibration off by this much (robustness)
AIR_MIN, F_ON = 0.15, 2.0  # s, N: a touchdown = the first contact after this long in the air
HOME_R = 0.15  # m: the jumper beeps only in its spot under the rope (before it walks in there are no beeps,
# and the follower keeps its camera logic: the correction holds at 0 until beeps come)


def start_offset(n: int, device) -> torch.Tensor:
  """(n,) the follower's clock offset when both turners start on the jumper's start sound (user, 2026-11-06):
  the waiting jumper is about as far from both; each turner starts its clock at 0 when it hears the sound,
  minus its own calibrated latency. Left over: each one's calibration error (CAL_SD) and detection timing
  (JIT), ~5 ms between them = ~2.3 deg (a 3-beep count-in survives a missed beep: not modelled)."""
  sd = math.sqrt(CAL_SD**2 + JIT**2)
  return OMEGA * (sd * (torch.randn(n, device=device) - torch.randn(n, device=device)) + START_BIAS)


def start_by_ear(env, env_ids: torch.Tensor | None) -> None:
  """Reset event: the follower's clock offset after the start sound (start_offset)."""
  if not hasattr(env, "_ys_off"):
    env._ys_off = torch.zeros(env.num_envs, device=env.device)
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  env._ys_off[ids] = start_offset(len(ids), env.device)


def wrap(a: torch.Tensor) -> torch.Tensor:
  return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def circ_est(buf: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
  """(N,) circular median of the first min(n, W) ring entries (the entry nearest to all others; the
  circular mean below 3 entries). Angles cannot be averaged as numbers: 350 and 10 deg mean 0, not 180."""
  valid = torch.arange(buf.shape[1], device=buf.device)[None] < n.clamp(max=buf.shape[1])[:, None]
  mean = torch.atan2((buf.sin() * valid).sum(1), (buf.cos() * valid).sum(1))
  cost = (wrap(buf[:, :, None] - buf[:, None, :]).abs() * valid[:, None, :]).sum(2).masked_fill(~valid, float("inf"))
  return torch.where(n >= 3, buf.gather(1, cost.argmin(1, keepdim=True))[:, 0], mean)


class BeatFollower:
  """Per-env offset estimate from beat samples, and the correction c (rad) dragged toward it."""

  def __init__(self, n: int, device):
    self.buf = torch.zeros(n, W, device=device)
    self.n = torch.zeros(n, dtype=torch.long, device=device)  # samples taken (ring index = n % W)
    self.rej = torch.zeros(n, dtype=torch.long, device=device)
    self.c = torch.zeros(n, device=device)
    self.lock = torch.zeros(n, dtype=torch.bool, device=device)

  def reset(self, ids: torch.Tensor) -> None:
    self.n[ids], self.rej[ids], self.c[ids], self.lock[ids] = 0, 0, 0.0, False

  def observe(self, o: torch.Tensor, m: torch.Tensor) -> None:
    """o (N,) offset samples, m (N,) where a beep was heard."""
    est = circ_est(self.buf, self.n)
    gate = torch.where(self.n >= W, GATE, GATE0)
    out = m & self.lock & (wrap(o - est).abs() > gate)
    self.rej = torch.where(out, self.rej + 1, torch.where(m, 0, self.rej))
    fresh = self.rej >= NREJ  # the beat really moved (a fall, a restart): unlock, start the window over
    self.n, self.rej, self.lock = torch.where(fresh, 0, self.n), torch.where(fresh, 0, self.rej), self.lock & ~fresh
    take = (m & ~out) | fresh
    i = (self.n % W)[take]
    self.buf[take.nonzero()[:, 0], i] = o[take]
    self.n = self.n + take.long()
    k = torch.arange(LOCK_OF, device=o.device)[None]
    last = self.buf.gather(1, (self.n[:, None] - 1 - k) % W)
    med = circ_est(last, self.n.clamp(max=LOCK_OF))
    inl = (k < self.n[:, None]) & (wrap(last - med[:, None]).abs() < LOCK_TOL)
    new = m & ~self.lock & (inl.sum(1) >= LOCK_N)
    self.buf[new, :LOCK_OF] = torch.where(inl, last, med[:, None])[new]  # the window = that beat
    self.n[new], self.lock[new] = LOCK_OF, True

  def est(self) -> torch.Tensor:
    return circ_est(self.buf, self.n)

  def drag(self, dt: float) -> None:
    e = wrap(self.est() - self.c)
    self.c = torch.where(self.lock, self.c + (K * e).clamp(-RMAX, RMAX) * dt, self.c)


class Beat:
  """The jumper's beeps through the air into the follower's estimator. step() once per env step ->
  (correction (N,) rad to add to the follower's clock, circle amp (N,) or None)."""

  def __init__(self, env):
    n, dev = env.num_envs, env.device
    self.env, self.cs, self.dt, self.rob = env, env.scene["feet_ground_contact"], env.step_dt, env.scene["robot"]
    self.f = BeatFollower(n, dev)
    z = lambda: torch.zeros(n, device=dev)
    self.air, self.lat, self.lcal, self.heard = z(), z(), z(), z()
    self.t_arr = torch.full((n,), math.inf, device=dev)  # one beep in flight at most (they are 0.8 s apart)
    self.t_go = torch.full((n,), math.inf, device=dev)  # LISTEN: when the follower started turning
    self.step_id = -1

  def _reset(self, ids: torch.Tensor) -> None:
    u = lambda lo_hi: torch.empty(len(ids), device=ids.device).uniform_(*lo_hi)
    self.lat[ids] = u(LAT_SPK) + u(LAT_MIC) + AIR
    self.lcal[ids] = self.lat[ids] + DET / 2 + CAL_ERR + CAL_SD * torch.randn(len(ids), device=ids.device)
    self.f.reset(ids)
    self.air[ids], self.heard[ids], self.t_arr[ids], self.t_go[ids] = 0.0, 0.0, math.inf, math.inf

  def _sample(self, t_hear: torch.Tensor, now: torch.Tensor) -> torch.Tensor:
    """Offset sample from a beep heard at t_hear: where the leader 'should' be (landing phase + the
    calibrated latency) minus the follower's own clock then."""
    env = self.env
    lead = env.command_manager.get_term("head_pose").phase - getattr(env, "_ys_delta", math.radians(60))
    off = env.__dict__.get("_ys_off")
    local = lead + (0.0 if off is None else off) - OMEGA * (now - t_hear)  # the follower's own clock
    return wrap(MU_LAND + OMEGA * self.lcal - local)

  def step(self):
    env = self.env
    if self.step_id != env.common_step_counter:
      self.step_id = env.common_step_counter
      fresh = env.episode_length_buf <= 1
      if fresh.any():
        self._reset(fresh.nonzero()[:, 0])
      now = env.episode_length_buf * self.dt
      m = self.t_arr <= now  # the beep in flight lands
      self.f.observe(self._sample(torch.where(m, self.t_arr, now), now), m)
      self.heard += m.float()
      self.t_arr = torch.where(m, math.inf, self.t_arr)
      m = torch.rand_like(now) < FALSE * self.dt  # a false beep
      self.f.observe(self._sample(now - self.dt * torch.rand_like(now), now), m)
      on = self.cs.data.force[:, 0].norm(dim=-1) > F_ON  # LEFT foot (sensor order: left, right)
      land = on & (self.air >= AIR_MIN)
      self.air = torch.where(on, 0.0, self.air + self.dt)
      home = (self.rob.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]).norm(dim=-1) < HOME_R
      send = land & home & (torch.rand_like(now) >= MISS)  # only a jumper in its spot beeps
      self.t_arr = torch.where(send, now + DET * torch.rand_like(now) + self.lat + JIT * torch.randn_like(now), self.t_arr)
      if LISTEN:
        go = (self.f.n >= 3) & self.t_go.isinf()
        self.f.c = torch.where(go, self.f.est(), self.f.c)  # not turning yet: take the phase outright
        self.t_go = torch.where(go, now, self.t_go)
      self.f.drag(self.dt)
      self._amp = ((now - self.t_go) / 1.0).clamp(0, 1) if LISTEN else None
    return self.f.c, self._amp


if __name__ == "__main__":  # self-check: estimator only, CPU
  d = math.radians
  b = torch.tensor([[d(350), d(10), d(0), 0, 0, 0, 0, 0]])
  assert abs(circ_est(b, torch.tensor([3]))[0]) < d(1)  # median of 350, 10, 0 is 0, not 180
  torch.manual_seed(0)
  f = BeatFollower(1, "cpu"); one = torch.ones(1, dtype=torch.bool)
  g = torch.Generator().manual_seed(1)
  for _ in range(40):  # false beeps alone (uniform): no lock, no correction
    f.observe(torch.rand(1, generator=g) * 2 * math.pi - math.pi, one); f.drag(0.02)
  assert not f.lock[0] and f.c[0] == 0
  for k in range(10):  # then 10 beats at -120 deg (4 deg noise) and two false beeps
    f.observe(torch.tensor([d(-120) + d(4) * torch.randn(()).item()]), one)
    if k in (5, 7):
      f.observe(torch.tensor([d(100)]), one)
  assert abs(wrap(f.est() - d(-120))[0]) < d(5), math.degrees(f.est()[0])
  for _ in range(25):  # 0.5 s of drag: at most RMAX / 2
    f.drag(0.02)
  assert abs(f.c[0] + RMAX / 2) < d(0.5), math.degrees(f.c[0])
  for _ in range(400):
    f.drag(0.02)
  assert abs(wrap(f.c - d(-120))[0]) < d(5)
  for _ in range(NREJ + 3):  # the beat moved to +60: the window starts over and follows
    f.observe(torch.tensor([d(60)]), one)
  assert abs(wrap(f.est() - d(60))[0]) < d(1), math.degrees(f.est()[0])
  print("ys_beat self-check ok")
