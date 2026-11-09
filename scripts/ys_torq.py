"""ys_torq.py <ckpt> <N> <out.json> (user 2026-11-09: baseline v3, "조인트토크나 실물테스트를 고려해서 추가작업이 필요한 부분"): the
full game (scripts/ys_fullset.py setup; CLIP, VIN, SEED from env) recording J's total motor current per phase, J's joint-stop hits
while jumping (time at a stop, joint speed at each hit), and L / F torque, heat and current during the walk-out (episode start
.. J's call) and while turning. Writes the game summary too."""
import sys, os, json
FS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ys_fullset.py")
CK, NN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [FS, "x", CK, NN]  # neither stats nor video: N envs, no renderer, fullset's own loop not run
exec(compile(open(FS).read().split("# ---------------- video")[0], FS, "exec"))
import numpy as np
KT_, CAP_ = SC.KT, SC.KT * SC.I_MAX; TOL = math.radians(0.5)
CUR = {k: [] for k in ("walk", "wait", "run", "jump")}; LF = {k: [] for k in ("walk_L", "walk_F", "turn_L", "turn_F")}
JN = list(rob.joint_names); nj = len(JN); at_n = torch.zeros(nj, device=dev); jump_steps = [0]; HIT = [[] for _ in range(nj)]
prev = {"at": None, "qd": None}


def on_step(k, M):
  alive = M["dead"] < 0; st, go, arr, call = M["start"], M["go"], M["arr"], M["call"]
  ended = (call >= 0) & (call <= k); kst = torch.where(call >= 0, call + KSTOP, 10 ** 9)
  ph = {"walk": alive & (st < 0), "wait": alive & (st >= 0) & (go < 0), "run": alive & (go >= 0) & (arr < 0) & ~ended,
        "jump": alive & (arr >= 0) & (k >= arr + 25) & ~ended}
  cj = (rob.data.qfrc_actuator.abs() / KT_).sum(-1)
  for nm, m in ph.items(): CUR[nm].append(cj[m])
  turn = alive & (st >= 0) & (k >= st + 50) & (k < kst)
  for w, ent in (("L", tu), ("F", pa)):
    t = ent.data.qfrc_actuator.abs(); LF["walk_" + w].append(t[ph["walk"]]); LF["turn_" + w].append(t[turn])
  q, qd, lim = rob.data.joint_pos, rob.data.joint_vel, rob.data.joint_pos_limits
  at = (q >= lim[..., 1] - TOL) | (q <= lim[..., 0] + TOL); jm = ph["jump"]
  at_n.add_(at[jm].float().sum(0)); jump_steps[0] += int(jm.sum())
  if prev["at"] is not None:
    hit = at & ~prev["at"] & jm[:, None]
    for j in range(nj):
      if hit[:, j].any(): HIT[j].append(prev["qd"][hit[:, j], j].abs())
  prev["at"], prev["qd"] = at, qd.clone()
  return bool((~alive | ((call >= 0) & (k > call + KSTOP + int(AFTER / dt)))).all())


with torch.inference_mode(): M = episode(on_step)
R = records(M); S = summary(R); out = dict(ckpt=CK, seed=SEED, clip=os.environ.get("CLIP"), vin=os.environ.get("VIN"), summary=S)
pc = lambda x, p: round(float(torch.quantile(x.float(), p)), 3) if len(x) else None
out["J_current_A"] = {nm: dict(n=len(x := torch.cat(v)), p50=pc(x, 0.5), p99=pc(x, 0.99), max=round(float(x.max()), 3) if len(x) else None) for nm, v in CUR.items()}
js = max(jump_steps[0], 1); dur = js * dt
out["J_stops_jump"] = {JN[j]: dict(at_stop_pct=round(100 * float(at_n[j]) / js, 1), hits_per_s=round(sum(len(h) for h in HIT[j]) / dur, 2),
                                   hit_speed_p50=pc(torch.cat(HIT[j]), 0.5) if HIT[j] else None, hit_speed_p95=pc(torch.cat(HIT[j]), 0.95) if HIT[j] else None)
                       for j in range(nj) if float(at_n[j]) / js > 0.01}
for k_, v in LF.items():
  x = torch.cat(v) if v else torch.zeros(0, nj, device=dev)
  if not len(x): continue
  heat = ((x / KT_) ** 2 * SC.R_OHM).mean(0); cur = (x / KT_).sum(-1)
  out["LF_" + k_] = dict(n=len(x), heat_sum=round(float(heat.sum()), 2), current_p99=pc(cur, 0.99), current_max=round(float(cur.max()), 2),
                         top=[(JN[j], round(float(heat[j]), 2), round(100 * float((x[:, j] >= 0.98 * CAP_).float().mean()), 1)) for j in torch.argsort(-heat)[:4].tolist()])
json.dump(out, open(OUT, "w"), indent=1)
print("SUMMARY", json.dumps({k: out[k] for k in ("summary", "J_current_A")}), flush=True)
print("STOPS", json.dumps(out["J_stops_jump"]), flush=True)
print("LF", json.dumps({k: v for k, v in out.items() if k.startswith("LF_")}), flush=True)
