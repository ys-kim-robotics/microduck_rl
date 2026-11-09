"""ys_odo_ana.py <rec.npz> <out stem>: replay odo_rec.py's samples through the runtime odometry (odo_replay, unchanged runtime
code) and compare with the true trunk path from J's start call: error at the set-off, at arrival and every few seconds of
jumping, the drift rate while jumping (cm/sqrt(s), vs the s13 tolerance runs), and plots (odo.png)."""
import sys, os, subprocess, json, math
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FONT = "/usr/share/fonts/truetype/nanum/NanumSquare_acR.ttf"; fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name(); plt.rcParams["axes.unicode_minus"] = False
BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ys_odo_replay/target/release/odo_replay")  # cargo build --release there
z = np.load(sys.argv[1]); STEM = sys.argv[2]; dt = float(z["dt"]); on = z["on"]; T, N = on.shape
ms, mg, ma, mc = z["m_start"], z["m_go"], z["m_arr"], z["m_call"]


def replay(key):
  lines = []
  for n in range(N):
    for t in np.nonzero(on[:, n])[0]:
      lines.append(f"{n} {t} " + " ".join(f"{v:.7f}" for v in z[key][t, n]) + " " + " ".join(f"{v:.7f}" for v in z["quat"][t, n]))
  out = subprocess.run([BIN], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True).stdout.split("\n")
  est = np.full((T, N, 2), np.nan)
  for l in out:
    if l: e, t, x, y, *_ = l.split(); est[int(t), int(e)] = (float(x), float(y))
  return est


res = {}
for key in ("enc", "true"):
  est = replay(key); err = np.full((T, N, 2), np.nan)
  for n in range(N):
    if ms[n] < 0: continue
    t0 = ms[n]; ok = on[:, n]
    err[ok, n] = est[ok, n] - (z["pos"][ok, n, :2] - z["pos"][t0, n, :2])
  res[key] = err
E = np.linalg.norm(res["enc"], axis=-1)  # (T, N) m


def at(n, k): return E[k, n] if 0 <= k < T and np.isfinite(E[k, n]) else np.nan


rows = {"set-off": [], "arrival": []} | {f"jump +{s} s": [] for s in (4, 8, 12, 16)}
rate = []
for n in range(N):
  if mg[n] < 0 or ma[n] < 0: continue
  rows["set-off"].append(at(n, mg[n])); rows["arrival"].append(at(n, ma[n]))
  for s in (4, 8, 12, 16): rows[f"jump +{s} s"].append(at(n, ma[n] + int(s / dt)))
  ks = np.arange(ma[n], T); ks = ks[np.isfinite(E[ks, n])]
  if len(ks) > 100:  # growth after arrival: |e(t) - e(arr)| vs sqrt(t - t_arr), least squares through 0
    d = np.linalg.norm(res["enc"][ks, n] - res["enc"][ma[n], n], axis=-1); s_ = np.sqrt((ks - ma[n]) * dt)
    rate.append(float((d * s_).sum() / (s_ * s_).sum()))
summ = {k: dict(n=int(np.isfinite(v).sum()), p50_cm=round(100 * float(np.nanmedian(v)), 2), p90_cm=round(100 * float(np.nanpercentile(v, 90)), 2)) for k, v in rows.items()}
summ["jump drift cm/sqrt(s)"] = dict(p50=round(100 * float(np.median(rate)), 3), p90=round(100 * float(np.percentile(rate, 90)), 3), n=len(rate))
Et = np.linalg.norm(res["true"], axis=-1); summ["true-angle replay, jump +16 s p50 cm"] = round(100 * float(np.nanmedian([Et[ma[n] + int(16 / dt), n] for n in range(N) if ma[n] >= 0 and ma[n] + int(16 / dt) < T and np.isfinite(Et[ma[n] + int(16 / dt), n])])), 2)
json.dump(summ, open(STEM + ".json", "w"), indent=1); print(json.dumps(summ, ensure_ascii=False))

fig, ax = plt.subplots(1, 3, figsize=(19, 6), gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
for n in range(N):
  if ma[n] < 0: continue
  ks = np.nonzero(np.isfinite(E[:, n]))[0]; ax[0].plot((ks - ma[n]) * dt, 100 * E[ks, n], color="#2e86c1", alpha=0.25, lw=0.8)
tt = np.linspace(0, 16, 100)
for s_, c_ in ((0.3, "#27ae60"), (0.5, "#e67e22"), (1.0, "#c0392b")):
  ax[0].plot(tt, s_ * np.sqrt(tt), color=c_, lw=2, ls="--", label=f"{s_} cm/√s 드리프트 (평균 크기)")
ax[0].axvline(0, color="k", lw=0.8); ax[0].text(0.1, ax[0].get_ylim()[1] * 0.95 if ax[0].get_ylim()[1] > 0 else 5, "도착", fontsize=9)
ax[0].set_xlabel("도착부터 시간 (s, 음수 = 대기·달리기)"); ax[0].set_ylabel("위치 추정 오차 (cm)"); ax[0].set_xlim(-6, 17); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
ax[0].set_title("런타임 오도메트리의 위치 오차 (시뮬 64판, 엔코더 오차 포함)", fontsize=12)
ks = list(rows); ax[1].bar(range(len(ks)), [summ[k]["p50_cm"] for k in ks], color="#5dade2", label="중앙값")
ax[1].bar(range(len(ks)), [summ[k]["p90_cm"] - summ[k]["p50_cm"] for k in ks], bottom=[summ[k]["p50_cm"] for k in ks], color="#aed6f1", label="상위 10%까지")
ax[1].set_xticks(range(len(ks))); ax[1].set_xticklabels(ks, fontsize=9, rotation=20); ax[1].set_ylabel("cm"); ax[1].legend(fontsize=9); ax[1].grid(axis="y", alpha=0.3)
ax[1].set_title("시점별 오차", fontsize=12)
pick = [n for n in range(N) if ma[n] >= 0][:4]
for i, n in enumerate(pick):
  ok = np.isfinite(E[:, n]); tp = z["pos"][ok, n, :2] - z["pos"][ms[n], n, :2]; ep = tp + res["enc"][ok, n]
  ax[2].plot(100 * tp[:, 0], 100 * tp[:, 1], color=f"C{i}", lw=1.5); ax[2].plot(100 * ep[:, 0], 100 * ep[:, 1], color=f"C{i}", lw=1, ls="--")
ax[2].set_aspect("equal"); ax[2].set_xlabel("x (cm, 시작 소리 때 위치 기준)"); ax[2].set_ylabel("y (cm)"); ax[2].grid(alpha=0.3)
ax[2].set_title("실제 경로 (실선) vs 오도메트리 (점선), 4판", fontsize=12)
fig.tight_layout(); fig.savefig(STEM + ".png", dpi=95); print("saved")
