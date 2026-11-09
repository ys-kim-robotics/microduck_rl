"""history_v3.py -> history_v3.png: how the jump rope trio got to baseline v3 (2026-11-02 .. 11-09).
Top: the checkpoint lineage per lane (L / F turners, J generation 1 and 2) with each node's number at the time (each on the
yardstick of its day) and the turns where the yardstick or the hit rule changed. Bottom: the J candidates re-measured on one
yardstick (full game, no push, user's hit rule, 2 seeds pooled) and baseline v3 under real-robot conditions (DEPLOY-ys.md).
Numbers come from STAGES-ys.md (k / n from the stats files of 7-13 and the checks). Run: python docs/ys/history_v3.py"""
import math, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FD = "/usr/share/fonts/truetype/nanum/"
for f in ("NanumSquare_acR.ttf", "NanumSquare_acB.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams["font.family"] = fm.FontProperties(fname=FD + "NanumSquare_acR.ttf").get_name(); plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_v3.png")
LF, J1, J2, RULE, USER, ROOT = "#2e86c1", "#e67e22", "#1e8449", "#7f8c8d", "#8e44ad", "#555555"
Y = {"lf": 4.25, "j1": 2.95, "j2": 1.65}

fig = plt.figure(figsize=(22, 14.5))
ax = fig.add_axes([0.01, 0.37, 0.98, 0.56]); ax.set_xlim(-6.5, 101); ax.set_ylim(-0.45, 5.0); ax.axis("off")

# days (bands are approximate: most of the work is 11/06 .. 11/09)
DAYS = [("11/02 (월)", -0.3, 6), ("11/03 (화)", 6, 38), ("11/04 (수)", 38, 45.3), ("11/05 (목)", 45.3, 53), ("11/06 (금)", 53, 60),
        ("11/07 (토)", 60, 77.3), ("11/08 (일)", 77.3, 91.2), ("11/09 (월)", 91.2, 101)]
for i, (d, a, b) in enumerate(DAYS):
  ax.axvspan(a, b, color="#eef1f3" if i % 2 else "white", zorder=0); ax.text((a + b) / 2, 4.9, d, ha="center", fontsize=11, color="#555")
for k, lab, col in (("lf", "L·F\n줄 돌리기", LF), ("j1", "J 1세대\n제자리\n줄넘기", J1), ("j2", "J 2세대\n입장부터\n같이", J2)):
  ax.text(-6.3, Y[k], lab, ha="left", va="center", fontsize=10.5, color=col, fontweight="bold")
ax.text(-6.3, 0.55, "잣대·판정이\n바뀐 때", ha="left", va="center", fontsize=10.5, color=RULE, fontweight="bold")


def node(x, k, title, lines, w, col=None, star=None, dead=False, user=None, y=None):
  col = col or {"lf": LF, "j1": J1, "j2": J2}[k]; y = Y[k] if y is None else y; h = 0.36 + 0.2 * len(lines)
  ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec=col,
                              lw=3.2 if star else 1.6, zorder=3, ls="--" if dead else "-"))
  ax.text(x, y + h / 2 - 0.18, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=col, zorder=4)
  for i, t in enumerate(lines):
    red = t.startswith("!"); t = t.lstrip("!")
    ax.text(x, y + h / 2 - 0.41 - 0.2 * i, t, ha="center", va="center", fontsize=9, color="#c0392b" if red else "#222", zorder=4)
  if star: ax.text(x, y + h / 2 + 0.04, "★ " + star, ha="center", va="bottom", fontsize=11, fontweight="bold", color="#b7950b", zorder=4)
  if dead: ax.text(x, y + h / 2 + 0.04, "X 막힘", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c0392b", zorder=4)
  if user: ax.text(x, y - h / 2 - 0.07, "사용자: " + user, ha="center", va="top", fontsize=8.5, color=USER, zorder=4)
  return dict(l=x - w / 2, r=x + w / 2, x=x, y=y, t=y + h / 2, b=y - h / 2)


def link(a, b, col, ls="-"):  # left-to-right, same height
  ax.add_patch(FancyArrowPatch((a["r"], a["y"]), (b["l"], b["y"]), arrowstyle="-|>", mutation_scale=14, color=col, lw=1.8, ls=ls,
                               zorder=2, shrinkA=1, shrinkB=1))


def path(pts, col, ls="-", lw=1.6):  # polyline ending in an arrow head
  xs, ys = zip(*pts); ax.plot(xs[:-1] + (xs[-1],), ys, color=col, lw=lw, ls=ls, zorder=1)
  ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=14, color=col, lw=lw, zorder=2, shrinkA=0, shrinkB=1))


# root: the walking policy every lane starts from
root = node(2.6, "j1", "full2500", ["걷기 정책", "(1~3단계)"], 5.0, col=ROOT)
# L / F (stage 4 .. 6)
hc4 = node(10.6, "lf", "hc4_load@6000", ["머리로 원 + 부리 하중", "4단계"], 8.0)
bc5 = node(18.8, "lf", "bc5@14997", ["몸 전체로 큰 원", "!반경 7 cm에서 멈춤"], 7.6)
rope5 = node(27.1, "lf", "rope5", ["T핸들로 줄 돌리기", "줄 회전 1.00"], 8.2, user="T핸들 설계")
rope14 = node(34.9, "lf", "rope11 → 14", ["두 마리 self-play", "넘어짐 0%"], 6.6, user="거리 0.6 m")
path([(root["x"], root["t"]), (root["x"], Y["lf"]), (hc4["l"], Y["lf"])], ROOT)
link(hc4, bc5, LF); link(bc5, rope5, LF); link(rope5, rope14, LF)
ax.annotate("", xy=(100.6, Y["lf"]), xytext=(rope14["r"] + 0.2, Y["lf"]), arrowprops=dict(arrowstyle="-|>", color=LF, lw=6, alpha=0.3), zorder=1)
ax.text(69, Y["lf"] + 0.14, "rope14@19200 = L·F, 7단계 내내 고정 (앞 L = 빨간 공, 뒤 F = J의 시작 소리로 박자)", ha="center", fontsize=10.5, color=LF)
# J generation 1: in-place jumping, put in by teleport
jmp = node(41.6, "j1", "jump13 → side5", ["제자리 → 옆으로 뛰기", "발 높이 35~38 mm"], 7.4)
trio = node(49.2, "j1", "trio1 → B1", ["셋이 한 장면", "카메라로 박자"], 7.0)
g1 = node(56.6, "j1", "G1@12700", ["순간이동 입장", "20줄 이상 36%"], 6.6, star="J/F/L baseline", user="앞 L, 뒤 F는 소리로")
e5 = node(63.6, "j1", "E5@13199", ["제자리 + 500 it", "20줄 이상 61%"], 6.6, star="baseline v2")
ent = node(71.6, "j1", "E1~E3b, E6b~E8b", ["돌고 있는 줄로 입장", "!10줄 이상 0~7%"], 8.4, dead=True, user="잽싸게 들어가야")
link(root, jmp, ROOT); link(jmp, trio, J1); link(trio, g1, J1); link(g1, e5, J1); link(e5, ent, J1)
ax.text(21, Y["j1"] + 0.1, "줄넘기 점프 (11/04부터)", ha="center", fontsize=9.5, color=ROOT)
# J generation 2: entry and jumping learned together, from the walking policy again
n1 = node(65.1, "j2", "N1 → N1c", ["A·B·C 섞어 입장부터", "!10줄 이상 0~2%"], 7.4, user="처음부터 같이")
n24 = node(73.3, "j2", "N2 → N4", ["A·B·C 따로, 판정 교체", "10줄 이상 2~18%"], 8.0, user="분리, 걸림 = 중간")
n5 = node(80.7, "j2", "N5Af@4996", ["진짜 걸림 벌점", "10줄 이상 27%"], 6.4)
d1a = node(87.6, "j2", "D1A@5495", ["+ E5 따라하기(BC)", "풀세트 55%"], 6.6, user="밀기 없이 재기")
arms = node(96.1, "j2", "D1A+ 세 갈래", ["시작 순서·배치·약한 밀기", "① 전 관절 자르기 44%", "② head·yaw 자르기 67%", "③ ② + 하늘 보기 74%",
                                         "③ = D1Asky@5994"], 8.8, star="baseline v3 = ③", user="관절 범위 원칙, 하늘 감점")
path([(root["x"], root["b"]), (root["x"], Y["j2"]), (n1["l"], Y["j2"])], ROOT, ls=(0, (5, 3)))
ax.text(31, Y["j2"] + 0.1, "11/07 걷기 정책에서 새로 시작 (입력 61 → 73: 카메라 추적 12칸)", ha="center", fontsize=9.5, color=ROOT)
link(n1, n24, J2); link(n24, n5, J2); link(n5, d1a, J2); link(d1a, arms, J2)
yb = 2.3; path([(e5["x"], e5["b"]), (e5["x"], yb), (d1a["x"], yb), (d1a["x"], d1a["t"])], J1, ls=(0, (4, 2)), lw=1.4)
ax.text(80.7, yb + 0.07, "E5를 선생으로 (BC)", ha="center", va="bottom", fontsize=9, color=J1)

# yardstick / hit-rule turns: numbered diamonds on the timeline, explained in the empty lower-left corner
RULES = [(58.5, "걸림 = 종료 (11/06 저녁)", "옛 지표 '깨끗 80%'가 입장 실패(첫 줄 걸림)를 숨겼음"),
         (76.0, "사용자 걸림 판정 (11/07 밤)", "60 ms 넘게 닿거나 줄이 90° 넘게 끌리면 실패, 스침은 허용"),
         (83.0, "평가에 학습용 밀기가 켜져 있었음 (11/08)", "끄니 D1A 입장 평가 26 → 72%"),
         (86.5, "풀세트 잣대 (11/08)", "손잡이 물기부터 끝 소리까지 전부: D1A 72 → 55%"),
         (90.0, "관절 끝 밀기 발견 (11/08~09)", "대기 중 head_roll 0.64 N·m(전류 한계), 열 19 W"),
         (99.0, "실물 전 점검 9개 (11/09)", "위치·명령 지연·줄 무게에 매우 민감 (DEPLOY-ys.md)")]
for i, (x, t, d) in enumerate(RULES):
  ax.plot([x], [0.5], marker="D", color="white", mec=RULE, mew=1.5, ms=17, zorder=3)
  ax.text(x, 0.5, str(i + 1), ha="center", va="center", fontsize=10, fontweight="bold", color="#444", zorder=4)
  cx, cy = 7.5 + 26 * (i // 3), 1.0 - 0.48 * (i % 3)
  ax.plot([cx], [cy], marker="D", color="white", mec=RULE, mew=1.5, ms=15, zorder=3)
  ax.text(cx, cy, str(i + 1), ha="center", va="center", fontsize=9.5, fontweight="bold", color="#444", zorder=4)
  ax.text(cx + 0.9, cy + 0.08, t, ha="left", va="center", fontsize=10, fontweight="bold", color="#444")
  ax.text(cx + 0.9, cy - 0.14, d, ha="left", va="center", fontsize=9, color="#555")
fig.text(0.5, 0.972, "baseline v3까지의 히스토리 (2026-11-02 ~ 11-09): 세 마리 줄넘기 L·F·J", ha="center", fontsize=18, fontweight="bold")
fig.text(0.5, 0.95, "상자 = 체크포인트, 화살표 = 이어서 학습. 상자 안 숫자는 그때의 잣대로 잰 값이라 서로 바로 비교하지 않습니다. 보라 글씨 = 방향을 바꾼 사용자 결정.",
         ha="center", fontsize=11, color="#444")

# bottom: one yardstick
bx = fig.add_axes([0.055, 0.07, 0.925, 0.245])


def wilson(k, n, z=1.96):
  p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return 100 * p, 100 * (c - h), 100 * (c + h)


A = [("N5Af", 68, 246), ("N5Cf", 81, 247), ("N5Ce", 39, 245), ("D1A", 136, 249), ("N5Af→E5tA\n교체", 141, 247),
     ("① 전 관절\n자르기", 110, 248), ("② head·yaw\n자르기", 167, 248), ("③ ② + 하늘\n= v3", 180, 243)]
B = [("카메라 없음\n(61차원)", 305, 489), ("박자 ±20°", 296, 493), ("자리 1 cm\n틀림", 79, 125), ("자리 3 cm\n틀림", 45, 115),
     ("명령 지연\n30~45 ms", 50, 122), ("줄 15 g\n(-25%)", 13, 115), ("줄 25 g\n(+25%)", 27, 120), ("도착 뒤 위치\n보정 없음", 16, 121)]
xs, labs = [], []
for i, (lab, k, n) in enumerate(A):
  p, lo, hi = wilson(k, n); last = i == len(A) - 1; xs.append(i); labs.append(lab)
  bx.bar(i, p, 0.66, color=J2 if last else "#a9dfbf", ec=J2); bx.plot([i, i], [lo, hi], color="k", lw=1.1)
  bx.text(i, hi + 1.5, f"{p:.0f}%", ha="center", fontsize=10.5, fontweight="bold" if last else "normal")
for j, (lab, k, n) in enumerate(B):
  p, lo, hi = wilson(k, n); x = len(A) + 0.8 + j; xs.append(x); labs.append(lab)
  bx.bar(x, p, 0.66, color="#f5b7b1" if p > 50 else "#e74c3c", ec="#922b21"); bx.plot([x, x], [lo, hi], color="k", lw=1.1)
  bx.text(x, hi + 1.5, f"{p:.0f}%", ha="center", fontsize=10.5)
bx.axhline(74.1, color=J2, ls="--", lw=1); bx.text(len(A) + len(B) + 0.55, 76, "v3 74%", color=J2, fontsize=10, ha="right")
bx.axvline(len(A) - 0.1, color="#888", lw=0.8)
bx.set_xticks(xs); bx.set_xticklabels(labs, fontsize=9.5); bx.set_ylim(0, 100); bx.set_xlim(-0.6, len(A) + len(B) + 0.6); bx.grid(axis="y", alpha=0.3)
bx.set_ylabel("10줄 이상 (%)\n선 = 95% 범위", fontsize=10.5)
bx.text(-0.5, 104, "같은 잣대로 다시 잰 J 후보 (풀세트, 밀기 없음, 사용자 판정, 약 245판씩)", fontsize=12, color=J2, fontweight="bold")
bx.text(len(A) + 0.3, 104, "baseline v3를 실물 조건으로 흔들면 (시뮬, 학습 없음, 115~493판) → DEPLOY-ys.md", fontsize=12, color="#922b21", fontweight="bold")
fig.text(0.055, 0.012, "1세대(G1, E5)는 순간이동으로 넣었고 스스로 입장하지 못해(입장 평가 10줄 이상 0~2%) 풀세트에서는 재지 않았습니다. "
         "숫자 출처: STAGES-ys.md 7단계, DEPLOY-ys.md. 그림 코드: docs/ys/history_v3.py", fontsize=10, color="#555")
fig.savefig(OUT, dpi=100); print("saved", OUT)
