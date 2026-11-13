"""history_v4.py -> history_v4.png, baseline_lineage.png: the model history up to 2026-11-13 and the baseline family tree.
history_v4.png = history_v3.png (kept as is: up to baseline v3) stacked on a new part for 2026-11-09 .. 11-13: every J retraining
after baseline v3 (what it continues from, what was changed, its number on the yardstick of its day), the procedure / yardstick
turns, and the candidates on one yardstick (realistic conditions, 10 ropes or more, 2 seeds pooled; k / n from the standard-set
stats files of 7-16 .. 7-23). baseline_lineage.png = the four baselines, what each descends from, what changed, who decided.
Numbers come from STAGES-ys.md 7-7 .. 7-25. Run: uv run python docs/ys/history_v4.py"""
import io, math, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from PIL import Image

FD = "/usr/share/fonts/truetype/nanum/"
for f in ("NanumSquare_acR.ttf", "NanumSquare_acB.ttf"): fm.fontManager.addfont(FD + f)
plt.rcParams["font.family"] = fm.FontProperties(fname=FD + "NanumSquare_acR.ttf").get_name(); plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
LF, J1, J2, J4, RULE, USER, ROOT, RED, GOLD = "#2e86c1", "#e67e22", "#1e8449", "#117a8b", "#7f8c8d", "#8e44ad", "#555555", "#c0392b", "#b7950b"


def node(ax, U, x, y, title, lines, w, col, star=None, dead=None, lw=1.6):
  """Box with a bold title and 9 pt lines ('!' first = red, '@' first = purple, the user's decision). dead = dashed border, with an
  'X <dead>' label when not empty. U = inches per y unit. Returns its edges."""
  h = (0.36 + 0.225 * len(lines)) / U; t = y + h / 2
  ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0.02,rounding_size={0.15 / U:.3f}", fc="white", ec=col,
                              lw=3.2 if star else lw, ls="-" if dead is None else "--", zorder=3))
  ax.text(x, t - 0.19 / U, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=col, zorder=4)
  for i, s in enumerate(lines):
    ax.text(x, t - (0.43 + 0.225 * i) / U, s.lstrip("!@"), ha="center", va="center", fontsize=9, color={"!": RED, "@": USER}.get(s[0], "#222"), zorder=4)
  if star: ax.text(x, t + 0.04 / U, "★ " + star, ha="center", va="bottom", fontsize=11, fontweight="bold", color=GOLD, zorder=4)
  if dead: ax.text(x + w / 2, t + 0.04 / U, "X " + dead, ha="right", va="bottom", fontsize=9.5, fontweight="bold", color=RED, zorder=4)
  return dict(l=x - w / 2, r=x + w / 2, x=x, y=y, t=t, b=y - h / 2)


def path(ax, pts, col, ls="-", lw=1.7):
  """Polyline ending in an arrow head (the head sits on the second half of the last segment so dashes stay dashes)."""
  xs, ys = zip(*pts); ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
  (ax_, ay), (bx_, by) = pts[-2], pts[-1]
  ax.add_patch(FancyArrowPatch(((ax_ + bx_) / 2, (ay + by) / 2), (bx_, by), arrowstyle="-|>", mutation_scale=14, color=col, lw=lw, zorder=2, shrinkA=0, shrinkB=1))


def wilson(k, n, z=1.96):
  p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return 100 * p, 100 * (c - h), 100 * (c + h)


# ---------------------------------------------------------------- part 2 of the history: after baseline v3
# (label, colour = what it continues from, realistic k/n, realistic + delay 30~45 ms k/n, true position k/n)
STD = [("baseline v3", J2, (111, 243), (24, 244), (188, 246)), ("재학습 7\n= baseline v4", J4, (118, 239), (111, 242), (175, 247)),
       ("재학습 9", J4, (95, 244), (70, 243), (169, 240)), ("재학습 10", J4, (67, 242), (53, 237), (160, 244)),
       ("재학습 10b", J4, (92, 248), (69, 239), (189, 243)), ("대조", J4, (88, 246), (80, 240), (186, 247)),
       ("재학습 11", J4, (140, 242), (88, 242), (183, 243)), ("재학습 11v3", J2, (124, 240), (102, 244), (158, 245)),
       ("이어 학습\n(머리 벌점 있음)", J4, (61, 246), (48, 241), (141, 247)), ("벌점 있음\n+ 사진 놓침 0%", J4, (76, 242), (105, 242), (157, 244)),
       ("이어 학습\n(머리 벌점 없음)", J4, (116, 237), (136, 249), (166, 243)), ("다시 학습 2단계\n(머리 벌점 있음)", J1, (65, 239), (95, 250), (151, 243)),
       ("다시 학습 1단계\n(머리 벌점 없음)", J1, (77, 240), None, None), ("다시 학습 2단계\n(머리 벌점 없음)", J1, (103, 244), (77, 242), (149, 247))]


def part2():
  FH, AH = 16.4, 10.35; fig = plt.figure(figsize=(22, FH)); U = AH / 9.0
  ax = fig.add_axes([0.01, 5.3 / FH, 0.98, AH / FH]); ax.set_xlim(-6.5, 101); ax.set_ylim(-0.05, 8.95); ax.axis("off")
  YU, YM, YL, YL2, YV1 = 6.05, 4.75, 3.45, 2.3, 0.85
  N = lambda x, y, title, lines, w, col, **k: node(ax, U, x, y, title, lines, w, col, **k)
  for i, (d, a, b) in enumerate([("11/09 (월) 밤 ~ 11/10 새벽", -0.5, 35.5), ("11/10 (화)", 35.5, 55.3), ("11/11 (수)", 55.3, 77.3),
                                 ("11/12 (목)", 77.3, 89.6), ("11/13 (금)", 89.6, 101)]):
    ax.axvspan(a, b, color="#eef1f3" if i % 2 else "white", zorder=0); ax.text((a + b) / 2, 8.78, d, ha="center", va="center", fontsize=11, color="#555")
  # L / F: frozen; one retraining tried and dropped
  ax.annotate("", xy=(100.8, 7.75), xytext=(-0.5, 7.75), arrowprops=dict(arrowstyle="-|>", color=LF, lw=6, alpha=0.3), zorder=1)
  ax.text(62, 7.86, "L·F = rope14@19200, 계속 고정 (앞 L = 빨간 공, 뒤 F)", ha="center", va="bottom", fontsize=10.5, color=LF)
  lf = N(22, 7.08, "L·F 줄 무게 재학습", ["rope14 + 500 it, 줄 10~40 g", "넘어짐 41 → 3% (줄 10 g)", "!줄이 낮아져 J 74 → 0.4%"], 10.6, LF, dead="버림")
  path(ax, [(18.5, 7.75), (18.5, lf["t"])], LF)
  # baseline v3 and what continued from it (+500 it each)
  v3 = N(3.2, YM, "D1Asky@5994", ["풀세트 74%", "실물 46% · 지연 10%"], 8.0, J2, star="baseline v3 (11/09)")
  r1 = N(12.3, YU, "재학습 1", ["도착 뒤 위치 명령 끄기", "기본 73% · 오도 24%"], 8.8, J2)
  r6 = N(40.0, YU, "재학습 6", ["재학습 1 + 500 it", "!기본 40% · 오도 10%"], 8.2, J2, dead="멈춤")
  r2 = N(12.3, YL, "재학습 2", ["재학습 1 조건 + 판의", "절반은 카메라 끄기", "!기본 10% · 오도 0%"], 8.8, J2, dead="무너짐")
  r3 = N(21.5, YL, "재학습 3", ["줄 15~25 g 흔들기", "!15 g 7% · 25 g 26%", "(v3: 11% · 25%)"], 8.6, J2, dead="효과 없음")
  r4 = N(30.7, YL, "재학습 4", ["지연 15~45 ms", "지연 37 → 68%", "!기본 74 → 61%"], 8.6, J2, dead="기본 손해")
  r7 = N(45.0, YM, "재학습 7 @6493", ["지연 15~40 ms", "기본 71% · 지연 67%", "실물 49% (v3 46%)", "실물 + 지연 46% (v3 10%)"], 9.4, J4,
         star="baseline v4 (11/11 사용자 결정)")
  path(ax, [(v3["r"], YM), (r7["l"], YM)], J2, lw=1.8)
  for n_, up in ((r1, True), (r2, False), (r3, False), (r4, False)): path(ax, [(n_["x"], YM), (n_["x"], n_["b"] if up else n_["t"])], J2)
  path(ax, [(r1["r"], YU), (r6["l"], YU)], J2)
  ax.text(24, YM + 0.07, "baseline v3 + 500 it씩", ha="center", va="bottom", fontsize=9.5, color=J2)
  # baseline v4 (= 재학습 7) and what continued from it (+500 it each)
  ax.plot([r7["r"], 99.5], [YM, YM], color=J4, lw=1.8, zorder=1)
  ax.annotate("", xy=(100.9, YM), xytext=(98.5, YM), arrowprops=dict(arrowstyle="-|>", color=J4, lw=1.8), zorder=2)
  ax.text(55.0, YM + 0.07, "baseline v4 + 500 it씩", ha="center", va="bottom", fontsize=9.5, color=J4)
  ax.text(96.6, YM - 0.1, "지금도 baseline v4", ha="center", va="top", fontsize=10.5, fontweight="bold", color=J4)
  kids = [N(50.6, YL, "재학습 8", ["카메라 칸 0, 박자는 절차", "위치 정답 70%", "!카메라 보정 42 → 15%"], 9.0, J4, dead="버림"),
          N(60.2, YU, "재학습 9", ["위치 오차 모델 넣기", "!실물 39% · 지연 29%"], 8.6, J4, dead="버림"),
          N(70.7, YU, "재학습 10 · 10b", ["두 박자 (1.25 / 1.42 Hz)", "!실물 28% · 37%", "@사용자: 회전 속도는 두 가지만, 소리로"], 11.0, J4, dead="실패"),
          N(60.5, YL, "대조", ["아무것도 안 바꾸고 +500 it", "!실물 36% (위치 정답 75%)", "→ 이어 학습만 해도 떨어짐"], 9.0, J4),
          N(71.0, YL, "재학습 11", ["실제 추정 오차 궤적 넣기", "실물 58% (+8.5±8.9)", "!지연 46 → 36%"], 9.6, J4, dead="지연에서 탈락"),
          N(83.4, YU, "이어 학습 (머리 벌점 있음)", ["놓침·지연·추정 오차·줄 무게", "+ head_roll 벌점", "!실물 25% (중간 34%)", "@사용자 승인: 조건 전부 넣기"], 11.0, J4, dead="무너짐"),
          N(83.4, YL, "벌점 있음 + 놓침 0%", ["!실물 31%", "→ 사진 놓침은 원인 아님"], 8.6, J4, dead=""),
          N(95.4, YU, "이어 학습 (머리 벌점 없음)", ["실물 49% · 지연 55%", "머리 기울임 그대로 (96%)", "!기준(실물 +6%p) 미달"], 10.6, J4)]
  for n_ in kids: path(ax, [(n_["x"], YM), (n_["x"], n_["b"] if n_["y"] > YM else n_["t"])], J4)
  r11v3 = N(71.0, YL2, "재학습 11v3", ["같은 것을 baseline v3에서", "실물 52% · 위치 정답 64%"], 9.6, J2)
  path(ax, [(v3["x"], v3["b"]), (v3["x"], YL2), (r11v3["l"], YL2)], J2, lw=1.4)
  ax.text(36, YL2 + 0.07, "baseline v3에서 다시 한 번 (추정 오차 궤적 넣기)", ha="center", va="bottom", fontsize=9.5, color=J2)
  # baseline v1 again, with baseline v4 as the teacher for the entry only
  v1 = N(66.0, YV1, "G1@12700", ["제자리 줄넘기", "(순간이동 입장)"], 8.6, J1, star="baseline v1 (11/06)")
  a = N(83.5, YV1, "다시 학습 (머리 벌점 있음)", ["1단계 입장 따라 하기: 실물 26%", "!2단계 PPO만: 중간 50 → 끝 27%"], 11.5, J1)
  b = N(95.3, YV1, "다시 학습 (머리 벌점 없음)", ["1단계: 실물 32%", "2단계: 실물 42% · 지연 32%", "머리 바로 섬 (끝 닿음 14%)"], 11.0, J1)
  path(ax, [(v1["r"], YV1), (a["l"], YV1)], J1); path(ax, [(v1["x"], v1["b"]), (v1["x"], 0.08), (b["x"], 0.08), (b["x"], b["b"])], J1)
  DS = (0, (4, 2)); path(ax, [(91.2, YM), (91.2, b["t"])], J4, ls=DS, lw=1.4); path(ax, [(91.2, 1.62), (87.5, 1.62), (87.5, a["t"])], J4, ls=DS, lw=1.4)
  ax.text(92.0, 2.62, "재학습 7을 선생으로\n(입장만 따라 하기, BC)", ha="left", va="center", fontsize=9, color=J4, linespacing=1.4)
  # procedure / yardstick turns: numbered diamonds on the timeline, explained in the lower-left corner
  RULES = [(6.0, "오도메트리 풀세트 잣대 (11/09 밤)", "J가 자기 위치를 추정으로만 알면: baseline v3 6%"),
           (37.5, "같은 시드도 흔들림 (11/10)", "74%와 67%는 같은 정책 → 측정을 합쳐서 잣대로"),
           (43.5, "박자 창 + 카메라 위치 보정 (11/10)", "절차만 고쳐서 끝까지 위치 명령 13 → 42%"),
           (52.5, "서서 기다리기 (11/10 밤)", "출발 때 위치 오차 2.0 → 0.7 cm, 38 → 49%"),
           (57.5, "사용자 결정: 실물 조건 잣대 (11/11)", "정책 73차원 + 위치는 절차, baseline은 실물 조건으로만"),
           (65.4, "회전 속도와 줄 무게 (11/11)", "줄 15 g은 1.42 Hz에서 6 → 29%, 25 g은 못 고침"),
           (75.0, "실물 조건 쪼개기 (11/11 밤)", "손해는 전부 추정 위치에서 (위치 정답 71 → 추정 39%)"),
           (92.0, "머리 벌점이 무너짐의 원인 (11/13)", "벌점만 빼니 실물 25 → 49%"),
           (98.5, "방법 조사와 확인 두 가지 (11/13)", "v4의 49%는 운이 아님(새 시드 50%), 이어 학습 때 학습률이 튐")]
  for i, (x, t, d) in enumerate(RULES):
    cx, cy = -5.2 + 22.3 * (i // 3), 1.36 - 0.45 * (i % 3)
    for px, py, ms in ((x, 8.4, 17), (cx, cy, 15)):
      ax.plot([px], [py], marker="D", color="white", mec=RULE, mew=1.5, ms=ms, zorder=3)
      ax.text(px, py, str(i + 1), ha="center", va="center", fontsize=10 if ms == 17 else 9.5, fontweight="bold", color="#444", zorder=4)
    ax.text(cx + 0.9, cy + 0.09, t, ha="left", va="center", fontsize=10, fontweight="bold", color="#444")
    ax.text(cx + 0.9, cy - 0.1, d, ha="left", va="center", fontsize=9, color="#555")
  fig.text(0.5, 1 - 0.3 / FH, "baseline v3 이후의 히스토리 (2026-11-09 밤 ~ 11-13): baseline v4와 그 뒤에 해 본 학습", ha="center", va="center", fontsize=18, fontweight="bold")
  fig.text(0.5, 1 - 0.62 / FH, "상자 = 체크포인트(각 500 it, 점선 상자 = 버림), 가지 = 이어서 학습, 점선 화살표 = 선생으로만 씀. 초록 = baseline v3에서, 청록 = baseline v4에서, 주황 = baseline v1에서 (이어 학습 · 다시 학습은 사용자가 붙인 이름). "
           "기본 = 기본 풀세트(위치 정답), 오도 = 오도메트리 풀세트, 실물 = 실물 조건, 지연 = 실물 + 지연 30~45 ms (10줄 이상).",
           ha="center", va="center", fontsize=10.5, color="#444")
  # bottom: one yardstick
  bx = fig.add_axes([0.055, 0.95 / FH, 0.925, 3.3 / FH]); p4, lo4, hi4 = wilson(*STD[1][2])
  bx.axhspan(lo4, hi4, color=J4, alpha=0.08, zorder=0); bx.axhline(p4, color=J4, ls="--", lw=1, zorder=1)
  bx.text(len(STD) - 0.45, 93, "띠 = baseline v4 실물 조건의 95% 범위", ha="right", va="center", fontsize=9.5, color=J4)
  for i, (lab, col, real, dly, true) in enumerate(STD):
    for k_n, dx, kw in ((real, -0.2, dict(color=col, ec=col)), (dly, 0.2, dict(color=col, ec=col, alpha=0.38, hatch="///"))):
      if not k_n: continue
      p, lo, hi = wilson(*k_n); bx.bar(i + dx, p, 0.36, zorder=2, **kw); bx.plot([i + dx] * 2, [lo, hi], color="k", lw=1.1, zorder=3)
      bx.text(i + dx, hi + 1.5, f"{p:.0f}", ha="center", fontsize=10.5 if dx < 0 else 9.5, fontweight="bold" if dx < 0 else "normal", zorder=3)
    if true:
      p = 100 * true[0] / true[1]; bx.plot([i], [p], marker="D", color="white", mec="#333", mew=1.3, ms=7, zorder=4)
      bx.text(i + 0.12, p, f"{p:.0f}", ha="left", va="center", fontsize=8.5, color="#333")
  bx.set_xticks(range(len(STD))); bx.set_xticklabels([s[0] for s in STD], fontsize=9.5); bx.set_ylim(0, 100); bx.set_xlim(-0.6, len(STD) - 0.4); bx.grid(axis="y", alpha=0.3)
  bx.set_ylabel("10줄 이상 (%)\n선 = 95% 범위", fontsize=10.5)
  bx.text(-0.55, 104, "같은 잣대로 잰 정책들 (풀세트, 시드 2개 합침, 약 240판씩):  진한 막대 = 실물 조건,  빗금 막대 = 실물 조건 + 지연 30~45 ms,  ◇ = 참고: 위치 정답. "
          "baseline은 실물 조건 성적으로만 정합니다.", fontsize=12, color="#333", fontweight="bold")
  fig.text(0.055, 0.2 / FH, "재학습 1~6, 8은 실물 조건 잣대가 생기기 전(11/11 이전)에 끝나 이 잣대로 재지 않았습니다. 재학습 5는 돌리지 않았습니다. 다시 학습 1단계는 실물 조건만 쟀습니다. "
           "숫자 출처: STAGES-ys.md 7-14 ~ 7-25. 그림 코드: docs/ys/history_v4.py", fontsize=10, color="#555")
  buf = io.BytesIO(); fig.savefig(buf, dpi=100); plt.close(fig); buf.seek(0); return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------- baseline family tree
def card(ax, x, top, name, sub, col, rows, w=20.5, nrow=10):
  """Baseline card. rows = (kind, text): b bold, n normal, g grey label, u user decision, r limit."""
  h = 5.3 + 1.22 * nrow; l = x - w / 2
  ax.add_patch(FancyBboxPatch((l, top - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.8", fc="white", ec=col, lw=3.2, zorder=3))
  ax.text(x, top - 1.5, "★ " + name, ha="center", va="center", fontsize=15, fontweight="bold", color=GOLD, zorder=4)
  ax.text(x, top - 3.35, sub, ha="center", va="center", fontsize=9.5, color=USER, zorder=4)
  ax.plot([l + 0.7, l + w - 0.7], [top - 4.4] * 2, color=col, lw=0.9, zorder=4)
  for i, (k, s) in enumerate(rows):
    ax.text(l + 0.9, top - 5.6 - 1.22 * i, s, ha="left", va="center", fontsize=9.3, zorder=4, color={"r": RED, "u": USER, "g": "#777"}.get(k, "#222"),
            fontweight="bold" if k == "b" else "normal")
  return dict(l=l, r=l + w, x=x, t=top, b=top - h)


def lineage():
  fig = plt.figure(figsize=(20, 12.8)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 64); ax.axis("off"); U = 0.2
  N = lambda x, y, title, lines, w, col, **k: node(ax, U, x, y, title, lines, w, col, **k)
  DS = (0, (4, 2)); TOP = 40
  root = N(14.5, 56.5, "걷기 정책 full2500", ["1~3단계 (11/02)"], 12, ROOT)
  g1 = N(14.5, 48.6, "J 1세대: jump13 → side5 → trio1 → B1", ["제자리 점프 → 옆으로 뛰기 → 셋이 한 장면", "→ 카메라로 박자 (11/04 ~ 11/06)"], 20.5, J1)
  g2 = N(62.45, 48.6, "J 2세대: N1 → N1c → N2 → N4 → N5Af → D1A", ["입장부터 같이 → 판정 교체 → 진짜 걸림 벌점", "→ + v2 따라 하기 (11/07 ~ 11/08)"], 23, J2)
  v1 = card(ax, 13.25, TOP, "baseline v1", "2026-11-06 · 사용자 명명 (J/F/L baseline)", J1,
            [("b", "J = G1@12700"), ("n", "줄 안에서 제자리 줄넘기"), ("n", "박자는 머리 카메라로 (L 머리, 빨간 공)"), ("n", "입장은 뛰는 상태로 순간이동 (대역)"),
             ("g", "숫자 (순간이동 입장, 걸리면 끝, 128판)"), ("b", "20줄 이상 36% · 첫 줄 걸림 27%"),
             ("u", "사용자: 앞이 L(빨간 공), 뒤 F는 소리로"), ("r", "한계: 스스로 줄에 못 들어감")])
  v2 = card(ax, 37.85, TOP, "baseline v2", "2026-11-07 · 사용자 승인", J1,
            [("b", "J = E5@13199 = v1 + 500 it"), ("n", "J만 바뀜 (L·F, 시계, 소리, 판정 그대로)"), ("n", "줄 출발 구간 벌점 4배, v1 동작에 붙잡기"),
             ("n", "v1과 행동 차이 RMS 0.19 (모션 유지)"), ("g", "숫자 (같은 잣대, 시드 1 / 2)"), ("b", "20줄 이상 36 / 31 → 61 / 54%"),
             ("r", "한계: 돌고 있는 줄로 입장을 덧붙이는"), ("r", "학습(E1~E8b)은 10줄 이상 0~7%")])
  v3 = card(ax, 62.45, TOP, "baseline v3", "2026-11-09 · 사용자 승인", J2,
            [("b", "J = D1Asky@5994 (2세대)"), ("n", "입장(달려 들어가기)부터 줄넘기까지 한 정책"), ("n", "v2는 줄넘기 선생으로만 (따라 하기)"),
             ("n", "배포 때 head·yaw 목표를 관절 범위로 자름"), ("n", "대기 중 하늘 보기 벌점"), ("g", "숫자 (풀세트, 사용자 판정, 243판)"),
             ("b", "10줄 이상 74% · 약한 밀기 68%"), ("u", "사용자: 처음부터 같이, 관절 범위 원칙"), ("r", "한계: 위치 추정, 명령 지연, 줄 무게")])
  v4 = card(ax, 87.05, TOP, "baseline v4", "2026-11-11 · 사용자 결정 (지금 기준)", J4,
            [("b", "J = 재학습 7 @6493 = v3 + 500 it"), ("n", "학습 때 J의 명령 지연 15~40 ms"), ("g", "숫자 (실물 조건, 약 240판, v3 → v4)"),
             ("b", "실물 46 → 49% · + 지연 10 → 46%"), ("b", "+ 줄 25 g 10 → 16% · + 밀기 39 → 44%"), ("n", "참고: 위치 정답 76 → 71%"),
             ("u", "사용자: baseline은 실물 조건 성적으로만,"), ("u", "정책 73차원 + 위치 계산은 절차"),
             ("r", "한계: 줄 무게(15 g 4%), 추정 위치에서"), ("r", "약 20%p 손해, 머리 기울임(끝 닿음 95%)")])
  # J: weights (solid) and teachers (dashed)
  path(ax, [(root["x"], root["b"]), (g1["x"], g1["t"])], ROOT); path(ax, [(14.5, g1["b"]), (14.5, TOP)], J1)
  ax.text(15.3, 43.3, "+ L 머리 보기 = G1", ha="left", va="center", fontsize=9.5, color=J1)
  path(ax, [(root["r"], 56.5), (g2["x"], 56.5), (g2["x"], g2["t"])], ROOT, ls=(0, (5, 3)))
  ax.text(41.5, 57.1, "11/07 걷기 정책에서 새로 시작 (입장과 줄넘기를 처음부터 같이)", ha="center", va="bottom", fontsize=9.5, color=ROOT)
  path(ax, [(g2["x"], g2["b"]), (g2["x"], TOP)], J2)
  ax.text(63.2, 43.3, "+ 관절 범위 자르기, 하늘 보기 벌점 = D1Asky", ha="left", va="center", fontsize=9.5, color=J2)
  path(ax, [(45.0, TOP), (45.0, 48.6), (g2["l"], 48.6)], J1, ls=DS, lw=1.4)
  ax.text(44.3, 44.3, "v2를 줄넘기 선생으로\n(따라 하기, BC)", ha="right", va="center", fontsize=9.5, color=J1, linespacing=1.4)
  for a_, b_, col in ((v1, v2, J1), (v3, v4, J2)):
    path(ax, [(a_["r"], 31), (b_["l"], 31)], col, lw=2.2); ax.text((a_["r"] + b_["l"]) / 2, 31.9, "+500 it", ha="center", va="bottom", fontsize=9, color=col)
  xm = (v2["r"] + v3["l"]) / 2; ax.text(xm, 31.2, "X", ha="center", va="center", fontsize=15, fontweight="bold", color=RULE)
  ax.text(xm, 28.9, "가중치는\n안 이음", ha="center", va="center", fontsize=8.5, color=RULE, linespacing=1.3)
  # after baseline v4: nothing adopted
  ax.text(89, 58.7, "baseline v4 뒤에 해 본 것 (채택 안 됨)", ha="center", va="center", fontsize=11, fontweight="bold", color="#444")
  again = N(89, 55.6, "다시 학습 (11/12 ~ 11/13)", ["baseline v1에서 다시, v4가 입장 선생", "실물 42% · 지연 32% · 머리는 바로 섬"], 18, J1, dead="")
  cont = N(89, 49.6, "이어 학습 (11/12 ~ 11/13)", ["v4 + 500 it, 실물 조건 묶음을 넣고", "실물 49% · 지연 55% (기준 +6%p 미달)"], 18, J4, dead="")
  rest = N(89, 43.6, "재학습 8 ~ 11, 대조 (11/10 ~ 11/11)", ["v4 + 500 it씩, 조건 하나씩", "실물 28~58%, 기준을 모두 통과한 것 없음"], 18, J4, dead="")
  ax.plot([78.3, 78.3], [TOP, 49.6], color=J4, lw=1.7, zorder=1); ax.plot([78.3, 78.3], [49.6, 55.6], color=J4, lw=1.4, ls=DS, zorder=1)
  path(ax, [(78.3, 43.6), (rest["l"], 43.6)], J4); path(ax, [(78.3, 49.6), (cont["l"], 49.6)], J4); path(ax, [(78.3, 55.6), (again["l"], 55.6)], J4, ls=DS, lw=1.4)
  ax.text(78.9, 52.6, "v4가 입장 선생 (BC)", ha="left", va="center", fontsize=8.5, color=J4)
  path(ax, [(6, v1["b"]), (6, 20.2), (99.3, 20.2), (99.3, 55.6), (again["r"], 55.6)], J1, lw=1.5)
  ax.text(50, 19.85, "다시 학습은 baseline v1에서 다시 시작", ha="center", va="top", fontsize=9.5, color=J1)
  # L / F: one frozen pair for all four baselines
  ax.add_patch(Rectangle((3, 15.0), 94.3, 2.5, fc="#d6eaf8", ec=LF, lw=1.2, zorder=2))
  ax.text(50.15, 16.25, "네 baseline 모두 L(앞, 빨간 공)·F(뒤) = rope14@19200 그대로 (7단계 내내 고정: 줄 0.85 m · 20 g, 박자 1.25 Hz)", ha="center", va="center",
          fontsize=11, fontweight="bold", color=LF, zorder=3)
  ax.text(4.9, 11.5, "L·F 계열 (4~6단계, 11/02 ~ 11/03)", ha="left", va="center", fontsize=10.5, fontweight="bold", color=LF)
  ch = [N(12, 8, "hc4_load@6000", ["머리로 원 + 부리 하중", "4단계"], 14.5, LF), N(28.5, 8, "bc5@14997", ["몸 전체로 큰 원", "5단계"], 13, LF),
        N(43.5, 8, "rope5", ["T핸들로 줄 돌리기", "5단계"], 13, LF), N(58.5, 8, "rope11", ["두 마리 self-play", "6단계"], 13, LF),
        N(75, 8, "rope14@19200", ["두 오리 거리 0.6 m", "넘어짐 0% (11/03 밤)"], 15.5, LF, lw=3.2)]
  path(ax, [(root["l"], 56.5), (1.3, 56.5), (1.3, 8), (ch[0]["l"], 8)], ROOT)
  for a_, b_ in zip(ch, ch[1:]): path(ax, [(a_["r"], 8), (b_["l"], 8)], LF)
  path(ax, [(75, ch[-1]["t"]), (75, 15.0)], LF, lw=2.2)
  fig.text(0.5, 0.972, "베이스라인 계보도 (2026-11-13 현재): baseline v1 → v2 → v3 → v4", ha="center", va="center", fontsize=18, fontweight="bold")
  fig.text(0.5, 0.948, "실선 = 가중치를 이어받아 계속 학습, 점선 = 선생으로만 씀(따라 하기, BC). 보라 글씨 = 사용자 결정. L = 리더(앞), F = 팔로워(뒤), J = 점퍼(가운데).",
           ha="center", va="center", fontsize=11, color="#444")
  fig.text(0.03, 0.034, "카드의 숫자는 그 baseline을 정한 날의 잣대로 잰 값입니다. v1·v2(줄 안에 순간이동으로 넣고 20줄 이상)와 v3·v4(풀세트: 손잡이 물기 → 걸어 나가기 → 시작 소리 → 입장 → 줄넘기, 10줄 이상)는 "
           "잣대가 달라 서로 비교하지 않습니다.", fontsize=10, color="#555")
  fig.text(0.03, 0.014, "실물 = 실물 조건(위치 추정 + 카메라 보정 + 서서 기다리기, 사진 놓침 5%), 지연 = 실물 조건 + 명령 지연 30~45 ms. 숫자 출처: STAGES-ys.md 7단계. 그림 코드: docs/ys/history_v4.py",
           fontsize=10, color="#555")
  fig.savefig(os.path.join(HERE, "baseline_lineage.png"), dpi=100); plt.close(fig)


if __name__ == "__main__":
  top, bot = Image.open(os.path.join(HERE, "history_v3.png")).convert("RGB"), part2(); assert top.width == bot.width, (top.size, bot.size)
  out = Image.new("RGB", (top.width, top.height + bot.height), "white"); out.paste(top, (0, 0)); out.paste(bot, (0, top.height))
  out.save(os.path.join(HERE, "history_v4.png")); lineage(); print("saved", HERE + "/history_v4.png", out.size, HERE + "/baseline_lineage.png")
