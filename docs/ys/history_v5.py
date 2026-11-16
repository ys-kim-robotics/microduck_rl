"""history_v5.py -> history_v5.png, baseline_lineage_v5.png: the model history up to 2026-11-17 and the baseline v5 family tree.
history_v5.png = history_v4.png (kept as is: up to 2026-11-13) stacked on a new part for 11-13 .. 11-17: how baseline v5-1 and v5-2 were
found, the L·F retrainings tried (LF2 / LF3 / LF4, all dropped), the yardstick / procedure turns, and the policies on one yardstick (realistic
conditions and the truth-free yardstick, seeds 21-24 pooled). baseline_lineage_v5.png = baseline v3 / v4 and the two v5s they lead to.
Numbers come from STAGES-ys.md 7-29 .. 7-41. Drawing helpers from history_v4.py. Run: uv run python docs/ys/history_v5.py"""
import io, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from history_v4 import HERE, J2, J4, LF, RED, RULE, USER, card, node, path, wilson

# one yardstick, seeds 21-24 pooled (k, n): realistic conditions (7-32, 7-33), truth-free (7-36, 7-40), truth-free with J's spot from
# L's and F's own estimates + the fixed gyro filter (7-41)
BARS = [("baseline v4", J4, (236, 508), (53, 506), "안 잼"), ("baseline v5-1\n(R14@6950)", J4, (283, 508), (101, 511), (216, 508)),
        ("baseline v5-2\n(P1b@6100)", J2, (316, 507), (114, 505), (221, 509))]


def part3():
  FH, AH = 15.2, 9.6; fig = plt.figure(figsize=(22, FH)); U = AH / 9.0
  ax = fig.add_axes([0.01, 4.85 / FH, 0.98, AH / FH]); ax.set_xlim(-6.5, 101); ax.set_ylim(-0.05, 8.95); ax.axis("off")
  N = lambda x, y, title, lines, w, col, **k: node(ax, U, x, y, title, lines, w, col, **k)
  for i, (d, a, b) in enumerate([("11/13 (금) ~ 11/14 (토)", -0.5, 24), ("11/15 (일)", 24, 52), ("11/16 (월)", 52, 86), ("11/17 (화) 새벽", 86, 101)]):
    ax.axvspan(a, b, color="#eef1f3" if i % 2 else "white", zorder=0); ax.text((a + b) / 2, 8.78, d, ha="center", va="center", fontsize=11, color="#555")
  # L / F: still rope14; three retrainings for the rope's mass, all dropped
  ax.annotate("", xy=(100.8, 7.75), xytext=(-0.5, 7.75), arrowprops=dict(arrowstyle="-|>", color=LF, lw=6, alpha=0.3), zorder=1)
  ax.text(20, 7.86, "L·F = rope14@19200, 계속 고정", ha="center", va="bottom", fontsize=10.5, color=LF)
  for x, t, ls in ((60.5, "LF2 (11/16 오전)", ["줄 15~25 g + 손잡이 간격 보상", "!세 무게 모두 줄이 늦게 옴"]),
                   (75.0, "LF3 (11/16 저녁, 3-a)", ["+ 줄 무게를 L·F 입력으로", "무게 차이 32 → 0~5°", "!20 g이 7~25° 늦음, J와 7%"]),
                   (93.5, "LF4 (11/17, 나)", ["+ 줄 박자를 직접 보상", "!300 it 자동 멈춤", "!20 g이 6~22° 늦음"])):
    n_ = N(x, 6.85, t, ls, 11.0, LF, dead="버림"); path(ax, [(x - 4.5, 7.75), (x - 4.5, n_["t"])], LF)
  # J from baseline v4: R14 (이어 학습, 11/13) -> its checkpoint 6950 = baseline v5-1
  YM, YU, YL, Y3 = 4.6, 5.75, 3.45, 1.95
  v4 = N(3.6, YM, "재학습 7 @6493", ["실물 조건 46.5%", "(시드 21~24)"], 8.4, J4, star="baseline v4")
  r14 = N(16.0, YM, "R14 = 이어 학습", ["v4 + 500 it, 머리 벌점 없음", "최종본 6992: 49.4%"], 10.4, J4)
  pick = N(40.0, YM, "R14 @6950", ["R14 안의 저장본 고르기 (11/15)", "처음 쓰는 시드 59.8% (v4 45.2%)", "부분 2: 25점 중 나빠짐 0"], 12.0, J4,
           star="baseline v5-1 (11/16 사용자 결정)")
  path(ax, [(v4["r"], YM), (r14["l"], YM)], J4, lw=1.8); path(ax, [(r14["r"], YM), (pick["l"], YM)], J4, lw=1.8)
  kids = N(31.0, YU + 0.45, "R16 · R17 · R18 (11/15 밤)", ["R14 + 500 it씩 (카메라 범위, 오차 녹화)", "!실물 35~47%, 모두 R14보다 낮음"], 13.0, J4, dead="버림")
  path(ax, [(22.0, YM), (22.0, kids["b"] - 0.05), (kids["l"] + 0.6, kids["b"] - 0.05)], J4, lw=1.4)
  # J from baseline v3: 재학습 1 (11/10) -> its checkpoint 6100 = baseline v5-2
  v3 = N(3.6, Y3, "D1Asky@5994", ["실물 41% (시드 1·2)"], 8.4, J2, star="baseline v3")
  p1 = N(16.0, Y3, "재학습 1 (11/10)", ["v3 + 위치를 3~10 cm", "틀리게 알고 학습"], 10.4, J2)
  p6 = N(40.0, Y3, "재학습 1 @6100 (+100 it)", ["실패했던 학습의 저장본 115개 훑기", "실물 62.3% (v5-1보다 +6.6%p)", "!명령 지연 30·35 ms에서 v4보다 낮음"], 12.6, J2,
         star="baseline v5-2 (11/16 사용자 결정)")
  path(ax, [(v3["r"], Y3), (p1["l"], Y3)], J2, lw=1.8); path(ax, [(p1["r"], Y3), (p6["l"], Y3)], J2, lw=1.8)
  rep = N(75.0, YL, "재학습 1의 방법 다시 (11/16)", ["v3에서 재현됨", "!v4·v5-1에 적용하면 55~62%를 못 넘음"], 12.6, J2, dead="버림")
  path(ax, [(p6["r"], Y3), (75.0, Y3), (75.0, rep["b"])], J2, lw=1.4)
  ax.text(57.5, (YM + Y3) / 2 + 0.25, "@사용자: 둘 다 baseline으로 두고\n실물 명령 지연을 재서 고른다\n(20 ms 이하면 v5-2, 길면 v5-1)", ha="center", va="center",
          fontsize=10, color=USER, linespacing=1.4)
  for n_ in (pick, p6): ax.annotate("", xy=(50.5, (YM + Y3) / 2 + 0.25), xytext=(n_["r"], n_["y"]), arrowprops=dict(arrowstyle="-", color=USER, lw=1, ls=(0, (3, 2))))
  # procedure / yardstick turns: numbered diamonds on the timeline, explained below
  RULES = [(8.0, "실물 카메라 범위 잣대 (11/14~15)", "시뮬 카메라가 안 보일 때도 참값을 줬음 → v4 47%"),
           (28.0, "Full check 부분 2·3 (11/15)", "후보 25점 중 나빠짐 0, 저장본 115개에서 재학습 1 찾음"),
           (55.0, "부분 4: 참값 지우기 (11/16)", "J가 참값을 하나도 모르면 세 정책 모두 10~23%"),
           (58.5, "줄 무게 약점의 원인 (11/16)", "물리엔진 아님: 꼭대기에서 팽팽한 줄이 L·F 손잡이를 끎"),
           (82.0, "J 자리 치우침 빼기, 4-c (11/16 밤)", "평균만 빠지고 흩어짐은 남음: +5~9%p"),
           (88.5, "자이로 칼만 필터, 4-a (11/17)", "틀어짐만 있으면 완전 회복 (6 → 58%)"),
           (95.5, "L·F가 자기 위치를 J에게 보냄 (11/17)", "참값 없는 잣대 v5-1 20 → 43%, v5-2 23 → 43% (+ 칼만 필터)")]
  for i, (x, t, d) in enumerate(RULES):
    cx, cy = -5.2 + 27.0 * (i // 2), 0.86 - 0.47 * (i % 2)
    for px, py, ms in ((x, 8.4, 17), (cx, cy, 15)):
      ax.plot([px], [py], marker="D", color="white", mec=RULE, mew=1.5, ms=ms, zorder=3)
      ax.text(px, py, str(i + 1), ha="center", va="center", fontsize=10 if ms == 17 else 9.5, fontweight="bold", color="#444", zorder=4)
    ax.text(cx + 0.9, cy + 0.09, t, ha="left", va="center", fontsize=10, fontweight="bold", color="#444")
    ax.text(cx + 0.9, cy - 0.12, d, ha="left", va="center", fontsize=9, color="#555")
  fig.text(0.5, 1 - 0.3 / FH, "baseline v4 이후의 히스토리 (2026-11-13 ~ 11-17): baseline v5-1·v5-2와 그 뒤에 해 본 것", ha="center", va="center", fontsize=18, fontweight="bold")
  fig.text(0.5, 1 - 0.62 / FH, "상자 = 체크포인트(점선 상자 = 버림), 가지 = 이어서 학습. 청록 = baseline v4에서, 초록 = baseline v3에서, 파랑 = L·F. 보라 글씨 = 사용자 결정. "
           "실물 = 실물 조건(7-30 잣대), 참값 없는 잣대 = J가 참값을 하나도 모르는 조건(7-36).", ha="center", va="center", fontsize=10.5, color="#444")
  # bottom: one yardstick, seeds 21-24
  bx = fig.add_axes([0.055, 0.85 / FH, 0.925, 3.1 / FH]); G = [("실물 조건", 2), ("참값 없는 잣대", 3), ("참값 없는 잣대\n+ L·F가 보낸 위치 + 칼만 필터", 4)]
  xt, xl = [], []
  for gi, (gl, col_i) in enumerate(G):
    for bi, b in enumerate(BARS):
      x = gi * 4.2 + bi; k_n = b[col_i]; xt.append(x); xl.append(b[0])
      if not isinstance(k_n, tuple): bx.text(x, 3, k_n or "재는 중", ha="center", fontsize=9, color="#999"); continue
      p, lo, hi = wilson(*k_n); bx.bar(x, p, 0.7, color=b[1], alpha=1.0 if gi == 0 else 0.75, hatch=None if gi < 2 else "///", ec=b[1], zorder=2)
      bx.plot([x, x], [lo, hi], color="k", lw=1.1, zorder=3); bx.text(x, hi + 1.5, f"{p:.0f}", ha="center", fontsize=10.5, fontweight="bold", zorder=3)
    bx.text(gi * 4.2 + 1, 78, gl, ha="center", va="center", fontsize=11, fontweight="bold", color="#333", linespacing=1.3)
  bx.set_xticks(xt); bx.set_xticklabels(xl, fontsize=9); bx.set_ylim(0, 90); bx.set_xlim(-0.7, 2 * 4.2 + 2.7); bx.grid(axis="y", alpha=0.3)
  bx.set_ylabel("10줄 이상 (%)\n선 = 95% 범위", fontsize=10.5); bx.spines[["top", "right"]].set_visible(False)
  bx.text(-0.65, 94, "같은 잣대로 잰 정책들 (풀세트, 줄 20 g, 시드 21~24 합침, 약 510판씩)", fontsize=12, color="#333", fontweight="bold")
  fig.text(0.055, 0.2 / FH, "참값 없는 잣대의 J 자리: 둘째 묶음 = L·F에게 준 걷기 목표의 가운데(지금 방식), 셋째 묶음 = L·F가 발로 센 자기 위치를 J에게 보냄(로봇 사이 통신 필요). "
           "숫자 출처: STAGES-ys.md 7-29 ~ 7-41. 그림 코드: docs/ys/history_v5.py", fontsize=10, color="#555")
  buf = io.BytesIO(); fig.savefig(buf, dpi=100); plt.close(fig); buf.seek(0); return Image.open(buf).convert("RGB")


def lineage():
  fig = plt.figure(figsize=(20, 7.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(11, 48); ax.axis("off")
  TOP = 40
  v3 = card(ax, 12.5, TOP, "baseline v3", "2026-11-09 · 사용자 승인", J2,
            [("b", "J = D1Asky@5994"), ("n", "입장부터 줄넘기까지 한 정책"), ("g", "실물 조건 (7-30 잣대, 시드 1·2)"), ("b", "41%"),
             ("r", "한계: 위치 추정, 명령 지연, 줄 무게")], nrow=6)
  v4 = card(ax, 37.5, TOP, "baseline v4", "2026-11-11 · 사용자 결정", J4,
            [("b", "J = 재학습 7 @6493 = v3 + 500 it"), ("n", "학습 때 명령 지연 15~40 ms"), ("g", "실물 조건 (시드 21~24)"), ("b", "46.5%"),
             ("r", "한계: 줄 무게, 참값 없이 10%")], nrow=6)
  w1 = card(ax, 64.0, TOP, "baseline v5-1", "2026-11-16 · 사용자 결정", J4,
            [("b", "J = R14 @6950 = v4 + 457 it"), ("n", "이어 학습, 머리 기울임 벌점 없음"), ("g", "실물 조건 / 참값 없는 잣대 (시드 21~24)"),
             ("b", "55.7% / 19.8%"), ("n", "+ L·F 위치 보내기 + 칼만 필터: 42.5%"), ("u", "실물 명령 지연이 20 ms보다 길면 이것"),
             ("r", "한계: 줄 15 g 3%, 25 g 21%")], w=23, nrow=8)
  w2 = card(ax, 88.5, TOP, "baseline v5-2", "2026-11-16 · 사용자 결정", J2,
            [("b", "J = 재학습 1 @6100 = v3 + 106 it"), ("n", "위치를 3~10 cm 틀리게 알고 학습"), ("g", "실물 조건 / 참값 없는 잣대 (시드 21~24)"),
             ("b", "62.3% / 22.6%"), ("n", "+ L·F 위치 보내기 + 칼만 필터: 43.4%"), ("u", "실물 명령 지연 20 ms 이하면 이것"),
             ("r", "한계: 지연 30·35 ms에서 v4보다 낮음")], w=21, nrow=8)
  path(ax, [(v3["r"], 31), (v4["l"], 31)], J2, lw=2.2); ax.text((v3["r"] + v4["l"]) / 2, 31.9, "+500 it", ha="center", va="bottom", fontsize=9, color=J2)
  path(ax, [(v4["r"], 31), (w1["l"], 31)], J4, lw=2.2); ax.text((v4["r"] + w1["l"]) / 2, 31.9, "+457 it", ha="center", va="bottom", fontsize=9, color=J4)
  path(ax, [(v3["x"], v3["b"]), (v3["x"], 21.5), (w2["x"], 21.5), (w2["x"], w2["b"])], J2, lw=1.8)
  ax.text(50, 22.1, "baseline v3 + 106 it (재학습 1, 11/10 학습, 11/15 저장본 훑기에서 찾음)", ha="center", va="bottom", fontsize=9.5, color=J2)
  ax.text(50, 17.0, "네 baseline 모두 L(앞, 빨간 공)·F(뒤) = rope14@19200 그대로. 줄 무게용 L·F 학습 LF2·LF3·LF4는 모두 버림(20 g 박자가 늦어짐).", ha="center", va="center",
          fontsize=11, fontweight="bold", color=LF)
  fig.text(0.5, 0.955, "베이스라인 계보도 (2026-11-17 현재): baseline v3 → v4 → v5-1, v3 → v5-2", ha="center", va="center", fontsize=18, fontweight="bold")
  fig.text(0.5, 0.91, "실선 = 가중치를 이어받아 계속 학습. 보라 글씨 = 사용자 결정. v1·v2와 그 앞은 baseline_lineage.png.", ha="center", va="center", fontsize=11, color="#444")
  fig.text(0.03, 0.035, "숫자 출처: STAGES-ys.md 7-30 ~ 7-41. 그림 코드: docs/ys/history_v5.py", fontsize=10, color="#555")
  fig.savefig(os.path.join(HERE, "baseline_lineage_v5.png"), dpi=100); plt.close(fig)


if __name__ == "__main__":
  top, bot = Image.open(os.path.join(HERE, "history_v4.png")).convert("RGB"), part3(); assert top.width == bot.width, (top.size, bot.size)
  out = Image.new("RGB", (top.width, top.height + bot.height), "white"); out.paste(top, (0, 0)); out.paste(bot, (0, top.height))
  out.save(os.path.join(HERE, "history_v5.png")); bot.save("/tmp/claude-1000/-home-roai22/aae78e4a-6740-4769-a2da-c932e7ef790d/scratchpad/s38/h5_part3.png")
  lineage(); print("saved", HERE + "/history_v5.png", out.size, HERE + "/baseline_lineage_v5.png")
