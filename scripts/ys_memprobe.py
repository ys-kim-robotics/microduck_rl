"""ys_memprobe.py <out.json> <run.npz> [<run.npz> ...]   (7-26, DEPLOY-ys.md 1-6: the memory probe; no policy is trained)
Can a memory of J's own observations tell how wrong its position estimate is? Recordings: scripts/ys_fullset.py with DIAG=<run.npz>
OBSLOG=1 in the realistic condition (<run>.json next to it gives each env's kind). Samples: every STEP-th step J is in play (0.5 s
after its arrival until its end call). Input: the last H actor observations (73-D each: 48 proprioception, 13 command, 12 camera),
or "sofar": the latest one + the mean of all since the arrival (what an unlimited memory could average). Target: true - estimated
xy (tr[..., 11:13], m). FOLDS folds split by game (the error is mostly one constant per game, so the steps of one game must not sit
on both sides). Ridge regression (lambda picked on a held-out fifth of the training games) and a small MLP (256-128, early stopping
on the same held-out games). Score: 2-D RMS error (cm) on the test games, 95% range by resampling games; next to it the estimate as
it is ("as is"), minus the training games' mean ("const"), minus each game's own mean ("game": the floor if the per-game constant
were known). x = along the rope (F -> L), y = across it. SELFTEST=1: a planted target (the observation 10 steps back) that only a memory of >= 11 steps can read.
7-27 (DEPLOY 1-8): FIT=<model.pt> ys_memprobe.py - <run.npz> ...: the net ys_fullset.py runs in the loop (CORR=<model.pt>): the last FIT_H observations + the
correction in use (cm) -> what is left of the error (cm). Recordings with CORR_EXPLORE (the correction wandering) teach it what a correction does to what J sees.
Every 5th game stops the training early, the next of 5 scores it. CHECK=<model.pt> ys_memprobe.py - <run.npz>: a CORR recording replayed offline: the
correction recomputed from the recorded observations must be the recorded one (the loop feeds the net the windows it was fitted on)."""
import json, os, sys
import numpy as np, torch

STEP, FOLDS, HS, HMAX, FIT_H = 5, 5, (1, 5, 25, 50, 100), 100, 50
FROM = os.environ.get("CORR_FROM", "arr")  # FIT: go = samples from J's set-off on (every 2nd step until 0.5 s after the arrival: that part is short)
TBINS = ((0.5, 1.5), (1.5, 3.0), (3.0, 8.0), (8.0, 99.0))  # s since J's arrival: the windows of "by time"
PARTS = {"all": slice(0, 73), "proprio": slice(0, 48), "command": slice(48, 61), "camera": slice(61, 73)}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load(runs):
  """-> X (samples, HMAX, 73) the last HMAX observations, C (samples, 73) their mean since the arrival, Y (samples, 2) m, G game id, T steps since the arrival,
  A (samples, 2) m the correction in use when the latest observation was made (0 without CORR; Y is then the error left over that one)."""
  X, C, Y, G, T, A = [], [], [], [], [], []
  for f in runs:
    d = np.load(f); kind = [r["kind"] for r in json.load(open(f[:-4] + ".json"))["games"]]
    tr, ob, arr, call = d["tr"], d["ob"], d["arr"], d["call"]; co = tr[:, :, 15:17] if tr.shape[2] > 15 else np.zeros_like(tr[:, :, 11:13])
    for n in range(tr.shape[1]):
      k0 = max(d["go"][n] + 1 if FROM == "go" else arr[n] + 25, HMAX - 1); ks = np.r_[np.arange(k0, min(arr[n] + 25, call[n]), 2), np.arange(max(k0, arr[n] + 25), call[n], STEP)]
      if kind[n] != "game" or arr[n] < 0 or len(ks) == 0: continue
      cm = np.cumsum(ob[k0:call[n], n], 0) / np.arange(1, call[n] - k0 + 1)[:, None]
      X.append(np.stack([ob[k - HMAX + 1:k + 1, n] for k in ks])); C.append(cm[ks - k0]); Y.append(tr[ks, n, 11:13] + co[ks, n] - co[ks - 1, n]); G.append(np.full(len(ks), len(G))); T.append(ks - arr[n]); A.append(co[ks - 1, n])
  return [np.concatenate(v) for v in (X, C, Y, G, T, A)]


def ridge(xa, ya, xv, yv, xt):
  A, B, best = xa.T @ xa, xa.T @ ya, (np.inf, None)
  for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):  # x the number of samples (inputs are standardised)
    w = torch.linalg.solve(A + lam * len(xa) * torch.eye(xa.shape[1], device=dev), B); e = float(((xv @ w - yv) ** 2).mean())
    if e < best[0]: best = (e, xt @ w)
  return best[1]


def make_net(d): return torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ELU(), torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 2))


def mlp(xa, ya, xv, yv, xt, keep=None):
  """keep: a dict that gets the best net's weights (FIT)."""
  torch.manual_seed(0)
  net = make_net(xa.shape[1]).to(dev)
  opt, best, bad = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2), (np.inf, None), 0
  for _ in range(60):
    for i in torch.randperm(len(xa), device=dev).split(1024):
      opt.zero_grad(); ((net(xa[i]) - ya[i]) ** 2).mean().backward(); opt.step()
    with torch.no_grad(): e = float(((net(xv) - yv) ** 2).mean())
    if e < best[0]:
      with torch.no_grad(): best, bad = (e, net(xt)), 0
      if keep is not None: keep["net"] = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
    else:
      bad += 1
      if bad >= 6: break
  return best[1]


def score(err, G, rng):
  """2-D RMS (cm) of err (samples, 2), its 95% range by resampling games, then the RMS of x and of y alone."""
  ng = G.max() + 1; sse, cnt = np.bincount(G, (err ** 2).sum(1), ng), np.bincount(G, minlength=ng); ok = cnt > 0; sse, cnt = sse[ok], cnt[ok]
  bs = [np.sqrt(sse[i].sum() / cnt[i].sum()) for i in rng.integers(0, len(sse), (1000, len(sse)))]
  return [round(float(v), 2) for v in (np.sqrt(sse.sum() / cnt.sum()), *np.percentile(bs, [2.5, 97.5]), *np.sqrt((err ** 2).mean(0)))]


def probe(F, Y, G, models=(("ridge", ridge), ("mlp", mlp)), T=None):
  """F: name -> (samples, d) features; Y in cm. -> name -> model -> [rms, lo, hi, x, y], plus the three references; with T (steps since the
  arrival) also "by time": the same errors' 2-D RMS (cm) inside each TBINS window."""
  rng = np.random.default_rng(0); ng = G.max() + 1; fold = rng.permutation(ng) % FOLDS
  pred = {(nm, m): np.zeros_like(Y) for nm in F for m, _ in models}; const = np.zeros_like(Y)
  for f in range(FOLDS):
    te = fold[G] == f; trg = np.flatnonzero(fold != f); val = np.isin(G, trg[::5]); fit = ~te & ~val; const[te] = Y[~te].mean(0)
    ya, yv = (torch.as_tensor(Y[m] - Y[~te].mean(0), dtype=torch.float32, device=dev) for m in (fit, val))
    for nm, x in F.items():
      mu, sd = x[fit].mean(0), x[fit].std(0) + 1e-6
      xa, xv, xt = (torch.as_tensor(((x[m] - mu) / sd).clip(-5, 5), dtype=torch.float32, device=dev) for m in (fit, val, te))
      for m, fn in models: pred[(nm, m)][te] = fn(xa, ya, xv, yv, xt).cpu().numpy() + Y[~te].mean(0)
  gm = np.stack([np.bincount(G, Y[:, i], ng) for i in (0, 1)], 1) / np.maximum(np.bincount(G, minlength=ng), 1)[:, None]
  out = {"as is": score(Y, G, rng), "const": score(Y - const, G, rng), "game": score(Y - gm[G], G, rng)}
  for (nm, m), p in pred.items(): out.setdefault(nm, {})[m] = score(Y - p, G, rng)
  if T is not None:
    rms = lambda e, k: round(float(np.sqrt((e[k] ** 2).sum(1).mean())), 2)
    out["by time"] = {f"{a}~{b} s": {"n": int(k.sum()), "as is": rms(Y, k), "const": rms(Y - const, k), "game": rms(Y - gm[G], k), **{f"{nm} {m}": rms(Y - p, k) for (nm, m), p in pred.items()}}
                      for a, b in TBINS for k in [(T >= 50 * a) & (T < 50 * b)]}
  return out


if __name__ == "__main__":
  if os.environ.get("SELFTEST") == "1":  # 200 games x 40 samples of noise; the target = observation 0, 10 steps back, + 0.2 noise
    r = np.random.default_rng(1); X = r.normal(size=(8000, HMAX, 73)).astype(np.float32); G = np.repeat(np.arange(200), 40)
    Y = np.stack([X[:, -11, 0], X[:, -11, 1]], 1) + 0.2 * r.normal(size=(8000, 2))
    o = probe({"H5": X[:, -5:].reshape(8000, -1), "H25": X[:, -25:].reshape(8000, -1)}, Y, G, (("ridge", ridge),))
    print(o); assert o["H5"]["ridge"][0] > 1.2 and o["H25"]["ridge"][0] < 0.45, o; print("SELFTEST OK"); sys.exit()
  if os.environ.get("FIT"):
    X, C, Y, G, T, A = load(sys.argv[2:]); Y, A = 100 * Y, 100 * A; x = X[:, -FIT_H:].reshape(len(Y), -1); del X, C
    va, te = G % 5 == 0, G % 5 == 1; fit = ~va & ~te; mu, sd, ym = x[fit].mean(0), x[fit].std(0) + 1e-6, Y[fit].mean(0); keep = {}
    f = lambda m: torch.as_tensor(np.concatenate([((x[m] - mu) / sd).clip(-5, 5), A[m]], 1), dtype=torch.float32, device=dev)
    t = lambda m: torch.as_tensor(Y[m] - ym, dtype=torch.float32, device=dev)
    p = mlp(f(fit), t(fit), f(va), t(va), f(te), keep).cpu().numpy() + ym; rms = lambda e: round(float(np.sqrt((e ** 2).sum(1).mean())), 2)
    torch.save(dict(net=keep["net"], mu=mu, sd=sd, ym=ym, H=FIT_H, frm=FROM), os.environ["FIT"])
    print("FIT samples", len(Y), "games", int(G.max()) + 1, "| test games: error left cm", rms(Y[te]), "-> after the net", rms(Y[te] - p),
          "| with no correction in use:", rms(Y[te][np.abs(A[te]).sum(1) == 0]), "->", rms((Y[te] - p)[np.abs(A[te]).sum(1) == 0]), "| correction in use, sd cm", A.std(0).round(2)); sys.exit()
  if os.environ.get("CHECK"):
    m = torch.load(os.environ["CHECK"], weights_only=False); net = make_net(len(m["mu"]) + 2); net.load_state_dict(m["net"]); d = np.load(sys.argv[2]); tr, ob = d["tr"], d["ob"]
    tau = float(os.environ.get("CORR_TAU", "0.2")); worst = 0.0; n_ = 0
    for n in np.flatnonzero((d["arr"] >= 0) & (d["call"] > d["arr"] + 60))[:8]:
      c = np.zeros(2)
      for k in range(d["go"][n] + 1 if m.get("frm") == "go" else d["arr"][n] + 25, d["call"][n]):  # the loop's first update
        z = np.concatenate([((ob[k - m["H"] + 1:k + 1, n].reshape(-1) - m["mu"]) / m["sd"]).clip(-5, 5), 100 * c]).astype(np.float32)
        with torch.no_grad(): c = np.clip(c + 0.02 / tau * (net(torch.as_tensor(z)[None])[0].numpy() + m["ym"]) / 100, -0.06, 0.06)
        worst = max(worst, float(np.abs(c - tr[k, n, 15:17]).max())); n_ += 1
    print("CHECK steps", n_, "max |offline - recorded correction| m", worst); assert n_ > 100 and worst < 2e-4, worst; print("CHECK OK"); sys.exit()
  X, C, Y, G, T, _ = load(sys.argv[2:]); Y = 100 * Y; n = len(Y); print("samples", n, "games", G.max() + 1, "bias cm", Y.mean(0).round(2), flush=True)
  F = {f"H{h}": X[:, -h:].reshape(n, -1) for h in HS}
  F.update({f"H50 {p}": X[:, -50:, s].reshape(n, -1) for p, s in PARTS.items() if p != "all"}); F["sofar"] = np.concatenate([X[:, -1], C], 1)
  out = probe(F, Y, G, T=T); out.update(samples=n, games=int(G.max() + 1), runs=sys.argv[2:]); json.dump(out, open(sys.argv[1], "w"), indent=1)
  for k, v in out.items(): print(f"{k:14s} {v}")
