"""ys_memprobe.py <out.json> <run.npz> [<run.npz> ...]   (7-26, DEPLOY-ys.md 1-6: the memory probe; no policy is trained)
Can a memory of J's own observations tell how wrong its position estimate is? Recordings: scripts/ys_fullset.py with DIAG=<run.npz>
OBSLOG=1 in the realistic condition (<run>.json next to it gives each env's kind). Samples: every STEP-th step J is in play (0.5 s
after its arrival until its end call). Input: the last H actor observations (73-D each: 48 proprioception, 13 command, 12 camera),
or "sofar": the latest one + the mean of all since the arrival (what an unlimited memory could average). Target: true - estimated
xy (tr[..., 11:13], m). FOLDS folds split by game (the error is mostly one constant per game, so the steps of one game must not sit
on both sides). Ridge regression (lambda picked on a held-out fifth of the training games) and a small MLP (256-128, early stopping
on the same held-out games). Score: 2-D RMS error (cm) on the test games, 95% range by resampling games; next to it the estimate as
it is ("as is"), minus the training games' mean ("const"), minus each game's own mean ("game": the floor if the per-game constant
were known). x = along the rope (F -> L), y = across it. SELFTEST=1: a planted target (the observation 10 steps back) that only a memory of >= 11 steps can read."""
import json, os, sys
import numpy as np, torch

STEP, FOLDS, HS, HMAX = 5, 5, (1, 5, 25, 50, 100), 100
TBINS = ((0.5, 1.5), (1.5, 3.0), (3.0, 8.0), (8.0, 99.0))  # s since J's arrival: the windows of "by time"
PARTS = {"all": slice(0, 73), "proprio": slice(0, 48), "command": slice(48, 61), "camera": slice(61, 73)}
dev = "cuda" if torch.cuda.is_available() else "cpu"


def load(runs):
  """-> X (samples, HMAX, 73) the last HMAX observations, C (samples, 73) their mean since the arrival, Y (samples, 2) m, G game id, T steps since the arrival."""
  X, C, Y, G, T = [], [], [], [], []
  for f in runs:
    d = np.load(f); kind = [r["kind"] for r in json.load(open(f[:-4] + ".json"))["games"]]
    tr, ob, arr, call = d["tr"], d["ob"], d["arr"], d["call"]
    for n in range(tr.shape[1]):
      k0 = max(arr[n] + 25, HMAX - 1); ks = np.arange(k0, call[n], STEP)
      if kind[n] != "game" or arr[n] < 0 or len(ks) == 0: continue
      cm = np.cumsum(ob[k0:call[n], n], 0) / np.arange(1, call[n] - k0 + 1)[:, None]
      X.append(np.stack([ob[k - HMAX + 1:k + 1, n] for k in ks])); C.append(cm[ks - k0]); Y.append(tr[ks, n, 11:13]); G.append(np.full(len(ks), len(G))); T.append(ks - arr[n])
  return [np.concatenate(v) for v in (X, C, Y, G, T)]


def ridge(xa, ya, xv, yv, xt):
  A, B, best = xa.T @ xa, xa.T @ ya, (np.inf, None)
  for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):  # x the number of samples (inputs are standardised)
    w = torch.linalg.solve(A + lam * len(xa) * torch.eye(xa.shape[1], device=dev), B); e = float(((xv @ w - yv) ** 2).mean())
    if e < best[0]: best = (e, xt @ w)
  return best[1]


def mlp(xa, ya, xv, yv, xt):
  torch.manual_seed(0)
  net = torch.nn.Sequential(torch.nn.Linear(xa.shape[1], 256), torch.nn.ELU(), torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 2)).to(dev)
  opt, best, bad = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2), (np.inf, None), 0
  for _ in range(60):
    for i in torch.randperm(len(xa), device=dev).split(1024):
      opt.zero_grad(); ((net(xa[i]) - ya[i]) ** 2).mean().backward(); opt.step()
    with torch.no_grad(): e = float(((net(xv) - yv) ** 2).mean())
    if e < best[0]:
      with torch.no_grad(): best, bad = (e, net(xt)), 0
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
  X, C, Y, G, T = load(sys.argv[2:]); Y = 100 * Y; n = len(Y); print("samples", n, "games", G.max() + 1, "bias cm", Y.mean(0).round(2), flush=True)
  F = {f"H{h}": X[:, -h:].reshape(n, -1) for h in HS}
  F.update({f"H50 {p}": X[:, -50:, s].reshape(n, -1) for p, s in PARTS.items() if p != "all"}); F["sofar"] = np.concatenate([X[:, -1], C], 1)
  out = probe(F, Y, G, T=T); out.update(samples=n, games=int(G.max() + 1), runs=sys.argv[2:]); json.dump(out, open(sys.argv[1], "w"), indent=1)
  for k, v in out.items(): print(f"{k:14s} {v}")
