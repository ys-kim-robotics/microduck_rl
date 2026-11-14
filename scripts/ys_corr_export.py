"""ys_corr_export.py <model.pt> <out.bin> [<diag.npz> [games]]: the learned position correction (7-27, DEPLOY 1-8; ys_fullset.py's CORR) as the
file the runtime's `odometry::Correction` loads (runtime-ys/odometry/src/correction.rs: "YSCORR01", then little-endian u32 window, observation
length, layer count; f32 means, standard deviations, 2 output means; per layer u32 outputs, inputs, f32 weights by output row, biases).

With a recording (a ys_fullset.py stats run with the same CORR, DIAG=<diag.npz> and OBSLOG=1): replays its first games (default 8) through the
runtime code (scripts/ys_odo_replay's corr_replay; build: `cargo build --release` there) and asserts it gives the correction the twin used.
"""
import struct, subprocess, sys
from pathlib import Path
import numpy as np, torch

m = torch.load(sys.argv[1], weights_only=False); H = int(m["H"]); mu, sd, ym = (np.asarray(m[k], dtype="<f4") for k in ("mu", "sd", "ym"))
W = [(v.cpu().numpy().astype("<f4"), m["net"][k.replace("weight", "bias")].cpu().numpy().astype("<f4")) for k, v in m["net"].items() if k.endswith("weight")]
assert len(mu) % H == 0 and W[0][0].shape[1] == len(mu) + 2 and W[-1][0].shape[0] == 2 and m.get("frm") == "go", "not a from-set-off correction model"
out = b"YSCORR01" + struct.pack("<3I", H, len(mu) // H, len(W)) + mu.tobytes() + sd.tobytes() + ym.tobytes()
for w, b in W: out += struct.pack("<2I", *w.shape) + w.tobytes() + b.tobytes()
Path(sys.argv[2]).write_bytes(out); print(f"{sys.argv[2]}: {len(out)} bytes, window {H} x {len(mu) // H}, layers {[w.shape for w, _ in W]}")

if len(sys.argv) > 3:
  d = np.load(sys.argv[3]); exe = Path(__file__).parent / "ys_odo_replay/target/release/corr_replay"; worst = 0.0; n = 0
  for g in np.flatnonzero(d["go"] >= 0)[: int(sys.argv[4]) if len(sys.argv) > 4 else 8]:
    k = np.arange(len(d["ob"])); act = (k > d["go"][g]) & ((d["call"][g] < 0) | (k <= d["call"][g]))  # ys_fullset.py: active from the tick after the set-off
    # is marked to the tick the end call is made
    txt = "\n".join(f"{int(a)} " + " ".join(map(repr, o.tolist())) for a, o in zip(act, d["ob"][:, g].astype(np.float32)))
    got = np.loadtxt(subprocess.run([exe, sys.argv[2]], input=txt, capture_output=True, text=True, check=True).stdout.splitlines()); ref = d["tr"][:, g, 15:17]
    e = np.abs(got - ref).max(); worst = max(worst, e); n += 1
    print(f"  game {g}: {int(act.sum())} active ticks, correction up to {100 * np.abs(ref).max():.2f} cm, runtime vs twin differ by at most {1e6 * e:.2f} um")
  assert n and worst < 1e-5, f"the runtime code and the twin disagree by {worst} m"
  print(f"OK: {n} games, largest difference {1e6 * worst:.2f} um (limit 10)")
