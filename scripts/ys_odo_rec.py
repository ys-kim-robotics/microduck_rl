"""ys_odo_rec.py <ckpt> <N> <out.npz> (7-13 check 1, user 2026-11-09): the full game (scripts/ys_fullset.py setup; CLIP, SEED, PUSH from
env) recording, for J from its start call to the end, what the runtime's odometry gets every 20 ms tick (the 14 servo angles
as the encoders read them, encoder bias included, in duck_ipc_proto JOINT_NAMES order with the mouth 0; the trunk quaternion
w x y z) plus the true trunk position and the game marks. Replayed through runtime/odometry by scripts/ys_odo_replay."""
import sys, os
FS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ys_fullset.py")
CK, NN, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [FS, "x", CK, NN]
exec(compile(open(FS).read().split("# ---------------- video")[0], FS, "exec"))
import numpy as np
JN = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch", "head_yaw", "head_roll",
      "mouth", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]  # duck_ipc_proto::JOINT_NAMES
COL = torch.tensor([rob.joint_names.index(n) if n != "mouth" else 0 for n in JN], device=dev); MOUTH = JN.index("mouth")
REC = {k: [] for k in ("enc", "true", "quat", "pos", "on")}


def on_step(k, M):
  alive = M["dead"] < 0; st, call = M["start"], M["call"]
  on = alive & (st >= 0) & (k <= torch.where(call >= 0, call + KSTOP + int(1.0 / dt), 10 ** 9))
  for nm, x in (("enc", rob.data.joint_pos_biased), ("true", rob.data.joint_pos)):
    j = x[:, COL].clone(); j[:, MOUTH] = 0.0; REC[nm].append(j.cpu().numpy())
  REC["quat"].append(rob.data.root_link_quat_w.cpu().numpy()); REC["pos"].append(rob.data.root_link_pos_w.cpu().numpy() - env.scene.env_origins.cpu().numpy())
  REC["on"].append(on.cpu().numpy())
  return bool((~alive | ((call >= 0) & (k > call + KSTOP + int(AFTER / dt)))).all())


with torch.inference_mode(): M = episode(on_step)
R = records(M)
mk = {k: M[k].cpu().numpy() for k in ("start", "go", "arr", "call", "hit")}
np.savez(OUT, **{k: np.stack(v) for k, v in REC.items()}, dt=dt, **{"m_" + k: v for k, v in mk.items()},
         streak=np.array([r.get("streak", -1) if r["kind"] == "game" else -1 for r in R]), kind=np.array([r["kind"] for r in R]),
         spot=env._ys_spot_rel.cpu().numpy())
print("DONE", OUT, "steps", len(REC["on"]), "summary", json.dumps(summary(R)), flush=True)
