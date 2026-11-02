"""Scenario eval: what training metrics don't show (ys personal tool).

Runs fixed-command scenarios on the GPU env (same physics/DR for every policy) and measures
what the commands asked for: straight-walk yaw drift, turn-in-place rate, standing still, falls.
Pushes and curricula are off; DR stays on. Accepts rsl_rl .pt checkpoints and exported .onnx.

  uv run scripts/ys_eval.py Mjlab-Velocity-Flat-MicroDuck ../policies/alpha_walking.onnx logs/.../model_2499.pt
"""

import argparse
from dataclasses import asdict

import numpy as np
import onnxruntime as ort
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

# name, (vx, vy, wz) command, seconds
SCENARIOS = [
  ("straight 0.3", (0.3, 0.0, 0.0), 10.0),
  ("turn +1.0", (0.0, 0.0, 1.0), 6.0),
  ("turn -1.0", (0.0, 0.0, -1.0), 6.0),
  ("stand", (0.0, 0.0, 0.0), 6.0),
]
WARMUP_S = 1.0  # skip the spawn transient


def make_env(task, cmd, n, drop=()):
  cfg = load_env_cfg(task, play=True)
  for k in drop:  # DR ablation: event names, or "imu" for the IMU mounting-error obs
    if k == "imu":
      for t in ("base_ang_vel", "projected_gravity"):
        if "max_angle_deg" in (cfg.observations["actor"].terms[t].params or {}):
          cfg.observations["actor"].terms[t].params["max_angle_deg"] = 0.0
    else:
      cfg.events.pop(k)
  cfg.scene.num_envs = n
  cfg.seed = 0
  fix_command(cfg, cmd)
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  return env, RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task).clip_actions)


def fix_command(cfg, cmd):
  """Hold one (vx, vy, wz) for the whole episode; no pushes, no curricula, neutral head/body."""
  cfg.curriculum = {}
  cfg.events.pop("push_robot", None)
  t = cfg.commands["twist"]
  t.ranges.lin_vel_x, t.ranges.lin_vel_y, t.ranges.ang_vel_z = ((c, c) for c in cmd)
  t.heading_command, t.ranges.heading = False, None
  t.rel_standing_envs = 0.0
  t.resampling_time_range = (1e9, 1e9)
  for k in ("rel_heading_envs", "rel_turn_in_place_envs", "rel_forward_envs", "rel_world_envs"):
    if hasattr(t, k):
      setattr(t, k, 0.0)
  for k in ("head_pose", "body_pose"):
    cfg.commands[k].ranges = tuple((0.0, 0.0) for _ in cfg.commands[k].ranges)


def make_policy(task, path, venv):
  if path.endswith(".onnx"):
    sess = ort.InferenceSession(path)
    def run(obs):
      x = obs["actor"].cpu().numpy().astype(np.float32)
      a = np.concatenate([sess.run(None, {"obs": x[i : i + 1]})[0] for i in range(len(x))])
      return torch.from_numpy(a).to("cuda:0")
    return run
  agent = load_rl_cfg(task)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(venv, asdict(agent), device="cuda:0")
  runner.load(path, load_cfg={"actor": True}, strict=True, map_location="cuda:0")
  return runner.get_inference_policy(device="cuda:0")


def run_scenario(task, path, cmd, seconds, n, drop=()):
  env, venv = make_env(task, cmd, n, drop)
  policy = make_policy(task, path, venv)
  robot = env.scene["robot"]
  steps, warm = int(seconds / env.step_dt), int(WARMUP_S / env.step_dt)
  v, falls = [], 0
  obs = venv.get_observations()
  with torch.inference_mode():
    for i in range(steps):
      obs, _, dones, extras = venv.step(policy(obs))
      falls += int((dones.bool() & ~extras["time_outs"].bool()).sum())
      if i >= warm:
        lin, ang = robot.data.root_link_lin_vel_b, robot.data.root_link_ang_vel_b
        v.append(torch.stack([lin[:, 0], lin[:, 1], ang[:, 2]], 1).cpu())
  env.close()
  per_robot = torch.stack(v).mean(0).numpy()  # (n, 3): each robot's own average
  # |wz|r: per-robot yaw rate, then |.| over robots. The signed mean alone hides robots
  # curving left and right that cancel out.
  return per_robot.mean(0), np.abs(per_robot[:, 2]).mean(), 100.0 * falls / n


def main():
  p = argparse.ArgumentParser()
  p.add_argument("task")
  p.add_argument("policies", nargs="+")
  p.add_argument("--envs", type=int, default=64)
  p.add_argument("--drop", nargs="*", default=[], help='DR to switch off: event names and/or "imu"')
  p.add_argument("--only", help="run only scenarios whose name starts with this")
  a = p.parse_args()
  rows = []
  for path in a.policies:
    for name, cmd, sec in SCENARIOS:
      if a.only and not name.startswith(a.only):
        continue
      (vx, vy, wz), awz, fall = run_scenario(a.task, path, cmd, sec, a.envs, a.drop)
      rows.append((path.split("/")[-2] + "/" + path.split("/")[-1], name, cmd, vx, vy, wz, awz, fall))
  print(f"\n{'policy':48s} {'scenario':13s} {'cmd vx,vy,wz':16s} {'vx':>6s} {'vy':>6s} {'wz':>6s} {'|wz|r':>6s} {'fall%':>6s}")
  for pol, name, cmd, vx, vy, wz, awz, fall in rows:
    print(f"{pol[-48:]:48s} {name:13s} {str(cmd):16s} {vx:6.3f} {vy:6.3f} {wz:6.3f} {awz:6.3f} {fall:6.1f}")


if __name__ == "__main__":
  main()
