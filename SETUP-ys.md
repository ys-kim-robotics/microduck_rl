# 내 로컬 셋업 기록 (ys-kim-robotics)

업스트림(pollen-robotics/microduck_rl)의 포크. 이 파일만 내가 추가한 것.

## 폴더 구조

```
~/YS/personal/microduck/
├── microduck_rl/   ← 이 repo (내 포크). RL 학습 환경
└── runtime/        ← pollen-robotics/microduck 클론. 로봇 온보드 Rust 런타임
                      학습된 ONNX 정책 9개가 runtime/policies/ 에 들어있음
```

`runtime/`은 참고용이라 git 추적 안 함. 다시 받으려면:

```bash
git clone https://github.com/pollen-robotics/microduck.git ~/YS/personal/microduck/runtime
```

## 환경

RTX 5080 (sm_120) / CUDA 12.8 / torch 2.9.1+cu128 / Python 3.12 (uv가 자동 설치).
`uv sync` 한 번이면 끝. `.venv`는 7.9GB.

주의: 프로젝트 폴더를 다른 경로로 옮기면 venv 실행 스크립트의 절대경로가 깨진다.
그때는 `uv sync --reinstall`.

## 검증된 명령

```bash
cd ~/YS/personal/microduck/microduck_rl

# 태스크 목록 (45개)
uv run list-envs

# 테스트 (154 passed)
uv run --with pytest pytest tests/ -q

# 학습된 정책을 CPU MuJoCo로 구경 (GPU 불필요, 학습 안 해도 됨)
uv run scripts/infer_policy.py \
    --walking ../runtime/policies/alpha_walking.onnx \
    --standing ../runtime/policies/alpha_stand.onnx \
    --new-cmd-obs

# 학습 스모크 테스트 (wandb 로그인 없이)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 512 \
    --agent.max-iterations 3 --agent.logger tensorboard --agent.run-name smoketest

# 본 학습 (512 env 기준 0.65 s/iter. VRAM 16GB라 4096에서 OOM 나면 2048로)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

기본 로거가 wandb라 `wandb login` 안 했으면 `--agent.logger tensorboard` 필요.

## git remote

| 이름 | 위치 | 용도 |
|---|---|---|
| `origin` | `git@github-personal:ys-kim-robotics/microduck_rl.git` | 내 포크. push |
| `upstream` | `https://github.com/pollen-robotics/microduck_rl.git` | 원본. 업데이트 당길 때만 |

업스트림 업데이트 받기: `git fetch upstream && git merge upstream/develop`
