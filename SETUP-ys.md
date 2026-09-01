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

## 계정 전환 세팅 (컴퓨터 전체 설정)

회사 계정(`ys-kim92` / `ys.kim@roai.im`)과 개인 계정(`ys-kim-robotics`)을 한 컴퓨터에서
같이 쓰기 위한 설정. **이 repo가 아니라 홈 디렉토리(`~`)에 있음.** 컴퓨터를 새로 세팅하면
아래를 다시 만들어야 함.

### 원리: 바꿔야 할 게 두 개

| | 무엇이 결정하나 | 자동? |
|---|---|---|
| 커밋 저자 (커밋에 박히는 이름/이메일) | **폴더 위치** | 자동 |
| push 인증 (어느 계정으로 접속하나) | **remote URL** | 수동 (클론할 때 선택) |

### 파일 목록

| 파일 | 하는 일 |
|---|---|
| `~/.ssh/id_ed25519` | 회사 키 (기존) |
| `~/.ssh/id_ed25519_personal` | 개인 키 (공개키는 개인 GitHub에 등록됨) |
| `~/.ssh/config` | `github-personal` 별명 정의 |
| `~/.gitconfig` | 기본 저자 = 회사 + 폴더별 전환 규칙 |
| `~/.gitconfig-personal` | 개인 저자 정보 (`~/YS/personal/` 아래에서만 적용) |

### `~/.ssh/config`

```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host github-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_personal
    IdentitiesOnly yes
```

`github-personal`은 서버가 아니라 **별명**. HostName은 둘 다 github.com이고 어느 키를
쓸지만 갈린다. `IdentitiesOnly yes`가 핵심 — 없으면 ssh가 키를 순서대로 다 던져보고
먼저 통과한 회사 키로 인증돼버린다.

### `~/.gitconfig` 에 추가된 부분

```
[includeIf "gitdir:~/YS/personal/"]
    path = /home/roai22/.gitconfig-personal
```

경로 끝의 `/` 필수 (없으면 하위 폴더에 적용 안 됨).

### `~/.gitconfig-personal`

```
[user]
	name = ys-kim-robotics
	email = 311407181+ys-kim-robotics@users.noreply.github.com
```

이메일은 GitHub가 계정마다 자동으로 주는 비공개 주소.
형식은 `<계정숫자ID>+<사용자명>@users.noreply.github.com`.
계정 ID는 `curl -s https://api.github.com/users/<사용자명> | grep '"id"'` 로 확인.
이걸 쓰면 잔디는 심기고 실제 이메일은 공개 저장소에 안 남는다.

### 사용법

```bash
# 개인 repo 클론 — github.com 대신 github-personal
git clone git@github-personal:ys-kim-robotics/repo.git ~/YS/personal/repo

# 회사는 하던 대로
git clone git@github.com:회사조직/repo.git ~/work/repo

# 이미 있는 repo를 개인 계정으로 돌리기
git remote set-url origin git@github-personal:ys-kim-robotics/repo.git

# 헷갈릴 때 확인 (개인 이메일 + github-personal 주소면 정상)
git config user.email && git remote -v

# 키가 어느 계정에 붙는지 확인
ssh -T git@github.com        # Hi ys-kim92!
ssh -T git@github-personal   # Hi ys-kim-robotics!
```

### 지켜야 할 규칙 하나

**회사 repo는 `~/YS/personal/` 밖에 둔다.**

안에 두면 인증은 회사 키로 통과하는데 저자만 개인 이름으로 박혀서, 회사 repo에
개인 이름 커밋이 조용히 올라간다. 반대 실수(개인 repo를 회사 주소로 클론)는
push가 `Permission denied`로 시끄럽게 실패하니 그때 `git remote set-url`로 고치면 된다.

### 주의: gh CLI

설치된 `gh`가 2.4.0(2022년)이라 다계정 기능(`gh auth switch`)이 없고 회사 계정에
묶여 있다. **`gh repo create` 하면 회사 계정 밑에 생긴다.** 개인 repo 생성은 웹에서
직접. clone/push/pull은 gh 없이 SSH로 다 되므로 급하지 않음.
