# 내 로컬 셋업 기록 (ys-kim-robotics)

업스트림(pollen-robotics/microduck_rl)의 포크. 이 파일만 내가 추가한 것.

## 폴더 구조

폴더 하나에 repo 둘. 둘 다 내 포크. 성격이 달라서(Python/GPU/학습 vs Rust/로봇/배포)
합치지 않고 나란히 둔다 — 합치면 각각의 업스트림 업데이트를 못 받는다.

```
~/YS/personal/microduck/          ← git repo 아님. 그냥 작업 폴더
├── microduck_rl/                 ← 내 포크. RL 학습 환경 (Python, GPU)
│                                    기본 브랜치: develop
└── runtime/                      ← 내 포크. 로봇 온보드 런타임 (Rust)
                                     폴더명은 runtime, repo명은 microduck
                                     기본 브랜치: main
                                     학습된 ONNX 정책 9개가 policies/ 에 있음
```

폴더명과 repo명이 달라도 상관없다. git은 폴더 이름을 안 보고 `origin` 주소만 본다.

다시 세팅하려면:

```bash
mkdir -p ~/YS/personal/microduck && cd ~/YS/personal/microduck
git clone git@github-personal:ys-kim-robotics/microduck_rl.git microduck_rl
git clone git@github-personal:ys-kim-robotics/microduck.git    runtime
# 각각 업스트림 추가
git -C microduck_rl remote add upstream https://github.com/pollen-robotics/microduck_rl.git
git -C runtime      remote add upstream https://github.com/pollen-robotics/microduck.git
cd microduck_rl && uv sync
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

# 학습된 정책 구경 (GPU 불필요) — 아래 "정책 구경하기" 섹션 참고
uv run scripts/infer_policy.py --walking ../runtime/policies/alpha_walking.onnx --new-cmd-obs

# 학습 스모크 테스트 (wandb 로그인 없이)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 512 \
    --agent.max-iterations 3 --agent.logger tensorboard --agent.run-name smoketest

# 본 학습 (512 env 기준 0.65 s/iter. VRAM 16GB라 4096에서 OOM 나면 2048로)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
```

기본 로거가 wandb라 `wandb login` 안 했으면 `--agent.logger tensorboard` 필요.

## 정책 구경하기 (infer_policy.py)

학습 안 해도 됨. `runtime/policies/`의 학습 완료 ONNX를 CPU MuJoCo로 돌린다.
실물 로봇의 정책 교체 방식(상황에 따라 다른 신경망을 바꿔 끼우는 것)을 그대로 흉내 낸다.

### 전체 모드 (모든 기능 활성화)

```bash
cd ~/YS/personal/microduck/microduck_rl
P=../runtime/policies
uv run scripts/infer_policy.py \
    --walking $P/alpha_walking.onnx \
    --standing $P/alpha_stand.onnx \
    --sitstand $P/alpha_sitstand.onnx \
    --ground-pick $P/alpha_ground_pick.onnx \
    --roulade $P/roulade.onnx \
    --kick-left $P/ball_kick_left.onnx \
    --kick-right $P/ball_kick_right.onnx \
    --new-cmd-obs
```

**정책 하나가 모든 걸 하는 게 아니다.** 걷기/서기/앉기/킥/구르기가 전부 따로 훈련된
별개 신경망이라, 넘겨주지 않은 정책의 기능은 아예 존재하지 않는다. 키를 눌러도
`kick_left unavailable: no --kick-left policy loaded` 같은 메시지만 나온다.

`--kick-left/right`를 주면 공이 있는 씬으로 바뀐다.
`--new-cmd-obs`는 61차원 관측 포맷 지정 — 이 정책들은 전부 61차원이라 필수.

### 조작법

**키는 MuJoCo 창이 아니라 명령을 입력한 터미널에 친다.** 창을 클릭하고 누르면 무반응.

| 키 | 동작 | 필요 옵션 |
|---|---|---|
| `↑` / `↓` | 전진 속도 증가 / 감소 (음수면 후진) | |
| `←` / `→` | 옆으로 이동 (strafe) | |
| `A` / `E` | 좌회전 / 우회전 | |
| `SPACE` | 정지 (모든 명령 0) | |
| `T` | 정책 on/off (끄면 마지막 자세 유지) | |
| `P` | 랜덤 방향 밀치기 (1.0 m/s) — 외란 버티기 테스트 | |
| `Q` | 종료 | |
| `G` | 바닥 물건 집기 | `--ground-pick` |
| `Y` | 앉기 ↔ 서기 | `--sitstand` |
| `K` / `L` | 왼발 / 오른발 킥 | `--kick-left` / `--kick-right` |
| `R` | 앞구르기 | `--roulade` |
| `B` | 몸통 자세 모드 토글 | |
| `H` | 머리 모드 토글 | |

`B`(몸통) / `H`(머리) 모드를 켜면 같은 방향키가 다른 의미가 된다:

| | `↑`/`↓` | `←`/`→` | `A`/`E` | `Z`/`S` |
|---|---|---|---|---|
| 몸통 모드 | 높이 ±10mm | 피치 ±10° | 롤 ±10° | 요 ±10° |
| 머리 모드 | head_pitch | head_yaw | head_roll | neck_pitch |

몸통 자세 제어는 **서 있는 정책(`alpha_stand`)의 기능**이라, 걷는 중에는 반응이 거의 없다.
`SPACE`로 세운 뒤 눌러야 한다.

### 넘어진 뒤 일어나기

**전용 버튼이 없다. 속도 명령을 주면 된다.**

`infer_policy.py`는 속도 명령 크기로 정책을 자동 교체한다 (`scripts/infer_policy.py:448`):

| 속도 명령 | 활성 정책 | 넘어졌을 때 |
|---|---|---|
| ≤ 0.05 (정지) | `alpha_stand` — 서 있기 + 자세 제어 | 못 일어남 (배운 적 없음) |
| > 0.05 | `alpha_walking` — 실제로는 **VelStand** = 걷기 + 넘어짐 복구 | 일어남 |

즉 `P`로 넘어뜨린 뒤 **`↑`를 누르면** walking 정책으로 바뀌면서 일어난다.
정지 상태에서도 항상 복구되게 하려면 `--standing`을 빼고 실행한다.

문턱값은 `--switch-threshold`로 조정 가능.

> RL 포인트: "일어나기"를 코딩한 사람은 없다. VelStand 훈련 때 에피소드 시작 자세에
> 엎어진/뒤집힌 상태를 섞고 보상은 "명령한 속도로 이동해라"만 줬을 뿐이다. 넘어진 채로는
> 속도를 못 내니 보상을 받으려면 일어날 수밖에 없었던 것. 동작을 지시하는 게 아니라
> 그 동작이 나올 수밖에 없는 상황과 보상을 만드는 것 — 이게 RL 설계의 핵심.

전용 일어서기 태스크(`Mjlab-StandUp-*`)도 있지만 학습된 ONNX가 없다. 직접 훈련해볼 첫 후보.

## git remote

두 repo 모두 같은 규칙: `origin` = 내 포크(push), `upstream` = 원본(읽기만).

| repo | origin (내 포크) | upstream (원본) | 기본 브랜치 |
|---|---|---|---|
| `microduck_rl/` | `github-personal:ys-kim-robotics/microduck_rl.git` | `pollen-robotics/microduck_rl` | `develop` |
| `runtime/` | `github-personal:ys-kim-robotics/microduck.git` | `pollen-robotics/microduck` | `main` |

업스트림 업데이트 받기 (브랜치 이름이 서로 다르니 주의):

```bash
git -C microduck_rl fetch upstream && git -C microduck_rl merge upstream/develop
git -C runtime      fetch upstream && git -C runtime      merge upstream/main
```

## 학습 결과를 실물 로봇에 넣기 (아직 미검증 — 로봇 없이 문서만 보고 정리)

runtime repo를 고치거나 빌드할 필요 없음. 로봇에서 돌고 있는 `robotd`가 설정 파일에
적힌 경로의 ONNX를 읽어가는 구조라, 파일 복사 + 설정 한 줄이면 끝.

```
내 PC:  uv run train ...                 → 체크포인트 .pt
        uv run scripts/export.py ...     → my_walk.onnx (약 800KB)
          ↓ scp
로봇:   /home/radxa/my_walk.onnx 에 두고
        robotd.toml 에  [policy] walk = "/home/radxa/my_walk.onnx"
        sudo systemctl restart robotd
```

주의:
- 반드시 `scripts/export.py`로 뽑을 것. 이 스크립트가 관측값 정규화를 ONNX 그래프에
  구워 넣는다. 체크포인트를 직접 변환하면 정책이 정규화 안 된 값을 보고 이상하게 움직인다.
- 관측 61차원이어야 한다. 아니면 로딩 단계에서 거부하고 로그에 이유를 찍는다
  (`observation width is 51, expected 61`).
- 로봇 소프트웨어 자체를 고치는 경우(새 동작을 패드 버튼에 붙이기 등)에만 runtime을
  건드리고 Rust 크로스컴파일 + 배포가 따라온다.

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
