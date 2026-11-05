from mjlab_microduck.train_hook import maybe_init_checkpoint, maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` submits to HF Jobs and exits here, before any
# of the cfg imports below: this module is what mjlab's plugin loader pulls
# in, and it is the only train path no install order can take from us (see
# train_hook.py). A no-op without the flag.
maybe_submit_to_hf_jobs()
# `--init-checkpoint <path | hf://bundle>`: resume/warm-start from a shared training bundle
# (checkpoints.py, docs/training_checkpoints.md). A no-op without the flag.
maybe_init_checkpoint()

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .backlash import make_backlash_variant

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand → robot_allcollisions_backlash.xml, StandUp → robot_groundcontact_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# groundcontact backlash robot — same model as their base task in each case.
_BL_GROUNDCONTACT = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BL_ALLCOLLISIONS = MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )

# ys personal: stage 4b head circle (STAGES-ys.md)
from .ys_head_circle_env_cfg import YsHeadCircleRlCfg, make_ys_head_circle_env_cfg

register_mjlab_task(
    task_id="Mjlab-YsHeadCircle-Flat-MicroDuck",
    env_cfg=make_ys_head_circle_env_cfg(),
    play_env_cfg=make_ys_head_circle_env_cfg(play=True),
    rl_cfg=YsHeadCircleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-YsHeadCircleTip-Flat-MicroDuck",
    env_cfg=make_ys_head_circle_env_cfg(tip_space=True),
    play_env_cfg=make_ys_head_circle_env_cfg(play=True, tip_space=True),
    rl_cfg=YsHeadCircleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-YsHeadCircleTipLoad-Flat-MicroDuck",
    env_cfg=make_ys_head_circle_env_cfg(tip_space=True, tip_load=True),
    play_env_cfg=make_ys_head_circle_env_cfg(play=True, tip_space=True, tip_load=True),
    rl_cfg=YsHeadCircleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-YsBodyCircle-Flat-MicroDuck",
    env_cfg=make_ys_head_circle_env_cfg(tip_space=True, tip_load=True, whole_body=True),
    play_env_cfg=make_ys_head_circle_env_cfg(play=True, tip_space=True, tip_load=True, whole_body=True),
    rl_cfg=YsHeadCircleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

from mjlab_microduck.tasks.ys_rope_env_cfg import YsRopeRlCfg, make_ys_rope_env_cfg  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-YsRope-Flat-MicroDuck",
    env_cfg=make_ys_rope_env_cfg(),
    play_env_cfg=make_ys_rope_env_cfg(play=True),
    rl_cfg=YsRopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

from mjlab_microduck.tasks.ys_rope_env_cfg import make_ys_rope_bob_env_cfg  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-YsRopeBob-Flat-MicroDuck",
    env_cfg=make_ys_rope_bob_env_cfg(),
    play_env_cfg=make_ys_rope_bob_env_cfg(play=True),
    rl_cfg=YsRopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-YsRopeLoop-Flat-MicroDuck",
    env_cfg=make_ys_rope_bob_env_cfg(loop=True),
    play_env_cfg=make_ys_rope_bob_env_cfg(play=True, loop=True),
    rl_cfg=YsRopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-YsRopeStay-Flat-MicroDuck",
    env_cfg=make_ys_rope_bob_env_cfg(stay=True),
    play_env_cfg=make_ys_rope_bob_env_cfg(play=True, stay=True),
    rl_cfg=YsRopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-YsRopeBoth-Flat-MicroDuck",
    env_cfg=make_ys_rope_bob_env_cfg(stay=True, reverse=True),
    play_env_cfg=make_ys_rope_bob_env_cfg(play=True, stay=True, reverse=True),
    rl_cfg=YsRopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

from mjlab_microduck.tasks.ys_rope_env_cfg import make_ys_rope_partner_env_cfg  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-YsRopePartner-Flat-MicroDuck",
    env_cfg=make_ys_rope_partner_env_cfg(),
    play_env_cfg=make_ys_rope_partner_env_cfg(play=True),
    rl_cfg=YsRopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

from mjlab_microduck.tasks.ys_rope_env_cfg import make_ys_rope_lock_env_cfg  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-YsRopeLock-Flat-MicroDuck",
    env_cfg=make_ys_rope_lock_env_cfg(),
    play_env_cfg=make_ys_rope_lock_env_cfg(play=True),
    rl_cfg=YsRopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

from mjlab_microduck.tasks.ys_rope_env_cfg import make_ys_rope_soft_env_cfg  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-YsRopeSoft-Flat-MicroDuck",
    env_cfg=make_ys_rope_soft_env_cfg(),
    play_env_cfg=make_ys_rope_soft_env_cfg(play=True),
    rl_cfg=YsRopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

from mjlab_microduck.tasks.ys_rope_env_cfg import make_ys_rope_slack_env_cfg  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-YsRopeSlack-Flat-MicroDuck",
    env_cfg=make_ys_rope_slack_env_cfg(),
    play_env_cfg=make_ys_rope_slack_env_cfg(play=True),
    rl_cfg=YsRopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

from mjlab_microduck.tasks.ys_rope_duo_env_cfg import make_ys_rope_duo_env_cfg  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-YsRopeDuo-Flat-MicroDuck",
    env_cfg=make_ys_rope_duo_env_cfg(),
    play_env_cfg=make_ys_rope_duo_env_cfg(play=True),
    rl_cfg=YsRopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

from mjlab_microduck.tasks.ys_rope_duo_env_cfg import YsSelfPlayRunner  # noqa: E402

register_mjlab_task(
    task_id="Mjlab-YsRopeSelf-Flat-MicroDuck",
    env_cfg=make_ys_rope_duo_env_cfg(),
    play_env_cfg=make_ys_rope_duo_env_cfg(play=True),
    rl_cfg=YsRopeRlCfg,
    runner_cls=YsSelfPlayRunner,
)

register_mjlab_task(
    task_id="Mjlab-YsRopeSelfNoisy-Flat-MicroDuck",
    env_cfg=make_ys_rope_duo_env_cfg(stochastic=True),
    play_env_cfg=make_ys_rope_duo_env_cfg(play=True, stochastic=True),
    rl_cfg=YsRopeRlCfg,
    runner_cls=YsSelfPlayRunner,
)

register_mjlab_task(
    task_id="Mjlab-YsRopeSelfBob-Flat-MicroDuck",
    env_cfg=make_ys_rope_duo_env_cfg(bob=2.0),
    play_env_cfg=make_ys_rope_duo_env_cfg(play=True, bob=2.0),
    rl_cfg=YsRopeRlCfg,
    runner_cls=YsSelfPlayRunner,
)

register_mjlab_task(
    task_id="Mjlab-YsRopeSelfShort-Flat-MicroDuck",
    env_cfg=make_ys_rope_duo_env_cfg(span=0.60),
    play_env_cfg=make_ys_rope_duo_env_cfg(play=True, span=0.60),
    rl_cfg=YsRopeRlCfg,
    runner_cls=YsSelfPlayRunner,
)

register_mjlab_task(
    task_id="Mjlab-YsRopeSelfBobShort-Flat-MicroDuck",
    env_cfg=make_ys_rope_duo_env_cfg(bob=2.0, span=0.60),
    play_env_cfg=make_ys_rope_duo_env_cfg(play=True, bob=2.0, span=0.60),
    rl_cfg=YsRopeRlCfg,
    runner_cls=YsSelfPlayRunner,
)

register_mjlab_task(  # 7.2 rope16: rope14 + handle-circle centre 5 cm lower (214 -> 164 mm)
    task_id="Mjlab-YsRope16-Flat-MicroDuck",
    env_cfg=make_ys_rope_duo_env_cfg(span=0.60, axis_z=0.164),
    play_env_cfg=make_ys_rope_duo_env_cfg(play=True, span=0.60, axis_z=0.164),
    rl_cfg=YsRopeRlCfg,
    runner_cls=YsSelfPlayRunner,
)

from .ys_jump_env_cfg import YsJumpRlCfg, make_ys_jump_env_cfg

register_mjlab_task(
    task_id="Mjlab-YsJump-Flat-MicroDuck",
    env_cfg=make_ys_jump_env_cfg(),
    play_env_cfg=make_ys_jump_env_cfg(play=True),
    rl_cfg=YsJumpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

from .ys_jump_env_cfg import YsSideJumpRlCfg, make_ys_side_jump_env_cfg

register_mjlab_task(
    task_id="Mjlab-YsSideJump-Flat-MicroDuck",
    env_cfg=make_ys_side_jump_env_cfg(),
    play_env_cfg=make_ys_side_jump_env_cfg(play=True),
    rl_cfg=YsSideJumpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

from .ys_jump_env_cfg import SIDE_OVERLAP_HALF_W, SIDE_SHORT_GAP

register_mjlab_task(
    task_id="Mjlab-YsSideJumpOverlap-Flat-MicroDuck",
    env_cfg=make_ys_side_jump_env_cfg(half_w=SIDE_OVERLAP_HALF_W, both=3.0),
    play_env_cfg=make_ys_side_jump_env_cfg(play=True, half_w=SIDE_OVERLAP_HALF_W, both=3.0),
    rl_cfg=YsSideJumpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-YsSideJumpGap-Flat-MicroDuck",
    env_cfg=make_ys_side_jump_env_cfg(gap=SIDE_SHORT_GAP),
    play_env_cfg=make_ys_side_jump_env_cfg(play=True, gap=SIDE_SHORT_GAP),
    rl_cfg=YsSideJumpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

from .ys_trio_env_cfg import YsTrioJumpRlCfg, make_ys_trio_jump_env_cfg

register_mjlab_task(
    task_id="Mjlab-YsTrioJump-Flat-MicroDuck",
    env_cfg=make_ys_trio_jump_env_cfg(),
    play_env_cfg=make_ys_trio_jump_env_cfg(play=True),
    rl_cfg=YsTrioJumpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-YsTrioJumpRand-Flat-MicroDuck",
    env_cfg=make_ys_trio_jump_env_cfg(delta_half=0.349),  # 20 deg
    play_env_cfg=make_ys_trio_jump_env_cfg(play=True, delta_half=0.349),
    rl_cfg=YsTrioJumpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
