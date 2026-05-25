# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg

from . import mdp
from .terrains import ROUGH_EVAL_CFG, ROUGH_TERRAINS_CFG, STAIRS_EVAL_CFG, WAVE_EVAL_CFG

_WORKSPACE_ROOT = os.path.abspath(os.path.dirname(__file__))
_USD_PATH = os.path.join(_WORKSPACE_ROOT, "custom_assets", "robot.usd")


def _to_sim_gains(real_kp: float, real_kd: float, kp_ratio: float, kd_ratio: float) -> tuple[float, float]:
    """Map real-world gain values to simulation gains."""
    return real_kp / kp_ratio, real_kd / kd_ratio


_GAIN_RATIOS = dict(kp_ratio=150.0, kd_ratio=16.0)
_WHEEL_GAINS_REAL = dict(kp_real=0.0, kd_real=2.928)
_LEG_GAINS_REAL = dict(kp_real=1500.0, kd_real=5.456)

_WHEEL_KP_SIM, _WHEEL_KD_SIM = _to_sim_gains(
    _WHEEL_GAINS_REAL["kp_real"],
    _WHEEL_GAINS_REAL["kd_real"],
    _GAIN_RATIOS["kp_ratio"],
    _GAIN_RATIOS["kd_ratio"],
)
_LEG_KP_SIM, _LEG_KD_SIM = _to_sim_gains(
    _LEG_GAINS_REAL["kp_real"],
    _LEG_GAINS_REAL["kd_real"],
    _GAIN_RATIOS["kp_ratio"],
    _GAIN_RATIOS["kd_ratio"],
)

AWM_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            "wheel_F_L": 0.0,
            "wheel_F_R": 0.0,
            "wheel_B_R": 0.0,
            "wheel_B_L": 0.0,
            "leg_F_L": 0.0,  # left closed: upper limit = 0 (flipped convention)
            "leg_F_R": 0.0,  # right closed: lower limit = 0
            "leg_B_L": 0.0,  # left closed: upper limit = 0 (flipped convention)
            "leg_B_R": 0.0,  # right closed: lower limit = 0
        },
    ),
    actuators={
        "wheel_drive": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*"],
            effort_limit_sim=1.94,
            velocity_limit_sim=7.6,
            stiffness=_WHEEL_KP_SIM,
            damping=_WHEEL_KD_SIM,
            armature=0.0022,
            friction=0.038,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["leg_.*"],
            effort_limit_sim=2.70,
            velocity_limit_sim=4.55,
            stiffness=_LEG_KP_SIM,
            damping=_LEG_KD_SIM,
            armature=0.0022,
            friction=0.015,
        ),
    },
)


@configclass
class AwmSceneCfg(InteractiveSceneCfg):
    """Configuration for the AWM scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="multiply",
            static_friction=2.0,
            dynamic_friction=2.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}"
                "/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = AWM_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*",
    #     history_length=3,
    #     track_air_time=True,
    #     track_pose=True,
    # )

    # Forward-looking terrain scanner: 7×5 grid offset 0.3 m ahead and 0.5 m
    # above the robot base, firing rays straight down.  attach_yaw_only keeps
    # the grid aligned with the robot heading so the scan always faces forward.
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body_assembly",
        offset=RayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 0.5)),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=(0.9, 0.6)),
        ray_alignment="yaw",
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    drive = mdp.AwmDriveActionCfg(
        asset_name="robot",
        wheel_joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"],
        leg_joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
        max_wheel_speed=8.0,
        leg_offset=0.0,
        use_auto_extension=False,
        closed_at_upper_limit=False,
        flip_wheel_joint_names=["wheel_F_R", "wheel_B_R"],
        flip_leg_joint_names=["leg_F_L", "leg_B_L"],
    )


@configclass
class CommandsCfg:
    """Velocity commands sampled each episode."""

    vel_cmd = mdp.UniformVelCommandCfg(
        resampling_time_range=(5.0, 10.0),
        vx_range=(0.2, 0.5),
        yaw_rate_range=(-0.8, 0.8),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Velocity commands from operator (what we're trying to track)
        commanded_velocity = ObsTerm(
            func=mdp.commanded_velocity,
            params={"command_name": "vel_cmd"},
        )
        # Robot velocity feedback
        # base_lin_vel_x: from ZED VIO or wheel odometry — moderate drift noise
        base_lin_vel_x = ObsTerm(func=mdp.base_lin_vel_x, noise=GaussianNoiseCfg(mean=0.0, std=0.05))
        # base_ang_vel_z: from ZED IMU gyroscope — small gyro noise
        base_ang_vel_z = ObsTerm(func=mdp.base_ang_vel_z, noise=GaussianNoiseCfg(mean=0.0, std=0.02))
        # Wheel state
        # wheel_velocities: Dynamixel encoders — accurate, minor noise
        wheel_velocities = ObsTerm(
            func=mdp.wheel_velocities,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"],
                ),
                "sign_mask": [1.0, -1.0, -1.0, 1.0],
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.1),
        )
        # Leg state
        # leg_positions: Dynamixel encoders — very accurate, minimal noise
        leg_positions = ObsTerm(
            func=mdp.leg_positions,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
                )
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )
        leg_actions = ObsTerm(func=mdp.leg_actions, params={"num_wheels": 4})
        # Terrain / stability feedback
        # projected_gravity: ZED IMU accelerometer — corrupted by motor/wheel vibrations
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=GaussianNoiseCfg(mean=0.0, std=0.05))
        progress_slip_hist = ObsTerm(
            func=mdp.progress_slip_history,
            params={
                "wheel_asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"],
                ),
                "wheel_radius": 0.0508,
                "ema_alpha": 0.1,
            },
        )
        # Forward terrain height map: 35 rays relative to robot base z.
        # Gives the policy direct look-ahead so it can extend legs proactively.
        # Gaussian noise (std=0.02m) matches ZED 2i depth noise for sim2real transfer.
        terrain_scan = ObsTerm(
            func=mdp.terrain_height_scan,
            params={"sensor_name": "ray_caster"},
            noise=GaussianNoiseCfg(mean=0.0, std=0.02),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for reset events."""

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.5, -0.5),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_wheels = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"],
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_legs = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Primary: track commanded velocity
    vel_tracking = RewTerm(
        func=mdp.velocity_tracking_reward,
        weight=2.0,
        params={"command_name": "vel_cmd", "std": 0.4},
    )
    yaw_tracking = RewTerm(
        func=mdp.yaw_rate_tracking_reward,
        weight=0.5,
        params={"command_name": "vel_cmd", "std": 0.4},
    )
    # Morphology adaptation: biphasic signal driven by forward terrain scan.
    # roughness_threshold=0.06 m: terrain height std of 6 cm → fully difficult.
    leg_extension_efficiency = RewTerm(
        func=mdp.leg_extension_efficiency,
        weight=-2.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
            ),
            "sensor_name": "ray_caster",
            "roughness_threshold": 0.04,
        },
    )
    rough_terrain_leg_bonus = RewTerm(
        func=mdp.rough_terrain_leg_bonus,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
            ),
            "sensor_name": "ray_caster",
            "roughness_threshold": 0.04,
        },
    )
    # Penalize extended legs when torques are low — teaches retraction on flat terrain.
    # Uses the same torque signal the ProprioTorque policy observes, so the policy
    # directly learns: torque drops → retract legs.
    torque_gated_retraction = RewTerm(
        func=mdp.torque_gated_retraction,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
            ),
            "sensor_name": "ray_caster",
            "roughness_threshold": 0.04,
            "torque_threshold": 0.3,
        },
    )
    # Penalize being stuck (low velocity tracking) on rough terrain with legs retracted.
    # Forces proactive leg extension before/during obstacles; scan-based so no feedback loops.
    stuck_with_retracted_legs = RewTerm(
        func=mdp.stuck_with_retracted_legs,
        weight=-1.5,
        params={
            "command_name": "vel_cmd",
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
            ),
            "sensor_name": "ray_caster",
            "roughness_threshold": 0.04,
        },
    )
    # Penalize tilt when legs are retracted: rewards keeping body parallel to ground.
    # tilt_threshold=1.0 → full penalty at ~5.8° tilt. Sim2real: IMU + joint encoders.
    body_tilt_with_retracted_legs = RewTerm(
        func=mdp.body_tilt_with_retracted_legs,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
            ),
            "tilt_threshold": 1.0,
        },
    )
    # Stability penalties
    wheel_slip = RewTerm(
        func=mdp.wheel_slip_penalty,
        weight=-1.5,
        params={
            "wheel_radius": 0.0508,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["wheel_F_L", "wheel_F_R", "wheel_B_R", "wheel_B_L"],
            ),
        },
    )
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_.*", "leg_.*"])},
    )
    action = RewTerm(func=mdp.action_l2, weight=-0.0005)


@configclass
class CurriculumCfg:
    """Terrain curriculum based on velocity tracking success."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel_tracking,
        params={"command_name": "vel_cmd", "promote_threshold": 0.7, "demote_threshold": 0.3},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    high_velocity = DoneTerm(func=mdp.high_base_velocity)


@configclass
class AwmEnvCfg(ManagerBasedRLEnvCfg):
    scene: AwmSceneCfg = AwmSceneCfg(num_envs=1024, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 20.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material


@configclass
class AwmWheelsOnlyCfg(AwmEnvCfg):
    """Ablation baseline: legs locked closed, wheels only.

    leg_offset=-0.5 guarantees desired_extension=0 regardless of policy output:
      policy_extension = 0.5 * leg_cmd + (-0.5), range [-1.0, 0.0]
      clamp([-1.0, 0.0], 0, 1) = 0.0 always → legs never leave closed position.
    Morphology reward terms are zeroed since they are meaningless without leg control.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        # leg_offset=-0.5 mathematically clamps desired_extension to 0 for any policy output
        self.actions.drive.leg_offset = -0.5
        # Zero out morphology rewards — not applicable for wheels-only baseline
        self.rewards.leg_extension_efficiency.weight = 0.0
        self.rewards.rough_terrain_leg_bonus.weight = 0.0
        self.rewards.stuck_with_retracted_legs.weight = 0.0
        self.rewards.body_tilt_with_retracted_legs.weight = 0.0
        self.rewards.torque_gated_retraction.weight = 0.0


@configclass
class AwmStairsEvalCfg(AwmEnvCfg):
    """Evaluation config: stairs-only terrain, small grid for easy observation.

    Use with num_envs=4 so all robots are visible in one view.
    All 10 terrain levels span step heights 0.02–0.15 m.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = STAIRS_EVAL_CFG
        # Start on easiest row (row 0, ~0.02m steps); curriculum advances to harder rows on success
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class AwmProprioOnlyCfg(AwmEnvCfg):
    """Ablation: proprioception-only — terrain_scan removed from observations.

    The ray_caster sensor stays in the scene so morphology rewards still work.
    Only the 35-dim terrain_scan observation is removed, so the policy must rely
    purely on proprioceptive signals (tilt, slip, velocity error) to decide when
    to extend legs. Compares against adaptive to quantify the value of look-ahead.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.observations.policy.terrain_scan = None


@configclass
class AwmProprioTorqueCfg(AwmProprioOnlyCfg):
    """Proprioception + leg torques — no terrain_scan.

    Same as ProprioOnly but keeps leg_torques in observations.
    Policy must use torque feedback (active morphological probing) to infer terrain.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        # terrain_scan already removed by AwmProprioOnlyCfg.__post_init__
        # Add leg_torques as the only terrain-inferring signal.
        # leg_extension_efficiency (scan-gated) is kept as privileged training signal —
        # scan in rewards is fine; policy learns retraction cue from observed torques.
        self.observations.policy.leg_torques = ObsTerm(
            func=mdp.leg_torques,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["leg_F_L", "leg_F_R", "leg_B_L", "leg_B_R"],
                )
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.05),  # real sensor noise ~0.02 normalized; 0.05 kept for sim2real robustness
        )


@configclass
class AwmProprioTorqueStairsEvalCfg(AwmProprioTorqueCfg):
    """Stairs-only terrain with ProprioTorque observations — for evaluating GRU/LSTM checkpoints."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = STAIRS_EVAL_CFG
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class AwmLegsOpenCfg(AwmEnvCfg):
    """Ablation baseline: legs locked fully open, wheels only.

    leg_offset=1.5 guarantees desired_extension=1 regardless of policy output:
      policy_extension = 0.5 * leg_cmd + 1.5, range [1.0, 2.0]
      clamp([1.0, 2.0], 0, 1) = 1.0 always → legs always fully extended.
    Morphology reward terms are zeroed since leg state is fixed.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        # leg_offset=1.5 mathematically clamps desired_extension to 1 for any policy output
        self.actions.drive.leg_offset = 1.5
        # Zero out morphology rewards — leg state is fixed, these are meaningless
        self.rewards.leg_extension_efficiency.weight = 0.0
        self.rewards.rough_terrain_leg_bonus.weight = 0.0
        self.rewards.stuck_with_retracted_legs.weight = 0.0
        self.rewards.body_tilt_with_retracted_legs.weight = 0.0
        self.rewards.torque_gated_retraction.weight = 0.0


# ---------------------------------------------------------------------------
# Unseen-terrain evaluation configs — three terrains × three obs conditions
# ---------------------------------------------------------------------------

@configclass
class AwmProprioOnlyStairsEvalCfg(AwmProprioOnlyCfg):
    """Stairs eval for the proprioception-only baseline (no camera, no torque)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = STAIRS_EVAL_CFG
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class AwmWaveEvalCfg(AwmEnvCfg):
    """Unseen terrain eval: pure wave terrain, camera (+terrain_scan) condition."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = WAVE_EVAL_CFG
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class AwmProprioOnlyWaveEvalCfg(AwmProprioOnlyCfg):
    """Unseen terrain eval: pure wave terrain, proprioception-only condition."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = WAVE_EVAL_CFG
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class AwmProprioTorqueWaveEvalCfg(AwmProprioTorqueCfg):
    """Unseen terrain eval: pure wave terrain, proprioception+torque condition (ours)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = WAVE_EVAL_CFG
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class AwmRoughEvalCfg(AwmEnvCfg):
    """Unseen terrain eval: dense random rough terrain, camera (+terrain_scan) condition."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_EVAL_CFG
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class AwmProprioOnlyRoughEvalCfg(AwmProprioOnlyCfg):
    """Unseen terrain eval: dense random rough terrain, proprioception-only condition."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_EVAL_CFG
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class AwmProprioTorqueRoughEvalCfg(AwmProprioTorqueCfg):
    """Unseen terrain eval: dense random rough terrain, proprioception+torque condition (ours)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_EVAL_CFG
        self.scene.terrain.max_init_terrain_level = 0
