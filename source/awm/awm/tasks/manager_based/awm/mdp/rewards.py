# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# # ── Unused rewards (goal-reaching, superseded by vel_tracking) ──────────────
# def _goal_distance_xy(env, goal_distance: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]
#     root_xy = asset.data.root_pos_w[:, :2]
#     goal_xy = env.scene.env_origins[:, :2] + torch.tensor([goal_distance, 0.0], device=env.device)
#     dist = torch.norm(goal_xy - root_xy, dim=-1)
#     return torch.nan_to_num(dist, nan=float(goal_distance), posinf=float(goal_distance))
#
# class progress_to_goal(ManagerTermBase):
#     def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
#         super().__init__(cfg, env)
#         self.prev_dist = torch.zeros(env.num_envs, device=env.device)
#     def reset(self, env_ids: torch.Tensor):
#         asset_cfg = self.cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
#         goal_distance = self.cfg.params["goal_distance"]
#         dist = _goal_distance_xy(self._env, goal_distance, asset_cfg)
#         self.prev_dist[env_ids] = dist[env_ids].clamp(max=20.0)
#     def __call__(self, env, goal_distance, asset_cfg=SceneEntityCfg("robot")):
#         dist = _goal_distance_xy(env, goal_distance, asset_cfg).clamp(max=20.0)
#         progress = self.prev_dist - dist
#         self.prev_dist[:] = dist
#         return torch.nan_to_num(torch.clamp(progress, -1.0, 1.0), nan=0.0)
#
# def goal_reached_bonus(env, goal_distance, goal_radius, asset_cfg=SceneEntityCfg("robot")):
#     dist = _goal_distance_xy(env, goal_distance, asset_cfg)
#     return (dist < goal_radius).float()
#
# def forward_velocity_reward(env, asset_cfg=SceneEntityCfg("robot")):
#     asset: Articulation = env.scene[asset_cfg.name]
#     vx = asset.data.root_lin_vel_b[:, 0]
#     return torch.nan_to_num(torch.clamp(vx, min=0.0, max=5.0), nan=0.0)


def lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical base velocity (clamped to prevent physics-blowup spikes)."""
    asset: Articulation = env.scene[asset_cfg.name]
    val = torch.square(asset.data.root_lin_vel_b[:, 2])
    return torch.nan_to_num(torch.clamp(val, max=25.0), nan=0.0)


def ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll/pitch angular velocity (clamped to prevent physics-blowup spikes)."""
    asset: Articulation = env.scene[asset_cfg.name]
    val = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return torch.nan_to_num(torch.clamp(val, max=25.0), nan=0.0)


def joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Small smoothness penalty on joint speeds."""
    asset: Articulation = env.scene[asset_cfg.name]
    val = torch.mean(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return torch.nan_to_num(torch.clamp(val, max=100.0), nan=0.0)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Small penalty on action magnitude."""
    return torch.mean(torch.square(env.action_manager.action), dim=1)


def _leg_extension(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Extension per leg, normalised to [0, 1].

    All legs close at joint_pos=0. Left legs open toward -π (lower limit);
    right legs open toward +π (upper limit). Using (pos - lower)/(upper - lower)
    gives 1.0 for closed left legs and 0.0 for closed right legs — constant
    regardless of actual extension, providing zero gradient.

    Correct formula: |joint_pos| / half_range, where half_range = max(|lower|, |upper|).
    This is 0 when closed (joint_pos=0) and 1 when fully open (joint_pos=±π) for all legs.
    Scale-invariant: works for any robot size with the same hinge mechanism.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]  # (N, L, 2)
    lower, upper = limits[..., 0], limits[..., 1]
    half_range = torch.maximum(torch.abs(lower), torch.abs(upper))  # π for all legs
    leg_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.clamp(torch.abs(leg_pos) / (half_range + 1e-6), 0.0, 1.0)


def _leg_extension_mean(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Helper: mean leg extension normalised to [0, 1] across all 4 legs."""
    return torch.mean(_leg_extension(env, asset_cfg), dim=1)


def _leg_extension_max(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Helper: max leg extension normalised to [0, 1] across all 4 legs.

    Unlike mean, max cannot be gamed by extending a single leg partially —
    any one leg sticking out gets the full penalty proportional to its extension.
    """
    return torch.amax(_leg_extension(env, asset_cfg), dim=1)


# # _locomotion_difficulty_from_state — superseded by _terrain_difficulty_from_scan
# def _locomotion_difficulty_from_state(env, asset_cfg, tilt_threshold):
#     asset: Articulation = env.scene[asset_cfg.name]
#     tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
#     return torch.nan_to_num(torch.clamp(tilt / tilt_threshold, 0.0, 1.0), nan=0.0)


def _terrain_difficulty_from_scan(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    roughness_threshold: float,
) -> torch.Tensor:
    """Compute terrain difficulty [0, 1] from ray caster scan.

    Uses the MAX of two signals:
      1. std(hit_z): detects rough/bumpy terrain in any direction.
      2. max(positive rel_heights): detects upward obstacles only.
         Downslopes produce only negative rel_heights → upward_height=0
         → no difficulty → legs retract on descents (correct behaviour).

    Flat terrain  → small std, no positive heights → difficulty ≈ 0.
    Downslope     → small std, no positive heights → difficulty ≈ 0.
    Upward stairs → large positive rel_heights     → difficulty → 1.
    Random rough  → high std                       → difficulty → 1.
    """
    from isaaclab.sensors import RayCaster
    sensor: RayCaster = env.scene.sensors[sensor_name]
    asset: Articulation = env.scene["robot"]
    hit_z = sensor.data.ray_hits_w[:, :, 2]                              # (N, num_rays)
    robot_z = asset.data.root_pos_w[:, 2:3]                               # (N, 1)
    rel_heights = hit_z - robot_z                                          # (N, num_rays)

    roughness = torch.std(hit_z, dim=1)                                    # bumpy terrain (all directions)
    upward_height = torch.amax(torch.clamp(rel_heights, min=0.0), dim=1)   # upward obstacles only

    difficulty = torch.clamp(
        torch.maximum(roughness / roughness_threshold, upward_height / roughness_threshold),
        0.0, 1.0,
    )
    return torch.nan_to_num(difficulty, nan=0.0)


def leg_extension_efficiency(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_name: str = "ray_caster",
    roughness_threshold: float = 0.04,
) -> torch.Tensor:
    """Penalize max leg extension when terrain ahead is flat.

      - flat scan  → difficulty ≈ 0 → ease ≈ 1 → penalised for any leg out
      - rough scan → difficulty ≈ 1 → ease ≈ 0 → no penalty
    """
    max_extension = _leg_extension_max(env, asset_cfg)
    difficulty = _terrain_difficulty_from_scan(env, sensor_name, roughness_threshold)
    terrain_ease = 1.0 - difficulty
    return torch.nan_to_num(max_extension * terrain_ease, nan=0.0)


def rough_terrain_leg_bonus(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_name: str = "ray_caster",
    roughness_threshold: float = 0.04,
) -> torch.Tensor:
    """Reward mean leg extension when terrain ahead is rough.

      - flat scan  → difficulty ≈ 0 → no bonus
      - rough scan → difficulty ≈ 1 → bonus proportional to mean extension
    """
    mean_extension = _leg_extension_mean(env, asset_cfg)
    difficulty = _terrain_difficulty_from_scan(env, sensor_name, roughness_threshold)
    return torch.nan_to_num(mean_extension * difficulty, nan=0.0)


def torque_gated_retraction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_name: str = "ray_caster",
    roughness_threshold: float = 0.04,
    torque_threshold: float = 0.3,
) -> torch.Tensor:
    """Penalize extended legs when both scan and torques confirm flat terrain.

    Uses scan as privileged training-time gate + torques as the observable proxy:
      - scan says rough → terrain_ease ≈ 0 → no penalty (robot may be probing)
      - torques high    → torque_ease ≈ 0 → no penalty (legs contacting terrain)
      - both say flat   → penalty fires → teaches: torque drop = retract

    At inference the policy uses the torque signal it observes; the scan gate
    during training prevents misfiring when legs are retracted before probing.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    mean_abs_torque = torch.mean(torch.abs(torques), dim=1)
    torque_ease = 1.0 - torch.clamp(mean_abs_torque / torque_threshold, 0.0, 1.0)
    terrain_ease = 1.0 - _terrain_difficulty_from_scan(env, sensor_name, roughness_threshold)
    max_extension = _leg_extension_max(env, asset_cfg)
    return torch.nan_to_num(max_extension * torque_ease * terrain_ease, nan=0.0)


# # rough_terrain_speed_penalty — unused, not wired into RewardsCfg
# def rough_terrain_speed_penalty(env, asset_cfg=SceneEntityCfg("robot"), tilt_threshold=0.15):
#     asset: Articulation = env.scene[asset_cfg.name]
#     speed = torch.nan_to_num(torch.abs(asset.data.root_lin_vel_b[:, 0]), nan=0.0)
#     tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
#     terrain_difficulty = torch.clamp(tilt / tilt_threshold, 0.0, 1.0)
#     return torch.clamp(speed * terrain_difficulty, max=5.0)


def velocity_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "vel_cmd",
    std: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for matching commanded forward speed. Uses exp(-error^2 / std^2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    vx_cmd = env.command_manager.get_command(command_name)[:, 0]
    vx_actual = asset.data.root_lin_vel_b[:, 0]
    error = torch.square(vx_cmd - vx_actual)
    return torch.nan_to_num(torch.exp(-error / (std ** 2)), nan=0.0)


def yaw_rate_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "vel_cmd",
    std: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for matching commanded yaw rate. Uses exp(-error^2 / std^2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    yaw_cmd = env.command_manager.get_command(command_name)[:, 1]
    yaw_actual = asset.data.root_ang_vel_b[:, 2]
    error = torch.square(yaw_cmd - yaw_actual)
    return torch.nan_to_num(torch.exp(-error / (std ** 2)), nan=0.0)


def wheel_slip_penalty(
    env: ManagerBasedRLEnv,
    wheel_radius: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize wheel slip: wheel linear speed exceeding base forward speed."""
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_omega = asset.data.joint_vel[:, asset_cfg.joint_ids]
    wheel_lin_speed = torch.abs(wheel_omega) * wheel_radius
    mean_wheel_speed = torch.mean(wheel_lin_speed, dim=1)
    base_vx = torch.abs(asset.data.root_lin_vel_b[:, 0])
    return torch.nan_to_num(torch.clamp(mean_wheel_speed - base_vx, min=0.0, max=5.0), nan=0.0)


def body_tilt_with_retracted_legs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tilt_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize body tilt when legs are retracted.

    Encourages the robot to keep the body parallel to the ground by extending legs.

    penalty = tilt_normalized × legs_retracted
      - flat terrain (tilt≈0)  → penalty≈0 → torque_gated_retraction keeps legs closed
      - rough terrain (tilt>0) → penalty fires → robot extends legs → tilt reduces
      - no perverse loop: extending legs always reduces legs_retracted regardless of tilt

    tilt_threshold=1.0 → penalty ramps linearly from 0 at 0° to 1.0 at ~5.8°.

    Sim2real: tilt from BNO085 IMU (projected_gravity roll/pitch components),
              legs_retracted from joint encoders. No extra sensors needed.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
    tilt_normalized = torch.clamp(tilt / tilt_threshold, 0.0, 1.0)

    mean_extension = _leg_extension_mean(env, asset_cfg)
    legs_retracted = 1.0 - mean_extension

    return torch.nan_to_num(tilt_normalized * legs_retracted, nan=0.0)


def stuck_with_retracted_legs(
    env: ManagerBasedRLEnv,
    command_name: str = "vel_cmd",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_name: str = "ray_caster",
    roughness_threshold: float = 0.04,
) -> torch.Tensor:
    """Penalize failing velocity tracking on rough terrain with legs retracted.

    Fires when all three conditions hold simultaneously:
      - vel_error is high (robot is stuck / not tracking commanded speed)
      - legs are retracted (mean_extension ≈ 0)
      - terrain ahead is difficult (scan difficulty ≈ 1)

    This directly incentivizes proactive leg extension before/during rough terrain.
    No perverse feedback loops: difficulty is scan-based (not leg-state-based).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    vx_cmd = env.command_manager.get_command(command_name)[:, 0]
    vx_actual = asset.data.root_lin_vel_b[:, 0]
    vel_error = torch.clamp(torch.abs(vx_cmd - vx_actual), max=2.0)

    mean_extension = _leg_extension_mean(env, asset_cfg)
    legs_retracted = 1.0 - mean_extension  # 1.0 when fully retracted, 0.0 when fully extended

    difficulty = _terrain_difficulty_from_scan(env, sensor_name, roughness_threshold)

    return torch.nan_to_num(vel_error * legs_retracted * difficulty, nan=0.0)
