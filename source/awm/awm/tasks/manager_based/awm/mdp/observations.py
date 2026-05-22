# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _goal_distance_xy(env, goal_distance: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_xy = asset.data.root_pos_w[:, :2]
    goal_xy = env.scene.env_origins[:, :2] + torch.tensor([goal_distance, 0.0], device=env.device)
    return torch.norm(goal_xy - root_xy, dim=-1)


def distance_to_goal(env, goal_distance: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    dist = _goal_distance_xy(env, goal_distance, asset_cfg)
    return torch.nan_to_num(torch.clamp(dist, max=20.0), nan=float(goal_distance)).unsqueeze(-1)


def base_lin_vel_x(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_lin_vel_b[:, 0:1], nan=0.0)


def wheel_velocities(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sign_mask: list[float] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vels = torch.nan_to_num(asset.data.joint_vel[:, asset_cfg.joint_ids], nan=0.0)
    if sign_mask is not None:
        signs = torch.tensor(sign_mask, device=vels.device, dtype=vels.dtype)
        vels = vels * signs
    return vels


def leg_positions(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def leg_torques(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Applied torque on leg joints — proxy for contact force / terrain hardness.

    In sim: computed_torques from the physics engine.
    On real robot (mode 5, current-based position control):
        torque_nm = current_LSB * 2.69e-3 * 3.83  (calibrated 2026-04-19, 18 trials)
        normalized = torque_nm / 2.70
    Clamped to [-2, 2] to handle transient spikes.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.nan_to_num(torch.clamp(torques / 2.70, -2.0, 2.0), nan=0.0)


def projected_gravity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def leg_actions(env, num_wheels: int) -> torch.Tensor:
    return env.action_manager.action[:, num_wheels:]


def goal_heading_error(
    env, goal_distance: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Sine and cosine of the yaw error between robot heading and direction to goal.

    Returns (sin(e), cos(e)) so the network sees a smooth, bounded 2D heading signal.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_xy = asset.data.root_pos_w[:, :2]
    goal_xy = env.scene.env_origins[:, :2] + torch.tensor([goal_distance, 0.0], device=env.device)
    goal_dir = goal_xy - root_xy
    goal_angle = torch.atan2(goal_dir[:, 1], goal_dir[:, 0])

    quat = asset.data.root_quat_w  # (N, 4) [w, x, y, z]
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    err = goal_angle - yaw
    return torch.nan_to_num(
        torch.stack([torch.sin(err), torch.cos(err)], dim=1), nan=0.0
    )


def commanded_velocity(env, command_name: str = "vel_cmd") -> torch.Tensor:
    """Return the current velocity command [vx_cmd, yaw_rate_cmd] of shape (N, 2)."""
    return env.command_manager.get_command(command_name)


def base_ang_vel_z(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the robot yaw rate (angular velocity about world z) of shape (N, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_ang_vel_b[:, 2:3], nan=0.0)


def wheel_contact_forces(env, sensor_name: str = "contact_forces", body_names: str | list[str] = "wheel_.*") -> torch.Tensor:
    sensor = env.scene.sensors[sensor_name]
    body_ids, _ = sensor.find_bodies(body_names, preserve_order=True)
    if len(body_ids) == 0:
        raise ValueError(f"No contact bodies matched {body_names}. Available: {sensor.body_names}")
    forces = sensor.data.net_forces_w[:, body_ids, :]
    result = torch.norm(forces, dim=-1)
    return torch.nan_to_num(torch.clamp(result, max=500.0), nan=0.0)


def terrain_height_scan(
    env,
    sensor_name: str = "ray_caster",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return ray caster hit heights relative to robot base z.

    Shape: (N, num_rays). Values clamped to [-0.5, 0.5] m.
    Gives the policy a direct look-ahead terrain map so it can extend legs
    before reaching rough terrain, not just after wheels start slipping.
    """
    from isaaclab.sensors import RayCaster
    sensor: RayCaster = env.scene.sensors[sensor_name]
    asset: Articulation = env.scene[asset_cfg.name]
    hit_z = sensor.data.ray_hits_w[:, :, 2]     # (N, num_rays) world z
    robot_z = asset.data.root_pos_w[:, 2:3]     # (N, 1)
    rel_heights = hit_z - robot_z                # (N, num_rays) relative heights
    return torch.nan_to_num(torch.clamp(rel_heights, -0.5, 0.5), nan=0.0)


class progress_slip_history(ManagerTermBase):
    """EMA history features for forward progress and wheel slip."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: ObservationTermCfg):
        super().__init__(cfg, env)
        self.prev_root_x = torch.zeros(env.num_envs, device=env.device)
        self.prog_ema = torch.zeros(env.num_envs, device=env.device)
        self.slip_ema = torch.zeros(env.num_envs, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        asset_cfg = self.cfg.params.get("wheel_asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = self._env.scene[asset_cfg.name]
        root_x = asset.data.root_pos_w[:, 0]
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        self.prev_root_x[env_ids] = root_x[env_ids]
        self.prog_ema[env_ids] = 0.0
        self.slip_ema[env_ids] = 0.0
        self.initialized[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        wheel_radius: float = 0.1,
        ema_alpha: float = 0.1,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[wheel_asset_cfg.name]
        root_x = asset.data.root_pos_w[:, 0]

        first_step = ~self.initialized
        self.prev_root_x[first_step] = root_x[first_step]
        self.initialized[first_step] = True

        step_progress = torch.clamp(root_x - self.prev_root_x, min=0.0, max=0.5)
        wheel_omega = asset.data.joint_vel[:, wheel_asset_cfg.joint_ids]
        wheel_lin_speed = torch.abs(wheel_omega) * wheel_radius
        mean_wheel_speed = torch.mean(wheel_lin_speed, dim=1)
        base_vx = torch.abs(asset.data.root_lin_vel_b[:, 0])
        slip = torch.clamp(mean_wheel_speed - base_vx, min=0.0)

        alpha = float(max(min(ema_alpha, 1.0), 0.0))
        self.prog_ema = (1.0 - alpha) * self.prog_ema + alpha * step_progress
        self.slip_ema = (1.0 - alpha) * self.slip_ema + alpha * slip

        self.prev_root_x[:] = root_x
        return torch.stack((self.prog_ema, self.slip_ema), dim=1)
