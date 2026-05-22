# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _goal_distance_xy(env, goal_distance: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    root_xy = asset.data.root_pos_w[:, :2]
    goal_xy = env.scene.env_origins[:, :2] + torch.tensor([goal_distance, 0.0], device=env.device)
    return torch.norm(goal_xy - root_xy, dim=-1)


def goal_reached(
    env: ManagerBasedRLEnv,
    goal_distance: float,
    goal_radius: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    dist = _goal_distance_xy(env, goal_distance, asset_cfg)
    return dist < goal_radius


def high_base_velocity(
    env: ManagerBasedRLEnv,
    lin_threshold: float = 8.0,
    ang_threshold: float = 20.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when base velocity is physically unrealistic (physics blowup detection)."""
    asset: Articulation = env.scene[asset_cfg.name]
    lin_speed = torch.norm(asset.data.root_lin_vel_w, dim=-1)
    ang_speed = torch.norm(asset.data.root_ang_vel_b, dim=-1)
    return (lin_speed > lin_threshold) | (ang_speed > ang_threshold)
