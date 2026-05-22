# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Maps normalized actions to per-wheel velocity and per-leg position targets.
class AwmDriveAction(ActionTerm):
    """Action term mapping per-wheel and per-leg inputs to actuator targets."""

    cfg: "AwmDriveActionCfg"

    def __init__(self, cfg: "AwmDriveActionCfg", env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        self._wheel_ids, self._wheel_names = self.robot.find_joints(cfg.wheel_joint_names, preserve_order=True)
        self._leg_ids, self._leg_names = self.robot.find_joints(cfg.leg_joint_names, preserve_order=True)

        if len(self._wheel_ids) == 0:
            raise ValueError("No wheel joints matched the provided names.")
        if len(self._leg_ids) == 0:
            raise ValueError("No leg joints matched the provided names.")

        print(f"[AwmDriveAction] wheel joints: {list(zip(self._wheel_ids, self._wheel_names))}")
        print(f"[AwmDriveAction] leg   joints: {list(zip(self._leg_ids, self._leg_names))}")
        all_ids, all_names = self.robot.find_joints(".*")
        print(f"[AwmDriveAction] ALL joints in USD: {list(zip(all_ids, all_names))}")

        # Right wheels spin around -y axis (mirrored geometry) so positive velocity target
        # produces backward motion. Negate targets for flipped wheels so positive action = forward.
        flip_wheel_names = set(cfg.flip_wheel_joint_names)
        self._flip_wheel_mask = torch.tensor(
            [name in flip_wheel_names for name in self._wheel_names],
            dtype=torch.bool, device=self.device,
        )

        # Left leg joints have closed=upper_limit, open=lower_limit (opposite of right legs).
        # Flip extension so extension=0 always means closed for all legs.
        flip_names = set(cfg.flip_leg_joint_names)
        self._flip_mask = torch.tensor(
            [name in flip_names for name in self._leg_names],
            dtype=torch.bool, device=self.device,
        )


        self._action_dim = len(self._wheel_ids) + len(self._leg_ids) #Action dimension is both for wheels and legs now which is going to be 8
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device) #Wheel targets
        self._wheel_targets = torch.zeros((self.num_envs, len(self._wheel_ids)), device=self.device)
        self._leg_targets = torch.zeros((self.num_envs, len(self._leg_ids)), device=self.device)
        self._prev_root_x = torch.zeros(self.num_envs, device=self.device)
        self._stuck_count = torch.zeros(self.num_envs, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)

        wheel_cmd = self._raw_actions[:, : len(self._wheel_ids)]
        leg_cmd = self._raw_actions[:, len(self._wheel_ids) :]

        self._wheel_targets = wheel_cmd * self.cfg.max_wheel_speed
        self._wheel_targets[:, self._flip_wheel_mask] *= -1

        # Convert policy leg action to normalized extension in [0, 1].
        # leg_offset sets the extension when action=0 (0.0=closed, 0.5=half, 1.0=open).
        policy_extension = 0.5 * leg_cmd + self.cfg.leg_offset
        desired_extension = torch.clamp(policy_extension, 0.0, 1.0)

        if self.cfg.use_auto_extension:
            root_x = self.robot.data.root_pos_w[:, 0]
            first_step = ~self._initialized
            self._prev_root_x[first_step] = root_x[first_step]
            self._initialized[first_step] = True

            step_progress = torch.clamp(root_x - self._prev_root_x, min=0.0)
            wheel_omega = self.robot.data.joint_vel[:, self._wheel_ids]
            wheel_lin_speed = torch.abs(wheel_omega) * self.cfg.wheel_radius
            mean_wheel_speed = torch.mean(wheel_lin_speed, dim=1)
            base_vx = torch.abs(self.robot.data.root_lin_vel_b[:, 0])
            slip = torch.clamp(mean_wheel_speed - base_vx, min=0.0)

            stuck_now = (step_progress < self.cfg.progress_threshold) & (slip > self.cfg.slip_threshold)
            self._stuck_count = torch.where(stuck_now, self._stuck_count + 1.0, torch.zeros_like(self._stuck_count))

            ramp = max(float(self.cfg.stuck_ramp_steps), 1.0)
            auto_extension = torch.clamp((self._stuck_count - float(self.cfg.stuck_steps)) / ramp, 0.0, 1.0)
            desired_extension = torch.maximum(desired_extension, auto_extension.unsqueeze(-1))
            self._prev_root_x[:] = root_x

        limits = self.robot.data.soft_joint_pos_limits[:, self._leg_ids]
        lower = limits[..., 0]
        upper = limits[..., 1]

        # For flipped joints: extension=0 must reach upper limit (closed), not lower.
        eff = desired_extension.clone()
        eff[:, self._flip_mask] = 1.0 - eff[:, self._flip_mask]

        if self.cfg.closed_at_upper_limit:
            self._leg_targets = upper - eff * (upper - lower)
        else:
            self._leg_targets = lower + eff * (upper - lower)

    def apply_actions(self):
        self.robot.set_joint_velocity_target(self._wheel_targets, joint_ids=self._wheel_ids)
        self.robot.set_joint_position_target(self._leg_targets, joint_ids=self._leg_ids)


@configclass
# Holds joint names and scales for the AWM drive action.
class AwmDriveActionCfg(ActionTermCfg):
    """Configuration for the AWM drive action term."""

    class_type: type[ActionTerm] = AwmDriveAction
    asset_name: str = MISSING
    wheel_joint_names: list[str] = MISSING
    leg_joint_names: list[str] = MISSING
    max_wheel_speed: float = 4.0
    leg_offset: float = 0.5
    use_auto_extension: bool = True
    wheel_radius: float = 0.0508
    progress_threshold: float = 0.002
    slip_threshold: float = 0.2
    stuck_steps: int = 30
    stuck_ramp_steps: int = 120
    closed_at_upper_limit: bool = False
    flip_wheel_joint_names: list[str] = []
    flip_leg_joint_names: list[str] = []
    debug_vis: bool = False
