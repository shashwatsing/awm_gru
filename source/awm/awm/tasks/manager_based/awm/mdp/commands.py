# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformVelCommand(CommandTerm):
    """Samples uniform forward speed and yaw rate commands per episode.

    vx_cmd    : commanded forward speed  [vx_range[0], vx_range[1]]  (m/s)
    yaw_rate  : commanded yaw rate       [yaw_range[0], yaw_range[1]] (rad/s)

    Commands are resampled at reset and periodically within an episode
    (controlled by resampling_time_range in the base class).
    """

    cfg: "UniformVelCommandCfg"

    def __init__(self, cfg: "UniformVelCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Shape: (num_envs, 2)  →  [:, 0]=vx_cmd, [:, 1]=yaw_rate_cmd
        self._vel_command = torch.zeros(env.num_envs, 2, device=env.device)

    # ------------------------------------------------------------------
    # Required interface
    # ------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """Return current command tensor of shape (num_envs, 2)."""
        return self._vel_command

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self._vel_command[env_ids, 0] = torch.empty(n, device=self.device).uniform_(
            self.cfg.vx_range[0], self.cfg.vx_range[1]
        )
        self._vel_command[env_ids, 1] = torch.empty(n, device=self.device).uniform_(
            self.cfg.yaw_rate_range[0], self.cfg.yaw_rate_range[1]
        )

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        pass

    def _debug_vis_callback(self, event) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        pass


@configclass
class UniformVelCommandCfg(CommandTermCfg):
    """Configuration for uniform velocity command sampling."""

    class_type: type = UniformVelCommand

    vx_range: tuple[float, float] = (0.3, 0.8)
    """Commanded forward speed range (m/s). Always positive (forward only)."""

    yaw_rate_range: tuple[float, float] = (-1.0, 1.0)
    """Commanded yaw rate range (rad/s)."""
