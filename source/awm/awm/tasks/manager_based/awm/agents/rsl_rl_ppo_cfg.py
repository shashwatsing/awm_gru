# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 200
    experiment_name = "awm_transformer"
    empirical_normalization = True
    logger = "wandb"
    wandb_project = "awm_gru"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPOLSTMRunnerCfg(PPORunnerCfg):
    """LSTM-based recurrent policy — captures observation history implicitly."""
    experiment_name = "awm_lstm"
    wandb_project = "awm_gru"
    num_steps_per_env = 48
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.3,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPOWheelsOnlyRunnerCfg(PPORunnerCfg):
    """Ablation: wheels-only baseline."""
    experiment_name = "awm_wheels_only"
    wandb_project = "awm_gru"


@configclass
class PPOProprioOnlyRunnerCfg(PPORunnerCfg):
    """Ablation: proprioception-only (no terrain scan, no torques)."""
    experiment_name = "awm_mlp_proprio_only"
    wandb_project = "awm_gru"


@configclass
class PPOProprioTorqueRunnerCfg(PPORunnerCfg):
    """MLP: proprioception + leg torques — no terrain scan."""
    experiment_name = "awm_mlp_proprio_torque"
    wandb_project = "awm_gru"


@configclass
class PPOLegsOpenRunnerCfg(PPORunnerCfg):
    """Ablation: legs-fully-open baseline."""
    experiment_name = "awm_legs_open"
    wandb_project = "awm_gru"


@configclass
class PPOLSTMProprioOnlyRunnerCfg(PPOLSTMRunnerCfg):
    """LSTM ablation: proprioception-only (no terrain scan)."""
    experiment_name = "awm_lstm_proprio_only"
    wandb_project = "awm_gru"


@configclass
class PPOLSTMProprioTorqueRunnerCfg(PPOLSTMRunnerCfg):
    """LSTM: proprioception + leg torques — active morphological probing."""
    experiment_name = "awm_lstm_proprio_torque"
    wandb_project = "awm_gru"


@configclass
class PPOGRURunnerCfg(PPORunnerCfg):
    """GRU-based recurrent policy."""
    experiment_name = "awm_gru"
    wandb_project = "awm_gru"
    num_steps_per_env = 48
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.3,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPOGRUScanRunnerCfg(PPOGRURunnerCfg):
    """GRU baseline: full observations with terrain_scan (camera condition)."""
    experiment_name = "awm_gru_scan"
    wandb_project = "awm_gru"


@configclass
class PPOGRUProprioOnlyRunnerCfg(PPOGRURunnerCfg):
    """GRU ablation: proprioception-only (no terrain scan, no torques)."""
    experiment_name = "awm_gru_proprio_only"
    wandb_project = "awm_gru"


@configclass
class PPOGRUProprioTorqueRunnerCfg(PPOGRURunnerCfg):
    """GRU: proprioception + leg torques — active morphological probing (ours)."""
    experiment_name = "awm_gru_proprio_torque"
    wandb_project = "awm_gru"


@configclass
class PPOLSTMWheelsOnlyRunnerCfg(PPOLSTMRunnerCfg):
    """LSTM ablation: wheels-only."""
    experiment_name = "awm_lstm_wheels_only"
    wandb_project = "awm_gru"


@configclass
class PPOLSTMLegsOpenRunnerCfg(PPOLSTMRunnerCfg):
    """LSTM ablation: legs-open."""
    experiment_name = "awm_lstm_legs_open"
    wandb_project = "awm_gru"


@configclass
class PPOFlatRunnerCfg(PPORunnerCfg):
    experiment_name = "awm_manager_flat"
    max_iterations = 10000
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.3,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
    )