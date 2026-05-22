# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Awm_Morph-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Awm_WheelsOnly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmWheelsOnlyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOWheelsOnlyRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_LegsOpen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmLegsOpenCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOLegsOpenRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_ProprioOnly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmProprioOnlyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOProprioOnlyRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_StairsEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmStairsEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

gym.register(
    id="Template-Awm_ProprioTorque-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmProprioTorqueCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOProprioTorqueRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_LSTM_ProprioOnly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmProprioOnlyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOLSTMProprioOnlyRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_LSTM_ProprioTorque-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmProprioTorqueCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOLSTMProprioTorqueRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_LSTM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOLSTMRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_LSTM_WheelsOnly-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmWheelsOnlyCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOLSTMWheelsOnlyRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_LSTM_LegsOpen-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmLegsOpenCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOLSTMLegsOpenRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_LSTM_StairsEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmStairsEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOLSTMRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_GRU_StairsEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmProprioTorqueStairsEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOGRUProprioTorqueRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_GRU_ProprioTorque-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmProprioTorqueCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOGRUProprioTorqueRunnerCfg",
    },
)

# ── Unseen-terrain evaluation tasks ─────────────────────────────────────────

# Terrain 1: Stairs — missing proprio-only variant
gym.register(
    id="Template-Awm_LSTM_ProprioOnly_StairsEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmProprioOnlyStairsEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOLSTMProprioOnlyRunnerCfg",
    },
)

# Terrain 2: Wave
gym.register(
    id="Template-Awm_LSTM_WaveEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmWaveEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOLSTMRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_LSTM_ProprioOnly_WaveEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmProprioOnlyWaveEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOLSTMProprioOnlyRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_GRU_WaveEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmProprioTorqueWaveEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOGRUProprioTorqueRunnerCfg",
    },
)

# Terrain 3: Rough
gym.register(
    id="Template-Awm_LSTM_RoughEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmRoughEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOLSTMRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_LSTM_ProprioOnly_RoughEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmProprioOnlyRoughEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOLSTMProprioOnlyRunnerCfg",
    },
)

gym.register(
    id="Template-Awm_GRU_RoughEval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.awm_env_cfg:AwmProprioTorqueRoughEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPOGRUProprioTorqueRunnerCfg",
    },
)
