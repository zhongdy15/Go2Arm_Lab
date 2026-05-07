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
    id="Isaac-Go2Arm-Flat",
    entry_point="Go2Arm_Lab.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2ARMFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ArmFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2Arm-Flat-Play",
    entry_point="Go2Arm_Lab.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2ARMFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ArmFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Go2Arm-Rough",
    entry_point="Go2Arm_Lab.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Go2ARMRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ArmRoughPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Go2Arm-Rough-Play",
    entry_point="Go2Arm_Lab.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:Go2ARMRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ArmRoughPPORunnerCfg",
    },
)


# =============================================================================
# Go2Arm v3 圆周运动任务（go2_arm_v3.usd, 14 DOF, 仅训练腿部）
# =============================================================================
gym.register(
    id="Isaac-Go2ArmV3-Circle",
    entry_point="Go2Arm_Lab.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "Go2Arm_Lab.tasks.manager_based.go2arm_lab.go2arm_v3_circle_env_cfg:"
            "Go2ArmV3CircleEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ArmV3CirclePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Isaac-Go2ArmV3-Circle-Play",
    entry_point="Go2Arm_Lab.env.manager_env:ManagerRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "Go2Arm_Lab.tasks.manager_based.go2arm_lab.go2arm_v3_circle_env_cfg:"
            "Go2ArmV3CircleEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ArmV3CirclePPORunnerCfg"
        ),
    },
)