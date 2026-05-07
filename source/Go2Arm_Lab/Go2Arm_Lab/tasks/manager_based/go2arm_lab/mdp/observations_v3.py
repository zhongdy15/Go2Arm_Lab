# =============================================================================
# observations_v3.py
# 适配 go2_arm_v3 (14 DOF) 的观测函数：
#   - 12 个腿关节 + 2 个臂关节 (waist / shoulder)
#   - 不再有 gripper_link，因此移除 get_mass_ee 相关观测
# =============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# v3 关节顺序：腿部 12 个 + 臂部 2 个，preserve_order=True 保证后续 action / obs 维度一致
V3_JOINT_NAMES = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "waist", "shoulder",
]


def joint_pos_rel_v3(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """关节位置相对默认值的偏差，shape=[num_envs, 14]"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(V3_JOINT_NAMES, preserve_order=True)
    return asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]


def joint_vel_rel_v3(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """关节速度相对默认速度的偏差，shape=[num_envs, 14]"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(V3_JOINT_NAMES, preserve_order=True)
    return asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]


def get_joints_torques_v3(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """实际作用力矩（特权观测），shape=[num_envs, 14]"""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(V3_JOINT_NAMES, preserve_order=True)
    return asset.data.applied_torque[:, joint_ids]
