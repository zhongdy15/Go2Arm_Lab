# =============================================================================
# events_v3.py
# 自定义 event：支持按 joint_names 过滤的 reset_joints_by_scale。
# IsaacLab 上游 reset_joints_by_scale 在传入 joint_ids 列表时使用
#   default_joint_pos[env_ids, joint_ids]
# 这种 fancy indexing 会因为 env_ids 与 joint_ids 长度不一致而广播失败。
# 这里用 [env_ids][:, joint_ids] 的两步索引规避该问题。
# =============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_joints_by_scale_filtered(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """按缩放系数随机化指定关节的初始位置 / 速度，支持 joint_names 过滤。

    与 isaaclab.envs.mdp.reset_joints_by_scale 行为一致，但允许只重置
    asset_cfg.joint_names 指定的子集。其它关节保持当前 sim 状态不变。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # joint_ids 既可能是 slice(None) 也可能是 list[int]，统一处理
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids_t = torch.tensor(joint_ids, device=asset.device, dtype=torch.long)
    elif joint_ids == slice(None):
        joint_ids_t = torch.arange(asset.num_joints, device=asset.device, dtype=torch.long)
    else:
        joint_ids_t = torch.as_tensor(joint_ids, device=asset.device, dtype=torch.long)

    # 取这批 env / 这批 joint 的默认值
    default_pos = asset.data.default_joint_pos[env_ids][:, joint_ids_t].clone()
    default_vel = asset.data.default_joint_vel[env_ids][:, joint_ids_t].clone()

    pos_scale = sample_uniform(*position_range, default_pos.shape, device=asset.device)
    vel_scale = sample_uniform(*velocity_range, default_vel.shape, device=asset.device)

    joint_pos = default_pos * pos_scale
    joint_vel = default_vel * vel_scale

    # 限位 clamp
    limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids_t]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

    asset.write_joint_state_to_sim(
        joint_pos, joint_vel, joint_ids=joint_ids_t, env_ids=env_ids
    )
