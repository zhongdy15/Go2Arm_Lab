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
from isaaclab.utils.math import quat_apply, sample_uniform

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


# =============================================================================
# 灭火枪后坐力事件
#   - 在指定 body（默认 upper_arm_link）上沿其本体 -X 方向施加冲击力，
#     模拟开枪 / 发射灭火弹时的反作用力。
#   - 支持「连发」：每次触发后连射 shots_per_burst 发，发与发之间隔
#     shot_interval_s 秒；一轮连射结束后进入 burst_cooldown_s 冷却。
#   - 力在 body 局部坐标系定义，自动通过 body 当前姿态四元数旋转到世界系再施加。
#   - 推荐以 mode="interval" + interval_range_s=(shot_interval_s, shot_interval_s)
#     注册：每个 shot_interval 周期调用一次，函数内部判定当前是否在连射阶段。
# =============================================================================

def apply_recoil_burst(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="upper_arm_link"),
    force_magnitude: float = 80.0,
    shots_per_burst: int = 3,
    shot_interval_s: float = 0.15,
    burst_cooldown_s: float = 3.0,
    direction_local: tuple[float, float, float] = (-1.0, 0.0, 0.0),
):
    """模拟灭火枪连发后坐力（仅适合 PLAY / 评估，不建议直接用于训练）。

    Args:
        asset_cfg: 指定施力的 body（默认 upper_arm_link 末端）。
        force_magnitude: 单发后坐力幅值 (N)。
        shots_per_burst: 一轮连发的弹数。
        shot_interval_s: 同一轮连发内相邻两发的间隔 (s)。注册 EventTerm 时
            interval_range_s 应与该值一致或更小。
        burst_cooldown_s: 两轮连发之间的冷却时间 (s)。
        direction_local: 力在 body 局部系的单位方向（默认 -X，向后）。
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 解析 body_id（取第一个匹配的 body）
    if isinstance(asset_cfg.body_ids, list) and len(asset_cfg.body_ids) > 0:
        body_id = int(asset_cfg.body_ids[0])
    else:
        body_ids_resolved, _ = asset.find_bodies(
            asset_cfg.body_names if isinstance(asset_cfg.body_names, list) else [asset_cfg.body_names]
        )
        body_id = int(body_ids_resolved[0])

    num_envs = env.num_envs
    device = env.device
    step_dt = float(getattr(env, "step_dt", env.physics_dt * env.cfg.decimation))

    # 首次调用：初始化每个环境的连发状态
    if not hasattr(env, "_recoil_state"):
        env._recoil_state = {
            "shots_fired": torch.zeros(num_envs, dtype=torch.long, device=device),
            "next_fire_time": torch.zeros(num_envs, device=device),
            "sim_time": torch.zeros(num_envs, device=device),
        }
    state = env._recoil_state

    # interval 模式下 env_ids 通常为 None / 全部环境；用步长累加时间，避免依赖 sim 时钟
    state["sim_time"] += step_dt
    t = state["sim_time"]

    # 触发条件：当前时间 >= next_fire_time
    fire_mask = t >= state["next_fire_time"]
    if env_ids is not None:
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        sel = torch.zeros(num_envs, dtype=torch.bool, device=device)
        sel[env_ids] = True
        fire_mask = fire_mask & sel

    # 准备力 / 力矩张量（必须 (num_envs, 1, 3) 与 body_ids=[body_id] 对齐）
    forces = torch.zeros(num_envs, 1, 3, device=device)
    torques = torch.zeros_like(forces)

    if fire_mask.any():
        local_dir = torch.tensor(direction_local, device=device, dtype=torch.float32).view(1, 3)
        body_quat_w = asset.data.body_quat_w[:, body_id]  # (num_envs, 4)
        world_dir = quat_apply(body_quat_w, local_dir.expand(num_envs, -1))
        forces[fire_mask, 0, :] = world_dir[fire_mask] * float(force_magnitude)

        # 推进各环境的连发状态机
        state["shots_fired"][fire_mask] += 1
        finished_burst = fire_mask & (state["shots_fired"] >= int(shots_per_burst))
        # 未结束连发：下一发在 shot_interval_s 后
        state["next_fire_time"][fire_mask] = t[fire_mask] + float(shot_interval_s)
        # 结束连发：清零计数 + 进入冷却
        if finished_burst.any():
            state["shots_fired"][finished_burst] = 0
            state["next_fire_time"][finished_burst] = t[finished_burst] + float(burst_cooldown_s)

    # 应用外力（持续到下次调用被覆盖；EventTerm interval 周期 ≈ shot_interval_s）
    asset.set_external_force_and_torque(forces, torques, body_ids=[body_id])
