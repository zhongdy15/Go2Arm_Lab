# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# flat_env_cfg.py - 平地环境专用配置
#
# 继承关系:
#   LocomotionVelocityEnvCfg (go2arm_lab_env_cfg.py)  ← 基类，定义全部 MDP 结构
#       └── Go2ARMFlatEnvCfg                           ← 平地训练配置（覆盖参数）
#               └── Go2ARMFlatEnvCfg_PLAY             ← 推理专用（禁用随机化）
#
# 主要改动（相对基类）:
#   - terrain_type = "plane"：平地，不生成随机地形（降低训练难度，加快收敛）
#   - 禁用 push_robot 事件（平地无需抗扰动推力训练）
#   - 末端位置/速度指令从窄范围出发，随课程系数 curriculum_coeff 逐渐扩展
#   - 覆盖各奖励项权重（平地场景的平衡调参结果）
# =============================================================================

from isaaclab.utils import configclass


from Go2Arm_Lab.tasks.manager_based.go2arm_lab.go2arm_lab_env_cfg import LocomotionVelocityEnvCfg
from Go2Arm_Lab.assets.go2arm_articulation_cfg import GO2ARM_CFG


@configclass
class Go2ARMFlatEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # 绑定机器人 USD 模型，{ENV_REGEX_NS} 会被替换为每个并行环境的命名空间
        self.scene.robot = GO2ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 平地训练不需要随机推力干扰（粗糙地形版本会启用）
        self.events.push_robot = None

        # 使用无限平面地形，不生成随机地貌（节省初始化时间）
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # ---- 速度指令课程学习 ----
        # curriculum_coeff=4000：在 4000 次迭代内将指令范围从 init 线性扩展到 final
        self.commands.base_velocity.curriculum_coeff = 4000
        # 初始阶段：机器人原地静止，不要求移动（让臂先学会稳定）
        self.commands.base_velocity.ranges_init.lin_vel_x  = (0.0, 0.0)
        self.commands.base_velocity.ranges_init.lin_vel_y  = (-0.0, 0.0)
        self.commands.base_velocity.ranges_init.ang_vel_z  = (-0.0, 0.0)
        # 最终阶段：前进最快 1 m/s，侧移 ±0.5 m/s，旋转 ±0.5 rad/s
        self.commands.base_velocity.ranges_final.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges_final.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges_final.ang_vel_z = (-0.5, 0.5)
  
        # ---- 末端位姿指令课程学习 ----
        # curriculum_coeff=5000：比速度课程稍慢，给腿部运动更多稳定时间
        self.commands.ee_pose.curriculum_coeff = 5000
        # 初始：末端保持在机身正前方较小区域（约 0.45~0.5 m 处）
        self.commands.ee_pose.ranges_init.pos_x = (0.45, 0.5)
        self.commands.ee_pose.ranges_init.pos_y = (-0.05, 0.05)
        self.commands.ee_pose.ranges_init.pos_z = (0.45, 0.5)
        # 最终：末端可达机身前方较大的工作空间
        self.commands.ee_pose.ranges_final.pos_x = (0.4, 0.65)
        self.commands.ee_pose.ranges_final.pos_y = (-0.35, 0.35)
        self.commands.ee_pose.ranges_final.pos_z = (0.15, 0.6)

        # ---- 奖励权重（平地场景调参结果）----
        # 机械臂相关奖励
        self.rewards.end_effector_position_tracking.weight = 3.0    # 末端位置追踪（正奖励，主要目标）
        self.rewards.end_effector_orientation_tracking.weight = -2.0 # 末端姿态误差惩罚
        self.rewards.end_effector_action_rate.weight = -0.01         # 臂关节动作变化率惩罚（抑制抖动）
        self.rewards.end_effector_action_smoothness.weight = -0.04   # 臂关节动作平滑度惩罚
        
        # 腿部运动相关奖励
        self.rewards.tracking_lin_vel_x_l1.weight = 3.5   # 前进速度追踪（L1，主要驱动力）
        self.rewards.track_ang_vel_z_exp.weight = 2.0     # 偏航角速度追踪（exp 形式，鼓励精准控制）
        self.rewards.lin_vel_z_l2.weight = -2.5           # 垂直速度惩罚（防止弹跳）
        self.rewards.ang_vel_xy_l2.weight = -0.05         # 侧倾/俯仰角速度惩罚（保持机身水平）
        self.rewards.dof_torques_l2.weight = -2.0e-5      # 关节力矩惩罚（节能）
        self.rewards.dof_acc_l2.weight = -2.5e-7          # 关节加速度惩罚（减少冲击）
        self.rewards.action_rate_l2.weight = -0.01        # 腿部动作变化率惩罚
        
        # 步态相关奖励
        self.rewards.feet_air_time.weight = 0.0           # 单脚离地时间（关闭，由 F/R 分别控制）
        self.rewards.F_feet_air_time.weight = 1.0         # 前腿离地时间奖励（鼓励抬腿步态）
        self.rewards.R_feet_air_time.weight = 1.0         # 后腿离地时间奖励

        self.rewards.feet_height.weight = -0.0            # 脚掌离地高度（暂时关闭，TODO）
        self.rewards.feet_height_body.weight = -3.0       # 脚掌相对机身高度惩罚（防止腿部折叠）
        self.rewards.foot_contact.weight = 0.003          # 脚掌接触奖励（保持稳定支撑）
        self.rewards.hip_deviation.weight = -0.2          # 髋关节偏离默认值惩罚（维持站立姿态）
        self.rewards.joint_deviation.weight = -0.01       # 所有关节偏离默认值惩罚
        self.rewards.action_smoothness.weight = -0.02     # 整体动作平滑度惩罚
        self.rewards.height_reward.weight = -2.0          # 机身高度偏离惩罚（维持目标站立高度）
        self.rewards.flat_orientation_l2.weight = -1.0    # 机身姿态偏离水平面惩罚


# Go2ARMFlatEnvCfg_PLAY：推理/演示专用配置（对应任务名 Isaac-Go2Arm-Flat-Play）
# 在训练配置基础上做以下修改：
#   1. 关闭观测噪声和域随机化（复现训练时真实感知，保证策略稳定输出）
#   2. 固定指令范围（不再使用课程扩展，直接使用目标范围）
#   3. 减少环境数量（50 个，降低显存占用，方便可视化）
class Go2ARMFlatEnvCfg_PLAY(Go2ARMFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # 推理时只需少量环境可视化，减少显存占用
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # 关闭观测噪声注入（训练时用于增强鲁棒性，推理时应关闭）
        self.observations.policy.enable_corruption = False
        # 关闭随机外力干扰事件（推理时不需要抗扰动测试）
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # self.events.reset_base = None

        # 推理时关闭课程学习标志（is_Go2ARM=False 表示直接使用 ranges 而非 ranges_init/final）
        self.commands.ee_pose.is_Go2ARM = False
        self.commands.base_velocity.is_Go2ARM = False
        
        # 启用推理专用模式（is_Go2ARM_Play=True 会激活固定目标点可视化等推理辅助逻辑）
        self.commands.ee_pose.is_Go2ARM_Play = True
        
        # 速度指令：每 5 秒重采样一次，10% 概率为静止指令
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        
        # 推理阶段速度指令范围（固定，不再课程扩展）
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
       
        # 末端位姿指令：每 4 秒重采样一次，范围略小于训练最终范围（保证可视化效果）
        self.commands.ee_pose.resampling_time_range = (4.0, 4.0)
        self.commands.ee_pose.ranges.pos_x = (0.45, 0.6)
        self.commands.ee_pose.ranges.pos_y = (-0.25, 0.25)
        self.commands.ee_pose.ranges.pos_z = (0.2, 0.5)