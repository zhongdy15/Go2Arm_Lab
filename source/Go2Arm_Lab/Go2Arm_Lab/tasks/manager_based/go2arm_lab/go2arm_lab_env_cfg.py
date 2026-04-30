# =============================================================================
# go2arm_lab_env_cfg.py
# 强化学习环境总配置文件
#
# 本文件采用 Isaac Lab 的 Manager-Based 框架，将环境分解为:
#   - MySceneCfg:       场景（地形、机器人、传感器、灯光）
#   - EventCfg:         随机化事件（域随机化 + 扰动）
#   - CommandsCfg:      指令生成（末端执行器目标位姿 + 底盘速度指令）
#   - ActionsCfg:       动作空间（腿部12关节 + 臂部6关节的目标位置）
#   - ObservationsCfg:  观测空间（本体感知 + 特权观测）
#   - RewardsCfg:       奖励函数（腿部运动 + 机械臂跟踪）
#   - TerminationsCfg:  终止条件（超时、机体碰撞等）
#   - CurriculumCfg:    课程学习（逐步扩大指令范围）
# =============================================================================
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.terrains as terrain_gen


import Go2Arm_Lab.tasks.manager_based.go2arm_lab.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##

# -----------------------------------------------------------------------------
# 地形生成器配置（用于粗糙地形训练）
# 混合使用平坦地形(30%)和随机起伏地形(70%)
# 地形课程学习：从低级别(平坦)逐步进阶到高级别(粗糙)
# -----------------------------------------------------------------------------
GO2ARM_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),          # 每个地形格子大小 8m×8m
    border_width=20.0,         # 边界宽度（防止机器人跑出地形）
    num_rows=10,               # 地形行数（难度级别数）
    num_cols=20,               # 地形列数（每级别的并行环境数）
    horizontal_scale=0.1,      # 水平分辨率 (m/格)
    vertical_scale=0.005,      # 垂直分辨率 (m/格)
    slope_threshold=0.75,      # 斜坡阈值（超过则视为垂直面）
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),  # 30%平坦地形
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.7, noise_range=(-0.05, 0.05), noise_step=0.01, border_width=0.25
        ),  # 70%随机起伏地形（高度噪声 ±5cm）
    },
)


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """场景配置：地形、机器人实体、传感器和光照"""

    # --- 地形 ---
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",           # 使用程序化地形生成器
        terrain_generator=GO2ARM_TERRAINS_CFG,
        max_init_terrain_level=5,           # 课程学习初始最高难度级别
        collision_group=-1,                 # -1 表示与所有物体碰撞
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,            # 静摩擦系数（域随机化会在此基础上扰动）
            dynamic_friction=1.0,           # 动摩擦系数
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # --- 机器人（由子类填充，例如 GO2ARM_CFG）---
    robot: ArticulationCfg = MISSING
    # --- 传感器 ---
    # 高度扫描仪：从机体正上方20m向下扫描，获取16×10=160个高度点
    # 用于地形感知（观测空间中的特权信息）
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 接触力传感器：检测所有刚体的接触状态，保留最近3帧历史
    # track_air_time=True 用于计算脚部离地时间（步态奖励需要）
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # --- 光照 ---
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """
    事件配置（域随机化 Domain Randomization）
    域随机化是 sim-to-real 迁移的关键，通过在训练时随机化物理参数，
    提高策略对真实世界不确定性的鲁棒性。
    
    事件分三类:
      - startup:  仿真启动时执行一次（随机化惯性参数）
      - reset:    每次 episode 重置时执行（随机化执行器增益、初始状态）
      - interval: 按时间间隔随机触发（推力扰动）
    """

    # ---- startup 事件（仿真启动时执行一次）----

    # 随机化所有刚体的摩擦系数（静摩擦 0.5~4.0，动摩擦 0.5~2.0）
    # 模拟不同地面材质（沙地、湿滑地板等）
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 4.0),
            "dynamic_friction_range": (0.5, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 随机化机体质量（±3kg），模拟携带不同载荷
    # 若换成消防水枪，可将此范围增大以模拟水枪充水/排水的质量变化
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )

    # base_com = EventTerm(
    #     func=mdp.randomize_rigid_body_com,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
    #     },
    # )

    # 随机化末端执行器（gripper_link）质量（-0.1~+0.5kg）
    # 若换成消防水枪，可增大上限以模拟水枪头部重量变化
    add_ee_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
            "mass_distribution_params": (-0.1, 0.5),
            "operation": "add",
        },
    )

    # ---- reset 事件（每个 episode 重置时执行）----

    # 向机体施加外部力和力矩（当前为 0，可用于模拟消防水枪后坐力的静态分量）
    # 【水枪后坐力拟合入口】: 将 force_range 设为 (-F_recoil, F_recoil) 可模拟
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # 重置机器人根部位姿到随机初始状态（位置±0.5m，偏航随机360°）
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    
    # 随机化执行器的刚度和阻尼（×0.8~1.2），模拟电机特性差异
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),  # 关节初始角度在默认值的 50%~150% 范围内随机
            "velocity_range": (0.0, 0.0),
        },
    )

    # ---- interval 事件（按时间间隔随机触发）----

    # 每隔 10~15 秒向机器人施加随机速度扰动（模拟被推撞）
    # 平地训练时会禁用此项（flat_env_cfg.py 中设置为 None）
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

##
# MDP settings
##

@configclass
class CommandsCfg:
    """
    指令配置（目标任务的描述）
    
    包含两类指令:
    1. ee_pose: 末端执行器目标位姿（7维：xyz + 四元数）
       - 在机器人底盘坐标系下定义目标位置
       - 课程学习: 初始阶段目标在正前方近处，逐步扩大到更远、更大范围
    2. base_velocity: 底盘目标速度（3维：vx, vy, wz）
       - 课程学习: 从慢速直行开始，逐步引入横移和转向
    """
    # --- 末端执行器位姿指令（用于机械臂控制）---
    ee_pose = mdp.command_cfg.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="gripper_link",       # 跟踪目标：夹爪连杆
        resampling_time_range=(6.0,8.0),# 每 6~8 秒重新采样一个新目标位姿
        debug_vis=True,                 # 在仿真中可视化目标位姿标记
        is_Go2ARM=True,                 # 启用课程学习模式
        curriculum_coeff = 1000,        # 课程系数（控制目标范围扩展速度）
        # 课程终态: 末端可达的最终范围（相对底盘坐标系，z 为世界坐标系高度）
        ranges_final =mdp.command_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),            # 前方 40~60cm
            pos_y=(-0.35, 0.35),         # 左右各 35cm
            pos_z=(0.1, 0.55),           # 高度 10~55cm（世界坐标系）
            roll=(-0.0, 0.0),
            pitch=(-3.14 / 9, 3.14 / 9),# ±20° 俯仰
            yaw=(-3.14 / 9, 3.14 / 9),  # ±20° 偏航
        ),
        ranges = mdp.command_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6),
            pos_y=(-0.35, 0.35),
            pos_z=(0.1, 0.55),
            roll=(-0.0, 0.0),
            pitch=(-3.14 / 9, 3.14 / 9),
            yaw=(-3.14 / 9, 3.14 / 9),
        ),
        # 课程初态: 训练初期目标在正前方固定位置（容易完成）
        ranges_init=mdp.command_cfg.UniformPoseCommandCfg.Ranges(
            pos_x=(0.45, 0.5),
            pos_y=(-0.05, 0.05),
            pos_z=(0.35, 0.4),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-0.0, 0.0),
        ),
    )

    # --- 底盘速度指令（用于腿部运动控制）---
    base_velocity = mdp.command_cfg.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0), # 每 10 秒重新采样速度指令
        rel_standing_envs=0.1,              # 10% 的环境保持静止（训练站立稳定性）
        debug_vis=True,
        is_Go2ARM=True,
        curriculum_coeff= 1000,
        ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.2, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5),heading=(-0.0, 0.0)
        ),
        # 课程终态: 最终速度范围
        ranges_final=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.8), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5),heading=(-0.0, 0.0)
        ),
        # 课程初态: 训练初期只走慢速直线
        ranges_init=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.35), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1),heading=(-0.0, 0.0)
        ),
    )

@configclass
class ActionsCfg:
    """
    动作空间配置
    
    总动作维度: 12（腿）+ 6（臂）= 18 维
    动作类型: 关节位置增量（相对默认姿态的偏移量）
    动作 = 默认关节角 + scale × 网络输出
    """
    # --- 腿部动作: 12 个关节的目标位置 ---
    # scale=0.25 表示最大偏移 ±0.25 rad（约 ±14°），防止步态过激
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", 
                                           joint_names=[
                                                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                                                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                                                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                                                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                                                    ],
                                           scale = {"FR_hip_joint": 0.25, "FR_thigh_joint": 0.25, "FR_calf_joint": 0.25,
                                                    "FL_hip_joint": 0.25, "FL_thigh_joint": 0.25, "FL_calf_joint": 0.25,
                                                    "RR_hip_joint": 0.25, "RR_thigh_joint": 0.25, "RR_calf_joint": 0.25,
                                                    "RL_hip_joint": 0.25, "RL_thigh_joint": 0.25, "RL_calf_joint": 0.25,}, 
                                         use_default_offset=True,
                                         preserve_order=True,
    )   
    arm_pose = mdp.JointPositionActionCfg(asset_name="robot",
                                          joint_names=[
                                              "waist", "shoulder", "elbow", 
                                              "forearm_roll", "wrist_angle", "wrist_rotate"],
                                           scale = {"waist":        0.5, # 0.8
                                                    "shoulder":     0.5, # 0.35
                                                    "elbow":        0.5, # 0.35
                                                    "forearm_roll": 0.5, # 0.35
                                                    "wrist_angle":  0.5, # 0.35
                                                    "wrist_rotate": 0.5}, # 0.35
                                            use_default_offset=True,
                                            preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """
    观测空间配置
    
    采用"本体感知 + 特权观测"的师生学习框架:
    - 普通观测 (无 priv_ 前缀): 真实机器人上可获取的传感器信息
      每项都带有 history_length=10，意味着堆叠最近10步，提供时序信息
    - 特权观测 (priv_ 前缀): 仿真中可获取但真实中无法直接测量的信息
      训练时教师网络使用，推理时由历史编码器(StateHistoryEncoder)估计
    
    观测维度（单步 × 10 步历史）:
      base_ang_vel:        3 × 10 = 30   机体角速度（IMU）
      joint_pos:          18 × 10 = 180  关节位置相对默认值
      joint_vel:          18 × 10 = 180  关节速度相对默认值
      actions:            18 × 10 = 180  上一步动作（作为历史）
      velocity_commands:   3 × 10 = 30   速度指令
      Go2_pose_command:    7 × 10 = 70   末端位姿指令（xyz+四元数）
      projected_gravity:   3 × 10 = 30   重力在机体系的投影（姿态感知）
      ------
      特权观测（不乘历史）:
      priv_mass_base:      1             机体质量偏差
      priv_mass_ee:        1             末端质量偏差
      priv_joint_torques: 18             实际关节力矩
      priv_base_lin_vel:   3             机体线速度（通常不可直接测量）
      priv_feet_contact:   4             4个脚部接触状态（bool）
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测组（输入到神经网络）"""
        # --- 本体感知观测（真实机器人可获取）---
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10,noise=Unoise(n_min=-0.0, n_max=0.0))  # dim = 3
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=10,noise=Unoise(n_min=-0.01, n_max=0.01)) # dim = 18
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=10, noise=Unoise(n_min=-0.5, n_max=0.5)) # dim = 18
        actions = ObsTerm(func=mdp.last_action, history_length=10) # dim = 18
        velocity_commands = ObsTerm(func=mdp.generated_commands, history_length=10,
                                    params={"command_name": "base_velocity"}) # dim = 3
        Go2_pose_command = ObsTerm(func=mdp.generated_commands, history_length=10,
                                   params={"command_name": "ee_pose"}) # dim = 7
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            history_length=10
        )        # dim = 3
        
        # --- 特权观测（仅训练时用，前缀必须为 "priv_"）---
        # ActorCritic 代码会自动识别 priv_ 前缀并分离处理
        priv_mass_base = ObsTerm(func=mdp.get_mass_base)     # dim = 1，机体质量偏差
        priv_mass_ee = ObsTerm(func=mdp.get_mass_ee)         # dim = 1，末端质量偏差
        priv_joint_torques = ObsTerm(func=mdp.get_joints_torques) # dim = 18，实际关节力矩
        priv_base_lin_vel = ObsTerm(func=mdp.base_lin_vel)   # dim = 3，机体线速度
        priv_feet_contact = ObsTerm(func=mdp.feet_contact,
                               params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")}) # dim = 4 bool
        # 可以在此添加更多特权观测，例如:
        # priv_xxx = xxx
        
        def __post_init__(self):
            self.enable_corruption = True   # 启用观测噪声（上面 noise= 参数生效）
            self.concatenate_terms = True   # 将所有观测拼接为单一向量
        
    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """
    奖励函数配置
    
    奖励分两大类:
    1. 机械臂任务奖励 (end_effector_ 前缀):
       - 位置跟踪：末端到目标的指数距离奖励（正值，越近越大）
       - 姿态跟踪：末端朝向误差惩罚（负值）
       - 动作平滑：防止抖动（负值）
    2. 腿部运动奖励:
       - 速度跟踪：跟随速度指令的奖励
       - 稳定性惩罚：抑制上下颠簸、横向倾斜等
       - 步态奖励：鼓励正确的足部离地时序
    
    注意: 带 "end_effector_" 前缀的奖励项由 on_policy_runner 用于课程学习判断
    """

    # -- 机械臂奖励 --
    # 【名称必须有 "end_effector_" 前缀，用于课程学习进度判断】
    # 末端位置跟踪：exp(-|pos_err|/std)，std=0.2m 时误差<20cm 奖励显著
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error_exp,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"),
                "command_name": "ee_pose",
                "std": 0.2},
    )

    # 末端姿态跟踪：四元数误差惩罚（负权重）
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="gripper_link"), 
                "command_name": "ee_pose"},
    )

    end_effector_action_rate = RewTerm(func=mdp.action_rate_l2_arm, weight=-0.005)

    end_effector_action_smoothness = RewTerm(func=mdp.arm_action_smoothness_penalty, weight=-0.02)

    # more rewards
    # end_effector_xxx = xxx


    # -- LEG
    tracking_lin_vel_x_l1 = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=1.5, 
        params={
                "command_name": "base_velocity", 
                "std":0.2}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=1.5,
         params={ 
                 "command_name": "base_velocity", 
                 "std": math.sqrt(0.2)}
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02) # -0.05
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2_Go2, weight=-2.0e-5) # - 0.0002
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2_Go2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_Go2, weight=-0.01)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight= 0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    F_feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight= 0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F.*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    R_feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight= 2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R.*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": 0.08,
            "command_name": "base_velocity",
        },
    )


    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": -0.2,
            "command_name": "base_velocity",
        },
    )

    foot_contact = RewTerm(
        func=mdp.standing_feet_contact_force,
        weight= 0.003,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R.*_foot"),
            "command_name": "base_velocity",
            "force_threshold": 7.5,
            "command_threshold": 0.1,
        },
    )

    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        # params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )

    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"])},
    )

    action_smoothness = RewTerm(
        func=mdp.leg_action_smoothness_penalty,
        weight=-0.02,
    )

    height_reward = RewTerm(func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.3})

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
   
   
    # thigh_contact = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-2.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 0.5},
    # )

    # calf_contact = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-2.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 0.5},
    # )

    # arm_contact = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-2.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link"), "threshold": 0.5},
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 0.5},
    )
    thigh_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold":0.5},
    )
    arm_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_link"), "threshold": 0.5},
    )
    calf_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 0.5},
    )



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    flat_ori_modify = CurrTerm(func=mdp.modify_reward_weight,
                               params={"term_name": "flat_orientation_l2",
                                       "num_steps": 2000,
                                       "weight": -0.00})

    flat_height_modify = CurrTerm(func=mdp.modify_reward_weight,
                               params={"term_name": "height_reward",
                                       "num_steps": 4000,
                                       "weight": -1.00})
    
##
# Environment configuration
##

@configclass
class LocomotionVelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

