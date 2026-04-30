import os
# 获取当前文件所在目录，用于拼接 USD 模型路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# go2_arm.usd 是包含 Unitree Go2 + WidowX 250s 机械臂的联合机器人模型文件
GO2ARM_USD = os.path.join(current_dir, "go2_arm.usd")

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

# =============================================================================
# GO2ARM_CFG: 机器人关节式铰接体的完整配置
#
# 机器人结构:
#   - 腿部: Unitree Go2 四足机器人，每条腿 3 自由度 (hip/thigh/calf)，共 12 关节
#   - 臂部: Interbotix WidowX 250s 6自由度机械臂
#           关节: waist(腰转) -> shoulder(肩) -> elbow(肘) ->
#                 forearm_roll(前臂转) -> wrist_angle(腕俯仰) -> wrist_rotate(腕旋转)
#
# 若要拟合消防水枪后坐力，可修改以下参数:
#   1. effort_limit / saturation_effort: 提高力矩上限以承受水枪反力
#   2. stiffness / damping: 调整关节刚度/阻尼以模拟水枪握持特性
#   3. init_state.joint_pos: 调整机械臂初始姿态
# =============================================================================
GO2ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2ARM_USD,
        # 启用接触传感器，用于脚部接触检测
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,         # 启用重力
            retain_accelerations=False,
            linear_damping=0.0,            # 线性阻尼（仿真稳定性用，设为0表示不额外衰减）
            angular_damping=0.0,           # 角阻尼
            max_linear_velocity=1000.0,    # 最大线速度限制（m/s）
            max_angular_velocity=1000.0,   # 最大角速度限制（rad/s）
            max_depenetration_velocity=1.0,# 碰撞穿透恢复最大速度
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,      # 启用自碰撞检测（腿和臂之间）
            solver_position_iteration_count=4, # 位置求解迭代次数（值越大越稳定但越慢）
            solver_velocity_iteration_count=0, # 速度求解迭代次数
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     collision_enabled=True,
        #     contact_offset=0.02,
        #     rest_offset=0.005 ,
        # ),
    ),
    # -------------------------------------------------------------------------
    # 初始状态：机器人在仿真开始/重置时的位置和关节角度
    # -------------------------------------------------------------------------
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),  # 初始质心高度约 0.35m（站立姿态）
        joint_pos={
            # --- 腿部初始关节角度 (弧度) ---
            ".*L_hip_joint": 0.1,       # 左侧髋关节（外展/内收）
            ".*R_hip_joint": -0.1,      # 右侧髋关节
            "F[L,R]_thigh_joint": 0.8,  # 前腿大腿关节
            "R[L,R]_thigh_joint": 1.0,  # 后腿大腿关节
            ".*_calf_joint": -1.5,      # 小腿关节（负值=弯曲）
            # --- 机械臂初始关节角度 (弧度) ---
            # 若换为消防水枪，此处需调整为水枪操作姿态
            "waist":0.0,          # 腰部旋转
            "shoulder":0.0,       # 肩关节
            "elbow":0.1,          # 肘关节
            "forearm_roll":-0.0,  # 前臂旋转
            "wrist_angle":-0.54,  # 腕部俯仰（约-31度，末端朝前偏下）
            "wrist_rotate":0.0,   # 腕部旋转
        },
        joint_vel={".*": 0.0},  # 所有关节初始速度为 0
    ),
    # 关节位置软限制因子：实际可用范围 = URDF限制 × 0.9，留出安全裕量
    soft_joint_pos_limit_factor=0.9,
    # -------------------------------------------------------------------------
    # 执行器配置 (Actuator)
    # DCMotorCfg 使用 PD 控制 + 力矩饱和模型:
    #   tau = clip(stiffness*(q_des - q) + damping*(dq_des - dq), -sat, +sat)
    #   最终力矩还会被 effort_limit 截断
    # -------------------------------------------------------------------------
    actuators={
        # --- 腿部执行器: 12个关节 (髋/大腿/小腿) ---
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=40.5,        # 最大输出力矩 (N·m)，Go2 电机峰值约 23.5N·m
            saturation_effort=23.5,   # 饱和力矩（超过此值后特性变软）
            velocity_limit=30.0,      # 最大关节速度 (rad/s)
            stiffness=40.0,           # PD 控制的 Kp（位置增益）
            damping=1.0,              # PD 控制的 Kd（速度增益/阻尼）
            friction=0.0,             # 关节摩擦（设为0，由域随机化添加）
        ),
        # --- 机械臂执行器: 6个关节 ---
        # 【水枪拟合关键参数区】
        # 若要拟合消防水枪后坐力（典型值: 喷水推力 50-200N，力臂约0.5m → 扭矩 25-100N·m）:
        #   - 增大 effort_limit 和 saturation_effort（建议 ≥ 30.0 N·m）
        #   - 增大 stiffness 以提高姿态维持能力
        #   - 后坐力本身可通过 apply_external_force_torque 事件或自定义执行器施加
        "widow_arm": DCMotorCfg(
            joint_names_expr=["waist","shoulder","forearm_roll",
                              "wrist_angle","wrist_rotate","elbow",
                              ],
            effort_limit=10.0,        # TODO: 当前值偏小，换水枪后需大幅提高
            saturation_effort=10.0,   # TODO: 同上
            velocity_limit=3.14,      # TODO: 约180度/s，可根据需求调整
            stiffness=10.0,           # 位置增益
            damping=0.5,              # 阻尼（影响动态响应，后坐力抑制需适当增大）
            friction=0.0,
        ),
    },
)

