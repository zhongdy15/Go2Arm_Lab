# =============================================================================
# go2arm_v3_articulation_cfg.py
# 基于 go2_arm_v3.usd 的机器人配置（去掉 elbow / forearm_roll / wrist_* 等关节，
# 只保留 12 条腿关节 + waist + shoulder，共 14 个 DOF）。
#
# v3 USD 中的执行器/刚体清单可参考 garbage/go2_arm_v3_info.txt
# =============================================================================
import os

# 当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
GO2ARM_V3_USD = os.path.join(current_dir, "go2_arm_v3.usd")

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg


# =============================================================================
# GO2ARM_V3_CFG
#   - 12 条腿关节: ".*_hip_joint" / ".*_thigh_joint" / ".*_calf_joint"
#   - 2  个臂关节: "waist", "shoulder"  （不进行实际任务训练，仅保持默认姿态）
# =============================================================================
GO2ARM_V3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO2ARM_V3_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            # --- 腿部关节默认角度（与原配置一致）---
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            # --- 仅剩的两个臂关节，保持中立姿态 ---
            "waist": 0.0,
            "shoulder": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 腿部 12 个关节
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=40.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        # 臂部 2 个关节（v3 仅保留 waist 与 shoulder）
        "widow_arm": DCMotorCfg(
            joint_names_expr=["waist", "shoulder"],
            effort_limit=10.0,
            saturation_effort=10.0,
            velocity_limit=3.14,
            stiffness=10.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
