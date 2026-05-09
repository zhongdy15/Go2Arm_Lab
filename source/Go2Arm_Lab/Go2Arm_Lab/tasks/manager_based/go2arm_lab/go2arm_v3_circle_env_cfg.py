# =============================================================================
# go2arm_v3_circle_env_cfg.py
#
# 简化版四足圆周运动训练环境（基于 go2_arm_v3.usd, 14 DOF）。
#
# 任务：让机器狗以恒定的前进速度 + 偏航角速度运动，从而绕一固定圆心做圆周运动；
#       机械臂仅保持默认姿态，不进行训练 / 跟踪。
#
# 与原 LocomotionVelocityEnvCfg 的主要差异：
#   1. 机器人换为 GO2ARM_V3_CFG（无 gripper_link / elbow / forearm / wrist 关节）
#   2. 移除 ee_pose 命令，仅保留 base_velocity（采用固定 ranges，不做课程）
#   3. 动作空间：12 腿 + 2 臂 (waist, shoulder) = 14 维
#   4. 观测：移除 ee_pose 命令观测、移除 priv_mass_ee 特权观测
#   5. 奖励：移除 end_effector_position/orientation 跟踪；保留 arm 动作平滑/微小
#            的关节偏离惩罚作为 end_effector_* 项（避免 arm 价值头退化）
#   6. 终止：移除 arm_contact（v3 上 *_link 仅含 shoulder_link / upper_arm_link）
# =============================================================================

import math

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
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import Go2Arm_Lab.tasks.manager_based.go2arm_lab.mdp as mdp
from Go2Arm_Lab.assets.go2arm_v3_articulation_cfg import GO2ARM_V3_CFG


##
# Scene
##

@configclass
class V3SceneCfg(InteractiveSceneCfg):
    """v3 圆周任务场景：平地 + 机器人 + 必要传感器。"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = GO2ARM_V3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 高度扫描仪（保留以兼容父类逻辑；本任务实际不使用其数据）
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# Events
##

@configclass
class V3EventCfg:
    """域随机化（精简）。注意：v3 没有 gripper_link，因此移除相关项。"""

    # startup
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

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

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
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # 平地任务不需要 push（通过 None 关闭 / 或在 PLAY 中关闭）
    push_robot = None


##
# Commands
##

@configclass
class V3CommandsCfg:
    """仅保留 base_velocity 命令；不使用课程，使用固定的 ranges 采样。

    设置 ang_vel_z 为正数，结合非零 lin_vel_x 即可让机器狗在世界坐标系中绕固定圆心
    做圆周运动，圆周半径 r = lin_vel_x / |ang_vel_z|。
    """

    base_velocity = mdp.command_cfg.UniformVelocityCommandCfg(
        asset_name="robot",
        # 单个 episode 长度内仅采样一次（episode_length_s = 20s）
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.0,
        heading_command=False,
        debug_vis=True,
        # 关闭课程，直接使用 ranges
        is_Go2ARM=False,
        is_Go2ARM_Flat=True,
        curriculum_coeff=1,
        ranges=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.3, 0.6),  # 始终逆时针圆周运动
            heading=(0.0, 0.0),
        ),
        # 课程占位（实际未使用）
        ranges_init=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.3, 0.6),
            heading=(0.0, 0.0),
        ),
        ranges_final=mdp.command_cfg.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.6),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.3, 0.6),
            heading=(0.0, 0.0),
        ),
    )


##
# Actions
##

@configclass
class V3ActionsCfg:
    """v3 共 14 DOF：12 腿 + 2 臂 (waist, shoulder)。"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        scale={
            "FR_hip_joint": 0.25, "FR_thigh_joint": 0.25, "FR_calf_joint": 0.25,
            "FL_hip_joint": 0.25, "FL_thigh_joint": 0.25, "FL_calf_joint": 0.25,
            "RR_hip_joint": 0.25, "RR_thigh_joint": 0.25, "RR_calf_joint": 0.25,
            "RL_hip_joint": 0.25, "RL_thigh_joint": 0.25, "RL_calf_joint": 0.25,
        },
        use_default_offset=True,
        preserve_order=True,
    )

    arm_pose = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["waist", "shoulder"],
        scale={"waist": 0.5, "shoulder": 0.5},
        use_default_offset=True,
        preserve_order=True,
    )


##
# Observations
##

@configclass
class V3ObservationsCfg:
    """观测：移除 ee_pose 与 priv_mass_ee；joint_pos / joint_vel / torques 改用 14 DOF 版本。"""

    @configclass
    class PolicyCfg(ObsGroup):
        # --- 本体感知（带历史）---
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, history_length=10, noise=Unoise(n_min=-0.0, n_max=0.0)
        )  # dim = 3
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_v3, history_length=10, noise=Unoise(n_min=-0.01, n_max=0.01)
        )  # dim = 14
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel_v3, history_length=10, noise=Unoise(n_min=-0.5, n_max=0.5)
        )  # dim = 14
        actions = ObsTerm(func=mdp.last_action, history_length=10)  # dim = 14
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            history_length=10,
            params={"command_name": "base_velocity"},
        )  # dim = 3
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            history_length=10,
        )  # dim = 3

        # --- 特权观测 ---
        priv_mass_base = ObsTerm(func=mdp.get_mass_base)  # dim = 1
        priv_joint_torques = ObsTerm(func=mdp.get_joints_torques_v3)  # dim = 14
        priv_base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # dim = 3
        priv_feet_contact = ObsTerm(
            func=mdp.feet_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )  # dim = 4

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Rewards
##

@configclass
class V3RewardsCfg:
    """奖励：腿部速度跟踪 + 稳定性；机械臂仅保留极小的姿态保持惩罚 (前缀 end_effector_)。

    end_effector_* 前缀的项会被 RewardManager 累计到 arm_reward_buf 中，
    使 PPO 的臂部价值头有非零的目标信号，避免梯度退化。
    """

    # ---- "机械臂" 奖励：仅保持默认姿态 / 抑制抖动 ----
    end_effector_arm_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist", "shoulder"])},
    )
    end_effector_action_rate = RewTerm(func=mdp.action_rate_l2_arm, weight=-0.005)
    end_effector_action_smoothness = RewTerm(func=mdp.arm_action_smoothness_penalty, weight=-0.02)

    # ---- 腿部速度跟踪（圆周运动的核心驱动）----
    tracking_lin_vel_x_l1 = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.2)},
    )

    # ---- 稳定性 / 平滑性 ----
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2_Go2, weight=-2.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2_Go2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_Go2, weight=-0.01)

    # ---- 步态 ----
    F_feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F.*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    R_feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R.*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": -0.2,
            "command_name": "base_velocity",
        },
    )

    foot_contact = RewTerm(
        func=mdp.standing_feet_contact_force,
        weight=0.003,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R.*_foot"),
            "command_name": "base_velocity",
            "force_threshold": 7.5,
            "command_threshold": 0.1,
        },
    )

    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"])},
    )

    action_smoothness = RewTerm(func=mdp.leg_action_smoothness_penalty, weight=-0.02)
    height_reward = RewTerm(func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.3})
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)


##
# Terminations
##

@configclass
class V3TerminationsCfg:
    """终止条件：超时、机体接触、腿部异常接触。"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 0.5},
    )
    thigh_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 0.5},
    )
    calf_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"), "threshold": 0.5},
    )


##
# Curriculum
##

@configclass
class V3CurriculumCfg:
    """关闭地形课程；仅保留 flat_ori / height 的简单调度，与原 flat 任务一致。"""

    flat_ori_modify = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "flat_orientation_l2", "num_steps": 2000, "weight": 0.0},
    )
    flat_height_modify = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "height_reward", "num_steps": 4000, "weight": -1.0},
    )


##
# Top-level env cfg
##

@configclass
class Go2ArmV3CircleEnvCfg(ManagerBasedRLEnvCfg):
    """Go2Arm v3 圆周运动训练环境（平地）。"""

    scene: V3SceneCfg = V3SceneCfg(num_envs=4096, env_spacing=2.5)
    observations: V3ObservationsCfg = V3ObservationsCfg()
    actions: V3ActionsCfg = V3ActionsCfg()
    commands: V3CommandsCfg = V3CommandsCfg()
    rewards: V3RewardsCfg = V3RewardsCfg()
    terminations: V3TerminationsCfg = V3TerminationsCfg()
    events: V3EventCfg = V3EventCfg()
    curriculum: V3CurriculumCfg = V3CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class Go2ArmV3CircleEnvCfg_PLAY(Go2ArmV3CircleEnvCfg):
    """推理 / 演示配置：减少环境数量，关闭噪声与扰动，固定速度命令。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # PLAY 模式下使用更窄的固定指令，便于观察一致的圆周运动
        self.commands.base_velocity.is_Go2ARM = False
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.5, 0.5)  # r = 1.0m 圆周


# =============================================================================
# 固定臂关节版本：waist 与 shoulder 在整个 episode 内保持给定常量值。
#
# 使用方式：
#   - 直接修改下方 fixed_waist / fixed_shoulder 的默认值即可；
#   - 或通过 Hydra 命令行覆盖：
#       python scripts/rsl_rl/train.py --task Isaac-Go2ArmV3-Circle-FixedArm \
#           --headless env.fixed_waist=0.3 env.fixed_shoulder=-0.2
#       python scripts/rsl_rl/play.py  --task Isaac-Go2ArmV3-Circle-FixedArm-Play \
#           --num_envs 1 env.fixed_waist=0.3 env.fixed_shoulder=-0.2
#
# 实现方式：
#   1. 在 __post_init__ 中用 .replace() 覆盖 GO2ARM_V3_CFG 的初始关节角度，
#      把 waist/shoulder 设为 fixed_waist/fixed_shoulder。
#   2. 把 arm_pose 动作的 scale 置为 0.0：动作 = default + 0 × policy_output = default，
#      使臂关节始终被 PD 控制器驱动到指定常量姿态。
#   3. reset_robot_joints 事件改为只随机化腿部 12 关节，避免重置时把臂关节随机化。
#   4. 移除 end_effector_arm_deviation 奖励（恒等于 0），减少不必要的计算。
# =============================================================================

@configclass
class Go2ArmV3CircleFixedArmEnvCfg(Go2ArmV3CircleEnvCfg):
    """臂关节 (waist / shoulder) 固定的圆周运动训练配置。"""

    # ---- 用户可调参数 ----
    fixed_waist: float = 0.0 #1.57 0.785 0.0 -0.758 -1.57 
    """waist 关节的固定角度 (rad)，建议范围与 USD 限制一致：[-π, π]"""

    fixed_shoulder: float = 0.0 #1.57 0.785 0.0 -0.758 -1.57
    """shoulder 关节的固定角度 (rad)，USD 限制约 [-1.88, 1.99]"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # 1) 覆盖机器人初始关节角度（waist/shoulder 改为用户指定值）
        robot_cfg = self.scene.robot
        new_joint_pos = dict(robot_cfg.init_state.joint_pos)
        new_joint_pos["waist"] = float(self.fixed_waist)
        new_joint_pos["shoulder"] = float(self.fixed_shoulder)
        self.scene.robot = robot_cfg.replace(
            init_state=robot_cfg.init_state.replace(joint_pos=new_joint_pos)
        )

        # 2) 关闭臂部动作输出（target = default + 0 * action = default 常量）
        self.actions.arm_pose.scale = {"waist": 0.0, "shoulder": 0.0}

        # 3) reset 时只随机化腿部 12 关节，避免把 waist/shoulder 抖出固定值。
        #    使用自定义 reset_joints_by_scale_filtered，绕过 IsaacLab 上游
        #    reset_joints_by_scale 在 joint_names 过滤时的 fancy-indexing bug。
        self.events.reset_robot_joints.func = mdp.reset_joints_by_scale_filtered
        self.events.reset_robot_joints.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_joint",
                    ".*_thigh_joint",
                    ".*_calf_joint",
                ],
            ),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        }

        # 4) 移除已恒等于 0 的臂关节偏离惩罚（保留动作平滑项以维持 arm 价值头有非零目标信号）
        self.rewards.end_effector_arm_deviation = None


@configclass
class Go2ArmV3CircleFixedArmEnvCfg_PLAY(Go2ArmV3CircleFixedArmEnvCfg):
    """固定臂关节圆周运动 - 推理 / 演示配置。

    可选：在 PLAY 模式下启用「灭火枪后坐力」事件，测试后坐力对圆周步态的影响。
    通过设置 ``recoil_force_magnitude > 0`` 启用（默认 0 即关闭）。

    可调超参（支持 Hydra CLI 覆盖）：
        env.recoil_force_magnitude  单发后坐力幅值 (N)，0 表示关闭
        env.recoil_shots_per_burst  一轮连发的弹数
        env.recoil_shot_interval_s  连发内相邻两发的时间间隔 (s)
        env.recoil_burst_cooldown_s 两轮连发之间的冷却时间 (s)

    用法示例：
        # 不开后坐力（保留原有圆周演示行为）
        python scripts/rsl_rl/play.py --task Isaac-Go2ArmV3-Circle-FixedArm-Play \
            --num_envs 1 --checkpoint /path/to/model.pt

        # 启用后坐力（80N 三连发，0.15s 间隔，3s 冷却）
        python scripts/rsl_rl/play.py --task Isaac-Go2ArmV3-Circle-FixedArm-Play \
            --num_envs 1 --checkpoint /path/to/model.pt \
            env.recoil_force_magnitude=80.0 env.recoil_shots_per_burst=3 \
            env.recoil_shot_interval_s=0.15 env.recoil_burst_cooldown_s=3.0
    """

    # ---- 后坐力超参（用户可调，0 表示关闭）----
    recoil_force_magnitude: float = 0.0
    recoil_shots_per_burst: int = 3
    recoil_shot_interval_s: float = 0.15
    recoil_burst_cooldown_s: float = 3.0

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.is_Go2ARM = False
        self.commands.base_velocity.resampling_time_range = (20.0, 20.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.5, 0.5)

        # PLAY 模式下：测试鲁棒性，机器狗倒了就让它倒下，不再因 base/thigh/calf 接触而 reset。
        self.terminations.base_contact = None
        self.terminations.thigh_contact = None
        self.terminations.calf_contact = None
        self.episode_length_s = 60.0  # 拉长 episode 便于长时间观察

        # 灭火枪后坐力事件（仅当 recoil_force_magnitude > 0 时启用）
        if float(self.recoil_force_magnitude) > 0.0:
            self.events.recoil_force = EventTerm(
                func=mdp.apply_recoil_burst,
                mode="interval",
                interval_range_s=(
                    float(self.recoil_shot_interval_s),
                    float(self.recoil_shot_interval_s),
                ),
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="upper_arm_link"),
                    "force_magnitude": float(self.recoil_force_magnitude),
                    "shots_per_burst": int(self.recoil_shots_per_burst),
                    "shot_interval_s": float(self.recoil_shot_interval_s),
                    "burst_cooldown_s": float(self.recoil_burst_cooldown_s),
                    "direction_local": (-1.0, 0.0, 0.0),  # body 局部 -X = 枪口反向
                },
            )


# =============================================================================
# 站立抗推 (Stand & Push) 变体：
#   - 继承 FixedArm 设定（waist / shoulder 固定）；
#   - base_velocity 命令恒为 0（站立不动）；
#   - 周期性向 base 施加随机水平速度扰动，模拟被推撞；
#   - 调整奖励权重：弱化步态相关奖励（feet_air_time / foot_contact），
#     强化姿态与高度保持，鼓励原地稳定站立。
#
# 用法示例：
#   python scripts/rsl_rl/train.py --task Isaac-Go2ArmV3-StandPush-FixedArm \
#       --headless env.fixed_waist=0.0 env.fixed_shoulder=0.0
#   python scripts/rsl_rl/play.py  --task Isaac-Go2ArmV3-StandPush-FixedArm-Play \
#       --num_envs 1 env.fixed_waist=0.0 env.fixed_shoulder=0.0
# =============================================================================

@configclass
class Go2ArmV3StandPushFixedArmEnvCfg(Go2ArmV3CircleFixedArmEnvCfg):
    """臂关节固定 + 原地站立抗推训练配置。"""

    # ---- 推撞参数（可通过 Hydra CLI 覆盖）----
    push_interval_s_min: float = 6.0
    """两次推撞之间的最小时间间隔 (s)。"""
    push_interval_s_max: float = 10.0
    """两次推撞之间的最大时间间隔 (s)。"""
    push_velocity_x: float = 0.8
    """x 方向推撞速度幅值 (m/s)，每次从 [-x, +x] 均匀采样。"""
    push_velocity_y: float = 0.8
    """y 方向推撞速度幅值 (m/s)，每次从 [-y, +y] 均匀采样。"""

    def __post_init__(self) -> None:
        super().__post_init__()

        # 1) 速度命令恒为 0（站立任务）
        zero_range = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = zero_range
        self.commands.base_velocity.ranges.lin_vel_y = zero_range
        self.commands.base_velocity.ranges.ang_vel_z = zero_range
        self.commands.base_velocity.ranges_init.lin_vel_x = zero_range
        self.commands.base_velocity.ranges_init.lin_vel_y = zero_range
        self.commands.base_velocity.ranges_init.ang_vel_z = zero_range
        self.commands.base_velocity.ranges_final.lin_vel_x = zero_range
        self.commands.base_velocity.ranges_final.lin_vel_y = zero_range
        self.commands.base_velocity.ranges_final.ang_vel_z = zero_range
        self.commands.base_velocity.rel_standing_envs = 1.0

        # 2) 启用周期性推撞事件（通过随机设定 base 线速度模拟外力扰动）
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(float(self.push_interval_s_min), float(self.push_interval_s_max)),
            params={
                "velocity_range": {
                    "x": (-float(self.push_velocity_x), float(self.push_velocity_x)),
                    "y": (-float(self.push_velocity_y), float(self.push_velocity_y)),
                },
            },
        )

        # 3) reset 时收紧本体扰动，使 episode 起始基本静止（仅小幅 yaw 与位置噪声）
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.5, 0.5)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }

        # 4) 调整奖励：站立任务下不再奖励抬脚 / 跨步
        if self.rewards.F_feet_air_time is not None:
            self.rewards.F_feet_air_time.weight = 0.0
        if self.rewards.R_feet_air_time is not None:
            self.rewards.R_feet_air_time.weight = 0.0
        if self.rewards.foot_contact is not None:
            self.rewards.foot_contact.weight = 0.0

        # 强化姿态保持（被推后能回到水平）
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.height_reward.weight = -3.0


@configclass
class Go2ArmV3StandPushFixedArmEnvCfg_PLAY(Go2ArmV3StandPushFixedArmEnvCfg):
    """臂关节固定 + 原地站立 - 推理 / 演示配置。

    PLAY 模式下：
      - 关闭 push_robot 推撞事件；
      - 启用「灭火枪后坐力」事件：在 upper_arm_link 上沿其本体 -X 方向
        周期性施加冲击力（连发模式）。

    可调超参（支持 Hydra CLI 覆盖）：
        env.recoil_force_magnitude  单发后坐力幅值 (N)
        env.recoil_shots_per_burst  一轮连发的弹数
        env.recoil_shot_interval_s  连发内相邻两发的时间间隔 (s)
        env.recoil_burst_cooldown_s 两轮连发之间的冷却时间 (s)

    用法示例：
        python scripts/rsl_rl/play.py --task Isaac-Go2ArmV3-StandPush-FixedArm-Play \
            --num_envs 1 --checkpoint /path/to/model.pt \
            env.fixed_waist=0.0 env.fixed_shoulder=0.0 \
            env.recoil_force_magnitude=120.0 env.recoil_shots_per_burst=5 \
            env.recoil_shot_interval_s=0.1 env.recoil_burst_cooldown_s=2.0
    """

    # ---- 后坐力超参（用户可调）----
    recoil_force_magnitude: float = 80.0
    recoil_shots_per_burst: int = 3
    recoil_shot_interval_s: float = 0.15
    recoil_burst_cooldown_s: float = 3.0

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None

        # 关闭推撞事件
        self.events.push_robot = None

        # 关闭"接触类"终止条件 + 拉长 episode：测试后坐力稳定性时，机器狗倒了就让它倒下，
        # 不再自动 reset 回站立姿态。time_out 仍然保留，episode 结束后会自然重置。
        self.terminations.base_contact = None
        self.terminations.thigh_contact = None
        self.terminations.calf_contact = None
        self.episode_length_s = 15.0  # 单局延长到 60 秒，便于观察长时间累计效果

        # 启用灭火枪后坐力事件
        self.events.recoil_force = EventTerm(
            func=mdp.apply_recoil_burst,
            mode="interval",
            interval_range_s=(
                float(self.recoil_shot_interval_s),
                float(self.recoil_shot_interval_s),
            ),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="upper_arm_link"),
                "force_magnitude": float(self.recoil_force_magnitude),
                "shots_per_burst": int(self.recoil_shots_per_burst),
                "shot_interval_s": float(self.recoil_shot_interval_s),
                "burst_cooldown_s": float(self.recoil_burst_cooldown_s),
                "direction_local": (-1.0, 0.0, 0.0),  # body 局部 -X = 枪口反向
            },
        )
