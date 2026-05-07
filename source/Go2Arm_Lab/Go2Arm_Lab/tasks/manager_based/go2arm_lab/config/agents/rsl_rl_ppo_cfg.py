from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from dataclasses import MISSING
from typing import Literal
from isaaclab.utils import configclass

# =============================================================================
# rsl_rl_ppo_cfg.py - PPO 算法和网络结构配置
#
# 本项目采用"双头Actor + 师生蒸馏"的网络架构:
#
# Actor 网络:
#   输入: 本体感知观测 (prop) + 特权信息编码 (训练时) 或 历史编码 (推理时)
#   主干: Backbone MLP (256维)
#   双头输出:
#     腿部控制头: [256, 128] → 12维关节位置偏移
#     臂部控制头: [256, 128] → 6维关节位置偏移
#
# Critic 网络:
#   输入: 本体感知 + 特权观测 (仅训练时)
#   双头: 腿部价值头 [256,128,64] + 臂部价值头 [256,128,64]
#
# 特权信息处理:
#   训练时: priv_encoder [num_priv → 32 → 18] 直接编码特权观测
#   推理时: StateHistoryEncoder (1D-Conv) 从历史观测估计特权信息
#   切换机制: mixing_schedule 控制从特权→历史的渐进混合
# =============================================================================

@configclass
class Go2ArmRslRlPpoActorCriticCfg(RslRlPpoActorCriticCfg):
    activation_out: str = MISSING
    """输出层激活函数（通常用 'elu' 或 'tanh'）"""

    leg_control_head_hidden_dims : list[int] = MISSING
    """腿部控制头的隐藏层维度，例如 [256, 128]"""

    arm_control_head_hidden_dims : list[int] = MISSING
    """臂部控制头的隐藏层维度"""

    critic_leg_control_head_hidden_dims : list[int] = MISSING
    """Critic 腿部头的隐藏层维度"""

    critic_arm_control_head_hidden_dims : list[int] = MISSING
    """Critic 臂部头的隐藏层维度"""

    priv_encoder_dims: list[int] = MISSING
    """特权信息编码器维度，例如 [32, 18]（输入→压缩为18维潜变量）"""    

    num_leg_actions : int = MISSING
    """腿部动作维度 = 12"""

    num_arm_actions : int = MISSING
    """臂部动作维度 = 6"""

    init_noise_std : float = MISSING
    """动作分布初始标准差（较大值→初期探索多）"""

@configclass
class Go2ArmRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    dagger_update_freq : int = MISSING
    """DAgger 更新频率（每 N 步更新一次历史编码器）"""

    priv_reg_coef_schedual: list = MISSING
    """特权正则化系数调度表 [起始值, 终值, 起始步数, 终止步数]
    控制训练时特权观测对历史编码器的监督强度"""

    mixing_schedule : list = MISSING
    """特权/历史混合系数调度表 [起始混合比, 终混合比, 步数]
    1.0→0.0 表示从完全使用特权到完全使用历史估计（实现 sim-to-real）"""

    eps : float = MISSING
    """数值稳定性 epsilon"""

    min_policy_std: list | None = None
    """策略动作分布的最小标准差下限。形状须与总动作维度匹配 (1, num_actions)。
    若为 None，则使用 PPO 内部默认值（对应 18 DOF 任务）。"""


@configclass
class Go2ArmRslRlOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    policy: Go2ArmRslRlPpoActorCriticCfg = MISSING
    """策略网络配置"""

    algorithm: Go2ArmRslRlPpoAlgorithmCfg = MISSING
    """PPO 算法配置"""


@configclass
class Go2ArmFlatPPORunnerCfg(Go2ArmRslRlOnPolicyRunnerCfg):
    """平坦地形训练的 PPO 运行器配置"""
    num_steps_per_env = 24          # 每个 env 每次收集 24 步数据再更新
    max_iterations = 15000          # 最大训练迭代次数（约 15000 × 4096 env × 24步）
    save_interval = 1000            # 每 1000 次迭代保存一次模型
    experiment_name = "unitree_Go2arm_flat"
    empirical_normalization = False
    policy = Go2ArmRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256],           # Actor 主干 MLP（单层256）
        critic_hidden_dims=[256],          # Critic 主干 MLP
        activation="elu",
        activation_out="elu",
        leg_control_head_hidden_dims = [256, 128],         # 腿部头
        arm_control_head_hidden_dims = [256, 128],         # 臂部头
        critic_leg_control_head_hidden_dims = [256, 128, 64],  # Critic 腿部头
        critic_arm_control_head_hidden_dims = [256, 128, 64],  # Critic 臂部头
        priv_encoder_dims = [32, 18],      # 特权编码器: num_priv → 32 → 18
        num_leg_actions = 12,
        num_arm_actions = 6,
    )
    
    algorithm = Go2ArmRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,          # 价值损失权重
        use_clipped_value_loss=True,  # 使用截断价值损失（PPO 稳定性）
        clip_param=0.2,               # PPO 截断参数 ε
        entropy_coef=0.005,           # 熵正则化系数（鼓励探索）
        num_learning_epochs=5,        # 每批数据学习 5 轮
        num_mini_batches=4,           # 每轮分 4 个 mini-batch
        learning_rate=1e-3,           # 初始学习率
        schedule="adaptive",          # 学习率自适应调整（基于 KL 散度）
        gamma=0.99,                   # 折扣因子
        lam=0.95,                     # GAE λ（广义优势估计）
        desired_kl=0.01,              # 目标 KL 散度（用于自适应学习率）
        max_grad_norm=1.0,            # 梯度裁剪上限
        dagger_update_freq = 20,      # 每 20 步更新一次历史编码器
        priv_reg_coef_schedual = [0, 0.1, 1500, 5000],   # 特权正则化: 0步→0, 1500步→0.1, 5000步→保持
        mixing_schedule=[1.0, 0, 4000],   # 混合调度: 0步→完全用特权, 4000步→完全用历史
        eps = 1e-5,
    )
 
@configclass
class Go2ArmRoughPPORunnerCfg(Go2ArmRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "unitree_Go2arm_rough"
    empirical_normalization = False
    policy = Go2ArmRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256],
        critic_hidden_dims=[256],
        activation="elu",
        activation_out="elu",
        leg_control_head_hidden_dims = [256, 128],
        arm_control_head_hidden_dims = [256, 128],
        critic_leg_control_head_hidden_dims = [256, 128, 64],
        critic_arm_control_head_hidden_dims = [256, 128, 64],
        priv_encoder_dims = [32, 18],
        num_leg_actions = 12,
        num_arm_actions = 6,
    )

    algorithm = Go2ArmRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        dagger_update_freq = 20,
        priv_reg_coef_schedual = [0, 0.1, 1500, 4000],
        mixing_schedule=[1.0, 0, 3000] ,
        eps = 1e-5,
    )


# =============================================================================
# Go2ArmV3 圆周运动任务专用 PPO 配置
#   - 14 DOF (12 腿 + 2 臂)
#   - num_leg_actions = 12, num_arm_actions = 2
# =============================================================================
@configclass
class Go2ArmV3CirclePPORunnerCfg(Go2ArmRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "unitree_Go2arm_v3_circle"
    empirical_normalization = False
    policy = Go2ArmRslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256],
        critic_hidden_dims=[256],
        activation="elu",
        activation_out="elu",
        leg_control_head_hidden_dims=[256, 128],
        arm_control_head_hidden_dims=[256, 128],
        critic_leg_control_head_hidden_dims=[256, 128, 64],
        critic_arm_control_head_hidden_dims=[256, 128, 64],
        priv_encoder_dims=[32, 18],
        num_leg_actions=12,
        num_arm_actions=2,  # v3 仅有 waist + shoulder
    )

    algorithm = Go2ArmRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        dagger_update_freq=20,
        priv_reg_coef_schedual=[0, 0.1, 1500, 4000],
        mixing_schedule=[1.0, 0, 3000],
        eps=1e-5,
        # v3 共 14 DOF：12 腿（hip/thigh/calf 各 4） + 2 臂 (waist, shoulder)
        # 形状必须为 (1, num_actions) 与策略 std 一致
        min_policy_std=[[0.15, 0.25, 0.25] * 4 + [0.2] * 2],
    )
