# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# train.py - 强化学习训练入口脚本
#
# 运行命令:
#   python scripts/rsl_rl/train.py --task Isaac-Go2Arm-Flat --headless
#
# 运行流程:
#   1. 解析命令行参数（任务名、环境数量、最大迭代次数等）
#   2. 启动 Isaac Sim 仿真器（AppLauncher 负责此步）
#   3. @hydra_task_config 装饰器自动加载对应的环境配置和PPO算法配置
#   4. gym.make 创建并行仿真环境（IsaacLab ManagerBasedRLEnv）
#   5. RslRlVecEnvWrapper 将环境包装成 RSL-RL 期望的接口格式
#   6. OnPolicyRunner 初始化 ActorCritic 网络和 PPO 优化器
#   7. runner.learn 开始训练循环（采集 → 计算优势 → PPO更新 → 保存）
# =============================================================================

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# ---- 第一阶段：解析命令行参数（Isaac Sim 启动前完成）----
# 注意：Isaac Sim 必须在所有 isaaclab/torch 导入之前先启动，因此参数解析在最前面
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")  # 并行环境数量（覆盖cfg中的默认值）
parser.add_argument("--task", type=str, default=None, help="Name of the task.")  # 例: Isaac-Go2Arm-Flat
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")  # 覆盖 cfg 中的 max_iterations=15000
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# 追加 RSL-RL 专用参数（resume、load_run、load_checkpoint 等，见 cli_args.py）
cli_args.add_rsl_rl_args(parser)
# 追加 AppLauncher 参数（--headless、--device 等）
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 录制视频时必须启用相机渲染
if args_cli.video:
    args_cli.enable_cameras = True

# 清理 sys.argv，避免 Hydra 配置系统受到干扰
sys.argv = [sys.argv[0]] + hydra_args

# ---- 第二阶段：启动 Isaac Sim 仿真器 ----
# AppLauncher 负责初始化 Omniverse/PhysX 引擎；--headless 表示无图形界面（提速）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 第三阶段：Isaac Sim 启动后，才能导入 torch / isaaclab 相关模块 ----
# 检查 RSL-RL 版本（多GPU分布式训练时要求 >= 2.3.1）

import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

import gymnasium as gym
import os
import torch
from datetime import datetime

# OnPolicyRunner: 负责训练循环（采集rollout、计算GAE优势、执行PPO更新、保存checkpoint）
from local_rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

# RslRlVecEnvWrapper: 将 IsaacLab gym 环境包装为 RSL-RL 期望的向量化接口
# 主要工作：拆分 obs 中的本体感知/特权/历史部分，转发 step/reset
from local_rsl_rl.wrappers import RslRlVecEnvWrapper
# Go2ArmRslRlOnPolicyRunnerCfg: 读取 rsl_rl_ppo_cfg.py 中定义的 PPO + 网络配置
from Go2Arm_Lab.tasks.manager_based.go2arm_lab.config.agents import Go2ArmRslRlOnPolicyRunnerCfg

# 触发 isaaclab_tasks 和 Go2Arm_Lab.tasks 的任务注册（将任务名映射到 cfg 类）
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import Go2Arm_Lab.tasks  # noqa: F401

# 允许 TF32 加速矩阵运算（A100/3090 等 Ampere 架构 GPU 有效）
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# @hydra_task_config 装饰器作用:
#   根据 --task Isaac-Go2Arm-Flat 自动查找并加载对应的:
#     env_cfg  → Go2ARMFlatEnvCfg（flat_env_cfg.py）
#     agent_cfg → Go2ArmFlatPPORunnerCfg（rsl_rl_ppo_cfg.py）
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: Go2ArmRslRlOnPolicyRunnerCfg):
    """训练主函数（由 @hydra_task_config 注入 env_cfg 和 agent_cfg）"""
    # ---- 步骤1：用命令行参数覆盖配置文件中的默认值 ----
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # 设置随机种子（影响域随机化的采样结果，保证可复现）
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # 多GPU分布式训练：每个进程绑定自己的 GPU，并使用不同种子
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # ---- 步骤2：确定日志保存路径 ----
    # 结构: logs/rsl_rl/unitree_Go2arm_flat/{时间戳}/
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # ---- 步骤3：创建 Isaac Lab 仿真环境 ----
    # gym.make 内部会:
    #   a. 实例化场景（MySceneCfg）：加载地形、生成机器人 USD、初始化传感器
    #   b. 初始化所有 Manager（ObsManager、RewardManager、EventManager 等）
    #   c. 重置所有并行环境到初始状态
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # 如果需要续训或使用 Distillation 算法，提前找到 checkpoint 路径
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # 可选：用 gym.RecordVideo 包装以定期录制训练视频
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ---- 步骤4：包装环境为 RSL-RL 接口 ----
    # RslRlVecEnvWrapper 主要工作:
    #   - 将 obs 字典拆分为 (prop_obs, priv_obs, hist_obs) 三部分
    #   - clip_actions: 将动作裁剪到 [-1, 1]（可选）
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---- 步骤5：创建 OnPolicyRunner（核心训练对象）----
    # 内部会初始化:
    #   - ActorCritic 网络（双头 + 历史编码器，见 actor_critic.py）
    #   - PPO 优化器
    #   - RolloutStorage（存储采集到的轨迹数据）
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)  # 记录当前 git 状态到日志
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)  # 加载已有模型权重继续训练

    # 将完整配置保存到日志目录（便于后续复现）
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # ---- 步骤6：开始训练循环 ----
    # runner.learn 内部每次迭代:
    #   a. 采集: 运行 num_steps_per_env=24 步，存入 RolloutStorage
    #   b. 计算: GAE 广义优势估计
    #   c. 更新: num_learning_epochs=5 轮，每轮分 4 个 mini-batch 做 PPO
    #   d. 课程: 根据末端追踪成功率扩展指令范围
    #   e. 保存: 每 save_interval=1000 次迭代保存一次 checkpoint
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
