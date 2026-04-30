# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =============================================================================
# play.py - 训练好的策略推理入口脚本
#
# 运行命令:
#   python scripts/rsl_rl/play.py --task Isaac-Go2Arm-Flat-Play --num_envs 1
#
# 与 train.py 的主要区别:
#   - 不做梯度更新，只做前向推理 (torch.inference_mode)
#   - 关闭域随机化（play cfg 中设置）
#   - 使用历史编码器替代特权观测（hist_encoding=True）
#   - 自动查找 logs/ 中最新的 checkpoint 加载
#
# 运行流程:
#   1. 启动仿真器
#   2. 加载 checkpoint 并提取推理策略
#   3. 将策略导出为 TorchScript (.pt) 与真实机器人部署对齐
#   4. 进入主推理循环：重排历史观测顺序 → 策略输出动作 → 仿真步进
# =============================================================================

"""Launch Isaac Sim Simulator first."""
import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# ---- 第一阶段：解析命令行参数 ----
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")  # 推理时通常设为 1
parser.add_argument("--task", type=str, default=None, help="Name of the task.")  # 例: Isaac-Go2Arm-Flat-Play
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# 追加 RSL-RL 专用参数（--checkpoint 可手动指定模型文件）
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# ---- 第二阶段：启动 Isaac Sim ----
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 第三阶段：导入训练相关模块 ----
import gymnasium as gym
import os
import time
import torch

from local_rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

# export_policy_as_jit: 将 PyTorch 网络导出为 TorchScript（.pt）
# TorchScript 格式可在无 Python 环境（如嵌入式系统）上运行
from isaaclab_rl.rsl_rl import export_policy_as_jit  # onnx not avaliable

# 同 train.py，使用项目定制的 Wrapper 和 Runner
from local_rsl_rl.wrappers import RslRlVecEnvWrapper
from Go2Arm_Lab.tasks.manager_based.go2arm_lab.config.agents import Go2ArmRslRlOnPolicyRunnerCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import numpy as np

import Go2Arm_Lab.tasks  # noqa: F401


def prepare_obs(env):
    """
    预处理观测索引映射表（只调用一次）

    问题背景:
        IsaacLab 的历史观测内部按“新到旧”顺序堆叠：
            ang_vel_t10, ang_vel_t9, ..., ang_vel_t1,
            joint_pos_t10, ..., joint_pos_t1, ...
        但神经网络输入期望的是“时序分组”顺序:
            [ang_vel_t1, joint_pos_t1, ...], [ang_vel_t2, ...], ..., [ang_vel_t10, ...]

    本函数返回一个索引数组 total[T, num_prop]，
    供后续的 change_obs_order 用来实时改变观测顺序。

        Example:
            In env.step, the original observation follows right-to-left order, after modification it becomes left-to-right

            For example, the original observation is:
                obs = ang_vel(3) * 10(num_history) + joint_pos(18) * 10(num_history) + joint_vel(18) * 10(num_history)
            After env.step, the observation (obs) becomes structured as follows:
                obs_old = ang_vel_timestep_10 -> ang_vel_timestep_1, joint_pos_timestep_10 -> joint_pos_timestep_1, joint_vel_timestep_10 -> joint_vel_timestep_1
            We need to modify the observation order to:
                obs_new = ang_vel_timestep_1, joint_pos_timestep_1, joint_vel_timestep_1 -> ang_vel_timestep_10, joint_pos_timestep_10, joint_vel_timestep_10
    """
    total = np.zeros((env.num_history, env.num_prop)) 
    obs_new = torch.zeros(env.num_envs, env.num_prop * env.num_history).to(env.device)
    
    lst, length = env.get_obs_list_length()
    lst = [item for item in lst if not item.startswith("policy-priv_")]

    result_dict = {}
    for i in range(len(lst)):
        c = np.array(list(range( sum(length[: i + 1 ]) - int(length[i] / env.num_history), sum(length[: i + 1]) )))
        result_dict[lst[i]] = c

    key_list = list(result_dict.keys())
    a1_list = []
    for i in range(env.num_history):
        for j in range(len(lst)):
            a1 = np.concatenate([
                result_dict[key_list[j]] - (i) * result_dict[key_list[j]].shape[0]])
            a1_list.append(a1)
            if j == len(lst) - 1:
                a1_list = np.concatenate(a1_list)
                total[i, :] = a1_list
                a1_list = []
    return total, obs_new

def change_obs_order(obs, obs_new, total, env):

    """
    实时改变观测顺序（每步调用）

    工作方式:
        利用 prepare_obs 返回的索引表 total，
        将原始 obs（新到旧）重排为网络期望的顺序（旧到新）。

    注意: 推理时仅使用本体感知 obs（不含特权 priv_obs），
             特权信息由 StateHistoryEncoder 从历史中自行估计。

        Example:
            In env.step, the original observation follows right-to-left order, after modification it becomes left-to-right

            For example, the original observation is:
                obs = ang_vel(3) * 10(num_history) + joint_pos(18) * 10(num_history) + joint_vel(18) * 10(num_history)
            After env.step, the observation (obs) becomes structured as follows:
                obs_old = ang_vel_timestep_10 -> ang_vel_timestep_1, joint_pos_timestep_10 -> joint_pos_timestep_1, joint_vel_timestep_10 -> joint_vel_timestep_1
            We need to modify the observation order to:
                obs_new = ang_vel_timestep_1, joint_pos_timestep_1, joint_vel_timestep_1 -> ang_vel_timestep_10, joint_pos_timestep_10, joint_vel_timestep_10
    """
    for i in range(10):
        obs_1 = obs[:, total[i, :]].to(env.device)
        obs_new = torch.cat([obs_new, obs_1], dim = -1)

    # 推理时只使用本体感知 obs（去掉特权部分）
    # 网络通过 hist_encoding=True 用历史编码器自行估计特权信息
    obs = obs_new[:, env.num_prop  * env.num_history:] 
    
    # prop and priv obs:
    # obs = torch.cat([obs_new[:, env.num_prop  * env.num_history :], 
    #                  obs[:, env.num_prop  * env.num_history :]], dim=-1)  
    
    obs_new = torch.zeros(env.num_envs, env.num_prop * env.num_history).to(env.device)
    return obs, obs_new



# @hydra_task_config 装饰器根据 --task Isaac-Go2Arm-Flat-Play 自动加载:
#   env_cfg  → Go2ARMFlatEnvCfg_PLAY（禁用域随机化、减少环境数量）
#   agent_cfg → Go2ArmFlatPPORunnerCfg
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: Go2ArmRslRlOnPolicyRunnerCfg):
    """推理主函数"""
    # 任务名去掉 -Play 后缀，用于在对应的训练日志目录中查找 checkpoint
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ---- 步骤1：确定要加载的 checkpoint 路径 ----
    # 优先级: --checkpoint 手动指定 > --use_pretrained > 自动查找 logs/最新训练结果
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        # 自动在 logs/rsl_rl/unitree_Go2arm_flat/ 下查找最新的 .pt 文件
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # ---- 步骤2：创建仿真环境（同 train.py）----
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ---- 步骤3：包装环境并加载 checkpoint ----
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # 初始化 runner（不需要 log_dir，推理时不保存）
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)  # 加载权重、obs normalizer 等

    # 获取封装好的推理函数（内部会创建 obs normalizer + actor 的复合函数）
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # 提取神经网络模块（用于导出）
    try:
        policy_nn = ppo_runner.alg.policy  # RSL-RL v2.3+
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic  # RSL-RL v2.2 及以下

    # ---- 步骤4：将策略导出为 TorchScript (.pt) ----
    # 导出文件将保存到 checkpoint 同目录的 exported/policy.pt
    # 这个文件可直接用于 Go2Arm_sim2sim（Gazebo）部署
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    
    # onnx not avaliable
    # export_policy_as_onnx(
    #     policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )

    dt = env.unwrapped.step_dt  # 仿真步长（秒），用于实时模式下控制渲染频率

    # ---- 步骤5：重置环境，获取初始观测 ----
    obs, _ = env.get_observations()
    total, obs_new = prepare_obs(env)  # 预计算观测索引映射表（只计算一次）
    
    timestep = 0
    # ---- 步骤6：推理主循环 ----
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():  # 禁用梯度计算，加快推理
            # a. 重排历史观测顺序：旧到新顺序 → 神经网络需要的顺序
            obs, obs_new = change_obs_order(obs, obs_new, total, env)

            # b. 策略推理：hist_encoding=True 表示使用 StateHistoryEncoder 估计特权信息
            #    输入 obs 为 [num_envs, num_prop * num_history]（旧到新重排后）
            #    输出 actions 为 [num_envs, 18]（腾12 + 臆18）
            actions = policy(obs, hist_encoding=True)

            # c. 仿真步进：将动作应用到所有并行环境
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # 实时模式：如果单步运行过快，等待剩余时间（与真实机器人节奏一致）
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
