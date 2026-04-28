# Go2Arm_Lab

**一个用于足式操作机器人的强化学习训练框架**
  
<img src="https://img.shields.io/badge/IsaacSim-v4.5.0-blue" alt="IsaacSim-v4.2.0" /> <img src="https://img.shields.io/badge/IsaacLab-v2.1.0-blue" alt="IsaacLab v2.1.0" /> <img src="https://img.shields.io/badge/Python-3.10-blue" alt="Python 3.10" /> <img src="https://img.shields.io/badge/Platform-Linux--64-orange" alt="Linux-64" /> <img src="https://img.shields.io/badge/License-Apache2.0-yellow" alt="Apache2.0T License" />

## 🚀 概述

Go2Arm_Lab 使足式操作机器人的强化学习训练成为可能:

- **基础平台**: Unitree Go2 四足机器人
- **操作臂**: Interbotix WidowX 250s 机械臂

> **版本兼容性**  
> 当前仓库依赖于 **IsaacLab v2.2.0**。
> 对于 IsaacLab v2.1.0，请使用此仓库的[v2.1.0](https://github.com/zzzJie-Robot/Go2Arm_Lab/releases/tag/v2.1)版本。
> 对于 IsaacLab v1.4.1，请使用此仓库的[v1.4.1](https://github.com/zzzJie-Robot/Go2Arm_Lab/releases/tag/v1.4)版本。

> **Gazebo 部署**  
> 如果您想在 Gazebo 中部署您的策略，请使用：
> [Go2Arm_sim2sim](https://github.com/zzzJie-Robot/Go2Arm_sim2sim)

| IsaacLab 仿真 (v2.2)                                                       | Gazebo 仿真 Simulation                                                      |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| ![IsaacLab](https://github.com/zzzJie-Robot/Go2Arm_Lab/blob/v2.1/video/IsaacLab%20v2.1.gif)  | ![Gazebo](https://github.com/zzzJie-Robot/Go2Arm_Lab/blob/main/video/Gazebo.gif)  |

**更多视频**欢迎访问我的[Bilibili 主页](https://space.bilibili.com/400627082?spm_id_from=333.1007.0.0).

## 📦 安装

1. 按照[官方指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)安装 IsaacLab v2.1.0。
2. 将此仓库克隆到您的 IsaacLab 同级目录:
    ```
    git clone https://github.com/zzzJie-Robot/Go2Arm_Lab.git
    ```

3. 使用已安装 Isaac Lab 的 Python 解释器，安装该库
    ```
    python -m pip install -e source/Go2Arm_Lab
    ```



### ⚙️ 训练与推理

#### 训练

在`headless`模式下运行强化学习训练，以提高训练效率:

```
# Activate IsaacLab environment
conda activate your_isaaclab_env

# Navigate to Go2Arm_Lab
cd /path/to/Go2Arm_Lab

# Launch training (headless)
python scripts/rsl_rl/train.py --task Isaac-Go2Arm-Flat --headless 
```

#### 推理

在单个环境中部署训练好的策略：

```
# Activate IsaacLab environment  
conda activate your_isaaclab_env

# Navigate to IsaacLab root  
cd /path/to/Go2Arm_Lab

# Run inference
python scripts/rsl_rl/play.py --task Isaac-Go2Arm-Flat-Play --num_envs 1 
```


## 🙏 致谢
本项目的强化学习算法实现参考了[Deep-Whole-Body-Control](https://github.com/MarkFzp/Deep-Whole-Body-Control)，特此致谢。