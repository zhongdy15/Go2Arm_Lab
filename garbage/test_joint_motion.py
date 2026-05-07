"""
test_joint_motion.py
在 Isaac Sim 中对 go2_wx250s 的单个关节进行往复运动测试，可视化观察效果。
用法:
    python test_joint_motion.py
可在下方 USER CONFIG 区域修改目标关节名称和运动参数。
"""

# ──────────────────────────────────────────────
#  USER CONFIG  ← 根据需要修改这里
# ──────────────────────────────────────────────
USD_PATH      = "/data/zdy/Go2Arm_Lab/source/Go2Arm_Lab/Go2Arm_Lab/assets/go2_weapon_v2.usd"
ROBOT_PRIM    = "/go2_wx250s"          # USD Stage 中机器人根节点路径

# 填入你想测试的关节名称（dof name），留空则脚本会打印所有可用关节后退出
TARGET_JOINT  = "FR_hip_joint"#"shoulder"#"waist"#"shoulder"                      # 例如: "waist" / "FR_hip_joint" / ...

# 运动参数
AMPLITUDE_RAD = 0.8    # 往复运动幅度（弧度）
PERIOD_STEPS  = 200    # 完成一次完整往复所需的仿真步数
TOTAL_STEPS   = 0    # 总仿真步数（0 表示无限循环直到关闭窗口）
PHYSICS_DT    = 1/60   # 物理步长 (秒)
# ──────────────────────────────────────────────

import math
from isaacsim import SimulationApp

# 以有头模式启动 Isaac Sim（可视化）
sim_app = SimulationApp({
    "headless": False,
    "width":    1280,
    "height":   720,
})

# ── 在 SimulationApp 启动之后再 import omni/isaac 相关包 ──
import omni
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation

# 创建仿真世界
world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)

# 添加地面（可选，避免机器人悬空）
world.scene.add_default_ground_plane()

# 将 USD 加入 Stage
stage_utils.add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM)

# 把机器人注册为 Articulation
robot = world.scene.add(
    Articulation(
        prim_path=ROBOT_PRIM,
        name="go2_wx250s",
    )
)

# 重置世界（初始化物理 & 关节状态）
world.reset()

# ── 获取关节信息 ──
dof_names = robot.dof_names
print("\n========== 所有可用 DOF (关节) ==========")
for i, name in enumerate(dof_names):
    print(f"  [{i:2d}] {name}")
print("==========================================\n")

if not TARGET_JOINT:
    print("TARGET_JOINT 为空，已打印所有关节名称，脚本退出。")
    print("请在脚本顶部 USER CONFIG 中填写 TARGET_JOINT 后重新运行。")
    sim_app.close()
    raise SystemExit(0)

if TARGET_JOINT not in dof_names:
    print(f"[ERROR] 关节 '{TARGET_JOINT}' 不存在！请检查名称拼写。")
    sim_app.close()
    raise SystemExit(1)

joint_idx = dof_names.index(TARGET_JOINT)
print(f"测试关节: '{TARGET_JOINT}'  (index={joint_idx})")
print(f"幅度={AMPLITUDE_RAD:.2f} rad，周期={PERIOD_STEPS} steps，总步数={TOTAL_STEPS}\n")

# 获取关节限位（如果有）
# SingleArticulation 使用 get_articulation_controller 或直接从 prim 读取限位
try:
    from omni.isaac.core.utils.types import ArticulationAction
    import omni.physics.tensors as physx_tensors
    dof_limits = robot.get_dof_limits()  # 部分版本支持
    low, high = float(dof_limits[joint_idx][0]), float(dof_limits[joint_idx][1])
except AttributeError:
    # 回退：从 USD prim 读取 physics:lowerLimit / physics:upperLimit
    import math as _math
    from pxr import UsdPhysics
    joint_prim = None
    for prim in robot.prim.GetAllChildren():
        if prim.GetName() == TARGET_JOINT or prim.GetPath().pathString.endswith("/" + TARGET_JOINT):
            joint_prim = prim
            break
    if joint_prim and joint_prim.HasAPI(UsdPhysics.RevoluteJoint):
        rev = UsdPhysics.RevoluteJoint(joint_prim)
        low  = _math.radians(rev.GetLowerLimitAttr().Get() or -360)
        high = _math.radians(rev.GetUpperLimitAttr().Get() or  360)
    else:
        low, high = -_math.pi, _math.pi  # 默认 ±π
print(f"关节限位: [{low:.3f}, {high:.3f}] rad")

# ── 主仿真循环 ──
import numpy as np

num_dofs = len(dof_names)
step = 0

while sim_app.is_running():
    # 计算目标位置：正弦往复
    angle = AMPLITUDE_RAD * math.sin(2 * math.pi * step / PERIOD_STEPS)
    # 裁剪到关节限位（若限位有效）
    if not (low == 0.0 and high == 0.0):
        angle = float(np.clip(angle, low, high))

    # 构造全关节目标位置（其他关节保持当前值，避免干扰）
    positions = robot.get_joint_positions()
    positions[joint_idx] = angle

    robot.set_joint_positions(positions)

    world.step(render=True)
    step += 1

    # 每 50 步打印一次当前角度
    if step % 50 == 0:
        cur_pos = robot.get_joint_positions()
        print(f"step={step:4d}  target={angle:+.4f} rad  actual={cur_pos[joint_idx]:+.4f} rad")

    if TOTAL_STEPS > 0 and step >= TOTAL_STEPS:
        print(f"\n已完成 {TOTAL_STEPS} 步仿真，退出。")
        break

sim_app.close()
