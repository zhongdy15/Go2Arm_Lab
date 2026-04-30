from isaacsim import SimulationApp
from pxr import Usd, UsdGeom

# 启动 Isaac Sim
sim = SimulationApp({"headless": True})

# 打开 USD 文件
stage = Usd.Stage.Open("/data/zdy/Go2Arm_Lab/source/Go2Arm_Lab/Go2Arm_Lab/assets/go2_arm.usd")

# 获取根节点 go2_wx250s
root_joint = UsdGeom.Xform.Get(stage, "/go2_wx250s")

# 遍历根节点下的所有子节点，获取关节信息
def get_joint_info(joint_path):
    joint = UsdGeom.Xform.Get(stage, joint_path)
    if isinstance(joint, UsdGeom.Xform):
        # 获取关节位置
        translation = joint.GetMatrixAttr().Get().Transform((0, 0, 0))
        print(f"关节名称: {joint_path}, 位置: {translation}")
        
        # 获取关节的旋转信息
        rotation = joint.GetRotateAttr().Get()
        if rotation:
            print(f"关节旋转: {rotation}")

# 遍历所有子节点
for child in root_joint.GetChildren():
    # 获取子节点的关节信息
    get_joint_info(child.GetPath())

sim.close()
