"""
print_usd_info.py
针对一个 USD 文件，打印其 Articulation 的 Joint(关节) 信息以及 Rigid Body(刚体) 信息。
用法:
    python print_usd_info.py
可在下方 USER CONFIG 区域修改 USD 路径和机器人 prim 路径。
"""

# ──────────────────────────────────────────────
#  USER CONFIG
# ──────────────────────────────────────────────
USD_PATH    = "/data/zdy/Go2Arm_Lab/source/Go2Arm_Lab/Go2Arm_Lab/assets/go2_arm_v3.usd"
ROBOT_PRIM  = "/go2_wx250s"   # USD Stage 中机器人根节点路径
PHYSICS_DT  = 1 / 60
HEADLESS    = True            # 仅打印信息，无需可视化
# ──────────────────────────────────────────────

import math

from isaacsim import SimulationApp

sim_app = SimulationApp({
    "headless": HEADLESS,
    "width":    1280,
    "height":   720,
})

# ── 在 SimulationApp 启动之后再 import omni/isaac 相关包 ──
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from pxr import Usd, UsdPhysics, UsdGeom


def _try_get(attr):
    """安全地读取 USD attribute 的值，失败返回 None。"""
    try:
        return attr.Get() if attr else None
    except Exception:
        return None


def print_section(title: str):
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


def collect_joints(stage):
    """遍历 stage，收集所有带有 UsdPhysics.Joint API 的 prim 信息。"""
    joints = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue

        joint = UsdPhysics.Joint(prim)
        info = {
            "name": prim.GetName(),
            "path": prim.GetPath().pathString,
            "type": prim.GetTypeName(),
            "body0": None,
            "body1": None,
            "lower_limit": None,
            "upper_limit": None,
            "axis": None,
            "drive_target_pos": None,
            "drive_target_vel": None,
            "drive_stiffness": None,
            "drive_damping": None,
            "drive_max_force": None,
        }

        # body0 / body1 (rel)
        try:
            rels0 = joint.GetBody0Rel().GetTargets()
            info["body0"] = rels0[0].pathString if rels0 else None
        except Exception:
            pass
        try:
            rels1 = joint.GetBody1Rel().GetTargets()
            info["body1"] = rels1[0].pathString if rels1 else None
        except Exception:
            pass

        # 针对 Revolute / Prismatic 读取限位与轴向
        if prim.IsA(UsdPhysics.RevoluteJoint):
            rev = UsdPhysics.RevoluteJoint(prim)
            info["lower_limit"] = _try_get(rev.GetLowerLimitAttr())  # 度
            info["upper_limit"] = _try_get(rev.GetUpperLimitAttr())  # 度
            info["axis"]        = _try_get(rev.GetAxisAttr())
            info["limit_unit"]  = "deg"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            pri = UsdPhysics.PrismaticJoint(prim)
            info["lower_limit"] = _try_get(pri.GetLowerLimitAttr())
            info["upper_limit"] = _try_get(pri.GetUpperLimitAttr())
            info["axis"]        = _try_get(pri.GetAxisAttr())
            info["limit_unit"]  = "m"
        else:
            info["limit_unit"] = ""

        # DriveAPI（angular / linear）
        for drive_name in ("angular", "linear", "transX", "transY", "transZ",
                           "rotX", "rotY", "rotZ"):
            if UsdPhysics.DriveAPI.CanApply(prim, drive_name) and \
               prim.HasAPI(UsdPhysics.DriveAPI, drive_name):
                drive = UsdPhysics.DriveAPI(prim, drive_name)
                info["drive_target_pos"] = _try_get(drive.GetTargetPositionAttr())
                info["drive_target_vel"] = _try_get(drive.GetTargetVelocityAttr())
                info["drive_stiffness"]  = _try_get(drive.GetStiffnessAttr())
                info["drive_damping"]    = _try_get(drive.GetDampingAttr())
                info["drive_max_force"]  = _try_get(drive.GetMaxForceAttr())
                info["drive_kind"]       = drive_name
                break

        joints.append(info)
    return joints


def collect_rigid_bodies(stage):
    """遍历 stage，收集带有 RigidBodyAPI 的 prim 信息。"""
    bodies = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        rb = UsdPhysics.RigidBodyAPI(prim)
        info = {
            "name": prim.GetName(),
            "path": prim.GetPath().pathString,
            "type": prim.GetTypeName(),
            "kinematic": _try_get(rb.GetKinematicEnabledAttr()),
            "rb_enabled": _try_get(rb.GetRigidBodyEnabledAttr()),
            "mass": None,
            "density": None,
            "com": None,
            "has_collision_api": prim.HasAPI(UsdPhysics.CollisionAPI),
        }

        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            info["mass"]    = _try_get(mass_api.GetMassAttr())
            info["density"] = _try_get(mass_api.GetDensityAttr())
            info["com"]     = _try_get(mass_api.GetCenterOfMassAttr())

        bodies.append(info)
    return bodies


# ── 加载 USD 到 Stage ──
world = World(physics_dt=PHYSICS_DT, rendering_dt=PHYSICS_DT)
stage_utils.add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM)

robot = world.scene.add(Articulation(prim_path=ROBOT_PRIM, name="robot"))
world.reset()

stage = stage_utils.get_current_stage()

# ── 1. Articulation 视角下的 DOF / Link 名称 ──
print_section(f"USD: {USD_PATH}")
print(f"Robot prim path: {ROBOT_PRIM}")

print_section("Articulation DOF (关节顺序，可用于控制索引)")
dof_names = list(robot.dof_names) if robot.dof_names is not None else []
print(f"DOF count: {len(dof_names)}")
for i, name in enumerate(dof_names):
    print(f"  [{i:2d}] {name}")

# Articulation 中的 link / body 名称
try:
    body_names = list(robot.body_names)
except Exception:
    body_names = []
print_section("Articulation Body / Link 名称")
print(f"Body count: {len(body_names)}")
for i, name in enumerate(body_names):
    print(f"  [{i:2d}] {name}")

# ── 2. USD 视角下的 Joint 详细信息 ──
joints = collect_joints(stage)
print_section(f"USD Physics Joint 详细信息  (共 {len(joints)} 个)")
for i, j in enumerate(joints):
    print(f"\n  [{i:2d}] {j['name']}  ({j['type']})")
    print(f"       path : {j['path']}")
    print(f"       body0: {j['body0']}")
    print(f"       body1: {j['body1']}")
    if j.get("axis") is not None:
        print(f"       axis : {j['axis']}")
    if j.get("lower_limit") is not None or j.get("upper_limit") is not None:
        unit = j.get("limit_unit", "")
        print(f"       limit: [{j['lower_limit']}, {j['upper_limit']}] {unit}")
    if j.get("drive_kind"):
        print(f"       drive[{j['drive_kind']}]: "
              f"target_pos={j['drive_target_pos']}, target_vel={j['drive_target_vel']}, "
              f"stiffness={j['drive_stiffness']}, damping={j['drive_damping']}, "
              f"max_force={j['drive_max_force']}")

# ── 3. USD 视角下的 Rigid Body 详细信息 ──
bodies = collect_rigid_bodies(stage)
print_section(f"USD Rigid Body 详细信息  (共 {len(bodies)} 个)")
for i, b in enumerate(bodies):
    print(f"\n  [{i:2d}] {b['name']}  ({b['type']})")
    print(f"       path        : {b['path']}")
    print(f"       rb_enabled  : {b['rb_enabled']}")
    print(f"       kinematic   : {b['kinematic']}")
    print(f"       mass        : {b['mass']}")
    print(f"       density     : {b['density']}")
    print(f"       center_mass : {b['com']}")
    print(f"       has_collider: {b['has_collision_api']}")

print_section("Done")
sim_app.close()
