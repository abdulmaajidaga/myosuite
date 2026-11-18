"""
Visualise IK marker sites on the model body.
Each site shown as a labelled sphere — confirm they sit at the correct anatomical locations.

  RED    = V_Shoulder
  GREEN  = V_Elbow
  BLUE   = V_Wrist
  YELLOW = V_Sternum
  PURPLE = V_Vector
  WHITE  = trunk_base hinge (L4/L5)
  LIME   = CS_1, CS_2, CS_3, CS_4  (chest cluster — drives trunk lean)
"""
import os, sys, time
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path

MODEL_PATH = get_path("mujoco_arm_model")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

IK_SITES = {
    "V_Shoulder": (np.array([1.0, 0.2, 0.2, 1.0]), 0.025),
    "V_Elbow":    (np.array([0.2, 1.0, 0.2, 1.0]), 0.022),
    "V_Wrist":    (np.array([0.2, 0.4, 1.0, 1.0]), 0.022),
    "V_Sternum":  (np.array([1.0, 1.0, 0.2, 1.0]), 0.022),
    "V_Vector":   (np.array([0.8, 0.2, 1.0, 1.0]), 0.018),
    # CS chest cluster — drives trunk_forward_lean
    "CS_1":       (np.array([0.2, 1.0, 0.2, 1.0]), 0.018),
    "CS_2":       (np.array([0.2, 1.0, 0.2, 1.0]), 0.018),
    "CS_3":       (np.array([0.2, 1.0, 0.2, 1.0]), 0.018),
    "CS_4":       (np.array([0.2, 1.0, 0.2, 1.0]), 0.018),
}

# Print positions
print(f"\n{'Site':<15} {'X':>8} {'Y':>8} {'Z':>8}")
print("─" * 45)
for name, (rgba, r) in IK_SITES.items():
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid >= 0:
        p = data.site_xpos[sid]
        print(f"{name:<15} {p[0]:>8.3f} {p[1]:>8.3f} {p[2]:>8.3f}")

# Trunk hinge
bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'trunk_base')
hinge = data.xpos[bid].copy()
print(f"{'trunk_base (hinge)':<22} {hinge[0]:>8.3f} {hinge[1]:>8.3f} {hinge[2]:>8.3f}")

# CS cluster — show rel to shoulder
sid_sho = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'V_Shoulder')
sho_pos = data.site_xpos[sid_sho]
print()
print(f"{'CS relative to shoulder (mm)'}")
for cs in ['CS_1','CS_2','CS_3','CS_4']:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cs)
    if sid >= 0:
        rel = (data.site_xpos[sid] - sho_pos) * 1000
        print(f"  {cs}: {rel[0]:+.1f}mm fwd  {rel[1]:+.1f}mm lat  {rel[2]:+.1f}mm vert")
print()


def add_sphere(scene, pos, rgba, radius):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    g.type     = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[:]  = [radius, radius, radius]
    g.pos[:]   = pos
    g.mat[:]   = np.eye(3)
    g.rgba[:]  = rgba
    g.dataid   = -1
    g.objtype  = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid    = -1
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    g.segid    = -1
    scene.ngeom += 1


print("Legend:")
print("  RED    = V_Shoulder")
print("  GREEN  = V_Elbow  (same green as CS cluster)")
print("  BLUE   = V_Wrist")
print("  YELLOW = V_Sternum  (CS centroid reference)")
print("  PURPLE = V_Vector")
print("  WHITE  = trunk_base hinge (L4/L5)")
print("  LIME   = CS_1, CS_2, CS_3, CS_4  (chest cluster — drives trunk lean)\n")

with mujoco.viewer.launch_passive(model, data) as v:
    with v.lock():
        v.opt.sitegroup[:] = 1
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    base_ngeom = v.user_scn.ngeom

    while v.is_running():
        mujoco.mj_forward(model, data)
        with v.lock():
            v.user_scn.ngeom = base_ngeom

            # IK sites
            for name, (rgba, radius) in IK_SITES.items():
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
                if sid >= 0:
                    add_sphere(v.user_scn, data.site_xpos[sid], rgba, radius)

            # Trunk hinge
            add_sphere(v.user_scn, hinge,
                       np.array([1.0, 1.0, 1.0, 0.9]), 0.030)

        v.sync()
        time.sleep(0.05)
