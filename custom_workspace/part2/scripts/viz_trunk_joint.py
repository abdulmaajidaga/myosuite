"""
Visualise the trunk hinge position vs anatomically correct lumbar position.

Colours drawn in the scene:
  RED    = current trunk_forward_lean hinge (full_body at Z=1.40 — shoulder height)
  GREEN  = anatomically correct lumbar/L5-S1 (~Z=0.90 for seated subject)
  YELLOW = approximate seated hip height (~Z=0.75)

IK marker sites are also shown (enable Sites in viewer with V -> Sites).

Run:
    python scripts/viz_trunk_joint.py
"""
import os, sys, time
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path

MODEL_PATH = get_path("mujoco_arm_model").replace("myoarm.xml", "myoarm_trunk.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# Print site positions so user can cross-check
print("\n── IK marker sites (world frame) ──")
for i in range(model.nsite):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    if name and any(k in name for k in ['V_Shoulder','V_Elbow','V_Wrist','V_Sternum','V_Vector']):
        pos = data.site_xpos[i]
        print(f"  {name:<20} Z={pos[2]:.3f}m")

bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'full_body')
hinge_pos = data.xpos[bid].copy()
print(f"\n── Trunk hinge (full_body) world pos ──")
print(f"  XYZ = {hinge_pos}  Z={hinge_pos[2]:.3f}m  ← current hinge (RED)")
print(f"  Correct lumbar Z~0.90m  ← (GREEN)")
print(f"  Hip level      Z~0.75m  ← (YELLOW)")
print(f"\nLaunching viewer...")
print("  Press V in viewer to toggle: Sites (s) and Joints (j)\n")

# Annotation spheres at 0.1m intervals down the spine
# Red = current hinge (Z=1.40, sternum level — too high)
# Then white spheres every 0.1m down so you can pick the right lumbar level visually
base = hinge_pos.copy()
markers = [
    (base + [0,0, 0.00], np.array([1.0, 0.0, 0.0, 0.8]), 0.05),  # Z=1.40 current (RED)
    (base + [0,0,-0.10], np.array([1.0, 0.5, 0.0, 0.7]), 0.03),  # Z=1.30
    (base + [0,0,-0.20], np.array([1.0, 1.0, 0.0, 0.7]), 0.03),  # Z=1.20 (V_Sternum level)
    (base + [0,0,-0.30], np.array([0.5, 1.0, 0.0, 0.7]), 0.03),  # Z=1.10
    (base + [0,0,-0.40], np.array([0.0, 1.0, 0.0, 0.8]), 0.04),  # Z=1.00 T12/L1 estimate (GREEN)
    (base + [0,0,-0.50], np.array([0.0, 0.8, 1.0, 0.7]), 0.03),  # Z=0.90 L5/S1 estimate
    (base + [0,0,-0.60], np.array([0.0, 0.5, 1.0, 0.7]), 0.03),  # Z=0.80 hip
]
print("Sphere heights (Z):")
for i,(p,c,r) in enumerate(markers):
    print(f"  Z={p[2]:.2f}m")

def add_sphere(scene, pos, rgba, radius):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    g.type       = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[:]    = [radius, radius, radius]
    g.pos[:]     = pos
    g.mat[:]     = np.eye(3)
    g.rgba[:]    = rgba
    g.dataid     = -1
    g.objtype    = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid      = -1
    g.category   = mujoco.mjtCatBit.mjCAT_DECOR
    g.segid      = -1
    scene.ngeom += 1

with mujoco.viewer.launch_passive(model, data) as v:
    with v.lock():
        v.opt.sitegroup[:] = 1          # show all site groups
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    while v.is_running():
        mujoco.mj_forward(model, data)
        with v.lock():
            for pos, rgba, radius in markers:
                add_sphere(v.user_scn, pos, rgba, radius)
        v.sync()
        time.sleep(0.05)
