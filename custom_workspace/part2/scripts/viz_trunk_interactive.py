"""
Interactive trunk hinge position finder.

Arrow keys move the marker column — skeleton stays fixed.
Find the spine position visually, then paste the printed coords into myoarm_trunk.xml.

Controls (click the viewer window first so it receives keys):
  LEFT / RIGHT  — shift markers in X (lateral)
  UP   / DOWN   — shift markers in Y (anterior / posterior)
  Q    / E      — no-op (Z is swept automatically)
  ESC           — quit

The 7 spheres span from current Z down 60cm in 10cm steps (shoulder → hip).
Goal: align the column so it runs along the spine.
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

bid       = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'trunk_base')
hinge_pos = data.xpos[bid].copy()

# Adjustable offset — modified by key callback
offset = np.array([hinge_pos[0], hinge_pos[1], 0.0])  # start at hinge XY
STEP   = 0.005  # 5 mm per keypress

GLFW_KEY_RIGHT = 262
GLFW_KEY_LEFT  = 263
GLFW_KEY_DOWN  = 264
GLFW_KEY_UP    = 265
GLFW_KEY_ESC   = 256

def on_key(key):
    if key == GLFW_KEY_RIGHT:
        offset[0] += STEP
    elif key == GLFW_KEY_LEFT:
        offset[0] -= STEP
    elif key == GLFW_KEY_UP:
        offset[1] += STEP
    elif key == GLFW_KEY_DOWN:
        offset[1] -= STEP
    print(f"  marker XY = ({offset[0]:+.3f}, {offset[1]:+.3f})  "
          f"  [← → = X lateral | ↑ ↓ = Y depth]")


# Sphere colours top→bottom: red, orange, yellow, yellow-green, green, cyan, blue
COLOURS = [
    np.array([1.0, 0.0, 0.0, 0.8]),
    np.array([1.0, 0.5, 0.0, 0.7]),
    np.array([1.0, 1.0, 0.0, 0.7]),
    np.array([0.5, 1.0, 0.0, 0.7]),
    np.array([0.0, 1.0, 0.0, 0.8]),
    np.array([0.0, 0.8, 1.0, 0.7]),
    np.array([0.0, 0.5, 1.0, 0.7]),
]
Z_OFFSETS = [0.0, -0.10, -0.20, -0.30, -0.40, -0.50, -0.60]

def add_sphere(scene, pos, rgba, radius=0.03):
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


print("\n── Controls ──────────────────────────────────────────────")
print("  Click the viewer window, then use arrow keys:")
print("  ← →  move column LEFT / RIGHT  (X, lateral)")
print("  ↑ ↓  move column FORWARD / BACK (Y, depth)")
print("  Goal: align the column to run along the spine")
print("  Current start XY:", f"({offset[0]:+.3f}, {offset[1]:+.3f})")
print("──────────────────────────────────────────────────────────\n")

with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as v:
    with v.lock():
        v.opt.sitegroup[:] = 1
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    with v.lock():
        base_ngeom = v.user_scn.ngeom  # geom count before we add anything

    while v.is_running():
        mujoco.mj_forward(model, data)
        with v.lock():
            v.user_scn.ngeom = base_ngeom  # clear our spheres from last frame
            pos = np.array([offset[0], offset[1], hinge_pos[2]])
            add_sphere(v.user_scn, pos, np.array([1.0, 0.0, 0.0, 0.9]), 0.05)
        v.sync()
        time.sleep(0.05)
