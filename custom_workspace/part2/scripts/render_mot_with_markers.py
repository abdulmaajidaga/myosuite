"""
Render a MOT file to MP4 with IK marker spheres overlaid.
Usage:
  python scripts/render_mot_with_markers.py [model.xml] [file.mot] [output.mp4]
Defaults to S5_12_1 test file.
"""
import os, sys
import numpy as np
import mujoco
import skvideo.io
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path

MODEL_PATH = get_path("mujoco_arm_model").replace("myoarm.xml", "myoarm_trunk.xml")
MOT_PATH   = "/tmp/test_S5_12_1.mot"
OUT_PATH   = "/tmp/test_S5_12_1_markers.mp4"

if len(sys.argv) > 3:
    MODEL_PATH, MOT_PATH, OUT_PATH = sys.argv[1], sys.argv[2], sys.argv[3]

# Camera
CAM_AZIMUTH  = 0
CAM_DISTANCE = 2.2
CAM_ELEV     = -15
CAM_LOOKAT   = [-0.05, 0.05, 1.35]
W, H = 640, 480

IK_SITES = {
    "V_Shoulder": (np.array([1.0, 0.2, 0.2, 0.9]), 0.022),
    "V_Elbow":    (np.array([0.2, 1.0, 0.2, 0.9]), 0.020),
    "V_Wrist":    (np.array([0.2, 0.4, 1.0, 0.9]), 0.020),
    "V_Vector":   (np.array([0.8, 0.2, 1.0, 0.8]), 0.015),
    "CS_1":       (np.array([0.2, 1.0, 0.5, 0.9]), 0.015),
    "CS_2":       (np.array([0.2, 1.0, 0.5, 0.9]), 0.015),
    "CS_3":       (np.array([0.2, 1.0, 0.5, 0.9]), 0.015),
    "CS_4":       (np.array([0.2, 1.0, 0.5, 0.9]), 0.015),
}


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


def read_mot(path):
    with open(path) as f:
        lines = f.readlines()
    end = next(i for i, l in enumerate(lines) if 'endheader' in l)
    cols = lines[end + 1].strip().split('\t')
    data = np.array([[float(x) for x in l.strip().split('\t')]
                     for l in lines[end + 2:] if l.strip()])
    return cols, data


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
cols, mot = read_mot(MOT_PATH)

renderer = mujoco.Renderer(model, height=H, width=W)
scene_option = mujoco.MjvOption()
scene_option.sitegroup[:] = 1          # show all site markers
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

cam = mujoco.MjvCamera()
cam.azimuth   = CAM_AZIMUTH
cam.distance  = CAM_DISTANCE
cam.elevation = CAM_ELEV
cam.lookat[:] = CAM_LOOKAT

joint_names = cols[1:]  # skip 'time'
joint_ids = []
for name in joint_names:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_ids.append(jid)

writer = skvideo.io.FFmpegWriter(OUT_PATH, outputdict={
    '-vcodec': 'libx264', '-pix_fmt': 'yuv420p', '-crf': '20'
})

n_frames = len(mot)
print(f"Rendering {n_frames} frames -> {OUT_PATH}")

for i in tqdm(range(n_frames)):
    for k, jid in enumerate(joint_ids):
        if jid != -1:
            data.qpos[model.jnt_qposadr[jid]] = mot[i, k + 1]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam, scene_option=scene_option)
    frame = renderer.render()
    writer.writeFrame(frame)

writer.close()
print(f"Saved: {OUT_PATH}")
