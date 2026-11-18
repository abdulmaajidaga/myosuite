"""
Interactive CS cluster placement — all 4 markers move as one rigid plate.

Controls
────────
  A / D      slide left / right     (frontal plane, world Y)
  W / X      slide up   / down      (frontal plane, world Z)
  Q / E      slide forward / back   (depth, world X)
  Z / C      rotate around sagittal axis (world X, front-to-back)
             → tilts the plate left/right like a steering wheel
  R / V      rotate around vertical axis (world Z) — spin in transverse plane
  P          save to myoarm_body.xml and print new coords
  Esc        quit without saving
"""
import os, sys, re, time
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path

MODEL_PATH = get_path("mujoco_arm_model").replace("myoarm.xml", "myoarm_trunk.xml")
XML_BODY   = os.path.join(os.path.dirname(MODEL_PATH), "assets", "myoarm_body.xml")

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

CS_NAMES = ['CS_1', 'CS_2', 'CS_3', 'CS_4']
site_ids  = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in CS_NAMES}

STEP     = 0.005          # 5 mm per key press
ROT_STEP = np.radians(3)  # 3° per key press

offsets = {n: np.zeros(3) for n in CS_NAMES}   # world-space delta from XML position
_dirty  = [False]
_quit   = [False]


def world_pos(name):
    return data.site_xpos[site_ids[name]].copy() + offsets[name]


def centroid():
    return np.mean([world_pos(n) for n in CS_NAMES], axis=0)


def rotate_all(axis_vec, angle):
    """Rotate all 4 markers as a rigid body around cluster centroid."""
    axis  = np.asarray(axis_vec, float)
    axis /= np.linalg.norm(axis)
    c, s  = np.cos(angle), np.sin(angle)
    ux, uy, uz = axis
    R = np.array([
        [c+ux*ux*(1-c),      ux*uy*(1-c)-uz*s,  ux*uz*(1-c)+uy*s],
        [uy*ux*(1-c)+uz*s,   c+uy*uy*(1-c),     uy*uz*(1-c)-ux*s],
        [uz*ux*(1-c)-uy*s,   uz*uy*(1-c)+ux*s,  c+uz*uz*(1-c)   ],
    ])
    cen = centroid()
    for n in CS_NAMES:
        p_new    = cen + R @ (world_pos(n) - cen)
        offsets[n] = p_new - data.site_xpos[site_ids[n]]
    _dirty[0] = True


def translate_all(delta):
    for n in CS_NAMES:
        offsets[n] += delta
    _dirty[0] = True


def print_status():
    cen = centroid()
    total = sum(offsets[n] for n in CS_NAMES) / 4
    print(f"\r  centroid=[{cen[0]:+.3f}, {cen[1]:+.3f}, {cen[2]:+.3f}]  "
          f"total offset Δ=[{total[0]*1000:+.1f},{total[1]*1000:+.1f},{total[2]*1000:+.1f}] mm   ",
          end="", flush=True)


def save_to_xml():
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'thorax')
    R   = data.xmat[bid].reshape(3, 3)
    p   = data.xpos[bid].copy()

    print("\n\nNew thorax-local positions:")
    new_locals = {}
    for n in CS_NAMES:
        lp = R.T @ (world_pos(n) - p)
        new_locals[n] = lp
        print(f"  {n}: pos=\"{lp[0]:.4f} {lp[1]:.4f} {lp[2]:.4f}\"")

    if not os.path.exists(XML_BODY):
        print(f"WARN: {XML_BODY} not found — paste coords above manually.")
        return

    with open(XML_BODY) as f:
        content = f.read()
    for n, lp in new_locals.items():
        pattern = rf'(<site name="{n}"[^>]*pos=")[^"]*(")'
        content, k = re.subn(pattern,
                             rf'\g<1>{lp[0]:.4f} {lp[1]:.4f} {lp[2]:.4f}\g<2>',
                             content)
        if k == 0:
            print(f"  WARN: {n} not found in XML.")
    with open(XML_BODY, 'w') as f:
        f.write(content)
    print(f"Saved → {XML_BODY}")
    _dirty[0] = False


def key_callback(keycode):
    print(f"\n[key] {keycode} = '{chr(keycode) if 32 < keycode < 127 else '?'}'", flush=True)
    if   keycode == 65:  translate_all(np.array([ 0, +STEP,  0]))   # A  left   (+Y)
    elif keycode == 68:  translate_all(np.array([ 0, -STEP,  0]))   # D  right  (-Y)
    elif keycode == 87:  translate_all(np.array([ 0,  0, +STEP]))   # W  up     (+Z)
    elif keycode == 88:  translate_all(np.array([ 0,  0, -STEP]))   # X  down   (-Z)
    elif keycode == 81:  translate_all(np.array([+STEP, 0,  0]))    # Q  fwd    (+X)
    elif keycode == 69:  translate_all(np.array([-STEP, 0,  0]))    # E  back   (-X)
    elif keycode == 90:  rotate_all([1, 0, 0], +ROT_STEP)           # Z  sagittal CCW
    elif keycode == 67:  rotate_all([1, 0, 0], -ROT_STEP)           # C  sagittal CW
    elif keycode == 82:  rotate_all([0, 0, 1], +ROT_STEP)           # R  vertical CCW
    elif keycode == 86:  rotate_all([0, 0, 1], -ROT_STEP)           # V  vertical CW
    elif keycode == 80:  save_to_xml()                               # P  save
    elif keycode == 256: _quit[0] = True                             # Esc quit
    print_status()


# ─── scene ────────────────────────────────────────────────────────────────────

GREEN  = np.array([0.2, 1.0, 0.4, 1.0])
WHITE  = np.array([1.0, 1.0, 1.0, 0.6])

def add_sphere(scene, pos, rgba, radius):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    g.type = mujoco.mjtGeom.mjGEOM_SPHERE
    g.size[:] = [radius]*3
    g.pos[:]  = pos
    g.mat[:]  = np.eye(3)
    g.rgba[:] = rgba
    g.dataid  = -1
    g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    g.objid   = -1
    g.category = mujoco.mjtCatBit.mjCAT_DECOR
    g.segid   = -1
    scene.ngeom += 1


print(__doc__)
print_status()

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as v:
    with v.lock():
        v.opt.sitegroup[:] = 0
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    base_ngeom = v.user_scn.ngeom

    while v.is_running() and not _quit[0]:
        mujoco.mj_forward(model, data)
        with v.lock():
            v.user_scn.ngeom = base_ngeom
            for n in CS_NAMES:
                add_sphere(v.user_scn, world_pos(n), GREEN, 0.020)
            add_sphere(v.user_scn, centroid(), WHITE, 0.010)
        v.sync()
        time.sleep(0.033)

if _dirty[0]:
    print("\nUnsaved changes — press S before closing next time.")
