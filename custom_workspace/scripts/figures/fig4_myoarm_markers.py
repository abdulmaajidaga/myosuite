#!/usr/bin/env python3
"""Figure 4: MyoArm model at Pick/Place and Drink phases (healthy trial).

Source MOT:  output/originals/mot/06_12_2.mot  (healthy subject 06, trial 12_2)
Phase index: output/phase_indices/healthy_phase_indices.json
  06_12_2.csv -> [268, 808, 1218]
  Pick:  frames   0 – 268  →  midpoint frame 134
  Drink: frames 268 – 808  →  max elbow_flexion = frame 579 (2.37 rad)
  Place: frames 808 – 1218

Left  panel (a): Pick/Place phase (frame 134) — arm extended, 7 DOFs labelled.
Right panel (b): Drink phase      (frame 579) — elbow maximally flexed,
                 5 virtual markers overlaid.

Outputs:
    figures/myoarm_markers.png  (300 DPI)
    figures/myoarm_markers.pdf  (vector)
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import mujoco

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.unicode_minus': False,
})

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
WORKSPACE    = os.path.join(SCRIPT_DIR, '..', '..')
XML_PATH     = os.path.join(WORKSPACE, 'models', 'model', 'myo_sim', 'arm', 'myoarm.xml')
MOT_PATH     = os.path.join(WORKSPACE, 'output', 'originals', 'mot', '12_12_1.mot')
PHASE_JSON   = os.path.join(WORKSPACE, 'output', 'phase_indices', 'healthy_phase_indices.json')
TRIAL_KEY    = '12_12_1.csv'

# ── Load phase indices ─────────────────────────────────────────────────────────
with open(PHASE_JSON) as f:
    phase_data = json.load(f)
pick_end, drink_end, _ = phase_data[TRIAL_KEY]
PICK_PLACE_FRAME = pick_end // 2          # midpoint of Pick phase → 172

# Find frame of max elbow_flexion within Drink phase
def parse_mot(path):
    with open(path) as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        if l.strip() == 'endheader':
            hend = i; break
    cols = lines[hend + 1].strip().split('\t')
    rows = [l.strip().split('\t') for l in lines[hend + 2:] if l.strip()]
    return cols, rows

cols, rows = parse_mot(MOT_PATH)
ef_idx = cols.index('elbow_flexion')
drink_ef = [float(rows[j][ef_idx]) for j in range(pick_end, drink_end)]
DRINK_FRAME = pick_end + int(np.argmax(drink_ef))

print(f'Trial: {TRIAL_KEY}')
print(f'Phase boundaries: pick=[0,{pick_end}], drink=[{pick_end},{drink_end}]')
print(f'Pick/Place frame: {PICK_PLACE_FRAME}')
print(f'Drink frame:      {DRINK_FRAME}  (elbow_flex={max(drink_ef):.3f} rad)')

# ── Load model ─────────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)
RENDER_W, RENDER_H = 480, 480


def set_qpos_from_row(m, d, cols, row):
    qpos = d.qpos.copy()
    for jnt_id in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        if name and name in cols:
            qpos[m.jnt_qposadr[jnt_id]] = float(row[cols.index(name)])
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)


# ── Camera: full arm visible ───────────────────────────────────────────────────
CAM_KW = dict(lookat=[-0.16, 0.09, 1.18], distance=0.78, azimuth=15, elevation=-5)

def make_camera(**kw):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = kw['lookat']
    cam.distance   = kw['distance']
    cam.azimuth    = kw['azimuth']
    cam.elevation  = kw['elevation']
    return cam


def render_scene(m, d, cam):
    r = mujoco.Renderer(m, height=RENDER_H, width=RENDER_W)
    r.update_scene(d, camera=cam)
    r._scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX]    = 0
    r._scene.flags[mujoco.mjtRndFlag.mjRND_FOG]       = 0
    r._scene.flags[mujoco.mjtRndFlag.mjRND_HAZE]      = 0
    r._scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    img = r.render().copy()
    r.close()
    return img


def get_scene_camera(m, d, cam):
    scene = mujoco.MjvScene(m, maxgeom=10000)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, cam,
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    return scene.camera[0]


def project(world_pts, scam, w, h):
    pos   = np.array(scam.pos)
    fwd   = np.array(scam.forward)
    up    = np.array(scam.up)
    right = np.cross(fwd, up)
    fnear = scam.frustum_near
    ftop  = scam.frustum_top
    asp   = w / h
    pxs   = []
    for P in world_pts:
        rel = np.array(P) - pos
        xc  = np.dot(rel, right)
        yc  = np.dot(rel, up)
        zc  = np.dot(rel, fwd)
        if zc <= 0:
            pxs.append(None); continue
        xndc = xc * fnear / (zc * ftop * asp)
        yndc = yc * fnear / (zc * ftop)
        pxs.append(((xndc + 1) / 2 * w, (1 - (yndc + 1) / 2) * h))
    return pxs


# ── DOF annotations ───────────────────────────────────────────────────────────
DOF_NAMES  = ['elv_angle', 'shoulder_elv', 'shoulder_rot',
               'elbow_flexion', 'pro_sup', 'deviation', 'flexion']
DOF_LABELS = ['elv angle', 'shoulder elv', 'shoulder rot',
              'elbow flex', 'pro/sup', 'deviation', 'flexion']
dof_jids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in DOF_NAMES]

def joint_world_pos(m, d, jid):
    bid  = m.jnt_bodyid[jid]
    jloc = m.jnt_pos[jid]
    return d.xpos[bid] + d.xmat[bid].reshape(3,3) @ jloc

# ── Virtual markers ───────────────────────────────────────────────────────────
VM_NAMES  = ['V_Shoulder', 'V_Elbow', 'V_Wrist', 'V_Vector', 'V_Sternum']
VM_LABELS = ['V Shoulder', 'V Elbow',  'V Wrist',  'V Vector', 'V Sternum']
VM_COLORS = ['#D62728', '#FF7F0E', '#2CA02C', '#9467BD', '#1F77B4']
vm_sids   = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in VM_NAMES]

# ── Render panel (a): Pick/Place ───────────────────────────────────────────────
mujoco.mj_resetData(model, data)
set_qpos_from_row(model, data, cols, rows[PICK_PLACE_FRAME])
cam_a  = make_camera(**CAM_KW)
img_a  = render_scene(model, data, cam_a)
scam_a = get_scene_camera(model, data, cam_a)
dof_w  = [joint_world_pos(model, data, j) for j in dof_jids]
dof_px = project(dof_w, scam_a, RENDER_W, RENDER_H)

# ── Render panel (b): Drink ────────────────────────────────────────────────────
mujoco.mj_resetData(model, data)
set_qpos_from_row(model, data, cols, rows[DRINK_FRAME])
cam_b  = make_camera(**CAM_KW)
img_b  = render_scene(model, data, cam_b)
scam_b = get_scene_camera(model, data, cam_b)
vm_w   = [data.site_xpos[s].copy() for s in vm_sids]
vm_px  = project(vm_w, scam_b, RENDER_W, RENDER_H)


# ── Annotation helper ─────────────────────────────────────────────────────────
def annotate(ax, img, labels, pixels, colors, dot_size, offsets):
    ax.imshow(img, origin='upper')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i, (lbl, pxy) in enumerate(zip(labels, pixels)):
        if pxy is None:
            continue
        px, py = pxy
        c = colors[i] if isinstance(colors, list) else colors
        ax.scatter(px, py, s=dot_size, c=c, zorder=6, edgecolors='white', linewidths=0.8)
        dx, dy = offsets[i % len(offsets)]
        t = ax.text(px + dx, py + dy, lbl, fontsize=5.8, color='#111111',
                    zorder=7, ha='left' if dx > 0 else 'right', va='center')
        t.set_path_effects([pe.Stroke(linewidth=2.2, foreground='white'), pe.Normal()])
        ax.plot([px, px + dx*0.55], [py, py + dy*0.55], color='#666666', lw=0.5, zorder=5)


DOF_OFFSETS = [(+58,-16), (-76,+14), (+58,+16), (+58,-16), (-76,+14), (+58,+12), (-76,-12)]
VM_OFFSETS  = [(+58,-16), (-76,+14), (-76,-14), (+58,+14), (+58, +4)]

# ── Compose figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(6.30, 3.94),
                         gridspec_kw={'wspace': 0.06})
fig.patch.set_facecolor('white')

annotate(axes[0], img_a, DOF_LABELS, dof_px,
         colors=['#D62728']*len(DOF_LABELS), dot_size=45, offsets=DOF_OFFSETS)
axes[0].set_title(f'(a) Pick/Place phase  [frame {PICK_PLACE_FRAME}]',
                  fontsize=7.5, pad=4, fontweight='bold')

annotate(axes[1], img_b, VM_LABELS, vm_px,
         colors=VM_COLORS, dot_size=50, offsets=VM_OFFSETS)
axes[1].set_title(f'(b) Drink phase  [frame {DRINK_FRAME}]',
                  fontsize=7.5, pad=4, fontweight='bold')

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = os.path.join(WORKSPACE, 'figures')
os.makedirs(out_dir, exist_ok=True)
fig.savefig(os.path.join(out_dir, 'myoarm_markers.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(out_dir, 'myoarm_markers.pdf'),
            bbox_inches='tight', facecolor='white')
print('Saved myoarm_markers.{png,pdf}')
plt.close(fig)
