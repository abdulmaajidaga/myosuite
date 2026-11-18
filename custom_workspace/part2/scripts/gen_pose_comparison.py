import os, numpy as np, math, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco

MOT_S   = "/home/abdul/Desktop/myosuite/custom_workspace/part2/output/neutral_start/stroke/mot/S14_12_3.mot"
MOT_H   = "/home/abdul/Desktop/myosuite/custom_workspace/part2/output/neutral_start/healthy/mot/09_12_2.mot"
TRC_S   = "/home/abdul/Desktop/myosuite/custom_workspace/part2/output/neutral_start/stroke/trc/S14_12_3.trc"
TRC_H   = "/home/abdul/Desktop/myosuite/custom_workspace/part2/output/neutral_start/healthy/trc/09_12_2.trc"
MODEL   = "/home/abdul/Desktop/myosuite/custom_workspace/models/model/myo_sim/arm/myoarm_trunk_viz.xml"
OUT     = ("/home/abdul/Desktop/myosuite/custom_workspace/part2/"
           "paper_publication_assets/latex_submission/pose_comparison.png")
R2D = 180.0 / np.pi

# ── IO ────────────────────────────────────────────────────────────────────────
def read_mot(path):
    with open(path) as f: lines = f.readlines()
    for i, l in enumerate(lines):
        if l.strip().startswith('time'):
            hdr = l.strip().split('\t'); ds = i+1; break
    rows = [[float(x) for x in l.strip().split('\t')]
            for l in lines[ds:] if len(l.strip().split('\t')) == len(hdr)]
    return {hdr[j]: np.array(rows)[:, j] for j in range(len(hdr))}

def drink_frame_from_trc(trc_path):
    """Frame index of max wrist height (col 9 = V_Wrist Y)."""
    with open(trc_path) as f: lines = f.readlines()
    wy = []
    for l in lines[5:]:
        v = l.strip().split('\t')
        if len(v) < 10: continue
        try: wy.append(float(v[9]))
        except: continue
    return int(np.argmax(wy))

mot_s = read_mot(MOT_S);  mot_h = read_mot(MOT_H)
f_rest    = 0
f_drink_s = drink_frame_from_trc(TRC_S)
f_drink_h = drink_frame_from_trc(TRC_H)

# Use actual peak trunk_forward_lean at the drinking frame from neutral_start data
TRUNK_OVERRIDE_DEG = float(abs(mot_s['trunk_forward_lean'][f_drink_s]) * R2D)

print(f"Frames → rest: 0  |  drink: S={f_drink_s} H={f_drink_h}")
print(f"Trunk override: {TRUNK_OVERRIDE_DEG:.1f}°")

# ── MuJoCo ────────────────────────────────────────────────────────────────────
def apply(model, data, mot, frame, trunk_override_deg=None):
    """Apply MOT frame to model. trunk_override_deg forces trunk lean (forward = positive)."""
    qpos = data.qpos.copy()
    off  = mot['trunk_lateral_lean'][0]
    for jname in ['trunk_lateral_lean','trunk_forward_lean',
                  'sternoclavicular_r2','sternoclavicular_r3',
                  'elv_angle','shoulder_elv','shoulder_rot',
                  'elbow_flexion','pro_sup','deviation','flexion']:
        if jname not in mot: continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0: continue
        val = mot[jname][min(frame, len(mot[jname])-1)]
        if jname == 'trunk_forward_lean':
            val = 0.0
        elif jname == 'trunk_lateral_lean':
            if trunk_override_deg is not None:
                val = -(trunk_override_deg / R2D)
            else:
                val = val - off
        elif jname in ('deviation', 'flexion'):
            val = 0.0
        qpos[model.jnt_qposadr[jid]] = val
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

PW, PH = 640, 776
def rend(model, data, cam):
    r = mujoco.Renderer(model, height=PH, width=PW)
    r.update_scene(data, camera=cam); img = r.render().copy(); r.close(); return img

def clean(img):
    bg = img[3,3].astype(float)
    m  = np.all(np.abs(img.astype(float)-bg) < 25, axis=2)
    o  = img.copy(); o[m] = (248, 249, 252); return o

model = mujoco.MjModel.from_xml_path(MODEL)
d     = mujoco.MjData(model)

# row 1: resting (frame 0, natural pose, both subjects)
apply(model, d, mot_s, f_rest); sr_f=clean(rend(model,d,'front')); sr_s=clean(rend(model,d,'side'))
apply(model, d, mot_h, f_rest); hr_f=clean(rend(model,d,'front')); hr_s=clean(rend(model,d,'side'))

# row 2: drinking frame — stroke with actual trunk lean, healthy natural
apply(model, d, mot_s, f_drink_s, trunk_override_deg=TRUNK_OVERRIDE_DEG)
sd_f=clean(rend(model,d,'front')); sd_s=clean(rend(model,d,'side'))

apply(model, d, mot_h, f_drink_h)
hd_f=clean(rend(model,d,'front')); hd_s=clean(rend(model,d,'side'))

# ── joint angles for table ────────────────────────────────────────────────────
def g(mot, key, fr):
    return float(mot[key][min(fr, len(mot[key])-1)]) * R2D if key in mot else 0.0

off_s = mot_s['trunk_lateral_lean'][0]; off_h = mot_h['trunk_lateral_lean'][0]

def angles(mot, frame, off, trunk_override_deg=None):
    trunk = trunk_override_deg if trunk_override_deg is not None \
            else abs(round((mot['trunk_lateral_lean'][frame]-off)*R2D, 1))
    return dict(
        trunk = round(trunk, 1),
        shld  = round(g(mot,'shoulder_elv',  frame), 1),
        elbow = round(g(mot,'elbow_flexion', frame), 1),
        pron  = round(g(mot,'pro_sup',       frame), 1),
        sc    = round(g(mot,'sternoclavicular_r3', frame), 1),
    )

A_sr = angles(mot_s, f_rest,    off_s)
A_hr = angles(mot_h, f_rest,    off_h)
A_sd = angles(mot_s, f_drink_s, off_s, trunk_override_deg=TRUNK_OVERRIDE_DEG)
A_hd = angles(mot_h, f_drink_h, off_h)

for lbl, A in [('S rest',A_sr),('H rest',A_hr),('S drink',A_sd),('H drink',A_hd)]:
    print(f"{lbl}: trunk={A['trunk']}° shld={A['shld']}° elbow={A['elbow']}° pron={A['pron']}° sc={A['sc']}°")

# ── colours ───────────────────────────────────────────────────────────────────
SBr = '#C0392B'   # stroke accent
HBr = '#2471A3'   # healthy accent

# ── layout ────────────────────────────────────────────────────────────────────
DPI    = 300
PAD    = 0.12
GAP    = 0.04
DIV    = 0.26
ROW_GAP= 0.16
HDR_H  = 0.24      # reduced: plain text only, no boxes
ROW_LBL= 0.26
VPAD_B = 0.08

PW_in = PW / DPI
PH_in = PH / DPI

fig_w = ROW_LBL + PAD + PW_in+GAP+PW_in + DIV + PW_in+GAP+PW_in + PAD
fig_h = HDR_H + PH_in + ROW_GAP + PH_in + VPAD_B

fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

def ax_(l, b, w, h):
    return fig.add_axes([l/fig_w, b/fig_h, w/fig_w, h/fig_h])

y_drink = VPAD_B
y_rest  = y_drink + PH_in + ROW_GAP
y_hdr   = y_rest  + PH_in

x0  = ROW_LBL + PAD
xSF = x0
xSS = xSF + PW_in + GAP
xHF = xSS + PW_in + DIV
xHS = xHF + PW_in + GAP

panel_w = PW_in*2 + GAP

def place_panel(x, y, img):
    a = ax_(x, y, PW_in, PH_in)
    a.imshow(img); a.set_xlim(0,PW-1); a.set_ylim(PH-1,0); a.axis('off')
    return a

# row 1 — resting
for x, img in [(xSF,sr_f),(xSS,sr_s),(xHF,hr_f),(xHS,hr_s)]:
    place_panel(x, y_rest, img)

# row 2 — drinking
for x, img in [(xSF,sd_f),(xSS,sd_s),(xHF,hd_f),(xHS,hd_s)]:
    place_panel(x, y_drink, img)

def draw_trunk_arc(ax, deg, col):
    cx, cy = PW*0.40, PH*0.20; r = PH*0.21
    ax.plot([cx,cx],[cy,cy+r*1.08], color=col, lw=1.2,
            ls=(0,(5,4)), alpha=0.6, zorder=5)
    if deg > 0.5:
        ang = math.radians(deg)
        ex  = cx + r*math.sin(ang); ey = cy + r*math.cos(ang)
        ax.annotate('', xy=(ex,ey), xytext=(cx,cy),
                    arrowprops=dict(arrowstyle='->', color=col,
                                   lw=2.0, mutation_scale=14), zorder=6)
        from matplotlib.patches import Arc as MArc
        ax.add_patch(MArc((cx,cy),r*0.70,r*0.70,angle=0,
                          theta1=90-deg,theta2=90,color=col,lw=1.7,zorder=6))
        mid = math.radians(deg/2)
        lx  = cx + r*0.50*math.sin(mid) + PW*0.04; ly = cy + r*0.50*math.cos(mid)
        ax.text(lx, ly, f'{deg:.1f}°', fontsize=10, color=col, fontweight='bold',
                ha='center', va='center', zorder=7,
                bbox=dict(fc='white', ec=col, lw=0.9,
                          boxstyle='round,pad=0.25', alpha=0.95))
    else:
        ax.text(cx+PW*0.12, cy+r*0.5, '≈0°', fontsize=9, color=col, fontweight='bold',
                ha='center', va='center', zorder=7,
                bbox=dict(fc='white', ec=col, lw=0.9,
                          boxstyle='round,pad=0.25', alpha=0.95))

# arc on stroke side panel (drink row)
a = ax_(xSS, y_drink, PW_in, PH_in)
a.imshow(sd_s); a.set_xlim(0,PW-1); a.set_ylim(PH-1,0); a.axis('off')
draw_trunk_arc(a, A_sd['trunk'], SBr)

# annotation on healthy side panel
a = ax_(xHS, y_drink, PW_in, PH_in)
a.imshow(hd_s); a.set_xlim(0,PW-1); a.set_ylim(PH-1,0); a.axis('off')
draw_trunk_arc(a, A_hd['trunk'], HBr)

# ── row labels ────────────────────────────────────────────────────────────────
for yR, lbl in [(y_rest, 'Resting\npose'), (y_drink, 'During\ndrinking')]:
    a = ax_(0, yR, ROW_LBL-0.05, PH_in)
    a.set_xlim(0,1); a.set_ylim(0,1); a.axis('off')
    a.text(0.85, 0.50, lbl, transform=a.transAxes,
           ha='right', va='center', fontsize=9, fontweight='bold',
           color='#444', linespacing=1.5)
    a.plot([0.88,0.88],[0.15,0.85], transform=a.transAxes,
           color='#CCCCCC', lw=1.5, solid_capstyle='round')

# ── view labels ───────────────────────────────────────────────────────────────
for yR in [y_rest, y_drink]:
    for x, lbl in [(xSF,'Front'),(xSS,'Side'),(xHF,'Front'),(xHS,'Side')]:
        fig.text((x+PW_in/2)/fig_w, (yR-0.07)/fig_h,
                 lbl, ha='center', va='top', fontsize=7.5, color='#888', style='italic')

# ── group headers — plain bold text, no boxes ─────────────────────────────────
for gx, title, col in [
        (xSF, 'Stroke  —  Subject S14_12_3',  SBr),
        (xHF, 'Healthy  —  Subject 09_12_2',  HBr)]:
    cx = (gx + panel_w/2) / fig_w
    cy = (y_hdr + HDR_H*0.50) / fig_h
    fig.text(cx, cy, title, ha='center', va='center',
             fontsize=10, fontweight='bold', color=col)

# ── row separator ─────────────────────────────────────────────────────────────
sep_y = (y_drink + PH_in + ROW_GAP/2) / fig_h
fig.add_artist(plt.Line2D([ROW_LBL/fig_w, (fig_w-PAD)/fig_w], [sep_y, sep_y],
               transform=fig.transFigure, color='#DDDDDD', lw=1.2, ls='--'))

# ── central divider ───────────────────────────────────────────────────────────
xd = (xHF - DIV/2) / fig_w
fig.add_artist(plt.Line2D([xd,xd], [VPAD_B/fig_h, (y_hdr+HDR_H-0.04)/fig_h],
               transform=fig.transFigure, color='#D0D3D7', lw=1.2, ls='--'))

fig.savefig(OUT, dpi=DPI, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"\nSaved → {OUT}")

print("\n--- LaTeX table values ---")
for lbl, A in [('S rest',A_sr),('H rest',A_hr),('S drink',A_sd),('H drink',A_hd)]:
    print(f"{lbl}: trunk={A['trunk']} shld={A['shld']} elbow={A['elbow']} pron={A['pron']} sc={A['sc']}")
