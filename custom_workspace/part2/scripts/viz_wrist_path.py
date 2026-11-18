"""
Animated wrist path comparison — Stroke vs Healthy.
Shows XZ (top-down) and XY (side view) of the wrist trajectory
from reach start to peak, normalised to start at origin.
Uses pre-computed phase indices to isolate only the forward reach phase.
"""
import matplotlib
matplotlib.use("Agg")
import os, sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root

ROOT            = get_project_root()
WORKSPACE       = os.path.dirname(ROOT)
STROKE_CSV_DIR  = os.path.join(WORKSPACE, get_path("data_stroke"))
HEALTHY_CSV_DIR = os.path.join(WORKSPACE, get_path("data_healthy"))
OUT             = os.path.join(ROOT, "output", "trunk_lean_validation", "wrist_path_animation.mp4")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

N_STROKE  = 6   # trials to show per group
N_HEALTHY = 6
SC = "#C0392B"; HC = "#2980B9"

PHASE_IDX_STROKE  = json.load(open(os.path.join(ROOT, "output", "phase_indices", "stroke_phase_indices.json")))
PHASE_IDX_HEALTHY = json.load(open(os.path.join(ROOT, "output", "phase_indices", "healthy_phase_indices.json")))


def load_csv(path):
    raw = pd.read_csv(path, header=[0, 1])
    current, tuples = None, []
    for col in raw.columns:
        m, ax = col
        if not str(m).startswith("Unnamed"):
            current = m.strip()
        tuples.append((current, ax))
    raw.columns = pd.MultiIndex.from_tuples(tuples)
    return raw.iloc[1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)


def get_wrist(df):
    avail = [m for m in ["WRA", "WRB"] if m in df.columns.get_level_values(0)]
    if not avail: return None
    wx = df[avail].xs("X", level=1, axis=1).mean(axis=1).values
    wy = df[avail].xs("Y", level=1, axis=1).mean(axis=1).values
    wz = df[avail].xs("Z", level=1, axis=1).mean(axis=1).values
    return np.stack([wx, wy, wz], axis=1)


def load_trials(csv_dir, phase_idx, n):
    trials = []
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"): continue
        phases = phase_idx.get(fname)
        if phases is None: continue
        sf, pf = int(phases[0]), int(phases[1])
        try:
            df  = load_csv(os.path.join(csv_dir, fname))
            xyz = get_wrist(df)
            if xyz is None: continue
            sf  = min(sf, len(xyz) - 1)
            pf  = min(pf, len(xyz) - 1)
            exc_peak = float(np.sqrt((xyz[pf, 0] - xyz[sf, 0])**2 +
                                     (xyz[pf, 2] - xyz[sf, 2])**2))
            seg = xyz[sf:pf+1]
            seg = seg - seg[0]
            path_len = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))
            straight  = np.linalg.norm(seg[-1] - seg[0])
            ratio = path_len / straight if straight > 1.0 else np.nan
            trials.append(dict(name=fname.replace(".csv",""), seg=seg,
                               ratio=ratio, exc=exc_peak))
        except:
            continue
        if len(trials) >= n:
            break
    return trials


stroke_trials  = load_trials(STROKE_CSV_DIR,  PHASE_IDX_STROKE,  N_STROKE)
healthy_trials = load_trials(HEALTHY_CSV_DIR, PHASE_IDX_HEALTHY, N_HEALTHY)
print(f"Loaded {len(stroke_trials)} stroke, {len(healthy_trials)} healthy trials")


# ── Figure setup ─────────────────────────────────────────────────────────────
# 2 rows × 2 cols: [top-down XZ | side XY] for each group
# Left col = top-down (XZ), right col = side view (YZ forward axis)

plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0D0D0D")
fig.patch.set_facecolor("#0D0D0D")
for ax in axes.flat:
    ax.set_facecolor("#111111")
    ax.tick_params(colors="white"); ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white"); ax.title.set_color("white")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")

axes[0,0].set_title("STROKE — Top-down (X vs Z)", fontsize=12, fontweight="bold", color=SC)
axes[1,0].set_title("HEALTHY — Top-down (X vs Z)", fontsize=12, fontweight="bold", color=HC)
axes[0,1].set_title("STROKE — Side view (Z vs Y)", fontsize=12, fontweight="bold", color=SC)
axes[1,1].set_title("HEALTHY — Side view (Z vs Y)", fontsize=12, fontweight="bold", color=HC)

for ax in axes[:, 0]:
    ax.set_xlabel("X (mm, lateral)", fontsize=10); ax.set_ylabel("Z (mm, forward)", fontsize=10)
for ax in axes[:, 1]:
    ax.set_xlabel("Z (mm, forward)", fontsize=10); ax.set_ylabel("Y (mm, vertical)", fontsize=10)

# compute axis limits across all trials
all_segs = [t["seg"] for t in stroke_trials + healthy_trials]
all_x = np.concatenate([s[:,0] for s in all_segs])
all_y = np.concatenate([s[:,1] for s in all_segs])
all_z = np.concatenate([s[:,2] for s in all_segs])
pad = 20
xlim = (all_x.min()-pad, all_x.max()+pad)
ylim = (all_y.min()-pad, all_y.max()+pad)
zlim = (all_z.min()-pad, all_z.max()+pad)

for ax in axes[:, 0]:
    ax.set_xlim(xlim); ax.set_ylim(zlim)
    ax.axhline(0, color="#333", lw=0.5); ax.axvline(0, color="#333", lw=0.5)
for ax in axes[:, 1]:
    ax.set_xlim(zlim); ax.set_ylim(ylim)
    ax.axhline(0, color="#333", lw=0.5); ax.axvline(0, color="#333", lw=0.5)

# Create line objects
stroke_colors  = plt.cm.Reds(np.linspace(0.4, 0.9, len(stroke_trials)))
healthy_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(healthy_trials)))

s_lines_xz, s_lines_yz, s_dots_xz, s_dots_yz = [], [], [], []
h_lines_xz, h_lines_yz, h_dots_xz, h_dots_yz = [], [], [], []

for i, t in enumerate(stroke_trials):
    c = stroke_colors[i]
    lxz, = axes[0,0].plot([], [], color=c, lw=1.8, alpha=0.9)
    lyz, = axes[0,1].plot([], [], color=c, lw=1.8, alpha=0.9)
    dxz, = axes[0,0].plot([], [], 'o', color=c, ms=6)
    dyz, = axes[0,1].plot([], [], 'o', color=c, ms=6)
    # start dot
    axes[0,0].plot(0, 0, 's', color=c, ms=5, alpha=0.6)
    axes[0,1].plot(0, 0, 's', color=c, ms=5, alpha=0.6)
    s_lines_xz.append(lxz); s_lines_yz.append(lyz)
    s_dots_xz.append(dxz); s_dots_yz.append(dyz)
    r = t["ratio"]; name = t["name"]
    axes[0,0].text(t["seg"][-1,0]+3, t["seg"][-1,2]+3,
                   f'{name}\nr={r:.2f}', fontsize=6.5, color=c, alpha=0.85)

for i, t in enumerate(healthy_trials):
    c = healthy_colors[i]
    lxz, = axes[1,0].plot([], [], color=c, lw=1.8, alpha=0.9)
    lyz, = axes[1,1].plot([], [], color=c, lw=1.8, alpha=0.9)
    dxz, = axes[1,0].plot([], [], 'o', color=c, ms=6)
    dyz, = axes[1,1].plot([], [], 'o', color=c, ms=6)
    axes[1,0].plot(0, 0, 's', color=c, ms=5, alpha=0.6)
    axes[1,1].plot(0, 0, 's', color=c, ms=5, alpha=0.6)
    h_lines_xz.append(lxz); h_lines_yz.append(lyz)
    h_dots_xz.append(dxz); h_dots_yz.append(dyz)
    r = t["ratio"]; name = t["name"]
    axes[1,0].text(t["seg"][-1,0]+3, t["seg"][-1,2]+3,
                   f'{name}\nr={r:.2f}', fontsize=6.5, color=c, alpha=0.85)

# Normalise all trials to the same frame count for smooth animation
N_FRAMES = 120
def resample(seg, n):
    idx = np.linspace(0, len(seg)-1, n).astype(int)
    return seg[idx]

s_segs = [resample(t["seg"], N_FRAMES) for t in stroke_trials]
h_segs = [resample(t["seg"], N_FRAMES) for t in healthy_trials]

# Progress text
prog_text = fig.text(0.5, 0.01, "", ha="center", color="white", fontsize=10)


def animate(frame):
    f = frame + 1
    for i, seg in enumerate(s_segs):
        s_lines_xz[i].set_data(seg[:f, 0], seg[:f, 2])
        s_lines_yz[i].set_data(seg[:f, 2], seg[:f, 1])
        s_dots_xz[i].set_data([seg[f-1, 0]], [seg[f-1, 2]])
        s_dots_yz[i].set_data([seg[f-1, 2]], [seg[f-1, 1]])
    for i, seg in enumerate(h_segs):
        h_lines_xz[i].set_data(seg[:f, 0], seg[:f, 2])
        h_lines_yz[i].set_data(seg[:f, 2], seg[:f, 1])
        h_dots_xz[i].set_data([seg[f-1, 0]], [seg[f-1, 2]])
        h_dots_yz[i].set_data([seg[f-1, 2]], [seg[f-1, 1]])
    prog_text.set_text(f"Reach progress: {f/N_FRAMES*100:.0f}%")
    return (s_lines_xz + s_lines_yz + s_dots_xz + s_dots_yz +
            h_lines_xz + h_lines_yz + h_dots_xz + h_dots_yz + [prog_text])


fig.suptitle("Wrist Path: Stroke vs Healthy  (square=start, circle=current position)\n"
             "r = path ratio (1.0 = perfectly straight)",
             color="white", fontsize=11, fontweight="bold")

anim = animation.FuncAnimation(fig, animate, frames=N_FRAMES,
                                interval=40, blit=True)

print("Saving animation...")
writer = animation.FFMpegWriter(fps=25, bitrate=1800)
anim.save(OUT, writer=writer)
print(f"Saved: {OUT}")
plt.show()
