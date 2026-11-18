"""
Animated CS marker plane visualiser.

Left  — 3D: four CS markers, fitted plane patch, normal arrow, current tilt angle.
Right — 2D: absolute tilt over full trial with animated playhead.

Usage:
    python scripts/viz_cs_plane.py              # default S5_12_1
    python scripts/viz_cs_plane.py S13_12_2
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root

ROOT            = get_project_root()
WORKSPACE       = os.path.dirname(ROOT)
STROKE_CSV_DIR  = os.path.join(WORKSPACE, get_path("data_stroke"))
HEALTHY_CSV_DIR = os.path.join(WORKSPACE, get_path("data_healthy"))

CS_MARKERS  = ["CS_1", "CS_2", "CS_3", "CS_4"]
UP_AXIS     = np.array([0.0, 1.0, 0.0])
MARKER_COLORS = ["#E74C3C", "#E67E22", "#9B59B6", "#1ABC9C"]

# Arm marker groups: label, raw marker names, display colour
ARM_GROUPS = [
    ("Shoulder", ["SA_1", "SA_2", "SA_3"],   "#F1C40F"),
    ("Elbow",    ["ELB_M", "ELB_L"],         "#2ECC71"),
    ("Wrist",    ["WRA", "WRB"],              "#3498DB"),
]


# ── Data helpers ──────────────────────────────────────────────────────────────

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


def get_cs_points(df):
    avail = [m for m in CS_MARKERS if m in df.columns.get_level_values(0)]
    x = df[avail].xs("X", level=1, axis=1).values
    y = df[avail].xs("Y", level=1, axis=1).values
    z = df[avail].xs("Z", level=1, axis=1).values
    return np.stack([x, y, z], axis=2), avail   # (N, k, 3)


def get_arm_points(df):
    """Returns dict: label -> (N, 3) mean position array."""
    result = {}
    for label, names, _ in ARM_GROUPS:
        avail = [m for m in names if m in df.columns.get_level_values(0)]
        if not avail:
            continue
        x = df[avail].xs("X", level=1, axis=1).mean(axis=1).values
        y = df[avail].xs("Y", level=1, axis=1).mean(axis=1).values
        z = df[avail].xs("Z", level=1, axis=1).mean(axis=1).values
        result[label] = np.stack([x, y, z], axis=1)  # (N, 3)
    return result


def fit_plane(pts):
    c = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    return Vt[-1], c


def compute_tilt(points):
    angles, prev_n = np.full(len(points), np.nan), None
    for i, pts in enumerate(points):
        if np.any(np.isnan(pts)):
            prev_n = None; continue
        n, _ = fit_plane(pts)
        if n[1] < 0:
            n = -n
        prev_n = n
        angles[i] = np.degrees(np.arccos(np.clip(abs(np.dot(n, UP_AXIS)), -1, 1)))
    return angles


def plane_patch_corners(centroid, normal, size=80):
    t1 = np.array([1., 0., 0.])
    t1 -= np.dot(t1, normal) * normal
    if np.linalg.norm(t1) < 1e-6:
        t1 = np.array([0., 0., 1.])
    t1 /= np.linalg.norm(t1)
    t2  = np.cross(normal, t1); t2 /= np.linalg.norm(t2)
    return [centroid + size * (a*t1 + b*t2)
            for a, b in [(-1,-1),(1,-1),(1,1),(-1,1)]]


# ── Load data ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("subject", nargs="?", default="S5_12_1")
args   = parser.parse_args()
stem   = args.subject

csv_path = os.path.join(STROKE_CSV_DIR, stem + ".csv")
if not os.path.exists(csv_path):
    csv_path = os.path.join(HEALTHY_CSV_DIR, stem + ".csv")
if not os.path.exists(csv_path):
    print(f"CSV not found for {stem}"); sys.exit(1)

df            = load_csv(csv_path)
points, avail = get_cs_points(df)          # (N, k, 3)
arm_pts       = get_arm_points(df)         # dict: label -> (N, 3)
tilt_angles   = compute_tilt(points)
time_vec    = np.arange(len(points)) / 200.0

valid_mask = ~np.any(np.isnan(points.reshape(len(points), -1)), axis=1)
frames     = np.where(valid_mask)[0]
step       = max(1, len(frames) // 120)
frames     = frames[::step]
print(f"  {len(frames)} animation frames  (total valid={valid_mask.sum()})")

# Axis limits — include arm markers
all_pts = points[valid_mask].reshape(-1, 3)
for ap in arm_pts.values():
    all_pts = np.vstack([all_pts, ap[~np.isnan(ap).any(axis=1)]])
cx, cy, cz = np.nanmean(all_pts, 0)
rng = np.nanmax(np.linalg.norm(all_pts - [cx,cy,cz], axis=1)) * 1.3


# ── Figure layout ─────────────────────────────────────────────────────────────

plt.rcParams.update({"font.family": "sans-serif"})
fig = plt.figure(figsize=(15, 7), facecolor="#1A1A2E")
gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.3, 1], wspace=0.05)

# 3D panel
ax3d = fig.add_subplot(gs[0], projection="3d")
ax3d.set_facecolor("#16213E")
for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor("#333355")
ax3d.tick_params(colors="#AAAACC", labelsize=7)
ax3d.set_xlabel("X", color="#AAAACC", fontsize=9, labelpad=5)
ax3d.set_ylabel("Y", color="#AAAACC", fontsize=9, labelpad=5)
ax3d.set_zlabel("Z", color="#AAAACC", fontsize=9, labelpad=5)
ax3d.set_xlim(cx-rng, cx+rng)
ax3d.set_ylim(cy-rng, cy+rng)
ax3d.set_zlim(cz-rng, cz+rng)
ax3d.set_title(f"CS Marker Plane  —  {stem}", color="white",
               fontsize=12, fontweight="bold", pad=10)

# 2D panel
ax2d = fig.add_subplot(gs[1])
ax2d.set_facecolor("#16213E")
ax2d.tick_params(colors="#AAAACC"); ax2d.spines[:].set_color("#333355")
for label in ax2d.get_xticklabels() + ax2d.get_yticklabels():
    label.set_color("#AAAACC")
ax2d.set_xlabel("Time (s)", color="white", fontsize=10)
ax2d.set_ylabel("Absolute trunk tilt (°)", color="white", fontsize=10)
ax2d.set_title("Trunk Tilt from Vertical", color="white", fontsize=11,
               fontweight="bold")
ax2d.grid(True, alpha=0.15, color="white", linestyle="--")

# Plot the full tilt curve with a gradient (green → red)
valid_t = time_vec[valid_mask]
valid_a = tilt_angles[valid_mask]
from matplotlib.collections import LineCollection
pts2d  = np.array([valid_t, valid_a]).T.reshape(-1, 1, 2)
segs   = np.concatenate([pts2d[:-1], pts2d[1:]], axis=1)
norm_c = plt.Normalize(valid_a.min(), valid_a.max())
lc     = LineCollection(segs, cmap="RdYlGn_r", norm=norm_c, lw=2, alpha=0.85)
lc.set_array(valid_a[:-1])
ax2d.add_collection(lc)
ax2d.set_xlim(valid_t[0], valid_t[-1])
ax2d.set_ylim(0, valid_a.max() * 1.15)

cbar = fig.colorbar(lc, ax=ax2d, fraction=0.03, pad=0.02)
cbar.ax.tick_params(colors="#AAAACC", labelsize=8)
cbar.set_label("Tilt (°)", color="#AAAACC", fontsize=8)

# Playhead + angle readout
vline    = ax2d.axvline(valid_t[0], color="#FFD700", lw=1.5, ls="--", alpha=0.8)
angle_txt= ax2d.text(0.97, 0.93, "", transform=ax2d.transAxes,
                     ha="right", va="top", fontsize=14, fontweight="bold",
                     color="#FFD700")


# ── 3D artists ────────────────────────────────────────────────────────────────

# CS marker scatters
scatters = [ax3d.plot([], [], [], 'o', color=MARKER_COLORS[k % 4],
                      ms=10, zorder=5, label=avail[k] if k < len(avail) else "")[0]
            for k in range(len(avail))]

# Arm marker scatters (one per group)
arm_scatters = {}
for label, _, color in ARM_GROUPS:
    if label in arm_pts:
        arm_scatters[label] = ax3d.plot([], [], [], 'o', color=color,
                                        ms=12, zorder=5, label=label)[0]

# Skeleton lines: shoulder→elbow→wrist
skel_lines = {}
for pair in [("Shoulder", "Elbow"), ("Elbow", "Wrist")]:
    if all(p in arm_pts for p in pair):
        skel_lines[pair] = ax3d.plot([], [], [], '-', color="#FFFFFF",
                                     lw=1.5, alpha=0.6)[0]

ax3d.legend(loc="upper left", fontsize=8, facecolor="#16213E",
            edgecolor="#333355", labelcolor="white")

# Plane patch placeholder
patch_holder = [None]

# Normal arrow (quiver)
quiver_holder = [None]

# Tilt angle arc text on 3D
tilt_3d_txt = ax3d.text2D(0.02, 0.95, "", transform=ax3d.transAxes,
                           color="#FFD700", fontsize=13, fontweight="bold")

prev_n = [None]


def update(fi):
    i   = frames[fi]
    pts = points[i]   # (k, 3)

    # Update marker positions
    for k, sc in enumerate(scatters):
        sc.set_data([pts[k, 0]], [pts[k, 1]])
        sc.set_3d_properties([pts[k, 2]])

    # Fit plane
    n, c = fit_plane(pts)
    if prev_n[0] is not None and np.dot(n, prev_n[0]) < 0:
        n = -n
    prev_n[0] = n.copy()

    # Plane patch
    if patch_holder[0] is not None:
        patch_holder[0].remove()
    corners = plane_patch_corners(c, n, size=70)
    verts = [[list(p) for p in corners]]
    poly  = Poly3DCollection(verts, alpha=0.20,
                             facecolor="#00BFFF", edgecolor="#00BFFF",
                             linewidth=0.8)
    ax3d.add_collection3d(poly)
    patch_holder[0] = poly

    # Normal arrow (quiver)
    if quiver_holder[0] is not None:
        quiver_holder[0].remove()
    scale = 90
    q = ax3d.quiver(c[0], c[1], c[2],
                    n[0]*scale, n[1]*scale, n[2]*scale,
                    color="#FFD700", linewidth=2.5, arrow_length_ratio=0.25)
    quiver_holder[0] = q

    # Arm markers
    for label, sc in arm_scatters.items():
        p = arm_pts[label][i]
        if not np.any(np.isnan(p)):
            sc.set_data([p[0]], [p[1]])
            sc.set_3d_properties([p[2]])

    # Skeleton lines
    for (a, b), line in skel_lines.items():
        pa, pb = arm_pts[a][i], arm_pts[b][i]
        if not np.any(np.isnan(pa)) and not np.any(np.isnan(pb)):
            line.set_data([pa[0], pb[0]], [pa[1], pb[1]])
            line.set_3d_properties([pa[2], pb[2]])

    # Tilt angle
    ang = tilt_angles[i]
    txt = f"{ang:.1f}°" if not np.isnan(ang) else "—"
    tilt_3d_txt.set_text(f"Tilt: {txt}")
    angle_txt.set_text(f"{txt}")

    # Playhead
    t_now = i / 200.0
    vline.set_xdata([t_now, t_now])

    return scatters + list(arm_scatters.values()) + list(skel_lines.values()) + \
           [vline, tilt_3d_txt, angle_txt]


ani = animation.FuncAnimation(fig, update, frames=len(frames),
                               interval=50, blit=False)
plt.tight_layout(pad=1.5)
plt.show()
