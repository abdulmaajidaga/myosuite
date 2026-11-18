"""
CS plane absolute tilt — Stroke vs Healthy group comparison.

Absolute tilt = angle between the CS plane normal and vertical (Y axis).
Zero = subject sitting perfectly upright. Higher = more trunk lean.

Each trial time-normalised to 0–100% of the reach.
"""
import os, sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root

ROOT            = get_project_root()
WORKSPACE       = os.path.dirname(ROOT)
STROKE_CSV_DIR  = os.path.join(WORKSPACE, get_path("data_stroke"))
HEALTHY_CSV_DIR = os.path.join(WORKSPACE, get_path("data_healthy"))

CS_MARKERS = ["CS_1", "CS_2", "CS_3", "CS_4"]
UP_AXIS    = np.array([0.0, 1.0, 0.0])
N_NORM     = 101

STROKE_COL  = "#C0392B"
HEALTHY_COL = "#2980B9"

PHASE_IDX_STROKE  = json.load(open(os.path.join(ROOT, "output", "phase_indices", "stroke_phase_indices.json")))
PHASE_IDX_HEALTHY = json.load(open(os.path.join(ROOT, "output", "phase_indices", "healthy_phase_indices.json")))

def get_phase(fname, group):
    idx = PHASE_IDX_STROKE if group == "Stroke" else PHASE_IDX_HEALTHY
    phases = idx.get(fname)
    if phases is None: return None, None
    return int(phases[0]), int(phases[1])


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    if len(avail) < 3:
        return None
    x = df[avail].xs("X", level=1, axis=1).values
    y = df[avail].xs("Y", level=1, axis=1).values
    z = df[avail].xs("Z", level=1, axis=1).values
    return np.stack([x, y, z], axis=2)  # (N, k, 3)


def compute_absolute_tilt(points):
    """Angle between CS plane normal and vertical (Y) — absolute, not relative."""
    angles = np.full(len(points), np.nan)
    prev_n = None
    for i, pts in enumerate(points):
        if np.any(np.isnan(pts)):
            prev_n = None
            continue
        c = pts.mean(0)
        _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
        n = Vt[-1]; n /= np.linalg.norm(n)
        if n[1] < 0:
            n = -n
        prev_n = n
        angles[i] = np.degrees(np.arccos(np.clip(abs(np.dot(n, UP_AXIS)), -1, 1)))
    return angles


def normalise(angles):
    valid = np.where(~np.isnan(angles))[0]
    if len(valid) < 10:
        return None
    t_orig = (valid - valid[0]) / (valid[-1] - valid[0])
    return interp1d(t_orig, angles[valid], kind="linear")(np.linspace(0, 1, N_NORM))


def process_dir(csv_dir, label):
    curves, peaks = [], []
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"):
            continue
        sf, pf = get_phase(fname, label)
        if sf is None:
            continue
        try:
            df  = load_csv(os.path.join(csv_dir, fname))
            pts = get_cs_points(df)
            if pts is None:
                continue
            sf = min(sf, len(pts) - 1)
            pf = min(pf, len(pts) - 1)
            ang = compute_absolute_tilt(pts[sf:pf+1])
            nrm = normalise(ang)
            if nrm is not None:
                curves.append(nrm)
                peaks.append(np.max(nrm))
        except Exception as e:
            print(f"  skip {fname}: {e}")
    print(f"{label}: {len(curves)} trials  mean peak={np.mean(peaks):.1f}°  std={np.std(peaks):.1f}°")
    return curves, peaks


# ── Run ───────────────────────────────────────────────────────────────────────

stroke_curves,  stroke_peaks  = process_dir(STROKE_CSV_DIR,  "Stroke")
healthy_curves, healthy_peaks = process_dir(HEALTHY_CSV_DIR, "Healthy")

t = np.linspace(0, 100, N_NORM)

plt.rcParams.update({
    "font.family": "sans-serif", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linestyle": "--",
})

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#F7F9FC")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── Top row: Stroke | Healthy ─────────────────────────────────────────────────
for col_idx, (curves, peaks, label, color) in enumerate([
    (stroke_curves,  stroke_peaks,  "Stroke",  STROKE_COL),
    (healthy_curves, healthy_peaks, "Healthy", HEALTHY_COL),
]):
    ax = fig.add_subplot(gs[0, col_idx])
    arr  = np.array(curves)
    mean = arr.mean(0); sd = arr.std(0)

    for c in arr:
        ax.plot(t, c, color=color, alpha=0.12, lw=0.7)
    ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.25)
    ax.plot(t, mean, color=color, lw=2.5, label=f"Mean  {mean.max():.1f}° peak")

    ax.set_xlim(0, 100); ax.set_ylim(0, None)
    ax.set_xlabel("Normalised time (%)", fontsize=10)
    ax.set_ylabel("Absolute trunk tilt (°)", fontsize=10)
    ax.set_title(f"{label}  (n={len(curves)})", fontsize=12, fontweight="bold",
                 color=color)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.set_facecolor("#F7F9FC")

# ── Top right: overlaid comparison ───────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
for curves, label, color in [
    (stroke_curves,  "Stroke",  STROKE_COL),
    (healthy_curves, "Healthy", HEALTHY_COL),
]:
    arr  = np.array(curves)
    mean = arr.mean(0); sd = arr.std(0)
    ax3.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.20)
    ax3.plot(t, mean, color=color, lw=2.5, label=f"{label} mean")

ax3.set_xlim(0, 100); ax3.set_ylim(0, None)
ax3.set_xlabel("Normalised time (%)", fontsize=10)
ax3.set_ylabel("Absolute trunk tilt (°)", fontsize=10)
ax3.set_title("Stroke vs Healthy", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9, framealpha=0.7)
ax3.set_facecolor("#F7F9FC")

# ── Bottom: peak tilt distributions ──────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
for i, (peaks, label, color) in enumerate([
    (stroke_peaks,  "Stroke",  STROKE_COL),
    (healthy_peaks, "Healthy", HEALTHY_COL),
]):
    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(peaks))
    ax4.scatter(peaks, np.full(len(peaks), i) + jitter,
                color=color, alpha=0.55, s=50, zorder=3)
    ax4.barh(i, np.mean(peaks), height=0.25, color=color,
             alpha=0.35, xerr=np.std(peaks), capsize=4,
             error_kw={"elinewidth": 1.5, "ecolor": color})
    ax4.text(np.mean(peaks) + np.std(peaks) + 0.5, i,
             f"  {np.mean(peaks):.1f}° ± {np.std(peaks):.1f}°",
             va="center", fontsize=10, color=color, fontweight="bold")

ax4.set_yticks([0, 1]); ax4.set_yticklabels(["Stroke", "Healthy"], fontsize=11)
ax4.set_xlabel("Peak absolute trunk tilt (°)", fontsize=10)
ax4.set_title("Peak tilt distribution per group", fontsize=12, fontweight="bold")
ax4.set_facecolor("#F7F9FC")
ax4.set_xlim(0, None)

# ── Bottom right: trial count & stats box ────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
stats_text = (
    f"Stroke\n"
    f"  N = {len(stroke_curves)} trials\n"
    f"  Mean peak = {np.mean(stroke_peaks):.1f}°\n"
    f"  SD         = {np.std(stroke_peaks):.1f}°\n"
    f"  Max        = {np.max(stroke_peaks):.1f}°\n\n"
    f"Healthy\n"
    f"  N = {len(healthy_curves)} trials\n"
    f"  Mean peak = {np.mean(healthy_peaks):.1f}°\n"
    f"  SD         = {np.std(healthy_peaks):.1f}°\n"
    f"  Max        = {np.max(healthy_peaks):.1f}°"
)
ax5.text(0.1, 0.95, stats_text, transform=ax5.transAxes,
         fontsize=10, va="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                   edgecolor="#CCCCCC", alpha=0.9))
ax5.set_facecolor("#F7F9FC")

fig.suptitle("CS Marker Plane — Absolute Trunk Tilt (Reach Phase Only)\nStroke vs Healthy",
             fontsize=14, fontweight="bold", y=0.98)

out = os.path.join(ROOT, "output", "trunk_lean_validation", "cs_group_comparison.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
plt.show()
