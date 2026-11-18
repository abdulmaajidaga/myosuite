"""
Five raw-marker compensation metrics — Stroke vs Healthy.

1. Shoulder elevation    — SA markers ΔY (mm) start→peak
2. Elbow height          — ELB_L/M ΔY (mm) start→peak
3. Reach path ratio      — actual wrist path / straight-line distance
4. Movement smoothness   — # velocity peaks in wrist speed (start→peak)
5. CS axial rotation     — trunk twist around vertical (°) start→peak

Bottom-right panel: reach-window verification on example trials.
"""
import os, sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import argrelmin, argrelmax, savgol_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root

ROOT            = get_project_root()
WORKSPACE       = os.path.dirname(ROOT)
STROKE_CSV_DIR  = os.path.join(WORKSPACE, get_path("data_stroke"))
HEALTHY_CSV_DIR = os.path.join(WORKSPACE, get_path("data_healthy"))
OUT             = os.path.join(ROOT, "output", "trunk_lean_validation", "real_data_metrics.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

PHASE_IDX_STROKE  = json.load(open(os.path.join(ROOT, "output", "phase_indices", "stroke_phase_indices.json")))
PHASE_IDX_HEALTHY = json.load(open(os.path.join(ROOT, "output", "phase_indices", "healthy_phase_indices.json")))

def get_phase(fname, group):
    idx = PHASE_IDX_STROKE if group == "Stroke" else PHASE_IDX_HEALTHY
    phases = idx.get(fname)
    if phases is None: return None, None
    return int(phases[0]), int(phases[1])

CS_MARKERS = ["CS_1", "CS_2", "CS_3", "CS_4"]
SA_MARKERS = ["SA_1", "SA_2", "SA_3"]
ELB_MARKERS = ["ELB_L", "ELB_M"]
UP_AXIS = np.array([0.0, 1.0, 0.0])
DATA_RATE = 200   # Hz
SC = "#C0392B"; HC = "#2980B9"


# ── Loader ────────────────────────────────────────────────────────────────────

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


# ── Reach window ──────────────────────────────────────────────────────────────

def wrist_excursion(df):
    avail = [m for m in ["WRA", "WRB"] if m in df.columns.get_level_values(0)]
    if not avail:
        return None, None
    wx = df[avail].xs("X", level=1, axis=1).mean(axis=1).values
    wy = df[avail].xs("Y", level=1, axis=1).mean(axis=1).values
    wz = df[avail].xs("Z", level=1, axis=1).mean(axis=1).values
    dx = wx - wx[0]; dz = wz - wz[0]
    exc = np.sqrt(dx**2 + dz**2)
    return exc, np.stack([wx, wy, wz], axis=1)



# ── Metric helpers ────────────────────────────────────────────────────────────

def marker_y(df, markers):
    avail = [m for m in markers if m in df.columns.get_level_values(0)]
    if not avail:
        return None
    return df[avail].xs("Y", level=1, axis=1).mean(axis=1).values


def reach_path_ratio(wrist_xyz, start_f, peak_f):
    segment = wrist_xyz[start_f:peak_f+1]
    if len(segment) < 2:
        return np.nan
    diffs = np.diff(segment, axis=0)
    path_len = np.sum(np.linalg.norm(diffs, axis=1))
    straight  = np.linalg.norm(segment[-1] - segment[0])
    return path_len / straight if straight > 1.0 else np.nan


MIN_SPEED_MM_S = 20   # ignore peaks below 20 mm/s — filters noise and micro-pauses

def velocity_peaks(wrist_xyz, start_f, peak_f):
    segment = wrist_xyz[start_f:peak_f+1]
    if len(segment) < 10:
        return np.nan
    speed = np.linalg.norm(np.diff(segment, axis=0), axis=1) * DATA_RATE  # mm/s
    win = min(21, (len(speed) // 2) * 2 - 1)
    if win < 5:
        return np.nan
    smooth = savgol_filter(speed, win, 3)
    # order scaled to window length so it doesn't find micro-peaks in long trials
    order = max(8, len(smooth) // 20)
    peaks = argrelmax(smooth, order=order)[0]
    return int(np.sum(smooth[peaks] > MIN_SPEED_MM_S))


def cs_axial_rotation(df):
    avail = [m for m in CS_MARKERS if m in df.columns.get_level_values(0)]
    if len(avail) < 3:
        return None
    x = df[avail].xs("X", level=1, axis=1).values
    y = df[avail].xs("Y", level=1, axis=1).values
    z = df[avail].xs("Z", level=1, axis=1).values
    pts = np.stack([x, y, z], axis=2)

    angles = np.full(len(df), np.nan)
    for i, row in enumerate(pts):
        if np.any(np.isnan(row)):
            continue
        c = row.mean(0)
        _, _, Vt = np.linalg.svd(row - c, full_matrices=False)
        n = Vt[-1]; n /= np.linalg.norm(n)
        if n[1] < 0:
            n = -n
        # horizontal component of normal → axial orientation of chest
        h = np.array([n[0], 0.0, n[2]])
        hn = np.linalg.norm(h)
        if hn < 1e-6:
            continue
        h /= hn
        angles[i] = np.degrees(np.arctan2(h[0], h[2]))
    return angles


# ── Process all trials ────────────────────────────────────────────────────────

rows = []
window_examples = []   # for verification panel

for csv_dir, group in [(STROKE_CSV_DIR, "Stroke"), (HEALTHY_CSV_DIR, "Healthy")]:
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"):
            continue
        stem = fname.replace(".csv", "")
        try:
            start_f, peak_f = get_phase(fname, group)
            if start_f is None:
                print(f"  skip {stem}: no phase index"); continue

            df = load_csv(os.path.join(csv_dir, fname))
            exc, wrist_xyz = wrist_excursion(df)
            if exc is None:
                continue
            start_f = min(start_f, len(exc) - 1)
            peak_f  = min(peak_f,  len(exc) - 1)

            # 1. Shoulder elevation (SA markers ΔY)
            sa_y = marker_y(df, SA_MARKERS)
            sho_elev = float(sa_y[peak_f] - sa_y[start_f]) if sa_y is not None else np.nan

            # 2. Elbow height (ELB ΔY)
            elb_y = marker_y(df, ELB_MARKERS)
            elb_ht = float(elb_y[peak_f] - elb_y[start_f]) if elb_y is not None else np.nan

            # 3. Path ratio
            path_r = reach_path_ratio(wrist_xyz, start_f, peak_f)

            # 4. Velocity peaks
            n_peaks = velocity_peaks(wrist_xyz, start_f, peak_f)

            # 5. CS axial rotation
            axial = cs_axial_rotation(df)
            ax_comp = float(axial[peak_f] - axial[start_f]) if axial is not None else np.nan
            # wrap to [-180, 180]
            ax_comp = ((ax_comp + 180) % 360) - 180

            if any(np.isnan(v) for v in [sho_elev, path_r]):
                print(f"  skip {stem}: missing SA or wrist"); continue

            rows.append(dict(
                subject=stem, group=group,
                sho_elev=round(sho_elev, 1),
                elb_ht=round(elb_ht, 1) if not np.isnan(elb_ht) else np.nan,
                path_ratio=round(path_r, 3),
                vel_peaks=n_peaks,
                ax_rot=round(ax_comp, 2) if not np.isnan(ax_comp) else np.nan,
            ))

            # Store a few examples for the window-verification panel
            if len(window_examples) < 6:
                window_examples.append(dict(
                    stem=stem, group=group,
                    exc=exc, start_f=start_f, peak_f=peak_f,
                ))

            print(f"  {stem:<16} {group:<8} "
                  f"ShoElev={sho_elev:+.1f}mm  ElbHt={elb_ht:+.1f}mm  "
                  f"PathR={path_r:.3f}  VPeaks={n_peaks}  AxRot={ax_comp:+.1f}°")

        except Exception as e:
            print(f"  skip {stem}: {e}")


results = pd.DataFrame(rows)
print(f"\n{len(results)} trials processed")

for g in ["Stroke", "Healthy"]:
    s = results[results.group == g]
    print(f"\n{g} (n={len(s)}):")
    for col, label in [("sho_elev","Shoulder elev (mm)"), ("elb_ht","Elbow height (mm)"),
                       ("path_ratio","Path ratio"), ("vel_peaks","Vel peaks"),
                       ("ax_rot","Axial rot (°)")]:
        v = s[col].dropna()
        print(f"  {label:<22} mean={v.mean():+.2f}  sd={v.std():.2f}")


# ── Plot ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(20, 12), facecolor="#F7F9FC")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle("Raw-Marker Compensation Metrics  —  Stroke vs Healthy",
             fontsize=14, fontweight="bold", y=0.99)

rng = np.random.default_rng(0)
stroke  = results[results.group == "Stroke"]
healthy = results[results.group == "Healthy"]

METRICS = [
    ("sho_elev",  "Shoulder elevation (mm)",  "SA marker ΔY  start→peak",  gs[0, 0]),
    ("elb_ht",    "Elbow height (mm)",         "ELB marker ΔY  start→peak", gs[0, 1]),
    ("path_ratio","Wrist path ratio",          "path length / straight line",gs[0, 2]),
    ("vel_peaks", "Velocity peaks (n)",        "submovements in wrist speed",gs[1, 0]),
    ("ax_rot",    "CS axial rotation (°)",     "chest twist  start→peak",    gs[1, 1]),
]

for col, ylabel, subtitle, loc in METRICS:
    ax = fig.add_subplot(loc)
    ax.set_facecolor("#F7F9FC")
    s_vals = stroke[col].dropna().values
    h_vals = healthy[col].dropna().values

    for xi, (vals, color) in enumerate([(s_vals, SC), (h_vals, HC)]):
        bp = ax.boxplot(vals, positions=[xi], widths=0.4, patch_artist=True,
                        medianprops=dict(color="white", lw=2),
                        whiskerprops=dict(color=color),
                        capprops=dict(color=color),
                        flierprops=dict(marker="o", color=color, alpha=0.4, ms=4))
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.35)
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(xi + jitter, vals, color=color, alpha=0.7, s=40, zorder=3)
        ax.text(xi, vals.max() + abs(vals.max() - vals.min()) * 0.07 + 0.01,
                f"μ={vals.mean():.1f}\nσ={vals.std():.1f}",
                ha="center", fontsize=8.5, color=color, fontweight="bold")

    t, p = stats.ttest_ind(s_vals, h_vals)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Stroke", "Healthy"], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{subtitle}\np={p:.3f}", fontsize=10)
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
    ax.grid(True, axis="y", alpha=0.25, ls="--")


# ── Window verification panel ─────────────────────────────────────────────────

ax_ver = fig.add_subplot(gs[1, 2])
ax_ver.set_facecolor("#F7F9FC")
t_norm = np.linspace(0, 100, 500)

for ex in window_examples:
    exc   = ex["exc"]
    sf    = ex["start_f"]
    pf    = ex["peak_f"]
    color = SC if ex["group"] == "Stroke" else HC
    n     = len(exc)
    t_raw = np.linspace(0, 100, n)

    # normalise excursion 0-1 for overlay
    exc_n = exc / (exc[pf] + 1e-6)
    ax_ver.plot(t_raw, exc_n, color=color, alpha=0.4, lw=1.2)
    ax_ver.axvline(t_raw[sf], color=color, alpha=0.6, lw=1, ls="--")
    ax_ver.axvline(t_raw[pf], color=color, alpha=0.9, lw=1.5, ls="-")
    ax_ver.scatter([t_raw[sf]], [exc_n[sf]], color=color, s=40, zorder=5, marker="o")
    ax_ver.scatter([t_raw[pf]], [exc_n[pf]], color=color, s=55, zorder=5, marker="^")

from matplotlib.lines import Line2D
legend_els = [
    Line2D([0], [0], color=SC, lw=2, label="Stroke"),
    Line2D([0], [0], color=HC, lw=2, label="Healthy"),
    Line2D([0], [0], color="gray", lw=1, ls="--", label="start (local min)"),
    Line2D([0], [0], color="gray", lw=1.5, ls="-",  label="peak"),
]
ax_ver.legend(handles=legend_els, fontsize=8, framealpha=0.8)
ax_ver.set_xlabel("Time (% of full trial)", fontsize=10)
ax_ver.set_ylabel("Wrist excursion (normalised)", fontsize=10)
ax_ver.set_title("Reach window verification\n(dashed=start, solid=peak)", fontsize=10)
ax_ver.set_xlim(0, 100); ax_ver.set_ylim(-0.05, 1.15)
ax_ver.grid(True, alpha=0.25, ls="--")

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved: {OUT}")
plt.show()
