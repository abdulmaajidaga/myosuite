"""
FMA score correlation with raw-marker compensation metrics.
Stroke subjects plotted against FMA score; healthy subjects shown as a
reference cluster at a notional x=22 (gap after FMA=20 max).
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
OUT             = os.path.join(ROOT, "output", "trunk_lean_validation", "fma_correlation.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

PHASE_IDX_STROKE  = json.load(open(os.path.join(ROOT, "output", "phase_indices", "stroke_phase_indices.json")))
PHASE_IDX_HEALTHY = json.load(open(os.path.join(ROOT, "output", "phase_indices", "healthy_phase_indices.json")))

def get_phase(fname, group):
    idx = PHASE_IDX_STROKE if group == "Stroke" else PHASE_IDX_HEALTHY
    phases = idx.get(fname)
    if phases is None: return None, None
    return int(phases[0]), int(phases[1])

FMA = {1:16, 2:20, 3:19, 4:19, 5:19, 6:14, 7:16, 8:19, 9:19,
       10:14, 11:18, 12:20, 13:19, 14:17, 16:20, 17:17, 18:20,
       19:18, 20:14, 21:17}

HEALTHY_X = 22   # notional x for healthy cluster (gap after stroke max FMA=20)
DATA_RATE = 200
SC = "#C0392B"; HC = "#2980B9"


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


def get_wrist_xyz(df):
    avail = [m for m in ["WRA", "WRB"] if m in df.columns.get_level_values(0)]
    if not avail: return None
    wx = df[avail].xs("X", level=1, axis=1).mean(axis=1).values
    wy = df[avail].xs("Y", level=1, axis=1).mean(axis=1).values
    wz = df[avail].xs("Z", level=1, axis=1).mean(axis=1).values
    return np.stack([wx, wy, wz], axis=1)



def shoulder_elev(df, sf, pf):
    if "SA_3" not in df.columns.get_level_values(0): return np.nan
    y = df["SA_3"]["Y"].values
    return float(y[pf] - y[sf])


def path_ratio(xyz, sf, pf):
    seg = xyz[sf:pf+1]
    if len(seg) < 2: return np.nan
    path = np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))
    straight = np.linalg.norm(seg[-1] - seg[0])
    return path / straight if straight > 1.0 else np.nan


MIN_SPEED_MM_S = 20

def vel_peaks(xyz, sf, pf):
    seg = xyz[sf:pf+1]
    if len(seg) < 10: return np.nan
    speed = np.linalg.norm(np.diff(seg, axis=0), axis=1) * DATA_RATE
    win = min(21, (len(speed) // 2) * 2 - 1)
    if win < 5: return np.nan
    smooth = savgol_filter(speed, win, 3)
    order = max(8, len(smooth) // 20)
    peaks = argrelmax(smooth, order=order)[0]
    return int(np.sum(smooth[peaks] > MIN_SPEED_MM_S))


def elbow_drop(df, sf, pf):
    avail = [m for m in ["ELB_L", "ELB_M"] if m in df.columns.get_level_values(0)]
    if not avail: return np.nan
    y = df[avail].xs("Y", level=1, axis=1).mean(axis=1).values
    return float(y[pf] - y[sf])


def process_dir(csv_dir, group):
    rows = []
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"): continue
        stem = fname.replace(".csv", "")
        try:
            sf, pf = get_phase(fname, group)
            if sf is None:
                print(f"  skip {stem}: no phase index"); continue
            df   = load_csv(os.path.join(csv_dir, fname))
            wxyz = get_wrist_xyz(df)
            if wxyz is None: continue
            sf = min(sf, len(wxyz) - 1)
            pf = min(pf, len(wxyz) - 1)
            rows.append(dict(
                subject=stem, group=group,
                sho_elev  = shoulder_elev(df, sf, pf),
                path_ratio= path_ratio(wxyz, sf, pf),
                vel_peaks = vel_peaks(wxyz, sf, pf),
                elb_drop  = elbow_drop(df, sf, pf),
            ))
        except Exception as e:
            print(f"  skip {stem}: {e}")
    return rows


# ── Process ───────────────────────────────────────────────────────────────────

stroke_rows = []
for row in process_dir(STROKE_CSV_DIR, "Stroke"):
    stem = row["subject"]
    try:
        subj_id = int(stem.split("_")[0].replace("S", ""))
    except:
        continue
    if subj_id not in FMA: continue
    row["fma"] = FMA[subj_id]
    stroke_rows.append(row)

healthy_rows = process_dir(HEALTHY_CSV_DIR, "Healthy")
for row in healthy_rows:
    row["fma"] = HEALTHY_X

stroke  = pd.DataFrame(stroke_rows).dropna()
healthy = pd.DataFrame(healthy_rows).dropna()

print(f"Stroke: {len(stroke)} trials | Healthy: {len(healthy)} trials")
print("\nFMA correlations (stroke only):")
METRICS = [
    ("sho_elev",   "Shoulder elevation (mm)\n[SA_3 ΔY, start→peak]"),
    ("path_ratio", "Wrist path ratio\n[path length / straight line]"),
    ("vel_peaks",  "Velocity peaks (n)\n[movement submovements]"),
    ("elb_drop",   "Elbow height change (mm)\n[ELB ΔY, negative = drop/extension]"),
]
for col, _ in METRICS:
    r, p = stats.pearsonr(stroke.fma, stroke[col])
    print(f"  {col:<15} r={r:+.3f}  p={p:.4f}")


# ── Plot ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(16, 10), facecolor="#F7F9FC")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.35)
fig.suptitle("Kinematic Metrics vs FMA Score  —  Stroke (coloured) + Healthy (blue cluster)\n"
             f"Stroke FMA range {stroke.fma.min()}–{stroke.fma.max()}/20  "
             f"| Healthy shown at x=22 as reference",
             fontsize=12, fontweight="bold")

rng = np.random.default_rng(42)

for idx, (col, ylabel) in enumerate(METRICS):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    ax.set_facecolor("#F7F9FC")

    sx = stroke.fma.values
    sy = stroke[col].values
    r, p = stats.pearsonr(sx, sy)

    # stroke scatter
    xj = sx + rng.uniform(-0.12, 0.12, len(sx))
    sc = ax.scatter(xj, sy, c=sx, cmap="RdYlGn", vmin=13, vmax=21,
                    s=60, alpha=0.85, zorder=4, label="Stroke")

    # regression line (stroke only, within FMA range)
    coef = np.polyfit(sx, sy, 1)
    xs   = np.linspace(sx.min()-0.3, sx.max()+0.3, 100)
    ax.plot(xs, np.polyval(coef, xs), color="tomato", lw=1.8, ls="--",
            alpha=0.9, label=f"Stroke OLS  r={r:+.3f}  p={p:.3f}")

    # healthy cluster
    hx = HEALTHY_X + rng.uniform(-0.2, 0.2, len(healthy))
    hy = healthy[col].values
    ax.scatter(hx, hy, color=HC, s=50, alpha=0.55, zorder=3, label="Healthy")
    ax.errorbar(HEALTHY_X, hy.mean(), yerr=hy.std(),
                fmt="D", color=HC, ms=8, lw=2, zorder=5,
                label=f"Healthy μ={hy.mean():.1f}±{hy.std():.1f}")

    # vertical separator
    ax.axvline(21, color="gray", lw=0.8, ls=":", alpha=0.6)

    ax.set_xlabel("FMA-UE score  (22 = Healthy reference)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=8, framealpha=0.85)
    ax.set_xticks(sorted(stroke.fma.unique()) + [HEALTHY_X])
    ax.set_xticklabels([str(v) for v in sorted(stroke.fma.unique())] + ["H"])
    ax.grid(True, alpha=0.25, ls="--")

    # colorbar for stroke
    cbar = plt.colorbar(sc, ax=ax, pad=0.01, fraction=0.03)
    cbar.set_label("FMA", fontsize=8)

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"\nSaved: {OUT}")
plt.show()
