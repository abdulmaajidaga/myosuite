"""
Validate IK trunk_forward_lean against CS-marker-derived trunk tilt.

For each trial, the COMPENSATION angle is the CHANGE from frame 0 at phase peak:
  cs_comp  = cs_tilt[peak_frame] - cs_tilt[0]   (radians, from CS plane SVD)
  ik_comp  = trunk_forward_lean[peak_frame] - trunk_forward_lean[0]  (radians, from MOT)

Pearson r across all trials quantifies how well IK recovers the CS-based trunk signal.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root
from scripts.inject_cs_trunk import (
    load_csv, load_mot, wrist_fwd_axis, cs_lean_rad,
    PHASE_IDX_STROKE, PHASE_IDX_HEALTHY,
    STROKE_CSV_DIR, HEALTHY_CSV_DIR,
)

ROOT     = get_project_root()
OUT_BASE = os.path.join(ROOT, "output", "neutral_start")
OUT_DIR  = os.path.join(ROOT, "output", "trunk_lean_validation")
os.makedirs(OUT_DIR, exist_ok=True)

OUTLIER_THRESH_DEG = 40.0   # filter physically impossible CS tilt values


def _cs_comp(df, phase_idx, fname):
    """CS tilt change at phase peak relative to frame 0 (radians)."""
    phases  = phase_idx.get(fname)
    peak_f  = int(phases[1]) if phases is not None else None
    fwd     = wrist_fwd_axis(df, peak_f)
    tilt    = cs_lean_rad(df, fwd)
    if tilt is None:
        return None
    baseline = np.nanmean(tilt[:5]) if np.isnan(tilt[0]) else tilt[0]
    if peak_f is None or peak_f >= len(tilt) or np.isnan(tilt[peak_f]):
        peak_f = int(np.nanargmax(np.abs(tilt - baseline)))
    peak_f = min(peak_f, len(tilt) - 1)
    return float(tilt[peak_f] - baseline)


def _ik_comp(mot_path, phase_idx, fname):
    """IK trunk_forward_lean change at phase peak relative to frame 0 (radians)."""
    phases = phase_idx.get(fname)
    peak_f = int(phases[1]) if phases is not None else None
    try:
        cols, arr, _ = load_mot(mot_path)
    except Exception:
        return None
    if "trunk_forward_lean" not in cols:
        return None
    col  = cols.index("trunk_forward_lean")
    vals = arr[:, col]
    if peak_f is None or peak_f >= len(vals):
        peak_f = int(np.argmax(np.abs(vals - vals[0])))
    peak_f = min(peak_f, len(vals) - 1)
    return float(vals[peak_f] - vals[0])


def process_group(group, csv_dir, mot_dir, phase_idx):
    results = []
    if not os.path.isdir(mot_dir):
        return results
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"):
            continue
        stem     = fname.replace(".csv", "")
        mot_path = os.path.join(mot_dir, stem + ".mot")
        if not os.path.exists(mot_path):
            continue
        try:
            df = load_csv(os.path.join(csv_dir, fname))
        except Exception:
            continue
        cs_val = _cs_comp(df, phase_idx, fname)
        ik_val = _ik_comp(mot_path, phase_idx, fname)
        if cs_val is None or ik_val is None:
            continue
        cs_deg = np.degrees(cs_val)
        ik_deg = np.degrees(ik_val)
        results.append((stem, group, cs_deg, ik_deg))
        print(f"  {stem:<18} {group:<8} CS={cs_deg:+.2f}°  IK_DOF={ik_deg:+.2f}°")
    return results


STROKE_COL  = '#d62728'
HEALTHY_COL = '#4472c4'

def main():
    rows = []
    rows += process_group("Stroke",  STROKE_CSV_DIR,  os.path.join(OUT_BASE, "stroke",  "mot"), PHASE_IDX_STROKE)
    rows += process_group("Healthy", HEALTHY_CSV_DIR, os.path.join(OUT_BASE, "healthy", "mot"), PHASE_IDX_HEALTHY)

    # filter physically impossible CS tilt values
    rows = [r for r in rows if abs(r[2]) <= OUTLIER_THRESH_DEG]

    print(f"\n{len(rows)} trials after outlier filter")
    if len(rows) < 3:
        print("Too few trials for correlation.")
        return

    cs_vals = np.array([r[2] for r in rows])
    ik_vals = np.array([r[3] for r in rows])
    groups  = [r[1] for r in rows]

    r, p = pearsonr(ik_vals, cs_vals)
    print(f"Pearson r={r:.3f}  p={p:.4f}  n={len(rows)}")

    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'DejaVu Sans'],
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.linewidth':    0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size':  3,
        'ytick.major.size':  3,
    })

    fig, ax = plt.subplots(figsize=(5, 4.5), facecolor='white')
    ax.set_facecolor('white')

    for g, cs, ik in zip(groups, cs_vals, ik_vals):
        col = STROKE_COL if g == 'Stroke' else HEALTHY_COL
        ax.scatter(cs, ik, color=col, alpha=0.75, s=30, linewidths=0, zorder=3)

    mn = min(cs_vals.min(), ik_vals.min()) - 1
    mx = max(cs_vals.max(), ik_vals.max()) + 1
    ax.plot([mn, mx], [mn, mx], color='#aaaaaa', lw=1.0, ls='--', zorder=1)

    p_str = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
    ax.text(0.05, 0.95, f'r = {r:.3f},  {p_str},  n = {len(rows)}',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8.5, color='#222222')

    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=STROKE_COL,
               markersize=7, label='Stroke'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=HEALTHY_COL,
               markersize=7, label='Healthy'),
    ]
    ax.legend(handles=handles, fontsize=8, frameon=False,
              handletextpad=0.4, loc='lower right')

    ax.set_xlabel('CS plate trunk tilt at peak (°)', fontsize=8.5, color='#444', labelpad=4)
    ax.set_ylabel('SVD-prescribed trunk lean at peak (°)', fontsize=8.5, color='#444', labelpad=4)
    ax.tick_params(labelsize=7.5, colors='#555555')
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.tick_params(color='#888888')

    plt.tight_layout(pad=0.8)

    out_path = os.path.join(ROOT, 'paper_publication_assets', 'figures_and_tables', 'ik_vs_cs_correlation.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
