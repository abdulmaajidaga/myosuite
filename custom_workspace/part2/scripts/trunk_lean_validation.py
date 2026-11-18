"""
Trunk lean validation: geometric CS-marker estimate vs IK-recovered trunk_forward_lean.

For each trial, computes trunk lean angle independently from raw CS cluster markers
(chest/sternum triad in the original Vicon data) and compares to the IK-recovered
trunk_forward_lean DOF from the weighted (neutral-start) MOT files.

Geometric method:
  - CS centroid = mean of CS_1..4 in lab frame (mm)
  - Forward axis = Vicon X (maps to MuJoCo X = forward in model frame)
  - Height reference = mean CS centroid Y across all frames (seated subject, stable)
  - Lean angle = atan2(CS_centroid_X - CS_centroid_X[0], height_ref)

Output:
  - Per-subject Pearson r and RMSE
  - Scatter plot (IK DOF vs geometric angle)
  - Bland-Altman plot
  - Summary statistics
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = get_project_root()  # = .../custom_workspace/part2
WORKSPACE = os.path.dirname(ROOT)
STROKE_CSV_DIR   = os.path.join(WORKSPACE, get_path("data_stroke"))
HEALTHY_CSV_DIR  = os.path.join(WORKSPACE, get_path("data_healthy"))
STROKE_MOT_DIR   = os.path.join(ROOT, "output", "neutral_start", "stroke", "mot")
HEALTHY_MOT_DIR  = os.path.join(ROOT, "output", "neutral_start", "healthy", "mot")
OUT_DIR          = os.path.join(ROOT, "output", "trunk_lean_validation")
os.makedirs(OUT_DIR, exist_ok=True)

CS_MARKERS = ["CS_1", "CS_2", "CS_3", "CS_4"]
MOT_SKIPROWS = 7   # 5 header lines + endheader + column name row

PHASE_IDX_STROKE  = json.load(open(os.path.join(ROOT, "output", "phase_indices", "stroke_phase_indices.json")))
PHASE_IDX_HEALTHY = json.load(open(os.path.join(ROOT, "output", "phase_indices", "healthy_phase_indices.json")))

def get_phase(stem, label):
    fname = stem + ".csv"
    idx = PHASE_IDX_STROKE if label == "Stroke" else PHASE_IDX_HEALTHY
    phases = idx.get(fname)
    if phases is None: return None, None
    return int(phases[0]), int(phases[1])


def _load_csv_markers(csv_path):
    """Return dict of marker_name -> DataFrame(X,Y,Z), skipping the axis label row."""
    raw = pd.read_csv(csv_path, header=[0, 1])
    # Rebuild multi-index: propagate marker names across Unnamed columns
    current = None
    tuples = []
    for col in raw.columns:
        name, axis = col
        if not str(name).startswith("Unnamed"):
            current = name.strip()
        tuples.append((current, axis))
    raw.columns = pd.MultiIndex.from_tuples(tuples)
    df = raw.iloc[1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    return df


def _cs_centroid(df):
    """Return (N,3) array of CS cluster centroid in mm [X,Y,Z]."""
    available = [m for m in CS_MARKERS if m in df.columns.get_level_values(0)]
    if len(available) < 2:
        return None
    x = df[available].xs("X", level=1, axis=1).mean(axis=1).values
    y = df[available].xs("Y", level=1, axis=1).mean(axis=1).values
    z = df[available].xs("Z", level=1, axis=1).mean(axis=1).values
    return np.stack([x, y, z], axis=1)


def _forward_axis_from_wrist(df):
    """
    Determine subject-specific forward direction (unit vector in XZ plane)
    from the wrist reach trajectory. During the drinking task the wrist moves
    unambiguously forward — WRA/WRB are always present in the raw CSV.
    Falls back to WRA alone if WRB is missing.
    """
    available = [m for m in ['WRA', 'WRB'] if m in df.columns.get_level_values(0)]
    if not available:
        return None
    wx = df[available].xs('X', level=1, axis=1).mean(axis=1).values
    wz = df[available].xs('Z', level=1, axis=1).mean(axis=1).values
    # Forward = direction of maximum wrist displacement in XZ plane
    delta_x = wx - wx[0]
    delta_z = wz - wz[0]
    # Peak reach frame = frame with maximum horizontal wrist displacement
    mag = np.sqrt(delta_x**2 + delta_z**2)
    peak_frame = np.nanargmax(mag)
    if peak_frame < 5:
        return None
    fwd = np.array([delta_x[peak_frame], delta_z[peak_frame]])
    norm = np.linalg.norm(fwd)
    return fwd / norm if norm > 1e-6 else None


def _geometric_lean(centroid, fwd_axis):
    """
    Forward lean angle (rad) from CS centroid trajectory.

    Projects the CS centroid horizontal displacement onto the subject-specific
    forward direction (derived from wrist reach) to handle arbitrary lab orientation.
    Angle = atan2(forward_projection, mean_height).
    """
    y_up  = centroid[:, 1]
    h_ref = np.nanmean(y_up)
    disp  = centroid[:, [0, 2]] - centroid[0, [0, 2]]   # XZ displacement from frame 0
    proj  = disp @ fwd_axis                              # signed scalar (mm)
    return np.arctan2(proj, h_ref)


def _load_mot_trunk(mot_path):
    """Return trunk_forward_lean column (rad) from weighted IK MOT."""
    with open(mot_path) as f:
        header = f.readline()
        while "endheader" not in header:
            header = f.readline()
        col_names = f.readline().strip().split("\t")
    data = np.loadtxt(mot_path, skiprows=MOT_SKIPROWS, delimiter="\t")
    col_idx = col_names.index("trunk_forward_lean")
    return data[:, col_idx]


def _match_length(a, b):
    """Truncate both arrays to the shorter length."""
    n = min(len(a), len(b))
    return a[:n], b[:n]


def _find_csv(stem, csv_dir):
    """Find CSV file matching the MOT stem (case-insensitive, handles spaces)."""
    target = stem.lower()
    for f in os.listdir(csv_dir):
        if f.lower().replace(" ", "_").replace(".csv", "") == target:
            return os.path.join(csv_dir, f)
    # direct match
    candidate = os.path.join(csv_dir, stem + ".csv")
    return candidate if os.path.exists(candidate) else None


# ── Per-trial processing ───────────────────────────────────────────────────────
def process_trial(mot_path, csv_dir, label):
    stem = os.path.splitext(os.path.basename(mot_path))[0]
    sf, pf = get_phase(stem, label)
    if sf is None:
        return None
    csv_path = _find_csv(stem, csv_dir)
    if csv_path is None or not os.path.exists(csv_path):
        return None

    try:
        df       = _load_csv_markers(csv_path)
        centroid = _cs_centroid(df)
        if centroid is None:
            return None
        # Use phase peak for forward axis direction
        avail = [m for m in ['WRA', 'WRB'] if m in df.columns.get_level_values(0)]
        wx = df[avail].xs('X', level=1, axis=1).mean(axis=1).values
        wz = df[avail].xs('Z', level=1, axis=1).mean(axis=1).values
        pf_clamp = min(pf, len(wx) - 1)
        delta_x = wx[pf_clamp] - wx[sf]; delta_z = wz[pf_clamp] - wz[sf]
        norm = np.sqrt(delta_x**2 + delta_z**2)
        fwd_axis = np.array([delta_x, delta_z]) / norm if norm > 1e-6 else None
        if fwd_axis is None:
            return None

        geo_lean = _geometric_lean(centroid, fwd_axis)
        ik_lean  = _load_mot_trunk(mot_path)

        # Slice to reach phase
        n = min(len(geo_lean), len(ik_lean))
        sf_c = min(sf, n - 1); pf_c = min(pf, n - 1)
        geo_lean = geo_lean[sf_c:pf_c+1]
        ik_lean  = ik_lean[sf_c:pf_c+1]

        mask = ~(np.isnan(geo_lean) | np.isnan(ik_lean))
        if mask.sum() < 10:
            return None
        geo_lean, ik_lean = geo_lean[mask], ik_lean[mask]

        r, p    = pearsonr(geo_lean, ik_lean)
        rmse    = np.sqrt(np.mean((geo_lean - ik_lean) ** 2))
        peak_geo = np.max(np.abs(geo_lean))
        peak_ik  = np.max(np.abs(ik_lean))
        return dict(stem=stem, group=label, r=r, p=p, rmse=rmse,
                    peak_geo=peak_geo, peak_ik=peak_ik,
                    geo=geo_lean, ik=ik_lean)
    except Exception as e:
        print(f"  skip {stem}: {e}")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    results = []
    for mot_dir, csv_dir, label in [
        (STROKE_MOT_DIR,  STROKE_CSV_DIR,  "Stroke"),
        (HEALTHY_MOT_DIR, HEALTHY_CSV_DIR, "Healthy"),
    ]:
        for fname in sorted(os.listdir(mot_dir)):
            if not fname.endswith(".mot"):
                continue
            r = process_trial(os.path.join(mot_dir, fname), csv_dir, label)
            if r:
                results.append(r)

    if not results:
        print("No matched trials found.")
        return

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'Subject':<18} {'Group':<8} {'r':>6} {'p':>8} {'RMSE(rad)':>10} {'Peak geo':>10} {'Peak IK':>9}")
    print("-" * 75)
    for res in results:
        print(f"{res['stem']:<18} {res['group']:<8} {res['r']:>6.3f} {res['p']:>8.4f} "
              f"{res['rmse']:>10.4f} {np.degrees(res['peak_geo']):>9.1f}° "
              f"{np.degrees(res['peak_ik']):>8.1f}°")

    rs = np.array([r["r"] for r in results])
    stroke_rs  = np.array([r["r"] for r in results if r["group"] == "Stroke"])
    healthy_rs = np.array([r["r"] for r in results if r["group"] == "Healthy"])
    print(f"\nOverall  r: mean={rs.mean():.3f}  std={rs.std():.3f}  median={np.median(rs):.3f}")
    print(f"Stroke   r: mean={stroke_rs.mean():.3f}  std={stroke_rs.std():.3f}")
    print(f"Healthy  r: mean={healthy_rs.mean():.3f}  std={healthy_rs.std():.3f}")

    # ── Pooled arrays for plots ────────────────────────────────────────────────
    all_geo = np.concatenate([r["geo"] for r in results])
    all_ik  = np.concatenate([r["ik"]  for r in results])
    r_pool, p_pool = pearsonr(all_geo, all_ik)

    # ── Figure: Scatter + Bland-Altman ────────────────────────────────────────
    colors = {"Stroke": "#C0392B", "Healthy": "#2980B9"}
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, wspace=0.35)

    # — Scatter —
    ax1 = fig.add_subplot(gs[0])
    for res in results:
        c = colors[res["group"]]
        ax1.scatter(np.degrees(res["geo"]), np.degrees(res["ik"]),
                    alpha=0.15, s=4, color=c)

    lim = max(np.abs(np.degrees(all_geo)).max(), np.abs(np.degrees(all_ik)).max()) * 1.1
    ax1.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="Identity")
    from numpy.polynomial import polynomial as P
    coef = np.polyfit(np.degrees(all_geo), np.degrees(all_ik), 1)
    xs = np.linspace(-lim, lim, 200)
    ax1.plot(xs, np.polyval(coef, xs), "gray", lw=1.5, ls="--",
             label=f"Fit  r={r_pool:.3f}")
    # Legend proxies
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor=v,
                      markersize=8, label=k) for k, v in colors.items()]
    handles += [Line2D([0],[0], color="k", ls="--", label="Identity"),
                Line2D([0],[0], color="gray", ls="--", label=f"OLS  r={r_pool:.3f}")]
    ax1.legend(handles=handles, fontsize=8)
    ax1.set_xlabel("Geometric lean — CS cluster (°)")
    ax1.set_ylabel("IK-recovered trunk_forward_lean (°)")
    ax1.set_title("Scatter: Geometric vs IK Trunk Lean")
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # — Bland-Altman —
    ax2 = fig.add_subplot(gs[1])
    mean_vals = np.degrees((all_geo + all_ik) / 2)
    diff_vals = np.degrees(all_ik - all_geo)
    bias  = diff_vals.mean()
    sd    = diff_vals.std()
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd

    for res in results:
        c = colors[res["group"]]
        m = np.degrees((res["geo"] + res["ik"]) / 2)
        d = np.degrees(res["ik"] - res["geo"])
        ax2.scatter(m, d, alpha=0.15, s=4, color=c)

    ax2.axhline(bias,   color="k",   lw=1.5, ls="-",  label=f"Bias {bias:+.2f}°")
    ax2.axhline(loa_hi, color="red", lw=1,   ls="--", label=f"+1.96 SD {loa_hi:+.2f}°")
    ax2.axhline(loa_lo, color="red", lw=1,   ls="--", label=f"−1.96 SD {loa_lo:+.2f}°")
    ax2.axhline(0, color="gray", lw=0.8, ls=":")
    ax2.legend(fontsize=8)
    ax2.set_xlabel("Mean of geometric and IK lean (°)")
    ax2.set_ylabel("IK − Geometric (°)")
    ax2.set_title("Bland-Altman: IK vs Geometric Trunk Lean")
    ax2.grid(True, alpha=0.3)

    # Legend proxies for groups
    handles2 = [Line2D([0],[0], marker="o", color="w", markerfacecolor=v,
                       markersize=8, label=k) for k, v in colors.items()]
    ax2.legend(handles=handles2 + ax2.get_lines()[:3], fontsize=8)

    plt.suptitle(
        f"Trunk Lean Validation  |  N={len(results)} trials  "
        f"|  Pooled r={r_pool:.3f} (p={p_pool:.2e})",
        fontsize=11, fontweight="bold"
    )

    out_png = os.path.join(OUT_DIR, "trunk_lean_validation.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {out_png}")

    # ── Per-subject peak Bland-Altman (peak values, not time series) ──────────
    peak_geo_deg = np.degrees(np.array([r["peak_geo"] for r in results]))
    peak_ik_deg  = np.degrees(np.array([r["peak_ik"]  for r in results]))
    peak_bias    = (peak_ik_deg - peak_geo_deg).mean()
    peak_sd      = (peak_ik_deg - peak_geo_deg).std()
    print(f"\nPeak-value agreement (per subject):")
    print(f"  Bias:     {peak_bias:+.2f}°")
    print(f"  ±1.96 SD: {peak_bias - 1.96*peak_sd:.2f}° to {peak_bias + 1.96*peak_sd:.2f}°")


if __name__ == "__main__":
    main()
