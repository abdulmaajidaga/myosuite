"""
Trunk lean plane validation.

For each trial, fits a plane through CS_1..4 each frame via SVD and tracks
how that plane's normal vector tilts in the sagittal direction. This gives a
geometry-only trunk lean angle that the IK solver never sees. Compare to the
recovered trunk_forward_lean DOF.

Why this works better than centroid displacement:
  - Trunk lean is a ROTATION — the chest plane tilts.
  - Shoulder protraction mostly TRANSLATES the markers without rotating the plane.
  - So plane normal orientation isolates the rotational (lean) component.

Sagittal plane angle:
  - Project n(t) onto the plane spanned by [fwd_axis, up_axis].
  - Angle of that projected vector relative to n(0) = forward lean.
  - fwd_axis is subject-specific from the wrist reach trajectory.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root

ROOT            = get_project_root()
WORKSPACE       = os.path.dirname(ROOT)
STROKE_CSV_DIR  = os.path.join(WORKSPACE, get_path("data_stroke"))
HEALTHY_CSV_DIR = os.path.join(WORKSPACE, get_path("data_healthy"))
STROKE_MOT_DIR  = os.path.join(ROOT, "output", "neutral_start", "stroke", "mot")
HEALTHY_MOT_DIR = os.path.join(ROOT, "output", "neutral_start", "healthy", "mot")
OUT_DIR         = os.path.join(ROOT, "output", "trunk_lean_validation")
os.makedirs(OUT_DIR, exist_ok=True)

CS_MARKERS = ["CS_1", "CS_2", "CS_3", "CS_4"]
UP_AXIS    = np.array([0.0, 1.0, 0.0])   # Vicon Y is vertical

PHASE_IDX_STROKE  = json.load(open(os.path.join(ROOT, "output", "phase_indices", "stroke_phase_indices.json")))
PHASE_IDX_HEALTHY = json.load(open(os.path.join(ROOT, "output", "phase_indices", "healthy_phase_indices.json")))

def get_phase(stem, label):
    fname = stem + ".csv"
    idx = PHASE_IDX_STROKE if label == "Stroke" else PHASE_IDX_HEALTHY
    phases = idx.get(fname)
    if phases is None: return None, None
    return int(phases[0]), int(phases[1])


# ── Data loading ──────────────────────────────────────────────────────────────

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


def load_mot_trunk(path):
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "endheader" in line:
            cols = lines[i + 1].strip().split("\t")
            break
    data = np.loadtxt(path, skiprows=i + 2, delimiter="\t")
    return np.degrees(data[:, cols.index("trunk_forward_lean")])


def find_csv(stem, csv_dir):
    candidate = os.path.join(csv_dir, stem + ".csv")
    if os.path.exists(candidate):
        return candidate
    for f in os.listdir(csv_dir):
        if f.lower().replace(" ", "_").replace(".csv", "") == stem.lower():
            return os.path.join(csv_dir, f)
    return None


# ── Geometry ──────────────────────────────────────────────────────────────────

def cs_plane_normals(df):
    """
    Return (N, 3) array of unit normal vectors for the CS marker plane each frame.
    Uses SVD on the 4 CS marker positions: the right-singular vector corresponding
    to the smallest singular value is the plane normal.
    """
    avail = [m for m in CS_MARKERS if m in df.columns.get_level_values(0)]
    if len(avail) < 3:
        return None
    x = df[avail].xs("X", level=1, axis=1).values   # (N, k)
    y = df[avail].xs("Y", level=1, axis=1).values
    z = df[avail].xs("Z", level=1, axis=1).values
    points = np.stack([x, y, z], axis=2)             # (N, k, 3)

    normals = np.full((len(df), 3), np.nan)
    prev = None
    for i in range(len(df)):
        pts = points[i]                              # (k, 3)
        if np.any(np.isnan(pts)):
            prev = None
            continue
        centered = pts - pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        n = Vt[-1]                                   # smallest singular vector
        n = n / np.linalg.norm(n)
        # SVD has sign ambiguity — enforce consistent orientation across frames
        if prev is not None and np.dot(n, prev) < 0:
            n = -n
        normals[i] = n
        prev = n
    return normals


def forward_axis_from_wrist(df):
    """Unit vector in XZ plane pointing in the subject's forward reach direction."""
    avail = [m for m in ["WRA", "WRB"] if m in df.columns.get_level_values(0)]
    if not avail:
        return None
    wx = df[avail].xs("X", level=1, axis=1).mean(axis=1).values
    wz = df[avail].xs("Z", level=1, axis=1).mean(axis=1).values
    dx, dz = wx - wx[0], wz - wz[0]
    peak = np.nanargmax(np.sqrt(dx**2 + dz**2))
    if peak < 5:
        return None
    fwd = np.array([dx[peak], 0.0, dz[peak]])
    norm = np.linalg.norm(fwd)
    return fwd / norm if norm > 1e-6 else None


def plane_forward_lean(normals, fwd_axis):
    """
    Sagittal forward lean angle (degrees) of the CS plane relative to frame 0.

    For each frame:
      1. Project n(t) onto the sagittal plane [fwd_axis, UP_AXIS].
      2. Compute the signed angle from n(0)'s projection to n(t)'s projection.

    Positive = plane tilted forward (lean toward the cup).
    """
    # Sagittal basis (orthonormal): fwd and up
    fwd = fwd_axis / np.linalg.norm(fwd_axis)
    up  = UP_AXIS.copy()
    # Make up orthogonal to fwd just in case
    up  = up - np.dot(up, fwd) * fwd
    up  = up / np.linalg.norm(up)

    # Project each normal onto the sagittal plane
    proj_fwd = normals @ fwd    # (N,)
    proj_up  = normals @ up     # (N,)

    # Reference direction from frame 0 (first valid frame)
    valid = np.where(~np.isnan(proj_fwd))[0]
    if len(valid) == 0:
        return None
    ref_f = proj_fwd[valid[0]]
    ref_u = proj_up[valid[0]]
    ref_angle = np.arctan2(ref_f, ref_u)

    # Angle at each frame relative to reference
    angles = np.arctan2(proj_fwd, proj_up) - ref_angle
    return np.degrees(angles)


# ── Per-trial ─────────────────────────────────────────────────────────────────

def process_trial(mot_path, csv_dir, label):
    stem     = os.path.splitext(os.path.basename(mot_path))[0]
    sf, pf   = get_phase(stem, label)
    if sf is None:
        return None
    csv_path = find_csv(stem, csv_dir)
    if not csv_path:
        return None
    try:
        df       = load_csv(csv_path)
        normals  = cs_plane_normals(df)
        if normals is None:
            return None

        # Forward axis from phase peak
        avail = [m for m in ["WRA", "WRB"] if m in df.columns.get_level_values(0)]
        if not avail:
            return None
        wx = df[avail].xs("X", level=1, axis=1).mean(axis=1).values
        wz = df[avail].xs("Z", level=1, axis=1).mean(axis=1).values
        pf_c = min(pf, len(wx) - 1); sf_c = min(sf, len(wx) - 1)
        dx = wx[pf_c] - wx[sf_c]; dz = wz[pf_c] - wz[sf_c]
        norm_v = np.sqrt(dx**2 + dz**2)
        if norm_v < 1e-6:
            return None
        fwd_axis = np.array([dx, 0.0, dz]) / norm_v

        plane_lean = plane_forward_lean(normals, fwd_axis)
        ik_lean    = load_mot_trunk(mot_path)
        if plane_lean is None:
            return None

        n = min(len(plane_lean), len(ik_lean))
        sf_c = min(sf, n - 1); pf_c = min(pf, n - 1)
        plane_lean = plane_lean[sf_c:pf_c+1]
        ik_lean    = ik_lean[sf_c:pf_c+1]

        mask = ~(np.isnan(plane_lean) | np.isnan(ik_lean))
        if mask.sum() < 10:
            return None

        pl, ik = plane_lean[mask], ik_lean[mask]
        r, p   = pearsonr(pl, ik)
        rmse   = np.sqrt(np.mean((pl - ik) ** 2))

        return dict(stem=stem, group=label, r=r, p=p, rmse=rmse,
                    peak_plane=np.max(np.abs(pl)),
                    peak_ik=np.max(np.abs(ik)),
                    plane=pl, ik=ik)
    except Exception as e:
        print(f"  skip {stem}: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

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
        print("No results.")
        return

    print(f"\n{'Subject':<18} {'Group':<8} {'r':>6} {'p':>8} {'RMSE(°)':>8} "
          f"{'Peak plane':>11} {'Peak IK':>9}")
    print("-" * 75)
    for res in results:
        print(f"{res['stem']:<18} {res['group']:<8} {res['r']:>6.3f} {res['p']:>8.4f} "
              f"{res['rmse']:>8.2f}° {res['peak_plane']:>10.1f}° {res['peak_ik']:>8.1f}°")

    rs          = np.array([r["r"] for r in results])
    stroke_rs   = np.array([r["r"] for r in results if r["group"] == "Stroke"])
    healthy_rs  = np.array([r["r"] for r in results if r["group"] == "Healthy"])
    print(f"\nOverall  r: mean={rs.mean():.3f}  median={np.median(rs):.3f}  std={rs.std():.3f}")
    print(f"Stroke   r: mean={stroke_rs.mean():.3f}  std={stroke_rs.std():.3f}  N={len(stroke_rs)}")
    print(f"Healthy  r: mean={healthy_rs.mean():.3f}  std={healthy_rs.std():.3f}  N={len(healthy_rs)}")

    # Pooled cross-subject peak correlation
    peak_plane = np.array([r["peak_plane"] for r in results])
    peak_ik    = np.array([r["peak_ik"]    for r in results])
    r_peak, p_peak   = pearsonr(peak_plane, peak_ik)
    rs_peak, ps_peak = spearmanr(peak_plane, peak_ik)
    print(f"\nCross-subject peak correlation:")
    print(f"  Pearson  r={r_peak:.3f}  p={p_peak:.2e}")
    print(f"  Spearman r={rs_peak:.3f}  p={ps_peak:.2e}")

    # ── Figure: Scatter + Bland-Altman ────────────────────────────────────────
    colors = {"Stroke": "#C0392B", "Healthy": "#2980B9"}
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, wspace=0.35)

    all_plane = np.concatenate([r["plane"] for r in results])
    all_ik    = np.concatenate([r["ik"]    for r in results])
    r_pool, p_pool = pearsonr(all_plane, all_ik)

    # Scatter
    ax1 = fig.add_subplot(gs[0])
    for res in results:
        ax1.scatter(res["plane"], res["ik"],
                    alpha=0.12, s=3, color=colors[res["group"]])
    lim = max(np.abs(all_plane).max(), np.abs(all_ik).max()) * 1.1
    ax1.plot([-lim, lim], [-lim, lim], "k--", lw=1)
    coef = np.polyfit(all_plane, all_ik, 1)
    xs   = np.linspace(-lim, lim, 200)
    ax1.plot(xs, np.polyval(coef, xs), "gray", lw=1.5, ls="--",
             label=f"OLS  r={r_pool:.3f}")
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor=v,
                      markersize=8, label=k) for k, v in colors.items()]
    handles += [Line2D([0],[0], color="k", ls="--", label="Identity"),
                Line2D([0],[0], color="gray", ls="--", label=f"OLS r={r_pool:.3f}")]
    ax1.legend(handles=handles, fontsize=8)
    ax1.set_xlabel("CS plane tilt — geometric (°)")
    ax1.set_ylabel("IK trunk_forward_lean (°)")
    ax1.set_title("Scatter: Plane-normal vs IK Trunk Lean")
    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # Bland-Altman
    ax2 = fig.add_subplot(gs[1])
    mean_vals = (all_plane + all_ik) / 2
    diff_vals = all_ik - all_plane
    bias = diff_vals.mean(); sd = diff_vals.std()
    for res in results:
        m = (res["plane"] + res["ik"]) / 2
        d = res["ik"] - res["plane"]
        ax2.scatter(m, d, alpha=0.12, s=3, color=colors[res["group"]])
    ax2.axhline(bias,            color="k",   lw=1.5, ls="-",
                label=f"Bias {bias:+.2f}°")
    ax2.axhline(bias + 1.96*sd, color="red", lw=1,   ls="--",
                label=f"+1.96SD {bias+1.96*sd:+.2f}°")
    ax2.axhline(bias - 1.96*sd, color="red", lw=1,   ls="--",
                label=f"−1.96SD {bias-1.96*sd:+.2f}°")
    ax2.axhline(0, color="gray", lw=0.8, ls=":")
    ax2.legend(fontsize=8)
    ax2.set_xlabel("Mean of plane and IK lean (°)")
    ax2.set_ylabel("IK − Plane (°)")
    ax2.set_title("Bland-Altman: IK vs Plane-normal Trunk Lean")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"CS Plane-normal Trunk Lean Validation  |  N={len(results)} trials  "
        f"|  Pooled r={r_pool:.3f} (p={p_pool:.2e})",
        fontsize=11, fontweight="bold"
    )
    out = os.path.join(OUT_DIR, "trunk_lean_plane_validation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
