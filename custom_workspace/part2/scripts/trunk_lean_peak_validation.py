"""
Trunk lean peak validation: cross-subject correlation between IK-recovered
peak trunk_forward_lean and CS-cluster forward displacement at peak lean.

Instead of time-series correlation (noisy due to shoulder motion), this compares
one number per subject: the CS centroid forward displacement at the exact frame
where trunk_forward_lean is maximum. Cross-subject scatter should be clean because
a subject who genuinely leans more physically moves their chest further forward.

Forward direction is determined per-subject from the wrist reach trajectory
(unambiguous forward motion during the drinking task).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root

ROOT      = get_project_root()
WORKSPACE = os.path.dirname(ROOT)
STROKE_CSV_DIR  = os.path.join(WORKSPACE, get_path("data_stroke"))
HEALTHY_CSV_DIR = os.path.join(WORKSPACE, get_path("data_healthy"))
STROKE_MOT_DIR  = os.path.join(ROOT, "output", "neutral_start", "stroke", "mot")
HEALTHY_MOT_DIR = os.path.join(ROOT, "output", "neutral_start", "healthy", "mot")
OUT_DIR         = os.path.join(ROOT, "output", "trunk_lean_validation")
os.makedirs(OUT_DIR, exist_ok=True)

CS_MARKERS = ["CS_1", "CS_2", "CS_3", "CS_4"]

PHASE_IDX_STROKE  = json.load(open(os.path.join(ROOT, "output", "phase_indices", "stroke_phase_indices.json")))
PHASE_IDX_HEALTHY = json.load(open(os.path.join(ROOT, "output", "phase_indices", "healthy_phase_indices.json")))

def get_phase(stem, label):
    fname = stem + ".csv"
    idx = PHASE_IDX_STROKE if label == "Stroke" else PHASE_IDX_HEALTHY
    phases = idx.get(fname)
    if phases is None: return None, None
    return int(phases[0]), int(phases[1])


def load_csv(path):
    df = pd.read_csv(path, header=[0, 1])
    current, tuples = None, []
    for col in df.columns:
        m, ax = col
        if not str(m).startswith("Unnamed"):
            current = m.strip()
        tuples.append((current, ax))
    df.columns = pd.MultiIndex.from_tuples(tuples)
    return df.iloc[1:].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)


def load_mot_trunk(path):
    with open(path) as f:
        cols = f.readlines()[6].strip().split("\t")
    data = np.loadtxt(path, skiprows=7, delimiter="\t")
    return data[:, cols.index("trunk_forward_lean")]


def cs_centroid_xz(df):
    """Return (N, 2) array of CS centroid in XZ plane (mm)."""
    avail = [m for m in CS_MARKERS if m in df.columns.get_level_values(0)]
    if len(avail) < 2:
        return None
    x = df[avail].xs("X", level=1, axis=1).mean(axis=1).values
    z = df[avail].xs("Z", level=1, axis=1).mean(axis=1).values
    return np.stack([x, z], axis=1)


def forward_axis_from_wrist(df):
    """Unit vector in XZ plane pointing in the subject's reaching direction."""
    avail = [m for m in ["WRA", "WRB"] if m in df.columns.get_level_values(0)]
    if not avail:
        return None
    wx = df[avail].xs("X", level=1, axis=1).mean(axis=1).values
    wz = df[avail].xs("Z", level=1, axis=1).mean(axis=1).values
    dx, dz = wx - wx[0], wz - wz[0]
    mag = np.sqrt(dx**2 + dz**2)
    peak = np.nanargmax(mag)
    if peak < 5:
        return None
    fwd = np.array([dx[peak], dz[peak]])
    norm = np.linalg.norm(fwd)
    return fwd / norm if norm > 1e-6 else None


def find_csv(stem, csv_dir):
    candidate = os.path.join(csv_dir, stem + ".csv")
    if os.path.exists(candidate):
        return candidate
    for f in os.listdir(csv_dir):
        if f.lower().replace(" ", "_").replace(".csv", "") == stem.lower():
            return os.path.join(csv_dir, f)
    return None


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
        centroid = cs_centroid_xz(df)
        if centroid is None:
            return None

        # Forward axis from phase peak frame
        avail = [m for m in ["WRA", "WRB"] if m in df.columns.get_level_values(0)]
        if not avail:
            return None
        wx = df[avail].xs("X", level=1, axis=1).mean(axis=1).values
        wz = df[avail].xs("Z", level=1, axis=1).mean(axis=1).values
        pf_c = min(pf, len(wx) - 1)
        dx = wx[pf_c] - wx[sf]; dz = wz[pf_c] - wz[sf]
        norm = np.sqrt(dx**2 + dz**2)
        if norm < 1e-6:
            return None
        fwd_axis = np.array([dx, dz]) / norm

        ik_lean = load_mot_trunk(mot_path)
        n       = min(len(df), len(ik_lean))
        sf_c    = min(sf, n - 1); pf_c = min(pf, n - 1)
        centroid = centroid[:n]

        # Use phase peak frame (not IK argmax which is near-zero)
        peak_ik_deg = np.degrees(ik_lean[pf_c])

        disp_at_peak = centroid[pf_c] - centroid[sf_c]
        cs_fwd_mm    = float(disp_at_peak @ fwd_axis)

        return dict(stem=stem, group=label,
                    peak_ik_deg=peak_ik_deg,
                    cs_fwd_mm=cs_fwd_mm,
                    peak_frame=pf_c)
    except Exception as e:
        print(f"  skip {stem}: {e}")
        return None


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

    ik   = np.array([r["peak_ik_deg"]  for r in results])
    cs   = np.array([r["cs_fwd_mm"]    for r in results])
    grps = np.array([r["group"]         for r in results])

    r_p, p_p   = pearsonr(ik, cs)
    r_sp, p_sp = spearmanr(ik, cs)

    print(f"\n{'Subject':<14} {'Group':<8} {'IK peak (°)':>12} {'CS disp (mm)':>13}")
    print("-" * 52)
    for res in results:
        print(f"{res['stem']:<14} {res['group']:<8} "
              f"{res['peak_ik_deg']:>12.1f} {res['cs_fwd_mm']:>13.1f}")

    print(f"\nPearson  r={r_p:.3f}  p={p_p:.2e}")
    print(f"Spearman r={r_sp:.3f}  p={p_sp:.2e}")
    for grp in ["Stroke", "Healthy"]:
        mask  = grps == grp
        r_g, p_g = pearsonr(ik[mask], cs[mask])
        print(f"  {grp}: r={r_g:.3f}  p={p_g:.2e}  N={mask.sum()}")

    # ── Figure ────────────────────────────────────────────────────────────────
    colors = {"Stroke": "#C0392B", "Healthy": "#2980B9"}
    fig, ax = plt.subplots(figsize=(8, 6))

    for grp, col in colors.items():
        mask = grps == grp
        ax.scatter(ik[mask], cs[mask], color=col, alpha=0.75, s=60,
                   edgecolors="white", linewidths=0.5, label=grp, zorder=3)

    # OLS fit line
    coef = np.polyfit(ik, cs, 1)
    xs   = np.linspace(ik.min() - 1, ik.max() + 1, 200)
    ax.plot(xs, np.polyval(coef, xs), "k--", lw=1.5,
            label=f"OLS fit  r={r_p:.3f}, ρ={r_sp:.3f}")

    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("IK-recovered peak trunk forward lean (°)", fontsize=12)
    ax.set_ylabel("CS-cluster forward displacement at peak lean (mm)", fontsize=12)
    ax.set_title(
        f"Cross-subject validation: IK trunk lean vs chest marker displacement\n"
        f"N={len(results)} trials  |  Pearson r={r_p:.3f} (p={p_p:.2e})  "
        f"|  Spearman ρ={r_sp:.3f}",
        fontsize=10
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate outliers (|cs_fwd_mm| < 0 with large IK lean)
    for res in results:
        if res["cs_fwd_mm"] < -10 and res["peak_ik_deg"] > 10:
            ax.annotate(res["stem"], (res["peak_ik_deg"], res["cs_fwd_mm"]),
                        fontsize=6, alpha=0.6)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "trunk_lean_peak_validation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
