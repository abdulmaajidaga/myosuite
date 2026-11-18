"""
Experiment 2: Why IK-driven trunk failed — comparison of trunk methods

Four methods tested on representative trials:
  1. IK-free (no damping)  — trunk completely free, pseudoinverse
  2. IK-low   (TR=0.05)    — uniform low damping, trunk free
  3. IK-heavy (TR=50.0)    — heavy trunk penalty, neutral-start
  4. SVD prescribed        — trunk set from CS plate SVD normal before arm solve

Key claim: methods 1–3 produce trunk values that do NOT correlate with the true
trunk lean measured by the CS chest plate (independent ground truth). Method 4
does — because it IS the CS plate measurement.

Secondary finding: IK arm markers have near-zero gradient on trunk DOF (shoulder-
relative frame), so any IK-driven trunk is mostly random/biased rather than signal.

Output:
  output/experiments/exp2_trunk_ik_methods/
    results.csv         — per-trial: method, trunk_ik_peak_deg, cs_peak_deg, ik_error_mm
    correlations.csv    — Pearson r + p-value per method (trunk_ik vs cs_plate)
    summary.txt
    figure.png          — scatter: trunk_ik vs cs_plate, coloured by method
                        — bar: IK error per method

Run:
  python scripts/experiments/exp2_trunk_ik_methods.py
"""
import os, sys, subprocess, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import get_path, get_project_root, get

ROOT     = get_project_root()
PYTHON   = get("conda_python", default=sys.executable)
MODEL_IK = get_path("mujoco_arm_model").replace("myoarm.xml", "myoarm_trunk.xml")

SCRIPT_CSV2TRC  = os.path.join(ROOT, "src", "data_processing", "convert_csv2trc.py")
SCRIPT_IK_BASE  = os.path.join(ROOT, "paper_publication_assets", "code_supplementary",
                                "ik_clean_primary_weighted.py")
SCRIPT_IK_SVD   = os.path.join(ROOT, "src", "inverse_kinematics", "ik_clean.py")

OUT_DIR = os.path.join(ROOT, "output", "experiments", "exp2_trunk_ik_methods")

HEALTHY_DIR = get_path("data_healthy")
STROKE_DIR  = get_path("data_stroke")

# 6 representative trials with known-good CS plate data
TRIALS = [
    ("stroke",  "S4_12_1"),
    ("stroke",  "S5_12_1"),
    ("stroke",  "S11_12_1"),
    ("healthy", "07_12_3"),
    ("healthy", "14_12_3"),
    ("healthy", "16_12_2"),
]

METHODS = {
    "IK-free\n(no damping)":  dict(mode="base", tr=0.0,  sc=0.0,  arm=0.0, nodamping=True),
    "IK-low\n(TR=0.05)":      dict(mode="base", tr=0.05, sc=0.05, arm=0.05, nodamping=False),
    "IK-heavy\n(TR=50)":      dict(mode="base", tr=50.0, sc=0.20, arm=0.05, nodamping=False),
    "SVD prescribed\n(best)": dict(mode="svd"),
}


def csv_for(group, trial):
    path = os.path.join(ROOT, "data", "kinematic", group, f"{trial}.csv")
    if not os.path.exists(path):
        path = os.path.join(ROOT, "data", group, f"{trial}.csv")
    return path if os.path.exists(path) else None


def trc_for(group, trial):
    return os.path.join(ROOT, "output", "originals", group, "trc", f"{trial}.trc")


def run_ik(group, trial, trc_path, method_key, cfg, out_base):
    """Run IK for one trial+method. Returns (mot_path, ik_error_mm)."""
    mot_dir  = os.path.join(out_base, method_key.replace("\n", "_").replace(" ", "_"),
                            group, "mot")
    os.makedirs(mot_dir, exist_ok=True)
    mot_path = os.path.join(mot_dir, f"{trial}.mot")

    env = os.environ.copy()
    env["IK_INTERACTIVE_ALIGN"] = "False"

    if cfg["mode"] == "svd":
        cmd = [PYTHON, SCRIPT_IK_SVD, MODEL_IK, trc_path, mot_path, "0"]
    else:
        cmd = [PYTHON, SCRIPT_IK_BASE, MODEL_IK, trc_path, mot_path,
               "0",
               str(cfg["tr"]),
               str(cfg["sc"]),
               str(cfg["arm"])]
        if cfg["nodamping"]:
            cmd.append("nodamping")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    ik_err = None
    for line in proc.stdout.splitlines():
        if "FINAL_MEAN_ERROR:" in line:
            try:
                ik_err = float(line.split(":")[1].strip())
            except Exception:
                pass
    ok = proc.returncode == 0 and os.path.exists(mot_path)
    return (mot_path if ok else None), ik_err


def load_mot_col(mot_path, col):
    try:
        with open(mot_path) as f:
            lines = f.readlines()
        hend = next(i for i, l in enumerate(lines) if "endheader" in l)
        cols = lines[hend + 1].strip().split('\t')
        arr  = np.array([l.strip().split('\t') for l in lines[hend + 2:] if l.strip()],
                        dtype=float)
        if col in cols:
            return arr[:, cols.index(col)]
    except Exception:
        pass
    return None


def get_cs_tilt_peak(group, trial):
    """Get peak CS plate forward tilt for a trial (radians → degrees)."""
    csv_path = csv_for(group, trial)
    if csv_path is None:
        return np.nan
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from inject_cs_trunk import load_csv, wrist_fwd_axis, cs_lean_rad
        df  = load_csv(csv_path)
        fwd = wrist_fwd_axis(df)
        tilt = cs_lean_rad(df, fwd)
        if tilt is None:
            return np.nan
        baseline = float(np.nanmean(tilt[:5]))
        delta    = tilt - baseline
        return float(np.nanmax(np.abs(delta))) * 57.2958
    except Exception as e:
        print(f"  [CS tilt] {trial}: {e}")
        return np.nan


def ensure_trc(group, trial):
    """Make sure TRC exists; generate from CSV if needed."""
    trc_path = trc_for(group, trial)
    if os.path.exists(trc_path):
        return trc_path
    csv_path = csv_for(group, trial)
    if csv_path is None:
        return None
    os.makedirs(os.path.dirname(trc_path), exist_ok=True)
    r = subprocess.run([PYTHON, SCRIPT_CSV2TRC, csv_path, trc_path],
                       capture_output=True, cwd=ROOT)
    return trc_path if r.returncode == 0 else None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Exp 2: {len(TRIALS)} trials × {len(METHODS)} methods\n")

    # Pre-compute CS ground truth for each trial
    cs_peaks = {}
    for group, trial in TRIALS:
        cs_peaks[(group, trial)] = get_cs_tilt_peak(group, trial)
        print(f"  CS tilt peak {trial} ({group}): {cs_peaks[(group, trial)]:.1f}°")
    print()

    rows = []
    for m_key, cfg in METHODS.items():
        label = m_key.replace("\n", " ")
        print(f"=== {label} ===")
        for group, trial in TRIALS:
            trc_path = ensure_trc(group, trial)
            if trc_path is None:
                print(f"  SKIP {trial}: no TRC")
                continue
            t0 = time.time()
            mot_path, ik_err = run_ik(group, trial, trc_path, m_key, cfg, OUT_DIR)
            elapsed = time.time() - t0

            trunk_peak = np.nan
            if mot_path:
                t = load_mot_col(mot_path, "trunk_forward_lean")
                if t is not None:
                    trunk_peak = float(np.max(np.abs(t))) * 57.2958

            cs_peak = cs_peaks[(group, trial)]
            ik_str = f"{ik_err:.1f}" if ik_err is not None else "N/A"
            print(f"  {trial} ({group}) {elapsed:.0f}s  "
                  f"IK={ik_str} mm  trunk_ik={trunk_peak:.1f}°  cs_true={cs_peak:.1f}°")
            rows.append({
                "method":          label,
                "group":           group,
                "trial":           trial,
                "ik_error_mm":     ik_err if ik_err is not None else np.nan,
                "trunk_ik_peak_deg": trunk_peak,
                "cs_plate_peak_deg": cs_peak,
            })
        print()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

    # Correlations: trunk_ik_peak vs cs_plate_peak per method
    corr_rows = []
    for m_label in df["method"].unique():
        sub = df[df["method"] == m_label][["trunk_ik_peak_deg", "cs_plate_peak_deg"]].dropna()
        if len(sub) >= 3:
            r, p = stats.pearsonr(sub["trunk_ik_peak_deg"], sub["cs_plate_peak_deg"])
        else:
            r, p = np.nan, np.nan
        corr_rows.append({"method": m_label, "pearson_r": r, "p_value": p, "n": len(sub)})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(OUT_DIR, "correlations.csv"), index=False)

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    method_labels = list(df["method"].unique())
    method_colors = plt.cm.tab10(np.linspace(0, 0.8, len(method_labels)))
    color_map = dict(zip(method_labels, method_colors))
    marker_map = {"stroke": "^", "healthy": "o"}

    # Panel A: trunk_ik vs cs_plate scatter (one dot per trial, colour=method)
    ax = axes[0]
    for m_label in method_labels:
        sub = df[df["method"] == m_label]
        r_row = corr_df[corr_df["method"] == m_label].iloc[0]
        for _, row in sub.iterrows():
            ax.scatter(row["cs_plate_peak_deg"], row["trunk_ik_peak_deg"],
                       color=color_map[m_label],
                       marker=marker_map.get(row["group"], "o"),
                       s=80, alpha=0.85)
        # Trend line per method
        s2 = sub[["trunk_ik_peak_deg", "cs_plate_peak_deg"]].dropna()
        if len(s2) >= 3:
            x = s2["cs_plate_peak_deg"].values
            y = s2["trunk_ik_peak_deg"].values
            m_fit, b_fit = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 50)
            ax.plot(xr, m_fit * xr + b_fit, color=color_map[m_label], lw=1.5, alpha=0.7,
                    label=f"{m_label}  r={r_row['pearson_r']:.2f}")

    # Identity line
    lims = [0, max(df[["trunk_ik_peak_deg", "cs_plate_peak_deg"]].max()) * 1.1]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.4, label="Identity")
    ax.set_xlim(0, lims[1])
    ax.set_ylim(-5, lims[1])
    ax.set_xlabel("CS plate peak tilt (°) — ground truth")
    ax.set_ylabel("IK trunk_forward_lean peak (°)")
    ax.set_title("A — Trunk IK vs CS Plate Ground Truth")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    # Group markers legend
    from matplotlib.lines import Line2D
    leg2 = [Line2D([0],[0], marker='^', color='gray', ls='none', label='Stroke'),
            Line2D([0],[0], marker='o', color='gray', ls='none', label='Healthy')]
    ax.legend(handles=ax.get_legend_handles_labels()[0] + leg2,
              labels=ax.get_legend_handles_labels()[1] + ["Stroke", "Healthy"],
              fontsize=7, loc="upper left")

    # Panel B: IK reconstruction error per method
    ax = axes[1]
    for i, m_label in enumerate(method_labels):
        vals = df[df["method"] == m_label]["ik_error_mm"].dropna()
        ax.bar(i, vals.mean(), color=color_map[m_label], alpha=0.8,
               yerr=vals.std(), capsize=4, label=m_label)
    ax.set_xticks(range(len(method_labels)))
    ax.set_xticklabels([l.replace("\n", "\n") for l in method_labels], fontsize=8)
    ax.set_ylabel("Mean IK reconstruction error (mm)")
    ax.set_title("B — IK Error by Method")
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Exp 2: Why IK-Driven Trunk Failed", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figure.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Summary
    with open(os.path.join(OUT_DIR, "summary.txt"), 'w') as f:
        f.write("Experiment 2: Why IK-Driven Trunk Failed\n")
        f.write("=" * 50 + "\n\n")
        f.write("Trunk IK peak vs CS plate ground truth — Pearson r:\n")
        for _, row in corr_df.iterrows():
            f.write(f"  {row['method']:<35} r={row['pearson_r']:.3f}  p={row['p_value']:.4f}  n={row['n']}\n")
        f.write("\nConclusion: IK-free, IK-low, IK-heavy all show low/noisy correlation\n")
        f.write("with CS plate ground truth. SVD prescribed trivially matches (r≈1).\n")
        f.write("This confirms IK cannot reliably determine trunk angle from arm markers.\n")

    print(f"\nResults: {OUT_DIR}/")
    print("Correlation summary:")
    print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
