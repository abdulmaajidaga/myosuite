"""
Experiment 1: Standard IK vs Best Method (SVD Two-Pass)

Standard IK:
  - All 8 markers in cost (arm + CS chest markers)
  - Trunk DOF completely free
  - Pure pseudoinverse, no damping/weighting
  - Trunk ends up at arbitrary values driven by arm residuals

Best Method (SVD two-pass):
  - Pass 1: trunk_forward_lean prescribed from CS plate SVD normal
  - Pass 2: arm IK with SC hierarchy (SC_DAMPING=0.20), trunk pinned

Output:
  output/experiments/exp1_standard_vs_svd/
    results.csv     — per-trial: method, group, ik_error_mm, trunk_peak_deg
    summary.txt     — group means ± std
    figure.png      — IK error + trunk lean comparison

Run:
  python scripts/experiments/exp1_standard_vs_svd.py
  python scripts/experiments/exp1_standard_vs_svd.py --subset   # 6 trials only
"""
import os, sys, subprocess, time, argparse
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

SCRIPT_CSV2TRC     = os.path.join(ROOT, "src", "data_processing", "convert_csv2trc.py")
SCRIPT_IK_STANDARD = os.path.join(ROOT, "paper_publication_assets", "code_supplementary",
                                   "ik_clean_primary_weighted.py")
SCRIPT_IK_BEST     = os.path.join(ROOT, "src", "inverse_kinematics", "ik_clean.py")

OUT_DIR = os.path.join(ROOT, "output", "experiments", "exp1_standard_vs_svd")

HEALTHY_DIR = get_path("data_healthy")
STROKE_DIR  = get_path("data_stroke")

# Representative subset used when --subset flag is passed
SUBSET_TRIALS = {
    "stroke":  ["S4_12_1", "S5_12_1", "S11_12_1"],
    "healthy": ["07_12_3", "14_12_3", "16_12_2"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", action="store_true",
                   help="Run on 6 representative trials instead of full dataset")
    return p.parse_args()


def find_csvs(d, subset=None):
    import glob
    all_csv = sorted(f for f in glob.glob(os.path.join(d, "*.csv")) if os.path.isfile(f))
    if subset is None:
        return all_csv
    return [f for f in all_csv if os.path.splitext(os.path.basename(f))[0] in subset]


def run_ik(csv_path, group, out_base, mode):
    """Run CSV->TRC->MOT. Returns (ik_error_mm or None)."""
    fid     = os.path.splitext(os.path.basename(csv_path))[0]
    trc_dir = os.path.join(out_base, group, "trc")
    mot_dir = os.path.join(out_base, group, "mot")
    os.makedirs(trc_dir, exist_ok=True)
    os.makedirs(mot_dir, exist_ok=True)
    trc_path = os.path.join(trc_dir, f"{fid}.trc")
    mot_path = os.path.join(mot_dir, f"{fid}.mot")

    # CSV -> TRC
    r = subprocess.run([PYTHON, SCRIPT_CSV2TRC, csv_path, trc_path],
                       capture_output=True, cwd=ROOT)
    if r.returncode != 0:
        return None, None

    env = os.environ.copy()
    env["IK_INTERACTIVE_ALIGN"] = "False"

    if mode == "standard":
        cmd = [PYTHON, SCRIPT_IK_STANDARD, MODEL_IK, trc_path, mot_path,
               "0", "0.0", "0.0", "0.0", "nodamping"]
    else:
        cmd = [PYTHON, SCRIPT_IK_BEST, MODEL_IK, trc_path, mot_path, "0"]

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    ik_err = None
    for line in proc.stdout.splitlines():
        if "FINAL_MEAN_ERROR:" in line:
            try:
                ik_err = float(line.split(":")[1].strip())
            except Exception:
                pass
    return mot_path if proc.returncode == 0 else None, ik_err


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


def collect(method_name, ik_script_mode, csvs_by_group, out_base):
    rows = []
    for group, csvs in csvs_by_group.items():
        for i, csv_path in enumerate(csvs):
            fid = os.path.splitext(os.path.basename(csv_path))[0]
            t0  = time.time()
            mot_path, ik_err = run_ik(csv_path, group, out_base, ik_script_mode)
            elapsed = time.time() - t0

            trunk_peak = np.nan
            if mot_path and os.path.exists(mot_path):
                t = load_mot_col(mot_path, "trunk_forward_lean")
                if t is not None:
                    trunk_peak = float(np.max(np.abs(t))) * 57.2958

            status = "OK" if mot_path and ik_err is not None else "FAIL"
            ik_str = f"{ik_err:.1f}" if ik_err is not None else "N/A"
            print(f"  [{method_name}] {group} {fid} ({i+1}/{len(csvs)}) "
                  f"{elapsed:.0f}s  IK={ik_str} mm  trunk={trunk_peak:.1f}°")
            rows.append({"method": method_name, "group": group, "trial": fid,
                         "status": status,
                         "ik_error_mm": ik_err if ik_err is not None else np.nan,
                         "trunk_peak_deg": trunk_peak})
    return pd.DataFrame(rows)


def plot_results(df, out_path):
    methods = ["Standard IK", "SVD Best"]
    colors  = {"stroke": "#d62728", "healthy": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Panel A: IK reconstruction error ---
    ax = axes[0]
    x_pos = 0
    xticks, xlabels = [], []
    for m in methods:
        for g in ["stroke", "healthy"]:
            vals = df[(df["method"] == m) & (df["group"] == g)]["ik_error_mm"].dropna()
            bp = ax.boxplot(vals, positions=[x_pos], widths=0.6,
                            patch_artist=True, showfliers=True,
                            medianprops=dict(color="black", lw=2),
                            boxprops=dict(facecolor=colors[g], alpha=0.7))
            xticks.append(x_pos)
            xlabels.append(f"{m}\n({g})")
            x_pos += 1
        x_pos += 0.5  # gap between methods
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("IK reconstruction error (mm)")
    ax.set_title("A — Reconstruction Error")
    ax.grid(axis='y', alpha=0.3)

    # --- Panel B: Trunk forward lean peak ---
    ax = axes[1]
    x_pos = 0
    xticks, xlabels = [], []
    for m in methods:
        for g in ["stroke", "healthy"]:
            vals = df[(df["method"] == m) & (df["group"] == g)]["trunk_peak_deg"].dropna()
            bp = ax.boxplot(vals, positions=[x_pos], widths=0.6,
                            patch_artist=True, showfliers=True,
                            medianprops=dict(color="black", lw=2),
                            boxprops=dict(facecolor=colors[g], alpha=0.7))
            xticks.append(x_pos)
            xlabels.append(f"{m}\n({g})")
            x_pos += 1
        x_pos += 0.5
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Peak trunk forward lean (°)")
    ax.set_title("B — Trunk Compensation Detected")
    ax.grid(axis='y', alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors["stroke"],  alpha=0.7, label="Stroke"),
               Patch(facecolor=colors["healthy"], alpha=0.7, label="Healthy")]
    fig.legend(handles=handles, loc="upper right", fontsize=9)

    fig.suptitle("Exp 1: Standard IK vs SVD Two-Pass", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {out_path}")


def write_summary(df, path):
    lines = ["Experiment 1: Standard IK vs SVD Two-Pass", "=" * 50, ""]
    for m in df["method"].unique():
        lines.append(f"Method: {m}")
        for g in ["stroke", "healthy"]:
            sub = df[(df["method"] == m) & (df["group"] == g)]
            ok  = sub[sub["status"] == "OK"]
            err = ok["ik_error_mm"].dropna()
            trk = ok["trunk_peak_deg"].dropna()
            lines.append(f"  {g}: n={len(ok)}")
            if len(err):
                lines.append(f"    IK error:   {err.mean():.2f} ± {err.std():.2f} mm  "
                              f"(p95={err.quantile(0.95):.2f})")
            if len(trk):
                lines.append(f"    Trunk peak: {trk.mean():.2f} ± {trk.std():.2f} °")
        lines.append("")

    # Between-group Mann-Whitney for trunk peak per method
    lines.append("Stroke vs Healthy trunk peak (Mann-Whitney U):")
    for m in df["method"].unique():
        s = df[(df["method"] == m) & (df["group"] == "stroke")]["trunk_peak_deg"].dropna()
        h = df[(df["method"] == m) & (df["group"] == "healthy")]["trunk_peak_deg"].dropna()
        if len(s) > 2 and len(h) > 2:
            u, p = stats.mannwhitneyu(s, h, alternative="greater")
            lines.append(f"  {m}: U={u:.0f}, p={p:.4f}")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Summary saved: {path}")


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    stroke_csvs  = find_csvs(STROKE_DIR,  SUBSET_TRIALS["stroke"]  if args.subset else None)
    healthy_csvs = find_csvs(HEALTHY_DIR, SUBSET_TRIALS["healthy"] if args.subset else None)
    csvs_by_group = {"stroke": stroke_csvs, "healthy": healthy_csvs}

    print(f"Exp 1: {len(stroke_csvs)} stroke + {len(healthy_csvs)} healthy trials")
    print(f"Subset mode: {args.subset}\n")

    all_dfs = []

    print("=== Standard IK (no damping, free trunk) ===")
    std_base = os.path.join(OUT_DIR, "standard")
    df_std = collect("Standard IK", "standard", csvs_by_group, std_base)
    all_dfs.append(df_std)

    print("\n=== SVD Two-Pass (best method) ===")
    svd_base = os.path.join(OUT_DIR, "svd_best")
    df_svd = collect("SVD Best", "svd_best", csvs_by_group, svd_base)
    all_dfs.append(df_svd)

    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    print(f"\nResults saved: {os.path.join(OUT_DIR, 'results.csv')}")

    plot_results(df, os.path.join(OUT_DIR, "figure.png"))
    write_summary(df, os.path.join(OUT_DIR, "summary.txt"))


if __name__ == "__main__":
    main()
