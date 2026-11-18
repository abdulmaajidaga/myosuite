"""
Experiment 3: Standard arm IK vs Weighted arm IK (SC damping hierarchy)

Both runs use SVD-prescribed trunk (Pass 1). Only Pass 2 (arm solve) differs:
  A. No SC hierarchy  — sternoclavicular DOFs at same cost as arm (SC_DAMPING=0.05)
  B. SC hierarchy     — sternoclavicular penalised 4x (SC_DAMPING=0.20)

Claim: without SC penalty, the IK exploits spurious sternoclavicular DOF excursion
(sternoclavicular_r3 up to ~9°) to reduce residual, giving unrealistic SC motion.
Penalising SC forces the solver to use real shoulder/elbow DOFs instead.

Output:
  output/experiments/exp3_sc_damping/
    results.csv         — per-trial: sc_config, ik_error_mm, sc_r3_peak_deg
    summary.txt
    figure.png          — IK error + SC r3 excursion side-by-side

Run:
  python scripts/experiments/exp3_sc_damping.py
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

SCRIPT_CSV2TRC = os.path.join(ROOT, "src", "data_processing", "convert_csv2trc.py")
SCRIPT_IK_SVD  = os.path.join(ROOT, "src", "inverse_kinematics", "ik_clean.py")

OUT_DIR = os.path.join(ROOT, "output", "experiments", "exp3_sc_damping")

# Configs: differ only in SC_DAMPING passed via env or script modification.
# ik_clean.py reads SC_DAMPING from argv[6] (if present).
SC_CONFIGS = {
    "No SC hierarchy\n(SC=0.05)":  "0.05",
    "SC hierarchy\n(SC=0.20)":     "0.20",
}

# 6 representative trials
TRIALS = [
    ("stroke",  "S4_12_1"),
    ("stroke",  "S5_12_1"),
    ("stroke",  "S11_12_1"),
    ("healthy", "07_12_3"),
    ("healthy", "14_12_3"),
    ("healthy", "16_12_2"),
]


def csv_for(group, trial):
    for d in [os.path.join(ROOT, "data", "kinematic", group),
              os.path.join(ROOT, "data", group)]:
        p = os.path.join(d, f"{trial}.csv")
        if os.path.exists(p):
            return p
    return None


def trc_for(group, trial):
    return os.path.join(ROOT, "output", "originals", group, "trc", f"{trial}.trc")


def ensure_trc(group, trial):
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


def run_ik(group, trial, trc_path, sc_label, sc_val):
    safe_label = sc_label.replace("\n", "_").replace("=", "").replace("(", "").replace(")", "").replace(" ", "_")
    mot_dir  = os.path.join(OUT_DIR, safe_label, group, "mot")
    os.makedirs(mot_dir, exist_ok=True)
    mot_path = os.path.join(mot_dir, f"{trial}.mot")

    env = os.environ.copy()
    env["IK_INTERACTIVE_ALIGN"] = "False"

    # ik_clean.py argv: model trc mot max_iters tr_damping sc_damping
    cmd = [PYTHON, SCRIPT_IK_SVD, MODEL_IK, trc_path, mot_path,
           "0",    # max_iters (0 = config default)
           "1.0",  # TR_DAMPING placeholder (trunk is prescribed, this is unused)
           sc_val] # SC_DAMPING

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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Exp 3: {len(TRIALS)} trials × {len(SC_CONFIGS)} SC configs\n")

    rows = []
    for sc_label, sc_val in SC_CONFIGS.items():
        label = sc_label.replace("\n", " ")
        print(f"=== {label} ===")
        for group, trial in TRIALS:
            trc_path = ensure_trc(group, trial)
            if trc_path is None:
                print(f"  SKIP {trial}: no TRC"); continue
            t0 = time.time()
            mot_path, ik_err = run_ik(group, trial, trc_path, sc_label, sc_val)
            elapsed = time.time() - t0

            sc_r3_peak = np.nan
            if mot_path:
                sc_r3 = load_mot_col(mot_path, "sternoclavicular_r3")
                if sc_r3 is not None:
                    sc_r3_peak = float(np.max(np.abs(sc_r3))) * 57.2958

            ik_str = f"{ik_err:.2f}" if ik_err is not None else "N/A"
            print(f"  {trial} ({group}) {elapsed:.0f}s  "
                  f"IK={ik_str} mm  SC_r3={sc_r3_peak:.1f}°")
            rows.append({
                "sc_config":    label,
                "group":        group,
                "trial":        trial,
                "ik_error_mm":  ik_err if ik_err is not None else np.nan,
                "sc_r3_peak_deg": sc_r3_peak,
            })
        print()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

    # --- Figure ---
    configs = list(SC_CONFIGS.keys())
    config_labels = [c.replace("\n", " ") for c in configs]
    colors = ["#7f7f7f", "#1f77b4"]  # grey=no hierarchy, blue=with hierarchy
    g_colors = {"stroke": "#d62728", "healthy": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: IK reconstruction error
    ax = axes[0]
    x  = np.arange(len(config_labels))
    w  = 0.35
    for gi, group in enumerate(["stroke", "healthy"]):
        means, stds = [], []
        for cfg in config_labels:
            sub = df[(df["sc_config"] == cfg) & (df["group"] == group)]["ik_error_mm"].dropna()
            means.append(sub.mean()); stds.append(sub.std())
        ax.bar(x + gi * w, means, w, yerr=stds, capsize=4,
               color=g_colors[group], alpha=0.8, label=group.capitalize())
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(config_labels, fontsize=9)
    ax.set_ylabel("Mean IK reconstruction error (mm)")
    ax.set_title("A — IK Error")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Panel B: SC r3 excursion
    ax = axes[1]
    for gi, group in enumerate(["stroke", "healthy"]):
        means, stds = [], []
        for cfg in config_labels:
            sub = df[(df["sc_config"] == cfg) & (df["group"] == group)]["sc_r3_peak_deg"].dropna()
            means.append(sub.mean()); stds.append(sub.std())
        ax.bar(x + gi * w, means, w, yerr=stds, capsize=4,
               color=g_colors[group], alpha=0.8, label=group.capitalize())
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(config_labels, fontsize=9)
    ax.set_ylabel("Peak sternoclavicular_r3 excursion (°)")
    ax.set_title("B — Spurious SC Excursion")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(5, color='red', ls='--', lw=1, alpha=0.6, label="5° threshold")

    fig.suptitle("Exp 3: SC Damping Hierarchy Prevents Spurious SC Excursion",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "figure.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Summary
    with open(os.path.join(OUT_DIR, "summary.txt"), 'w') as f:
        f.write("Experiment 3: Standard arm IK vs Weighted arm IK (SC hierarchy)\n")
        f.write("=" * 60 + "\n\n")
        for cfg in config_labels:
            f.write(f"Config: {cfg}\n")
            sub = df[df["sc_config"] == cfg]
            err = sub["ik_error_mm"].dropna()
            sc3 = sub["sc_r3_peak_deg"].dropna()
            f.write(f"  IK error:    {err.mean():.2f} ± {err.std():.2f} mm\n")
            f.write(f"  SC_r3 peak:  {sc3.mean():.2f} ± {sc3.std():.2f} °\n\n")

        # Paired comparison
        for metric in ["ik_error_mm", "sc_r3_peak_deg"]:
            vals = {cfg: df[df["sc_config"] == cfg][metric].dropna().values
                    for cfg in config_labels}
            v = list(vals.values())
            if len(v) == 2 and len(v[0]) > 1 and len(v[1]) > 1:
                t, p = stats.ttest_rel(v[0][:min(len(v[0]), len(v[1]))],
                                       v[1][:min(len(v[0]), len(v[1]))])
                f.write(f"Paired t-test {metric}: t={t:.3f}, p={p:.4f}\n")

    print(f"\nResults: {OUT_DIR}/")
    print(df.groupby("sc_config")[["ik_error_mm", "sc_r3_peak_deg"]].mean().to_string())


if __name__ == "__main__":
    main()
