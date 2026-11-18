"""
Compare IK results across SC damping configurations.

Tests three configs on a fixed set of trials:
  A — uniform:    all DOFs = 0.05
  B — hierarchy:  arm=0.05, SC=0.20  (current default)
  C — heavy SC:   arm=0.05, SC=0.50

Metrics:
  - IK position error (mean, p95, max) across all frames
  - SC joint angle range (sternoclavicular_r2, sternoclavicular_r3)
  - Trunk forward lean range (should be identical across configs)

Usage:
  python scripts/test_damping.py
"""
import os, sys, subprocess, tempfile, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_path, get_project_root, get

ROOT    = get_project_root()
PYTHON  = get("conda_python", default=sys.executable)
MODEL   = get_path("mujoco_arm_model").replace("myoarm.xml", "myoarm_trunk.xml")
IK_SCRIPT = os.path.join(ROOT, "src", "inverse_kinematics", "ik_clean.py")

TRIALS = [
    ("stroke",  "S5_12_1"),
    ("stroke",  "S11_12_1"),
    ("stroke",  "S4_12_1"),
    ("healthy", "07_12_3"),
    ("healthy", "14_12_3"),
    ("healthy", "16_12_2"),
]

CONFIGS = {
    "A_uniform":   0.05,
    "B_hierarchy": 0.20,   # current default
    "C_heavy_SC":  0.50,
}

OUT_DIR = os.path.join(ROOT, "output", "damping_test")
os.makedirs(OUT_DIR, exist_ok=True)

SC_JOINTS    = ["sternoclavicular_r2", "sternoclavicular_r3"]
TRUNK_JOINT  = "trunk_forward_lean"


def read_mot(path):
    with open(path) as f:
        lines = f.readlines()
    end  = next(i for i, l in enumerate(lines) if "endheader" in l)
    cols = lines[end + 1].strip().split("\t")
    data = np.array([[float(x) for x in l.strip().split("\t")]
                     for l in lines[end + 2:] if l.strip()])
    return cols, data


def run_ik(trc_path, mot_path, sc_damp):
    env = os.environ.copy()
    env["IK_INTERACTIVE_ALIGN"] = "False"
    proc = subprocess.run(
        [PYTHON, IK_SCRIPT, MODEL, trc_path, mot_path, "0", "1.0", str(sc_damp)],
        capture_output=True, text=True, env=env, cwd=ROOT
    )
    ik_error = None
    for line in proc.stdout.splitlines():
        if "FINAL_MEAN_ERROR:" in line:
            ik_error = float(line.split(":")[1].strip())
    return ik_error, proc.returncode == 0


results = []

for group, trial in TRIALS:
    trc_path = os.path.join(ROOT, "output", "originals", group, "trc", trial + ".trc")
    if not os.path.exists(trc_path):
        print(f"  [skip] {trial} — TRC not found")
        continue

    print(f"\n--- {trial} ({group}) ---")
    trial_mots = {}

    for cfg_name, sc_damp in CONFIGS.items():
        mot_path = os.path.join(OUT_DIR, f"{trial}_{cfg_name}.mot")
        print(f"  {cfg_name} (SC={sc_damp})...", end=" ", flush=True)
        ik_err, ok = run_ik(trc_path, mot_path, sc_damp)
        if not ok or not os.path.exists(mot_path):
            print("FAILED")
            continue
        print(f"IK={ik_err:.2f}mm")
        trial_mots[cfg_name] = mot_path
        results.append({"trial": trial, "group": group,
                        "config": cfg_name, "sc_damp": sc_damp,
                        "ik_error_mm": ik_err})

    # Joint range comparison
    if len(trial_mots) == len(CONFIGS):
        print(f"  {'Joint':<22}", end="")
        for cfg in CONFIGS:
            print(f"  {cfg:<18}", end="")
        print()
        for jname in SC_JOINTS + [TRUNK_JOINT]:
            print(f"  {jname:<22}", end="")
            for cfg, mot_path in trial_mots.items():
                cols, data = read_mot(mot_path)
                if jname in cols:
                    col  = cols.index(jname)
                    vals = np.degrees(data[:, col])
                    print(f"  [{vals.min():+.1f}, {vals.max():+.1f}]°   ", end="")
                else:
                    print(f"  N/A                ", end="")
            print()

# --- Summary table ---
df = pd.DataFrame(results)
if df.empty:
    print("\nNo results collected.")
    sys.exit(1)

print(f"\n{'='*60}")
print("IK ERROR SUMMARY (mm)")
pivot = df.pivot_table(index="trial", columns="config", values="ik_error_mm")
print(pivot.round(2).to_string())

print(f"\nMean across trials:")
print(df.groupby("config")["ik_error_mm"].mean().round(2).to_string())

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 4))
configs  = list(CONFIGS.keys())
x        = np.arange(len(TRIALS))
width    = 0.25
colors   = ["steelblue", "tomato", "seagreen"]

for i, (cfg, color) in enumerate(zip(configs, colors)):
    vals = [df[(df.trial==t) & (df.config==cfg)]["ik_error_mm"].values
            for _, t in TRIALS]
    vals = [v[0] if len(v) else np.nan for v in vals]
    ax.bar(x + i*width, vals, width, label=cfg, color=color, alpha=0.8)

ax.set_xticks(x + width)
ax.set_xticklabels([t for _, t in TRIALS], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("IK error (mm)")
ax.set_title("IK error by damping config")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out_fig = os.path.join(OUT_DIR, "damping_comparison.png")
plt.savefig(out_fig, dpi=150)
plt.close()
print(f"\nPlot: {out_fig}")
