"""
phase_a_extended.py — Extended kinematic analysis, no ID required.

Generates 5 figure sets into test/output/figures_phase_a/:

  1. fma_gradient_strength.png   — Spearman ρ (FMA vs each kinematic metric) per config
  2. segment_consistency.png     — Upper-arm & forearm length variance (physical plausibility)
  3. velocity_profiles.png       — Mean wrist speed profile (normalised time) by FMA group
  4. trunk_wrist_ratio.png       — Trunk/wrist compensation ratio vs FMA per config
  5. joint_rom_vs_fma.png        — Elbow flexion & shoulder elevation ROM from MOT files

Run from custom_workspace/:
  python test/scripts/phase_a_extended.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(SCRIPT_DIR)                    # test/
FINAL_DIR  = os.path.join(BASE_DIR, "output", "final")
OUT_DIR    = os.path.join(BASE_DIR, "output", "figures_phase_a")
os.makedirs(OUT_DIR, exist_ok=True)

STAGES  = [0, 1, 2, 3]
AUGS    = ["dtw", "smote", "linear"]
FMA_ALL = list(range(16, 67))

STAGE_COLORS = {0: "#4e79a7", 1: "#f28e2b", 2: "#59a14f", 3: "#e15759"}
AUG_STYLES   = {"dtw": "-", "smote": "--", "linear": ":"}
AUG_LABELS   = {"dtw": "DTW", "smote": "SMOTE", "linear": "Linear"}
STAGE_LABELS = {0: "Stage 0\n(Baseline)", 1: "Stage 1\n(+CFG)",
                2: "Stage 2\n(+FiLM)",   3: "Stage 3\n(+Residual)"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "figure.dpi": 150,
})

# ── Data loaders ──────────────────────────────────────────────────────────────

def load_csv(stage, aug, fma):
    p = os.path.join(FINAL_DIR, f"stage{stage}_{aug}", "csv", f"FMA_{fma}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def load_mot(stage, aug, fma):
    p = os.path.join(FINAL_DIR, f"stage{stage}_{aug}", "mot", f"FMA_{fma}.mot")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, sep="\t", comment="#", skiprows=6)


def build_master_table():
    """Pre-compute all kinematic metrics for all 612 files once."""
    rows = []
    for stage in STAGES:
        for aug in AUGS:
            for fma in FMA_ALL:
                df = load_csv(stage, aug, fma)
                if df is None:
                    continue

                wr = df[["Wr_x","Wr_y","Wr_z"]].values
                sh = df[["Sh_x","Sh_y","Sh_z"]].values
                el = df[["El_x","El_y","El_z"]].values
                tr = df[["Trunk_x","Trunk_y","Trunk_z"]].values

                dt = 1/200.0
                vel = np.linalg.norm(np.gradient(wr, dt, axis=0), axis=1)

                upper_arm_len = np.linalg.norm(el - sh, axis=1)
                forearm_len   = np.linalg.norm(wr - el, axis=1)
                trunk_d       = np.linalg.norm(tr - tr[0], axis=1).max()
                wrist_d       = np.linalg.norm(wr - wr[0], axis=1).max()

                rows.append({
                    "stage": stage, "aug": aug, "fma": fma,
                    "wrist_range":   wr[:,1].max() - wr[:,1].min(),
                    "peak_vel":      vel.max(),
                    "trunk_disp":    trunk_d,
                    "wrist_disp":    wrist_d,
                    "trunk_wrist":   trunk_d / wrist_d if wrist_d > 1 else np.nan,
                    "ua_len_std":    upper_arm_len.std(),
                    "fa_len_std":    forearm_len.std(),
                    "ua_len_mean":   upper_arm_len.mean(),
                    "fa_len_mean":   forearm_len.mean(),
                    "vel_profile":   vel,          # keep full array for profile plot
                    "n_frames":      len(df),
                })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — FMA gradient strength (Spearman ρ heatmap)
# ══════════════════════════════════════════════════════════════════════════════

def fig_fma_gradient(data):
    print("Generating: fma_gradient_strength.png")

    metrics   = ["wrist_range", "peak_vel", "trunk_disp", "trunk_wrist"]
    m_labels  = ["Wrist Range", "Peak Velocity", "Trunk Disp.", "Trunk/Wrist Ratio"]

    # Build 4×3×4 array: stage × aug × metric
    rho_vals = np.zeros((4, 3, len(metrics)))
    pval_vals = np.zeros((4, 3, len(metrics)))

    for i, stage in enumerate(STAGES):
        for j, aug in enumerate(AUGS):
            sub = data[(data.stage==stage) & (data.aug==aug)].sort_values("fma")
            for k, m in enumerate(metrics):
                col = sub[m].dropna()
                fma = sub.loc[col.index, "fma"]
                if len(col) > 5:
                    rho, p = stats.spearmanr(fma, col)
                    rho_vals[i,j,k]  = rho
                    pval_vals[i,j,k] = p

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))
    fig.suptitle("FMA Gradient Strength — Spearman ρ (FMA vs Kinematic Metric)\n"
                 "Darker = stronger correlation  |  * p<0.05  ** p<0.01",
                 fontweight="bold")

    for k, (ax, ml) in enumerate(zip(axes, m_labels)):
        mat = rho_vals[:,:,k]
        pmat = pval_vals[:,:,k]
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(ml, fontsize=10)
        ax.set_xticks(range(3)); ax.set_xticklabels([AUG_LABELS[a] for a in AUGS], fontsize=8)
        ax.set_yticks(range(4)); ax.set_yticklabels([f"S{s}" for s in STAGES])

        for i in range(4):
            for j in range(3):
                p = pmat[i,j]
                sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                ax.text(j, i, f"{mat[i,j]:.2f}{sig}",
                        ha="center", va="center", fontsize=8.5,
                        color="black" if abs(mat[i,j]) < 0.7 else "white",
                        fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "fma_gradient_strength.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Segment length consistency (physical plausibility)
# ══════════════════════════════════════════════════════════════════════════════

def fig_segment_consistency(data):
    print("Generating: segment_consistency.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey="row")
    fig.suptitle("Segment Length Consistency — Std Dev Across Frames\n"
                 "(lower = more physically plausible; ideal = 0)",
                 fontweight="bold")

    for col, aug in enumerate(AUGS):
        sub = data[data.aug == aug]

        for row, (metric, label, color_base) in enumerate([
            ("ua_len_std", "Upper Arm Std Dev (mm)", "#4e79a7"),
            ("fa_len_std", "Forearm Std Dev (mm)",   "#e15759"),
        ]):
            ax = axes[row][col]
            ax.set_title(f"{AUG_LABELS[aug]}" if row == 0 else "")

            for stage in STAGES:
                s = sub[sub.stage == stage].sort_values("fma")
                ax.plot(s["fma"], s[metric],
                        color=STAGE_COLORS[stage],
                        linestyle=AUG_STYLES[aug],
                        linewidth=1.5,
                        label=f"Stage {stage}")

            ax.set_xlabel("FMA Score" if row == 1 else "")
            ax.set_ylabel(label if col == 0 else "")
            ax.grid(True, alpha=0.25)

    handles = [plt.Line2D([0],[0], color=STAGE_COLORS[s], lw=2,
                           label=f"Stage {s}") for s in STAGES]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "segment_consistency.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Velocity profiles (normalised time) by FMA group
# ══════════════════════════════════════════════════════════════════════════════

def fig_velocity_profiles(data):
    print("Generating: velocity_profiles.png")

    fma_groups = {
        "FMA 16–20 (Severe)":   list(range(16, 21)),
        "FMA 21–40 (Moderate)": list(range(21, 41)),
        "FMA 41–66 (Mild/Healthy)": list(range(41, 67)),
    }
    N_NORM = 100  # normalised time steps

    fig, axes = plt.subplots(len(fma_groups), len(STAGES),
                             figsize=(14, 10), sharex=True, sharey=False)
    fig.suptitle("Mean Wrist Speed Profile (Normalised Time)\n"
                 "Bell-shaped = smooth healthy reach  |  Multi-peaked = impaired",
                 fontweight="bold", y=1.01)

    for col, stage in enumerate(STAGES):
        axes[0][col].set_title(STAGE_LABELS[stage], fontsize=9)

        for row, (grp_label, fma_list) in enumerate(fma_groups.items()):
            ax = axes[row][col]
            t  = np.linspace(0, 1, N_NORM)

            for aug in AUGS:
                profiles = []
                sub = data[(data.stage==stage) & (data.aug==aug) &
                           (data.fma.isin(fma_list))]
                for _, row_data in sub.iterrows():
                    vel = row_data["vel_profile"]
                    if len(vel) < 5:
                        continue
                    # Normalise time axis
                    t_orig = np.linspace(0, 1, len(vel))
                    resampled = np.interp(t, t_orig, vel)
                    # Normalise amplitude 0→1
                    rng = resampled.max() - resampled.min()
                    if rng > 0:
                        profiles.append((resampled - resampled.min()) / rng)

                if profiles:
                    arr  = np.array(profiles)
                    mean = arr.mean(axis=0)
                    std  = arr.std(axis=0)
                    ax.plot(t, mean, color=STAGE_COLORS[stage],
                            linestyle=AUG_STYLES[aug],
                            linewidth=1.5, label=AUG_LABELS[aug])
                    ax.fill_between(t, mean-std, mean+std,
                                    color=STAGE_COLORS[stage], alpha=0.12)

            if col == 0:
                ax.set_ylabel(grp_label, fontsize=8.5)
            ax.set_xlabel("Normalised Time" if row == len(fma_groups)-1 else "")
            ax.grid(True, alpha=0.25)
            ax.set_ylim(-0.05, 1.15)

    # Legend on last col
    handles = [plt.Line2D([0],[0], color="gray", linestyle=AUG_STYLES[a],
                           lw=1.5, label=AUG_LABELS[a]) for a in AUGS]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "velocity_profiles.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Trunk / wrist compensation ratio vs FMA
# ══════════════════════════════════════════════════════════════════════════════

def fig_trunk_wrist_ratio(data):
    print("Generating: trunk_wrist_ratio.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Trunk Compensation Ratio (Trunk Disp / Wrist Disp) vs FMA Score\n"
                 "Lower ratio at higher FMA = less trunk lean = clinical expectation",
                 fontweight="bold")

    for col, aug in enumerate(AUGS):
        ax = axes[col]
        ax.set_title(AUG_LABELS[aug])
        sub = data[data.aug == aug]

        for stage in STAGES:
            s = sub[sub.stage == stage].sort_values("fma")
            valid = s.dropna(subset=["trunk_wrist"])
            ax.plot(valid["fma"], valid["trunk_wrist"],
                    color=STAGE_COLORS[stage],
                    linestyle=AUG_STYLES[aug],
                    linewidth=1.5,
                    label=f"Stage {stage}")

        ax.set_xlabel("FMA Score")
        ax.set_ylabel("Trunk / Wrist Ratio" if col == 0 else "")
        ax.axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
        ax.grid(True, alpha=0.25)

        # Annotate ideal trend direction
        ax.annotate("← less compensation\n   (healthy)", xy=(60, ax.get_ylim()[0]),
                    fontsize=7.5, color="gray", ha="center")

    handles = [plt.Line2D([0],[0], color=STAGE_COLORS[s], lw=2,
                           label=f"Stage {s}") for s in STAGES]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "trunk_wrist_ratio.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Joint ROM from MOT files (elbow flexion + shoulder elevation)
# ══════════════════════════════════════════════════════════════════════════════

def fig_joint_rom():
    print("Generating: joint_rom_vs_fma.png  (reading MOT files...)")

    rows = []
    for stage in STAGES:
        for aug in AUGS:
            for fma in FMA_ALL:
                mot = load_mot(stage, aug, fma)
                if mot is None:
                    continue
                row = {"stage": stage, "aug": aug, "fma": fma}
                # Radians → degrees
                for col, lbl in [("elbow_flexion", "elbow_rom"),
                                  ("shoulder_elv",  "sh_elv_rom"),
                                  ("elv_angle",     "elv_angle_rom"),
                                  ("pro_sup",       "pro_sup_rom")]:
                    if col in mot.columns:
                        v = mot[col].values * (180 / np.pi)
                        row[lbl] = v.max() - v.min()
                    else:
                        row[lbl] = np.nan
                rows.append(row)

    data = pd.DataFrame(rows)

    joints    = ["elbow_rom", "sh_elv_rom", "elv_angle_rom", "pro_sup_rom"]
    j_labels  = ["Elbow Flexion ROM (°)", "Shoulder Elevation ROM (°)",
                 "Elev. Angle ROM (°)", "Pro/Sup ROM (°)"]

    fig, axes = plt.subplots(len(joints), len(AUGS),
                             figsize=(14, 13), sharex=True)
    fig.suptitle("Joint Range of Motion vs FMA Score (from MOT files)\n"
                 "Radians converted to degrees",
                 fontweight="bold", y=1.01)

    for col, aug in enumerate(AUGS):
        axes[0][col].set_title(AUG_LABELS[aug], fontsize=11, fontweight="bold")
        sub = data[data.aug == aug]

        for row, (jc, jl) in enumerate(zip(joints, j_labels)):
            ax = axes[row][col]
            for stage in STAGES:
                s = sub[sub.stage == stage].sort_values("fma")
                valid = s.dropna(subset=[jc])
                ax.plot(valid["fma"], valid[jc],
                        color=STAGE_COLORS[stage],
                        linestyle=AUG_STYLES[aug],
                        linewidth=1.5,
                        label=f"Stage {stage}")
            ax.set_ylabel(jl if col == 0 else "")
            ax.set_xlabel("FMA Score" if row == len(joints)-1 else "")
            ax.grid(True, alpha=0.25)

    handles = [plt.Line2D([0],[0], color=STAGE_COLORS[s], lw=2,
                           label=f"Stage {s}") for s in STAGES]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "joint_rom_vs_fma.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Output: {OUT_DIR}\n")
    print("Pre-computing metrics for all 612 files...")
    data = build_master_table()
    print(f"  loaded {len(data)} records\n")

    fig_fma_gradient(data)
    fig_segment_consistency(data)
    fig_velocity_profiles(data)
    fig_trunk_wrist_ratio(data)
    fig_joint_rom()

    print("\nPhase A extended complete. Figures saved:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")
