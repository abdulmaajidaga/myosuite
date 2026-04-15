"""
phase_a_visuals.py — Paper-ready figures that require no ID/inverse-dynamics.

Generates 4 figure sets into test/output/final/figures_phase_a/:

  1. training_curves.png        — val loss for all 12 configs, grouped by augmentation
  2. wrist_trajectories.png     — wrist path (Y vs Z) at FMA 20 / 40 / 60 across configs
  3. rom_trends.png             — kinematic metrics vs FMA 16→66 per config
  4. ablation_summary.png       — stage contribution (CCI Rho + LitVal) from known results

Run from custom_workspace/:
  python test/scripts/phase_a_visuals.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(SCRIPT_DIR)                   # test/
MODELS_DIR   = os.path.join(BASE_DIR, "models", "cvae")
FINAL_DIR    = os.path.join(BASE_DIR, "output", "final")
OUT_DIR      = os.path.join(BASE_DIR, "output", "figures_phase_a")
os.makedirs(OUT_DIR, exist_ok=True)

STAGES  = [0, 1, 2, 3]
AUGS    = ["dtw", "smote", "linear"]
FMA_ALL = list(range(16, 67))

# ── Style ──────────────────────────────────────────────────────────────────────
STAGE_COLORS = {0: "#4e79a7", 1: "#f28e2b", 2: "#59a14f", 3: "#e15759"}
AUG_STYLES   = {"dtw": "-", "smote": "--", "linear": ":"}
AUG_MARKERS  = {"dtw": "o", "smote": "s", "linear": "^"}
AUG_LABELS   = {"dtw": "DTW", "smote": "SMOTE", "linear": "Linear"}
STAGE_LABELS = {
    0: "Stage 0 — Baseline LSTM",
    1: "Stage 1 — + CFG",
    2: "Stage 2 — + FiLM",
    3: "Stage 3 — + Residual (SOTA)",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "figure.dpi": 150,
})

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_csv(stage, aug, fma):
    path = os.path.join(FINAL_DIR, f"stage{stage}_{aug}", "csv", f"FMA_{fma}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def wrist_range(df):
    """Peak-to-peak wrist displacement in mm (Y axis — forward reach)."""
    return df["Wr_y"].max() - df["Wr_y"].min()


def peak_velocity(df, dt=1/200.0):
    """Peak wrist speed (mm/s)."""
    vx = np.gradient(df["Wr_x"].values, dt)
    vy = np.gradient(df["Wr_y"].values, dt)
    vz = np.gradient(df["Wr_z"].values, dt)
    return np.sqrt(vx**2 + vy**2 + vz**2).max()


def trunk_disp(df):
    """Max trunk displacement from start (mm)."""
    t = df[["Trunk_x", "Trunk_y", "Trunk_z"]].values
    return np.linalg.norm(t - t[0], axis=1).max()


def smoothness(df, dt=1/200.0):
    """Mean squared jerk of wrist (lower = smoother)."""
    for ax in ["Wr_x", "Wr_y", "Wr_z"]:
        if ax not in df.columns:
            return np.nan
    jx = np.gradient(np.gradient(np.gradient(df["Wr_x"].values, dt), dt), dt)
    jy = np.gradient(np.gradient(np.gradient(df["Wr_y"].values, dt), dt), dt)
    jz = np.gradient(np.gradient(np.gradient(df["Wr_z"].values, dt), dt), dt)
    return float(np.mean(jx**2 + jy**2 + jz**2))


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Training curves
# ══════════════════════════════════════════════════════════════════════════════

def fig_training_curves():
    print("Generating: training_curves.png")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    fig.suptitle("Validation Loss During Training — All 12 Configurations", fontweight="bold")

    for col, aug in enumerate(AUGS):
        ax = axes[col]
        ax.set_title(f"Augmentation: {AUG_LABELS[aug]}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss") if col == 0 else None

        for stage in STAGES:
            hist_path = os.path.join(MODELS_DIR, f"cvae_stage{stage}_{aug}_history.csv")
            if not os.path.exists(hist_path):
                continue
            df = pd.read_csv(hist_path)
            ax.plot(df["epoch"], df["val"],
                    color=STAGE_COLORS[stage],
                    label=f"Stage {stage}",
                    linewidth=1.5)

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 300)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "training_curves.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Wrist trajectories at FMA 20, 40, 60
# ══════════════════════════════════════════════════════════════════════════════

def fig_wrist_trajectories():
    print("Generating: wrist_trajectories.png")
    fma_levels = [20, 40, 60]
    fig, axes = plt.subplots(len(fma_levels), len(AUGS),
                             figsize=(14, 11), sharex=False, sharey=False)
    fig.suptitle("Wrist Trajectories (Y–Z plane) by FMA Level & Augmentation Method",
                 fontweight="bold", y=1.01)

    for row, fma in enumerate(fma_levels):
        for col, aug in enumerate(AUGS):
            ax = axes[row][col]
            ax.set_title(f"{AUG_LABELS[aug]} | FMA {fma}", fontsize=9)

            for stage in STAGES:
                df = load_csv(stage, aug, fma)
                if df is None:
                    continue
                # Normalize so all start at (0, 0)
                y = df["Wr_y"].values - df["Wr_y"].values[0]
                z = df["Wr_z"].values - df["Wr_z"].values[0]
                ax.plot(y, z,
                        color=STAGE_COLORS[stage],
                        linewidth=1.4,
                        label=f"S{stage}")
                ax.plot(y[0], z[0], "o", color=STAGE_COLORS[stage], markersize=4)
                ax.plot(y[-1], z[-1], "x", color=STAGE_COLORS[stage], markersize=5)

            ax.set_xlabel("Forward (mm)" if row == len(fma_levels)-1 else "")
            ax.set_ylabel("Up (mm)" if col == 0 else "")
            ax.grid(True, alpha=0.25)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)

    # Shared legend
    handles = [plt.Line2D([0],[0], color=STAGE_COLORS[s], lw=2,
                           label=f"Stage {s}") for s in STAGES]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "wrist_trajectories.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Kinematic metric trends across FMA 16–66
# ══════════════════════════════════════════════════════════════════════════════

def fig_rom_trends():
    print("Generating: rom_trends.png  (computing metrics for all 612 files...)")

    metrics_list = []
    for stage in STAGES:
        for aug in AUGS:
            for fma in FMA_ALL:
                df = load_csv(stage, aug, fma)
                if df is None:
                    continue
                metrics_list.append({
                    "stage": stage, "aug": aug, "fma": fma,
                    "wrist_range": wrist_range(df),
                    "peak_vel":    peak_velocity(df),
                    "trunk_disp":  trunk_disp(df),
                    "smoothness":  smoothness(df),
                })

    data = pd.DataFrame(metrics_list)

    metric_cols  = ["wrist_range", "peak_vel", "trunk_disp", "smoothness"]
    metric_labels = ["Wrist Range (mm)", "Peak Velocity (mm/s)",
                     "Trunk Displacement (mm)", "Mean Squared Jerk"]

    fig, axes = plt.subplots(len(metric_cols), len(AUGS),
                             figsize=(14, 14), sharex=True)
    fig.suptitle("Kinematic Metric Trends vs FMA Score (16→66)", fontweight="bold", y=1.01)

    for col, aug in enumerate(AUGS):
        axes[0][col].set_title(AUG_LABELS[aug], fontsize=11, fontweight="bold")
        for row, (mc, ml) in enumerate(zip(metric_cols, metric_labels)):
            ax = axes[row][col]
            sub = data[data["aug"] == aug]
            for stage in STAGES:
                s = sub[sub["stage"] == stage].sort_values("fma")
                if s.empty:
                    continue
                ax.plot(s["fma"], s[mc],
                        color=STAGE_COLORS[stage],
                        linestyle=AUG_STYLES[aug],
                        linewidth=1.4,
                        label=f"Stage {stage}")
            ax.set_ylabel(ml if col == 0 else "")
            ax.set_xlabel("FMA Score" if row == len(metric_cols)-1 else "")
            ax.grid(True, alpha=0.25)

    handles = [plt.Line2D([0],[0], color=STAGE_COLORS[s], lw=2,
                           label=f"Stage {s}") for s in STAGES]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "rom_trends.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Ablation summary (from known sweep results)
# ══════════════════════════════════════════════════════════════════════════════

def fig_ablation_summary():
    print("Generating: ablation_summary.png")

    # Results from 12-way sweep (obsidian/stage.md)
    results = [
        {"stage": 0, "aug": "DTW",    "cci_rho": -0.600, "litval": 67},
        {"stage": 0, "aug": "SMOTE",  "cci_rho": -0.480, "litval": 61},
        {"stage": 0, "aug": "Linear", "cci_rho": -0.550, "litval": 72},
        {"stage": 1, "aug": "DTW",    "cci_rho": -1.000, "litval": 72},
        {"stage": 1, "aug": "SMOTE",  "cci_rho": -0.650, "litval": 67},
        {"stage": 1, "aug": "Linear", "cci_rho": -0.780, "litval": 72},
        {"stage": 2, "aug": "DTW",    "cci_rho": -0.900, "litval": 61},
        {"stage": 2, "aug": "SMOTE",  "cci_rho": -1.000, "litval": 67},
        {"stage": 2, "aug": "Linear", "cci_rho": -0.900, "litval": 72},
        {"stage": 3, "aug": "DTW",    "cci_rho": -0.600, "litval": 67},
        {"stage": 3, "aug": "SMOTE",  "cci_rho": -1.000, "litval": 72},
        {"stage": 3, "aug": "Linear", "cci_rho": -0.900, "litval": 72},
    ]
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Ablation: Architectural Stage vs Clinical Validity Metrics",
                 fontweight="bold")

    x      = np.arange(4)
    width  = 0.22
    aug_colors = {"DTW": "#4e79a7", "SMOTE": "#f28e2b", "Linear": "#59a14f"}

    for ax_idx, (metric, ylabel, title) in enumerate([
        ("cci_rho",  "CCI Rho (Spearman)",    "CCI Correlation with FMA\n(more negative = better)"),
        ("litval",   "Pass Rate (%)",          "Literature Validation Pass Rate"),
    ]):
        ax = axes[ax_idx]
        for i, aug in enumerate(["DTW", "SMOTE", "Linear"]):
            vals = [df[(df.stage==s) & (df.aug==aug)][metric].values[0] for s in STAGES]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width, label=aug,
                          color=aug_colors[aug], alpha=0.85, edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.01 if metric == "litval" else 0.01),
                        f"{val:.2f}" if metric == "cci_rho" else f"{val}%",
                        ha="center", va="bottom", fontsize=7.5)

        ax.set_title(title)
        ax.set_xlabel("Stage")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Stage {s}\n{STAGE_LABELS[s].split('—')[1].strip()}"
                            for s in STAGES], fontsize=8)
        ax.legend(title="Augmentation")
        ax.grid(True, axis="y", alpha=0.3)

        # Reference line: real human
        if metric == "cci_rho":
            ax.axhline(-0.911, color="red", linestyle="--", linewidth=1.2,
                       label="Real Human (−0.911)")
            ax.legend(title="Augmentation")
        elif metric == "litval":
            ax.axhline(89, color="red", linestyle="--", linewidth=1.2)
            ax.text(3.6, 90.5, "Real Human\n89%", color="red", fontsize=7.5)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "ablation_summary.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Training curves heatmap (final val loss per config)
# ══════════════════════════════════════════════════════════════════════════════

def fig_val_loss_heatmap():
    print("Generating: val_loss_heatmap.png")

    matrix = np.zeros((4, 3))
    for i, stage in enumerate(STAGES):
        for j, aug in enumerate(AUGS):
            hist_path = os.path.join(MODELS_DIR, f"cvae_stage{stage}_{aug}_history.csv")
            if os.path.exists(hist_path):
                df = pd.read_csv(hist_path)
                matrix[i, j] = df["val"].iloc[-10:].mean()  # avg last 10 epochs

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto")
    plt.colorbar(im, ax=ax, label="Final Val Loss (avg last 10 epochs)")

    ax.set_xticks(range(3))
    ax.set_xticklabels([AUG_LABELS[a] for a in AUGS])
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"Stage {s}" for s in STAGES])
    ax.set_title("Final Validation Loss Heatmap\n(lower = better fit)", fontweight="bold")

    for i in range(4):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                    color="black", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "val_loss_heatmap.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")
    fig_training_curves()
    fig_val_loss_heatmap()
    fig_wrist_trajectories()
    fig_rom_trends()
    fig_ablation_summary()
    print("\nPhase A complete. All figures saved to:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")
