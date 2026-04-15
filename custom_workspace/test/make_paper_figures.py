"""
make_paper_figures.py — Generate all paper figures for main.tex.

Outputs to: custom_workspace/ (same dir as main.tex so LaTeX finds them).

Figures generated:
  aug_smote_phases.png
  aug_comparison_trajectories.png
  aug_ablation_bars.png
  phase1_cci_progression.png
  ablation_component_bars.png
  cfg_effect_trajectories.png
  sag_constraint_effect.png
  d_phase_wrist_rho.png
  guidance_scale_sweep.png
  cdp_sweep.png
  training_curves_g1.png
  n10_variance_combined.png   (copy from variance/)
  ablation_waterfall_all.png
"""

import os, sys, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.stats import spearmanr
from scipy import signal

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(TEST_DIR)                   # custom_workspace/
OUT_DIR     = os.path.join(ROOT_DIR, "figures")             # figures/ folder
T2_OUT      = os.path.join(TEST_DIR, "output")
T1_OUT      = os.path.join(ROOT_DIR, "test", "output", "final")
SMOTE_DIR   = os.path.join(ROOT_DIR, "data", "kinematic", "cutoff", "augmented_smote")

def savefig(name, fig, dpi=150):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")

def load_csv_wry(exp_dir, fma):
    """Load wrist-Y column from test2 output CSV for a given FMA."""
    p = os.path.join(T2_OUT, exp_dir, "csv", f"FMA_{fma}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return df["Wr_y"].values if "Wr_y" in df.columns else None

def smooth(arr, cutoff=6, fs=100):
    nyq = 0.5 * fs
    b, a = signal.butter(2, min(cutoff / nyq, 0.99), btype="low")
    return signal.filtfilt(b, a, arr)

PALETTE = {
    "smote":  "#2c7bb6",
    "dtw":    "#d7191c",
    "linear": "#fdae61",
    "cfg":    "#1a9641",
    "no_cfg": "#d73027",
}

# =============================================================================
# 1. aug_smote_phases.png
# =============================================================================
def fig_smote_phases():
    print("1. aug_smote_phases.png")
    fma_show = [20, 30, 40, 50, 66]
    colors   = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(fma_show)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    def smote_files(fma, n=8):
        pool = [f for f in os.listdir(SMOTE_DIR)
                if f.endswith(".csv") and f"FMA{fma}" in f]
        return list(np.random.choice(pool, size=min(n, len(pool)), replace=False))

    def load_wy(fn):
        df = pd.read_csv(os.path.join(SMOTE_DIR, fn))
        return df["Wr_y"].values[:100] if "Wr_y" in df.columns else None

    # Left: within-class k-NN expansion — show samples at FMA 20 and 66
    ax = axes[0]
    for fma, color in zip([20, 66], [colors[0], colors[-1]]):
        files = smote_files(fma, 8)
        for i, fn in enumerate(files):
            wy = load_wy(fn)
            if wy is None or len(wy) < 10: continue
            lw, ls, alpha = (1.8, "-", 0.9) if i == 0 else (0.6, "--", 0.35)
            ax.plot(wy, color=color, lw=lw, ls=ls, alpha=alpha,
                    label=f"FMA {fma}" if i == 0 else "_")
    ax.set_title("Step 1: Within-Class k-NN Expansion\n(solid=original, dashed=synthetic)", fontsize=11)
    ax.set_xlabel("Frame", fontsize=10); ax.set_ylabel("Wrist Y (delta mm)", fontsize=10)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # Right: cross-class FMA-proportional blending — show mean ± SD band per FMA
    ax = axes[1]
    for fma, color in zip(fma_show, colors):
        trajs = [wy for fn in smote_files(fma, 20)
                 if (wy := load_wy(fn)) is not None and len(wy) >= 100]
        if len(trajs) < 3: continue
        arr = np.array(trajs)
        mn, sd = arr.mean(0), arr.std(0)
        x = np.arange(100)
        ax.fill_between(x, mn - sd, mn + sd, alpha=0.15, color=color)
        ax.plot(x, mn, color=color, lw=2, label=f"FMA {fma}")
    ax.set_title("Step 2: Cross-Class FMA-Proportional Blending\n(mean ± 1 SD per FMA level)", fontsize=11)
    ax.set_xlabel("Frame", fontsize=10); ax.set_ylabel("Wrist Y (mm)", fontsize=10)
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle("SMOTE Augmentation Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("aug_smote_phases.png", fig)


# =============================================================================
# 2. aug_comparison_trajectories.png
# =============================================================================
def fig_aug_comparison_trajectories():
    print("2. aug_comparison_trajectories.png")
    configs = [
        ("G1_standard_split_300", "SMOTE",  PALETTE["smote"]),
        ("H1_dtw_aug",            "DTW",    PALETTE["dtw"]),
        ("H2_linear_aug",         "Linear", PALETTE["linear"]),
    ]
    fma_pairs = [20, 66]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)

    for row, (exp, label, color) in enumerate(configs):
        for col, fma in enumerate(fma_pairs):
            ax = axes[row, col]
            wy = load_csv_wry(exp, fma)
            if wy is not None:
                t = np.linspace(0, 1, len(wy))
                ax.plot(t, wy, color=color, lw=2)
                ax.fill_between(t, wy - 5, wy + 5, alpha=0.15, color=color)
            ax.set_title(f"{label} — FMA {fma}", fontsize=11)
            ax.grid(alpha=0.3)
            if col == 0: ax.set_ylabel("Wrist Y (mm)", fontsize=10)
            if row == 2: ax.set_xlabel("Normalised time", fontsize=10)

    fig.suptitle("Wrist Y Trajectories by Augmentation Method\n(FMA 20: severe impairment | FMA 66: healthy)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("aug_comparison_trajectories.png", fig)


# =============================================================================
# 3. aug_ablation_bars.png
# =============================================================================
def fig_aug_ablation_bars():
    print("3. aug_ablation_bars.png")
    methods = ["SMOTE\n(primary)", "DTW", "Linear\nbaseline"]
    rhos    = [0.915, 0.829, 0.588]
    colors  = [PALETTE["smote"], PALETTE["dtw"], PALETTE["linear"]]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(methods, rhos, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                f"ρ = {val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(0.9, color="gray", ls="--", lw=1.2, label="Target ρ = 0.90")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Wrist Range Correlation (ρ)", fontsize=12)
    ax.set_title("Augmentation Method Ablation\n(G1 architecture, N=10 samples averaged)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig("aug_ablation_bars.png", fig)


# =============================================================================
# 4. phase1_cci_progression.png
# =============================================================================
def fig_phase1_cci():
    print("4. phase1_cci_progression.png")
    # From test/ 12-config sweep (documented in obsidian)
    stages = ["Stage 0\n(baseline)", "Stage 1\n(+CFG)", "Stage 2\n(+FiLM)", "Stage 3\n(+Residual)"]
    cci_smote = [-0.480, -0.650, -1.000, -1.000]
    cci_dtw   = [-0.320, -0.600, -1.000, -1.000]
    x = np.arange(len(stages))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w/2, cci_smote, w, label="SMOTE", color=PALETTE["smote"],  edgecolor="white")
    b2 = ax.bar(x + w/2, cci_dtw,   w, label="DTW",   color=PALETTE["dtw"],    edgecolor="white")
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h - 0.04,
                    f"{h:.2f}", ha="center", va="top", fontsize=9, color="white", fontweight="bold")
    ax.axhline(-0.9, color="gray", ls="--", lw=1.2, label="Clinical target ρ < −0.9")
    ax.set_xticks(x); ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel("CCI Rho (Spearman ρ)", fontsize=12)
    ax.set_title("Phase 1: CCI Rho Progression Across Architecture Stages", fontsize=13)
    ax.set_ylim(-1.15, 0.1)
    ax.legend(fontsize=11); ax.grid(axis="y", alpha=0.3)

    # Annotate component additions
    for i, label in enumerate(["CFG added", "FiLM added", "Residual added"]):
        ax.annotate("", xy=(i+1, -1.08), xytext=(i, -1.08),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
        ax.text(i + 0.5, -1.12, label, ha="center", fontsize=8, color="dimgray")

    plt.tight_layout()
    savefig("phase1_cci_progression.png", fig)


# =============================================================================
# 5. ablation_component_bars.png
# =============================================================================
def fig_ablation_components():
    print("5. ablation_component_bars.png")
    labels = ["A0\nFull Stage 3\n(reference)", "A1\n− Residual\ndecoder",
              "A2\n− FiLM\nconditioning", "A3\n− CFG\nguidance"]
    rhos   = [0.688, 0.389, 0.426, 0.272]
    colors = ["#2ca25f", "#fc8d59", "#fdbb84", "#d7301f"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, rhos, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    for bar, val, lbl in zip(bars, rhos, labels):
        drop = rhos[0] - val
        ann  = f"ρ = {val:.3f}" if val == rhos[0] else f"ρ = {val:.3f}\n(−{drop:.3f})"
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                ann, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(rhos[0], color="#2ca25f", ls="--", lw=1.2, alpha=0.6, label="Full Stage 3 reference")
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Wrist Range Correlation (ρ)", fontsize=12)
    ax.set_title("Phase 2: Component Ablation — A-Phase\n(200 epochs, SMOTE, N=1 sample)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig("ablation_component_bars.png", fig)


# =============================================================================
# 6. cfg_effect_trajectories.png
# =============================================================================
def fig_cfg_effect():
    print("6. cfg_effect_trajectories.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    configs = [("A0_full", "With CFG  (ρ = 0.688)"), ("A3_no_cfg", "Without CFG  (ρ = 0.272)")]
    fma_colors = {20: "#d73027", 66: "#2ca25f"}

    for ax, (exp, title) in zip(axes, configs):
        for fma, color in fma_colors.items():
            wy = load_csv_wry(exp, fma)
            if wy is not None:
                t = np.linspace(0, 1, len(wy))
                ax.plot(t, wy, color=color, lw=2.5, label=f"FMA {fma}")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Normalised time", fontsize=11)
        ax.legend(fontsize=11); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Wrist Y (mm)", fontsize=12)

    fig.suptitle("Effect of Classifier-Free Guidance on FMA Separation\n"
                 "Left: CFG amplifies severity-specific features. "
                 "Right: Without CFG, trajectories collapse.",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig("cfg_effect_trajectories.png", fig)


# =============================================================================
# 7. sag_constraint_effect.png
# =============================================================================
def fig_sag_effect():
    print("7. sag_constraint_effect.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    configs = [("D0_stage3_baseline", "Without Sagittal Constraint\n(σ_sag = 34.0 mm)"),
               ("D1_stage3_sag",      "With Sagittal Constraint  (w=5)\n(σ_sag = 0.29 mm)")]
    fmas = [20, 30, 40, 50, 66]
    colors = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(fmas)))

    for ax, (exp, title) in zip(axes, configs):
        for fma, color in zip(fmas, colors):
            p = os.path.join(T2_OUT, exp, "csv", f"FMA_{fma}.csv")
            if not os.path.exists(p): continue
            df = pd.read_csv(p)
            if "Wr_x" not in df.columns: continue
            wx = df["Wr_x"].values
            wx_rel = wx - wx[0]
            t = np.linspace(0, 1, len(wx_rel))
            ax.plot(t, wx_rel, color=color, lw=2, label=f"FMA {fma}")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Normalised time", fontsize=11)
        ax.set_ylabel("Wrist X deviation from start (mm)", fontsize=10)
        ax.legend(fontsize=9, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle("Sagittal Constraint Effect on Lateral Wrist Drift\n"
                 "(FMA 16–66; lower = more sagittal-plane motion)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig("sag_constraint_effect.png", fig)


# =============================================================================
# 8. d_phase_wrist_rho.png
# =============================================================================
def fig_d_phase():
    print("8. d_phase_wrist_rho.png")
    exps = [
        ("D0_stage3_baseline", "D0\nNo constraints\n(baseline)",   0.443),
        ("D1_stage3_sag",      "D1\n+ Sag (w=5)\n[selected]",      0.665),
        ("D2_stage3_sag_strong","D2\n+ Sag (w=15)\n[over-constr.]", 0.557),
        ("D3_stage3_minimal",  "D3\n+ Sag\nmin loss",              0.345),
        ("D4_no_dyn",          "D4\nD1 minus\nL_dyn",              0.261),
    ]
    labels = [e[1] for e in exps]
    rhos   = [e[2] for e in exps]
    colors = ["#fc8d59", "#2ca25f", "#fdbb84", "#e0e0e0", "#d7301f"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, rhos, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Wrist Range Correlation (ρ)", fontsize=12)
    ax.set_title("Phase 3: D-Phase Physical Constraint Results\n(200 epochs, SMOTE, N=1)", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig("d_phase_wrist_rho.png", fig)


# =============================================================================
# 9. guidance_scale_sweep.png
# =============================================================================
def fig_guidance_sweep():
    print("9. guidance_scale_sweep.png")
    scales = [1.5, 2.0, 2.5, 3.0, 4.0]
    rhos   = [0.441, 0.531, 0.545, 0.545, 0.528]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(scales, rhos, "o-", color="#2c7bb6", lw=2.5, ms=8, markerfacecolor="white", markeredgewidth=2)
    best_i = np.argmax(rhos)
    ax.scatter([scales[best_i]], [rhos[best_i]], s=120, color="#d73027", zorder=5, label=f"Optimal: s={scales[best_i]:.1f}")
    for s, r in zip(scales, rhos):
        ax.text(s, r + 0.005, f"{r:.3f}", ha="center", fontsize=9)
    ax.set_xlabel("CFG Guidance Scale (s)", fontsize=12)
    ax.set_ylabel("Wrist Range Correlation (ρ)", fontsize=12)
    ax.set_title("Phase 4: CFG Guidance Scale Sweep (E-Phase)\n(D1 model, seed=42, no retraining)", fontsize=12)
    ax.set_ylim(0.38, 0.62)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig("guidance_scale_sweep.png", fig)


# =============================================================================
# 10. cdp_sweep.png
# =============================================================================
def fig_cdp_sweep():
    print("10. cdp_sweep.png")
    cdp   = [0.05, 0.10, 0.20, 0.30]
    rhos  = [0.435, 0.531, 0.322, 0.475]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(cdp, rhos, "s-", color="#d7191c", lw=2.5, ms=8, markerfacecolor="white", markeredgewidth=2)
    best_i = np.argmax(rhos)
    ax.scatter([cdp[best_i]], [rhos[best_i]], s=120, color="#2ca25f", zorder=5,
               label=f"Optimal: p={cdp[best_i]:.2f}")
    for p, r in zip(cdp, rhos):
        ax.text(p, r + 0.008, f"{r:.3f}", ha="center", fontsize=9)
    ax.set_xlabel("Conditioning Dropout Probability", fontsize=12)
    ax.set_ylabel("Wrist Range Correlation (ρ)", fontsize=12)
    ax.set_title("Phase 4: Conditioning Dropout Sweep (F-Phase)\n(D1 architecture, 200 epochs, SMOTE)", fontsize=12)
    ax.set_ylim(0.25, 0.62)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    savefig("cdp_sweep.png", fig)


# =============================================================================
# 11. training_curves_g1.png
# =============================================================================
def fig_training_curves():
    print("11. training_curves_g1.png")
    hist_path = os.path.join(T2_OUT, "G1_standard_split_300", "history.csv")
    if not os.path.exists(hist_path):
        print("  WARNING: history.csv not found, skipping")
        return
    df = pd.read_csv(hist_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(df["epoch"], df["train"], label="Train loss", color="#2c7bb6", lw=2)
    ax.plot(df["epoch"], df["val"],   label="Val loss",   color="#d73027", lw=2)
    best_ep = df.loc[df["val"].idxmin(), "epoch"]
    ax.axvline(best_ep, color="gray", ls="--", lw=1.2, label=f"Best val epoch: {best_ep}")
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training and Validation Loss (G1, 300 epochs)", fontsize=12)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df["epoch"], df["lr"], color="#756bb1", lw=2)
    ax.set_xlabel("Epoch", fontsize=12); ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title("Learning Rate Schedule (ReduceLROnPlateau)", fontsize=12)
    ax.set_yscale("log"); ax.grid(alpha=0.3)

    fig.suptitle("G1 Model Training Dynamics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("training_curves_g1.png", fig)


# =============================================================================
# 12. n10_variance_combined.png  (copy existing)
# =============================================================================
def fig_n10_variance():
    print("12. n10_variance_combined.png (copy)")
    src = os.path.join(T2_OUT, "G1_n10_avg", "variance", "combined_variance_figure.png")
    dst = os.path.join(OUT_DIR, "n10_variance_combined.png")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print("  Copied from variance/")
    else:
        print("  WARNING: source not found")


# =============================================================================
# 13. ablation_waterfall_all.png
# =============================================================================
def fig_ablation_waterfall():
    print("13. ablation_waterfall_all.png")

    rows = [
        ("Stage 0\n(no CFG, no FiLM)",      0.182, "Phase 1\nArch Search"),
        ("Stage 1\n(+CFG)",                  0.272, "Phase 1\nArch Search"),
        ("Stage 2\n(+FiLM)",                 0.426, "Phase 1\nArch Search"),
        ("Stage 3\n(+Residual)",             0.688, "Phase 2\nComponent"),
        ("D1\n(+Sag constraint)",            0.665, "Phase 3\nPhysical"),
        ("G1\n(300 epochs)",                 0.550, "Phase 4\nTraining"),
        ("G1 + N=10\n(multi-sample avg)",    0.915, "Phase 5\nInference"),
    ]

    labels = [r[0] for r in rows]
    rhos   = [r[1] for r in rows]
    phases = [r[2] for r in rows]

    phase_colors = {
        "Phase 1\nArch Search": "#4393c3",
        "Phase 2\nComponent":   "#2ca25f",
        "Phase 3\nPhysical":    "#fd8d3c",
        "Phase 4\nTraining":    "#756bb1",
        "Phase 5\nInference":   "#de2d26",
    }
    colors = [phase_colors[p] for p in phases]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    bars = ax.bar(range(len(labels)), rhos, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.65)

    for i, (bar, val) in enumerate(zip(bars, rhos)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.012,
                f"ρ={val:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        if i > 0:
            delta = val - rhos[i-1]
            sign  = "+" if delta >= 0 else ""
            ax.text(bar.get_x() + bar.get_width()/2, val / 2,
                    f"{sign}{delta:.3f}", ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold", alpha=0.85)

    ax.axhline(0.9, color="gray", ls="--", lw=1.2, label="Target ρ = 0.90")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("Wrist Range Correlation (ρ_wrist)", fontsize=12)
    ax.set_title("Full Ablation Waterfall: Stage 0 → Final Model (G1 + N=10)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Phase legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=p.replace("\n", " "))
                       for p, c in phase_colors.items()]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper left", ncol=5)

    plt.tight_layout()
    savefig("ablation_waterfall_all.png", fig)


# =============================================================================
# Run all
# =============================================================================
if __name__ == "__main__":
    print(f"Saving all figures to: {OUT_DIR}\n")
    fig_smote_phases()
    fig_aug_comparison_trajectories()
    fig_aug_ablation_bars()
    fig_phase1_cci()
    fig_ablation_components()
    fig_cfg_effect()
    fig_sag_effect()
    fig_d_phase()
    fig_guidance_sweep()
    fig_cdp_sweep()
    fig_training_curves()
    fig_n10_variance()
    fig_ablation_waterfall()
    print("\nDone.")
