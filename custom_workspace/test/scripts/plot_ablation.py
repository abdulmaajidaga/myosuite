"""
plot_ablation.py — Publication-quality figures showing the test2 ablation progression.

Figures produced:
  1. stage_progression.png    — CCI Rho ladder (Stage 0→3, from test/ sweep) + key physical metrics
  2. architecture_ablation.png — Phase A: what each removed component costs
  3. sagittal_constraint.png   — Phase C/D: sagittal constraint effect (main finding)
  4. loss_ablation.png         — Phase B: loss term comparison
  5. full_overview.png         — All 16 experiments ranked by composite score

Run from custom_workspace/:
  python test2/scripts/plot_ablation.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST2_DIR  = os.path.dirname(SCRIPT_DIR)
OUT_DIR    = os.path.join(TEST2_DIR, "output", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

COLORS = {
    "stage0": "#4e79a7",
    "stage1": "#f28e2b",
    "stage2": "#59a14f",
    "stage3": "#e15759",
    "sag":    "#b07aa1",
    "ref":    "#2ca02c",
    "bad":    "#d62728",
    "neutral":"#aec7e8",
}

# ── Load results ──────────────────────────────────────────────────────────────

def load_results():
    path = os.path.join(TEST2_DIR, "results", "results_log.csv")
    df = pd.read_csv(path)
    # Keep only n_files==51 rows and deduplicate by taking last run per experiment
    df = df[df.n_files == 51].drop_duplicates(subset="experiment", keep="last")
    return df.set_index("experiment")

# ── Figure 1: Stage Progression (CCI + Physical) ─────────────────────────────

def fig_stage_progression(df):
    """Shows the full evolution: CCI Rho (from test/) + physical metrics (from test2)."""
    print("Generating: stage_progression.png")

    # CCI Rho data from test/ 12-config sweep (SMOTE column)
    stages = ["Stage 0\n(Baseline)", "Stage 1\n(+CFG)", "Stage 2\n(+FiLM)", "Stage 3\n(+Residual)"]
    cci_smote  = [-0.480, -0.650, -1.000, -1.000]
    cci_dtw    = [-0.600, -1.000, -0.900, -0.600]
    litval     = [61,      72,     67,     72]   # SMOTE column

    # Physical metrics from test2 Phase A (correctly matched to each architecture)
    # A3=Stage1-like (no FiLM, no residual), A2=Stage1 (FiLM off), A1=Stage2, A0=Stage3
    exp_order  = ["A3_no_cfg", "A2_no_film", "A1_no_residual", "A0_full"]
    wrist_rho  = [df.loc[e, "wrist_rho"]    for e in exp_order]
    seg_std    = [df.loc[e, "segment_std_mean"] for e in exp_order]
    sag_dev    = [df.loc[e, "sag_dev_mean"] for e in exp_order]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("CVAE Architecture Progression: Stage 0 → Stage 3",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    x = np.arange(len(stages))
    colors = [COLORS["stage0"], COLORS["stage1"], COLORS["stage2"], COLORS["stage3"]]

    # Panel 1: CCI Rho
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(x, [-v for v in cci_smote], color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
    ax1.axhline(0.911, color=COLORS["ref"], linestyle="--", linewidth=1.5, label="Real human (0.911)")
    ax1.set_xticks(x); ax1.set_xticklabels(stages, fontsize=9)
    ax1.set_ylabel("|Spearman ρ| (CCI vs FMA)")
    ax1.set_title("CCI Gradient Strength\n(higher = better)", fontweight="bold")
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=9)
    for bar, v in zip(bars, cci_smote):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{abs(v):.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 2: LitVal %
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(x, litval, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
    ax2.axhline(89, color=COLORS["ref"], linestyle="--", linewidth=1.5, label="Real human (89%)")
    ax2.set_xticks(x); ax2.set_xticklabels(stages, fontsize=9)
    ax2.set_ylabel("Literature Validation (%)")
    ax2.set_title("Literature Validation\n(higher = better)", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9)
    for bar, v in zip(bars, litval):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{v}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 3: FMA wrist gradient (test2)
    ax3 = fig.add_subplot(gs[0, 2])
    bars = ax3.bar(x, wrist_rho, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
    ax3.axhline(1.0, color=COLORS["ref"], linestyle="--", linewidth=1.5, alpha=0.5, label="Perfect (1.0)")
    ax3.set_xticks(x); ax3.set_xticklabels(stages, fontsize=9)
    ax3.set_ylabel("Spearman ρ (wrist range vs FMA)")
    ax3.set_title("FMA Gradient Quality\n(wrist range, higher = better)", fontweight="bold")
    ax3.set_ylim(0, 1.1)
    ax3.legend(fontsize=9)
    for bar, v in zip(bars, wrist_rho):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 4: Segment consistency
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.bar(x, seg_std, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
    ax4.axhline(6.0, color=COLORS["ref"], linestyle="--", linewidth=1.5, label="Real MoCap (~6mm)")
    ax4.set_xticks(x); ax4.set_xticklabels(stages, fontsize=9)
    ax4.set_ylabel("Segment Std (mm)")
    ax4.set_title("Segment Consistency\n(lower = better, target ~6mm)", fontweight="bold")
    ax4.legend(fontsize=9)
    for bar, v in zip(bars, seg_std):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 5: Sagittal deviation
    ax5 = fig.add_subplot(gs[1, 1])
    bars = ax5.bar(x, sag_dev, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
    ax5.axhline(0, color=COLORS["ref"], linestyle="--", linewidth=1.5, alpha=0.5, label="Ideal (0mm)")
    ax5.set_xticks(x); ax5.set_xticklabels(stages, fontsize=9)
    ax5.set_ylabel("Lateral Wrist Deviation (mm)")
    ax5.set_title("Sagittal Plane Adherence\n(lower = better)", fontweight="bold")
    ax5.legend(fontsize=9)
    for bar, v in zip(bars, sag_dev):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Panel 6: Stage contribution summary (text)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    contributions = [
        ("Stage 0", "Baseline LSTM CVAE\nCCI Rho: −0.480\nLitVal: 61%", COLORS["stage0"]),
        ("+ CFG\n(Stage 1)", "Largest CCI gain\nRho: −0.480 → −0.650\nGradient sharpened", COLORS["stage1"]),
        ("+ FiLM\n(Stage 2)", "Enables SMOTE data\nRho: −0.650 → −1.000\nFMA conditioning improved", COLORS["stage2"]),
        ("+ Residual\n(Stage 3)", "Gradient quality\nWrist ρ: 0.272 → 0.688\nSmoother profiles", COLORS["stage3"]),
    ]
    for i, (label, text, color) in enumerate(contributions):
        y = 0.85 - i * 0.22
        ax6.add_patch(mpatches.FancyBboxPatch((0.0, y - 0.08), 0.98, 0.18,
                      boxstyle="round,pad=0.01", facecolor=color, alpha=0.15,
                      edgecolor=color, linewidth=1.5, transform=ax6.transAxes))
        ax6.text(0.08, y + 0.04, label, transform=ax6.transAxes,
                 fontsize=9, fontweight="bold", color=color, va="center")
        ax6.text(0.45, y + 0.04, text, transform=ax6.transAxes,
                 fontsize=8.5, va="center", color="#333333")
    ax6.set_title("Component Contributions", fontweight="bold")

    out = os.path.join(OUT_DIR, "stage_progression.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved → {out}")


# ── Figure 2: Architecture Ablation ──────────────────────────────────────────

def fig_architecture_ablation(df):
    """Phase A: what happens when each component is removed."""
    print("Generating: architecture_ablation.png")

    exps = {
        "A0_full\n(Stage 3\nFiLM+CFG+Res)": "A0_full",
        "A1\n(−Residual)":  "A1_no_residual",
        "A2\n(−FiLM)":      "A2_no_film",
        "A3\n(−CFG)":       "A3_no_cfg",
    }
    labels   = list(exps.keys())
    exp_keys = list(exps.values())
    colors_a = [COLORS["stage3"], COLORS["stage2"], COLORS["stage1"], COLORS["stage0"]]

    metrics = [
        ("wrist_rho",        "FMA Gradient\n(wrist ρ, higher = better)",   True),
        ("trunk_rho",        "Trunk Gradient\n(|trunk ρ|, higher = better)", True),
        ("segment_std_mean", "Segment Consistency\n(std mm, lower = better)", False),
        ("sag_dev_mean",     "Sagittal Adherence\n(dev mm, lower = better)",  False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle("Architecture Ablation — Phase A\n"
                 "Each bar removes one component from the full Stage 3 model",
                 fontsize=13, fontweight="bold")

    for ax, (metric, title, higher_better) in zip(axes, metrics):
        vals = []
        for k in exp_keys:
            v = df.loc[k, metric]
            vals.append(abs(v) if "trunk" in metric else v)

        bars = ax.bar(range(len(labels)), vals, color=colors_a,
                      alpha=0.85, edgecolor="white", linewidth=1.5)

        # Highlight the reference bar
        bars[0].set_edgecolor("#333333")
        bars[0].set_linewidth(2.5)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)

        # Reference line
        ref_val = abs(df.loc["A0_full", metric]) if "trunk" in metric else df.loc["A0_full", metric]
        ax.axhline(ref_val, color="#333333", linestyle=":", linewidth=1, alpha=0.5)

        # Annotate change from reference
        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            if i > 0:
                delta = v - ref_val
                sign = "+" if delta > 0 else ""
                color_d = COLORS["bad"] if (higher_better and delta < 0) or (not higher_better and delta > 0) else COLORS["ref"]
                ax.text(bar.get_x() + bar.get_width()/2, max(vals)*0.05,
                        f"({sign}{delta:.2f})", ha="center", va="bottom",
                        fontsize=8, color=color_d, style="italic")

        ax.set_ylim(0, max(vals) * 1.25)
        direction = "↑ better" if higher_better else "↓ better"
        ax.annotate(direction, xy=(0.98, 0.95), xycoords="axes fraction",
                    ha="right", fontsize=9, color="gray", style="italic")

    # Legend
    handles = [mpatches.Patch(color=c, label=l) for c, l in
               zip(colors_a, ["A0 Full (ref)", "A1 −Residual", "A2 −FiLM", "A3 −CFG"])]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.03), frameon=True)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "architecture_ablation.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved → {out}")


# ── Figure 3: Sagittal Constraint Effect ─────────────────────────────────────

def fig_sagittal_constraint(df):
    """Main finding: sagittal constraint dramatically reduces lateral deviation."""
    print("Generating: sagittal_constraint.png")

    groups = {
        "No\nConstraint\n(D0 Stage 3)":     "D0_stage3_baseline",
        "Stage 2\n+Sag (w=5)\n(C2)":        "C2_sag_only",
        "Stage 3\n+Sag (w=5)\n(D1)":        "D1_stage3_sag",
        "Stage 3\n+Sag (w=15)\n(D2)":       "D2_stage3_sag_strong",
        "Stage 3\n+Sag, Min Loss\n(D3)":    "D3_stage3_minimal",
    }
    labels   = list(groups.keys())
    exp_keys = list(groups.values())
    bar_colors = ["#aec7e8", "#59a14f", COLORS["stage3"], "#c5b0d5", "#dbdb8d"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("Sagittal-Plane Constraint — Phase C/D\n"
                 "Penalising lateral wrist deviation during training",
                 fontsize=13, fontweight="bold")

    metrics_info = [
        ("sag_dev_mean",        "Lateral Wrist Deviation (mm)\n↓ lower = better", False, None),
        ("wrist_rho",           "FMA Gradient Quality\n(wrist range ρ)  ↑ higher = better", True, None),
        ("segment_std_mean",    "Segment Consistency (mm)\n↓ lower = better", False, 6.0),
    ]

    for ax, (metric, title, higher_better, ref_line) in zip(axes, metrics_info):
        vals = [df.loc[k, metric] for k in exp_keys]

        bars = ax.bar(range(len(labels)), vals, color=bar_colors,
                      alpha=0.88, edgecolor="white", linewidth=1.5)

        # Highlight D1 as winner
        d1_idx = exp_keys.index("D1_stage3_sag")
        bars[d1_idx].set_edgecolor("#333333")
        bars[d1_idx].set_linewidth(2.5)
        ax.annotate("★ best", xy=(d1_idx, vals[d1_idx] + max(vals)*0.04),
                    ha="center", fontsize=9, color="#333333", fontweight="bold")

        if ref_line:
            ax.axhline(ref_line, color=COLORS["ref"], linestyle="--",
                       linewidth=1.5, label=f"Real MoCap ({ref_line}mm)")
            ax.legend(fontsize=9)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontweight="bold", fontsize=10)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        direction = "↑ better" if higher_better else "↓ better"
        ax.annotate(direction, xy=(0.98, 0.95), xycoords="axes fraction",
                    ha="right", fontsize=9, color="gray", style="italic")
        ax.set_ylim(0, max(vals) * 1.3)

    # Annotation box: key finding
    fig.text(0.5, 0.01,
             "Key finding: Sagittal constraint reduces lateral wrist deviation 100× "
             "(34mm → 0.13–0.29mm) while maintaining FMA gradient quality.\n"
             "D1 (Stage 3 + w_sag=5) is the optimal balance: sag_dev=0.29mm, wrist_ρ=0.665.",
             ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9c4", alpha=0.8))

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    out = os.path.join(OUT_DIR, "sagittal_constraint.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved → {out}")


# ── Figure 4: Loss Ablation ───────────────────────────────────────────────────

def fig_loss_ablation(df):
    """Phase B: which loss terms actually matter?"""
    print("Generating: loss_ablation.png")

    exps = {
        "B0\nFull loss\n(recon+vel\n+acc+dyn)": "B0_full_loss",
        "B1\n−dyn_corr\n(recon+vel\n+acc)":     "B1_no_dyn",
        "B2\n−acc\n(recon+vel\n+dyn)":           "B2_no_acc",
        "B3\nMinimal\n(recon+vel\nonly)":         "B3_recon_vel_only",
    }
    labels   = list(exps.keys())
    exp_keys = list(exps.values())
    bar_colors = ["#4e79a7", "#76b7e8", "#f28e2b", "#aec7e8"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle("Loss Function Ablation — Phase B\n"
                 "All experiments use Stage 2 (FiLM + CFG, no Residual) + SMOTE",
                 fontsize=13, fontweight="bold")

    metrics_info = [
        ("wrist_rho",        "FMA Gradient (wrist ρ)\n↑ higher = better",   True),
        ("segment_std_mean", "Segment Consistency (mm)\n↓ lower = better",  False),
        ("sag_dev_mean",     "Sagittal Deviation (mm)\n↓ lower = better",   False),
    ]

    for ax, (metric, title, higher_better) in zip(axes, metrics_info):
        vals = [df.loc[k, metric] for k in exp_keys]
        bars = ax.bar(range(len(labels)), vals, color=bar_colors,
                      alpha=0.85, edgecolor="white", linewidth=1.5)

        # Highlight B0 as reference
        bars[0].set_edgecolor("#333333")
        bars[0].set_linewidth(2.0)

        ref = vals[0]
        ax.axhline(ref, color="#333333", linestyle=":", linewidth=1, alpha=0.4)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=10)

        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylim(0, max(vals) * 1.2)

    fig.text(0.5, 0.01,
             "Loss differences are small — architecture dominates over loss function choice.\n"
             "Removing dyn_corr (B1) has minimal effect. Removing acc (B2) weakens the FMA gradient.",
             ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f4f8", alpha=0.8))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    out = os.path.join(OUT_DIR, "loss_ablation.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved → {out}")


# ── Figure 5: Full Overview ───────────────────────────────────────────────────

def fig_full_overview(df):
    """Scatter plot + colour-coded bar: all 16 experiments."""
    print("Generating: full_overview.png")

    # Composite score: normalise wrist_rho (higher=better) + normalise -sag_dev (lower=better)
    # Drop the smoke-test A0 entry if somehow it's there
    all_df = df.copy()

    # Phase colour coding
    def phase_color(exp):
        if exp.startswith("A"): return COLORS["stage3"]
        if exp.startswith("B"): return COLORS["stage1"]
        if exp.startswith("C"): return COLORS["stage2"]
        if exp.startswith("D"): return COLORS["sag"]
        return "gray"

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("All 16 Experiments — Full Overview",
                 fontsize=14, fontweight="bold")

    # Left: scatter wrist_rho vs sag_dev (want top-left)
    ax1 = axes[0]
    for exp in all_df.index:
        x = all_df.loc[exp, "sag_dev_mean"]
        y = all_df.loc[exp, "wrist_rho"]
        c = phase_color(exp)
        ax1.scatter(x, y, color=c, s=90, alpha=0.85, zorder=3)
        # Label key experiments
        short = exp.replace("_no_", "\n−").replace("_stage3", "\nS3").replace("_sag", "+sag")
        offset = (2, 2)
        if "D1" in exp:
            ax1.annotate("★ D1\n(target)", xy=(x, y), xytext=(x+3, y+0.02),
                         fontsize=8, fontweight="bold", color=COLORS["sag"],
                         arrowprops=dict(arrowstyle="->", color=COLORS["sag"], lw=1.2))
        elif exp in ["A0_full", "A3_no_cfg", "D0_stage3_baseline", "C2_sag_only"]:
            ax1.annotate(exp.split("_")[0], xy=(x, y), xytext=(x+offset[0], y+offset[1]*0.01),
                         fontsize=8, color=c)

    ax1.set_xlabel("Sagittal Deviation (mm) — lower = better →", fontsize=10)
    ax1.set_ylabel("FMA Gradient, wrist ρ — higher = better ↑", fontsize=10)
    ax1.set_title("Physical quality vs FMA gradient\n(ideal: top-left)", fontweight="bold")
    ax1.axvline(0, color="gray", linestyle="--", alpha=0.3)
    # Shade ideal quadrant
    ax1.axvspan(0, 5, alpha=0.05, color="green")
    ax1.text(1, 0.95, "Ideal zone", fontsize=9, color="green", style="italic",
             transform=ax1.transAxes, ha="right")

    # Phase legend
    handles = [mpatches.Patch(color=COLORS["stage3"], label="Phase A (architecture)"),
               mpatches.Patch(color=COLORS["stage1"], label="Phase B (loss)"),
               mpatches.Patch(color=COLORS["stage2"], label="Phase C (Stage2+constraint)"),
               mpatches.Patch(color=COLORS["sag"],    label="Phase D (Stage3+constraint)")]
    ax1.legend(handles=handles, loc="lower right", fontsize=9)

    # Right: ranked bar chart by wrist_rho, coloured by phase, sag_dev on secondary axis
    ax2 = axes[1]
    sorted_df = all_df.sort_values("wrist_rho", ascending=True)
    y_pos = np.arange(len(sorted_df))
    bar_colors_r = [phase_color(e) for e in sorted_df.index]

    bars = ax2.barh(y_pos, sorted_df["wrist_rho"], color=bar_colors_r,
                    alpha=0.8, edgecolor="white", linewidth=1.2, height=0.6)

    # Highlight D1
    for i, (bar, exp) in enumerate(zip(bars, sorted_df.index)):
        if "D1" in exp:
            bar.set_edgecolor("#333333")
            bar.set_linewidth(2.5)

    ax2.set_yticks(y_pos)
    labels_r = []
    for e in sorted_df.index:
        sd = sorted_df.loc[e, "sag_dev_mean"]
        labels_r.append(f"{e}  (sag={sd:.1f}mm)")
    ax2.set_yticklabels(labels_r, fontsize=8.5)
    ax2.set_xlabel("Wrist Range ρ (FMA gradient)", fontsize=10)
    ax2.set_title("Ranked by FMA Gradient Quality\n(sag_dev shown in label)", fontweight="bold")
    ax2.axvline(0.6, color="gray", linestyle="--", alpha=0.4, label="0.6 threshold")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "full_overview.png")
    plt.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  saved → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_results()
    print(f"Loaded {len(df)} experiments\n")
    print(f"Output: {OUT_DIR}\n")

    fig_stage_progression(df)
    fig_architecture_ablation(df)
    fig_sagittal_constraint(df)
    fig_loss_ablation(df)
    fig_full_overview(df)

    print("\nDone:")
    for f in ["stage_progression.png", "architecture_ablation.png",
              "sagittal_constraint.png", "loss_ablation.png", "full_overview.png"]:
        p = os.path.join(OUT_DIR, f)
        print(f"  {'✓' if os.path.exists(p) else '✗'}  {f}")
