"""
make_corrected_figures.py — Regenerate 4 figures with real D-phase experiment data.

Replaces the old hardcoded/missing-data versions:
  ablation_component_bars.png  — A0→A4 architecture ablation (SMOTE wrist_rho)
  aug_ablation_bars.png        — SMOTE vs DTW vs Linear at best arch (A2)
  phase1_cci_progression.png   — wrist_rho progression A0→A2 (arch stages)
  effect_size_heatmap.png      — ATI & CCI vs FMA scatter (replaces empty heatmap)

All metrics loaded live from test/output/ and output/generated/id/.
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(TEST_DIR)
OUT_DIR  = os.path.join(ROOT_DIR, "figures")
T2_OUT   = os.path.join(TEST_DIR, "output")

sys.path.insert(0, ROOT_DIR)
from src.utils.config import get_path

TARGET = 0.9  # clinical target rho

PALETTE = {
    "smote":  "#2ca25f",
    "dtw":    "#2c7bb6",
    "linear": "#fc8d59",
    "select": "#d73027",
}


def load_rho(exp_dir):
    """Return wrist_rho from eval_summary_n10.json, or None."""
    p = os.path.join(T2_OUT, exp_dir, "eval_summary_n10.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p)).get("wrist_rho")
    except Exception:
        return None


def savefig(name, fig):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {path}")
    plt.close(fig)


# =============================================================================
# 1. ablation_component_bars.png
#    Architecture ablation: A0 (baseline) → A1 → A2 → A3 → A4, SMOTE data
# =============================================================================
def fig_ablation_components():
    print("1. ablation_component_bars.png")

    exps = [
        ("A0_smote", "A0\nBaseline\n(no FiLM, no CFG)"),
        ("A1_smote", "A1\n+ CFG"),
        ("A2_smote", "A2\n+ FiLM\n[selected]"),
        ("A3_smote", "A3\n+ Residual\ndecoder"),
        ("A4_smote", "A4\n+ Temporal\nconv"),
    ]

    rhos   = [load_rho(e) for e, _ in exps]
    labels = [lbl for _, lbl in exps]
    colors = []
    for i, (e, _) in enumerate(exps):
        if "A2" in e:
            colors.append(PALETTE["select"])   # selected
        elif rhos[i] is not None and rhos[i] >= TARGET:
            colors.append(PALETTE["smote"])
        else:
            colors.append("#aec7e8")

    valid = [(l, r, c) for l, r, c in zip(labels, rhos, colors) if r is not None]
    labels, rhos, colors = zip(*valid)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, rhos, color=colors, edgecolor="white", linewidth=1.5, width=0.6)
    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f"ρ = {val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.axhline(TARGET, color="gray", ls="--", lw=1.2, label=f"Target ρ = {TARGET}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Wrist Range Correlation (ρ)", fontsize=12)
    ax.set_title("Architecture Ablation — Phase A (SMOTE data, N=10 averaged samples)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig("ablation_component_bars.png", fig)


# =============================================================================
# 2. aug_ablation_bars.png
#    Augmentation method comparison at best architecture (A2)
# =============================================================================
def fig_aug_ablation():
    print("2. aug_ablation_bars.png")

    exps = [
        ("A2_smote",  "SMOTE\n[selected]", PALETTE["smote"]),
        ("A2_dtw",    "DTW",               PALETTE["dtw"]),
        ("A2_linear", "Linear\nbaseline",  PALETTE["linear"]),
    ]

    rhos   = [load_rho(e) for e, _, _ in exps]
    labels = [lbl for _, lbl, _ in exps]
    colors = [c for _, _, c in exps]

    valid = [(l, r, c) for l, r, c in zip(labels, rhos, colors) if r is not None]
    labels, rhos, colors = zip(*valid)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, rhos, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    for bar, val in zip(bars, rhos):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012,
                f"ρ = {val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(TARGET, color="gray", ls="--", lw=1.2, label=f"Target ρ = {TARGET}")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Wrist Range Correlation (ρ)", fontsize=12)
    ax.set_title("Augmentation Method Comparison\n(A2 architecture, N=10 averaged samples)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig("aug_ablation_bars.png", fig)


# =============================================================================
# 3. phase1_cci_progression.png  (now shows wrist_rho arch progression)
#    A0 → A1 → A2 : adding CFG then FiLM, per dataset
# =============================================================================
def fig_arch_progression():
    print("3. phase1_cci_progression.png")

    stages = ["A0\n(baseline)", "A1\n(+CFG)", "A2\n(+FiLM)\n[D_base arch]"]
    exps_smote  = ["A0_smote",  "A1_smote",  "A2_smote"]
    exps_dtw    = ["A0_dtw",    "A1_dtw",    "A2_dtw"]
    exps_linear = ["A0_linear", "A1_linear", "A2_linear"]

    rho_s = [load_rho(e) for e in exps_smote]
    rho_d = [load_rho(e) for e in exps_dtw]
    rho_l = [load_rho(e) for e in exps_linear]

    x = np.arange(len(stages))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w, rho_s, w, label="SMOTE", color=PALETTE["smote"],  edgecolor="white")
    b2 = ax.bar(x,     rho_d, w, label="DTW",   color=PALETTE["dtw"],    edgecolor="white")
    b3 = ax.bar(x + w, rho_l, w, label="Linear",color=PALETTE["linear"], edgecolor="white")

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(TARGET, color="gray", ls="--", lw=1.2, label=f"Target ρ = {TARGET}")
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_ylabel("Wrist Range Correlation (ρ)", fontsize=12)
    ax.set_title("Architecture Progression: Baseline → CFG → FiLM\n(wrist_rho, N=10 averaged samples)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    savefig("phase1_cci_progression.png", fig)


# =============================================================================
# 4. effect_size_heatmap.png  (replaces empty Cohen's d heatmap)
#    ATI & CCI vs FMA score scatter — meaningful with n=1 per level
# =============================================================================
def fig_ati_cci_scatter():
    print("4. effect_size_heatmap.png")

    gen_id_dir = os.path.join(get_path("output_generated"), "id")

    fma_vals, ati_vals, cci_vals = [], [], []
    for name in sorted(os.listdir(gen_id_dir)):
        p = os.path.join(gen_id_dir, name, "effort_metrics.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        ati = m.get("ATI")
        cci = m.get("CCI")
        if ati is None or cci is None:
            continue
        import re
        match = re.match(r"FMA_(\d+)$", name)
        if not match:
            continue
        fma_vals.append(int(match.group(1)))
        ati_vals.append(float(ati))
        cci_vals.append(float(cci))

    if not fma_vals:
        print("  No ATI/CCI data found — skipping")
        return

    fma = np.array(fma_vals)
    ati = np.array(ati_vals)
    cci = np.array(cci_vals)

    rho_ati, p_ati = spearmanr(fma, ati)
    rho_cci, p_cci = spearmanr(fma, cci)

    # Fit trend lines
    z_ati = np.polyfit(fma, ati, 1)
    z_cci = np.polyfit(fma, cci, 1)
    x_line = np.linspace(fma.min(), fma.max(), 200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, vals, z, rho, p, label, color in [
        (axes[0], ati, z_ati, rho_ati, p_ati, "ATI (Aggregate Torque Index)", "#E94F37"),
        (axes[1], cci, z_cci, rho_cci, p_cci, "CCI (Co-Contraction Index)",   "#2E86AB"),
    ]:
        ax.scatter(fma, vals, color=color, alpha=0.75, s=40, zorder=3)
        ax.plot(x_line, np.polyval(z, x_line), color=color, lw=2, ls="--",
                label=f"ρ = {rho:+.3f}  (p={p:.4f})")
        ax.set_xlabel("FMA Score", fontsize=12)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("Biomechanical Effort Metrics vs FMA Score (D_base, N=51 levels)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig("effect_size_heatmap.png", fig)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Saving to: {OUT_DIR}\n")
    fig_ablation_components()
    fig_aug_ablation()
    fig_arch_progression()
    fig_ati_cci_scatter()
    print("\nDone.")
