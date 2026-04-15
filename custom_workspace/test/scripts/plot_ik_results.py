"""
plot_ik_results.py — Visualises the full test2 pipeline results including IK-confirmed ROM metrics.

Generates 4 figures:
  1. session_summary.png    — What happened this session: 4-panel overview
  2. d1_vs_baseline.png     — D1 vs Stage3_SMOTE head-to-head on every metric
  3. ik_rom_by_fma.png      — elbow & pro/sup ROM across FMA 16–66 for both models
  4. sagittal_finding.png   — What the sagittal constraint actually achieved (corrected)
"""

import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import stats

ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR   = os.path.join(TEST2_DIR, "output", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Colours ──────────────────────────────────────────────────────────────────
C_BASELINE = "#5B7FA6"   # blue — Stage3_SMOTE
C_D1       = "#E06B3F"   # orange — D1
C_REAL     = "#4CAF50"   # green — real human / target
C_FAIL     = "#C0392B"   # red
C_OK       = "#27AE60"   # green
C_NEUTRAL  = "#7F8C8D"   # grey

# ─── Load MOT data ─────────────────────────────────────────────────────────────
def load_mot_dir(mot_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(mot_dir, "FMA_*.mot"))):
        fma = int(os.path.basename(f).replace("FMA_","").replace(".mot",""))
        mot = pd.read_csv(f, sep="\t", skiprows=6)
        r = {"fma": fma}
        for col, key in [("elbow_flexion","elbow_rom"), ("pro_sup","pro_sup_rom")]:
            if col in mot.columns:
                v = mot[col].values * (180/np.pi)
                r[key] = float(v.max() - v.min())
            else:
                r[key] = np.nan
        rows.append(r)
    return pd.DataFrame(rows).sort_values("fma").reset_index(drop=True)


mot_d1       = load_mot_dir(os.path.join(TEST2_DIR, "output", "D1_stage3_sag", "mot"))
mot_baseline = load_mot_dir(os.path.join(ROOT_DIR, "test", "output", "final", "stage3_smote", "mot"))

# ─── Load ablation CSV ─────────────────────────────────────────────────────────
results_csv = os.path.join(TEST2_DIR, "results", "results_log.csv")
df_all = pd.read_csv(results_csv)
# keep last run per experiment, only n_files==51
df_all = df_all[df_all["n_files"] == 51].groupby("experiment").last().reset_index()

def get_exp(name):
    row = df_all[df_all["experiment"] == name]
    return row.iloc[0] if len(row) else None

A0  = get_exp("A0_full")        # Stage 3 reference (no sag)
D0  = get_exp("D0_stage3_baseline")
D1  = get_exp("D1_stage3_sag")
D2  = get_exp("D2_stage3_sag_strong")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Session summary: what happened this session
# ══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
fig1.suptitle("Session Summary — What the IK Confirmed", fontsize=15, fontweight="bold", y=0.98)

# Panel A: Key marker metrics — D0 vs D1 vs A0 (ablation context)
ax = axes[0, 0]
experiments  = ["A0_full\n(Stage 3,\nno sag)", "D0_stage3\nbaseline", "D1_stage3\n+sag(w=5)"]
sag_devs     = [A0["sag_dev_mean"], D0["sag_dev_mean"], D1["sag_dev_mean"]]
colors_sag   = [C_NEUTRAL, C_FAIL, C_OK]
bars = ax.bar(experiments, sag_devs, color=colors_sag, edgecolor="white", linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Lateral Wrist Deviation (sag_dev)", fontweight="bold")
ax.set_ylabel("mm")
for b, v in zip(bars, sag_devs):
    ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}mm", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 42)
ax.annotate("Sagittal\nconstraint\nworks ✓", xy=(2, D1["sag_dev_mean"]), xytext=(1.5, 20),
            arrowprops=dict(arrowstyle="->", color=C_OK), color=C_OK, fontsize=9, fontweight="bold")

# Panel B: FMA gradient (wrist_rho) — D0 vs D1 vs A0
ax = axes[0, 1]
wrist_rhos = [A0["wrist_rho"], D0["wrist_rho"], D1["wrist_rho"]]
colors_rho = [C_NEUTRAL, C_FAIL, C_OK]
bars = ax.bar(experiments, wrist_rhos, color=colors_rho, edgecolor="white", linewidth=1.5)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("FMA Gradient (Wrist Spearman ρ)", fontweight="bold")
ax.set_ylabel("Spearman ρ")
for b, v in zip(bars, wrist_rhos):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 0.85)
ax.annotate("Improved\n+50% ✓", xy=(2, D1["wrist_rho"]), xytext=(1.4, 0.75),
            arrowprops=dict(arrowstyle="->", color=C_OK), color=C_OK, fontsize=9, fontweight="bold")

# Panel C: Pro/sup ROM — CORRECTED FINDING
ax = axes[1, 0]
categories  = ["Stage3_SMOTE\n(test/ baseline)", "D1_stage3_sag\n(new model)", "Real Human\n(target)"]
prosup_vals = [mot_baseline["pro_sup_rom"].mean(), mot_d1["pro_sup_rom"].mean(), 50.0]
colors_ps   = [C_NEUTRAL, C_FAIL, C_REAL]
bars = ax.bar(categories, prosup_vals, color=colors_ps, edgecolor="white", linewidth=1.5)
ax.axhline(50, color=C_REAL, linewidth=2, linestyle="--", label="Target ~50°")
ax.set_title("Pro/Sup ROM — IK Confirmed (HYPOTHESIS WRONG)", fontweight="bold", color=C_FAIL)
ax.set_ylabel("Degrees")
for b, v in zip(bars, prosup_vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.5, f"{v:.1f}°", ha="center", fontsize=10, fontweight="bold")
ax.set_ylim(0, 110)
ax.text(0.5, 95, "Sagittal constraint did NOT reduce\npro/sup — IK solver artifact", ha="center",
        fontsize=9, style="italic", color=C_FAIL,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEA", edgecolor=C_FAIL))
ax.legend(fontsize=9)

# Panel D: Summary table
ax = axes[1, 1]
ax.axis("off")
table_data = [
    ["Metric", "Stage3_SMOTE\n(baseline)", "D1\n(new)", "Result"],
    ["sag_dev (mm)", "~30mm", "0.29mm", "✓ Fixed"],
    ["wrist_rho", "0.688", "0.665", "✓ OK"],
    ["elbow ROM", "57.4°", "55.9°", "→ Same"],
    ["pro_sup ROM", "79.3°", "86.2°", "✗ WORSE"],
    ["segment_std", "11.9mm", "14.0mm", "→ Same"],
]
colors_t = [
    ["#E8E8E8"] * 4,
    ["white", "white", "white", "#D5F5E3"],
    ["white", "white", "white", "#D5F5E3"],
    ["white", "white", "white", "#FEF9E7"],
    ["white", "white", "white", "#FDECEA"],
    ["white", "white", "white", "#FEF9E7"],
]
table = ax.table(cellText=table_data, cellLoc="center", loc="center",
                  cellColours=colors_t)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)
ax.set_title("D1 vs Stage3_SMOTE — Full Comparison", fontweight="bold", pad=10)

plt.tight_layout()
out = os.path.join(FIG_DIR, "session_summary.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ session_summary.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — ROM across FMA scores (D1 vs Stage3_SMOTE)
# ══════════════════════════════════════════════════════════════════════════════
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Joint ROM Across FMA Scores: D1 vs Stage3_SMOTE Baseline", fontsize=13, fontweight="bold")

# Elbow flexion
ax1.scatter(mot_baseline["fma"], mot_baseline["elbow_rom"], color=C_BASELINE, alpha=0.6,
            s=40, label=f"Stage3_SMOTE (mean={mot_baseline['elbow_rom'].mean():.1f}°)")
ax1.scatter(mot_d1["fma"], mot_d1["elbow_rom"], color=C_D1, alpha=0.8,
            s=40, label=f"D1 (mean={mot_d1['elbow_rom'].mean():.1f}°)")
# trend lines
for data, color in [(mot_baseline, C_BASELINE), (mot_d1, C_D1)]:
    z = np.polyfit(data["fma"], data["elbow_rom"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(16, 66, 50)
    ax1.plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.8)
ax1.axhline(78, color=C_REAL, linewidth=2, linestyle="--", label="Real human mean ~78°")
ax1.fill_between([16, 66], 44, 115, alpha=0.1, color=C_REAL, label="Real human range [44–115°]")
ax1.set_xlabel("FMA Score"); ax1.set_ylabel("Elbow Flexion ROM (°)")
ax1.set_title("Elbow Flexion ROM vs FMA", fontweight="bold")
ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3); ax1.set_xlim(14, 68)

# Pro/sup
ax2.scatter(mot_baseline["fma"], mot_baseline["pro_sup_rom"], color=C_BASELINE, alpha=0.6,
            s=40, label=f"Stage3_SMOTE (mean={mot_baseline['pro_sup_rom'].mean():.1f}°)")
ax2.scatter(mot_d1["fma"], mot_d1["pro_sup_rom"], color=C_D1, alpha=0.8,
            s=40, label=f"D1 (mean={mot_d1['pro_sup_rom'].mean():.1f}°)")
for data, color in [(mot_baseline, C_BASELINE), (mot_d1, C_D1)]:
    z = np.polyfit(data["fma"], data["pro_sup_rom"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(16, 66, 50)
    ax2.plot(x_line, p(x_line), color=color, linewidth=2, alpha=0.8)
ax2.axhline(50, color=C_REAL, linewidth=2, linestyle="--", label="Real human mean ~50°")
ax2.fill_between([16, 66], 31, 96, alpha=0.1, color=C_REAL, label="Real human range [31–96°]")
ax2.set_xlabel("FMA Score"); ax2.set_ylabel("Pro/Sup ROM (°)")
ax2.set_title("Pro/Supination ROM vs FMA — Both Over-Estimate", fontweight="bold", color=C_FAIL)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_xlim(14, 68)
ax2.text(40, 102, "IK solver artifact\nnot fixable via CVAE", ha="center",
         fontsize=9, color=C_FAIL, style="italic",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDECEA", edgecolor=C_FAIL, alpha=0.8))

plt.tight_layout()
out = os.path.join(FIG_DIR, "ik_rom_by_fma.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ ik_rom_by_fma.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — What the sagittal constraint actually achieved (corrected)
# ══════════════════════════════════════════════════════════════════════════════
fig3 = plt.figure(figsize=(16, 8))
fig3.suptitle("Sagittal Constraint — What It Actually Achieves\n(D Phase: Stage 3 + sagittal constraint experiments)",
              fontsize=13, fontweight="bold")
gs = GridSpec(2, 3, figure=fig3, hspace=0.45, wspace=0.35)

D_exps = [
    ("D0\nno sag", get_exp("D0_stage3_baseline")),
    ("D1\nw_sag=5", get_exp("D1_stage3_sag")),
    ("D2\nw_sag=15", get_exp("D2_stage3_sag_strong")),
    ("D3\nmin loss", get_exp("D3_stage3_minimal")),
]
D_labels = [e[0] for e in D_exps]
D_data   = [e[1] for e in D_exps]

# 1. sag_dev
ax = fig3.add_subplot(gs[0, 0])
vals = [d["sag_dev_mean"] for d in D_data]
colors = [C_FAIL, C_OK, C_OK, C_OK]
bars = ax.bar(D_labels, vals, color=colors, edgecolor="white")
ax.set_title("Lateral Deviation ✓\n(Lower = better)", fontweight="bold", color=C_OK)
ax.set_ylabel("sag_dev (mm)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0, 40)

# 2. wrist_rho
ax = fig3.add_subplot(gs[0, 1])
vals = [d["wrist_rho"] for d in D_data]
colors = [C_FAIL, C_OK, C_NEUTRAL, C_FAIL]
bars = ax.bar(D_labels, vals, color=colors, edgecolor="white")
ax.set_title("FMA Gradient ✓\n(Higher = better)", fontweight="bold", color=C_OK)
ax.set_ylabel("wrist_rho")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0, 0.85)
ax.annotate("Sweet spot:\nD1 is best", xy=(1, D1["wrist_rho"]), xytext=(2.2, 0.72),
            arrowprops=dict(arrowstyle="->", color=C_OK), color=C_OK, fontsize=8)

# 3. segment_std
ax = fig3.add_subplot(gs[0, 2])
vals = [d["segment_std_mean"] for d in D_data]
colors = [C_NEUTRAL] * 4
bars = ax.bar(D_labels, vals, color=colors, edgecolor="white")
ax.axhline(6, color=C_REAL, linewidth=2, linestyle="--", label="Real MoCap ~6mm")
ax.set_title("Segment Consistency\n(Lower = better — none achieve target)", fontweight="bold")
ax.set_ylabel("segment_std (mm)")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
ax.legend(fontsize=8)

# 4. pro/sup ROM — the corrected finding
ax = fig3.add_subplot(gs[1, 0])
vals_ps = [mot_d1["pro_sup_rom"].mean()]  # only D1 has IK run
vals_baseline = [mot_baseline["pro_sup_rom"].mean()]
ax.bar(["Stage3_SMOTE\nbaseline", "D1\n+sag(w=5)", "Real human\ntarget"],
       [vals_baseline[0], vals_ps[0], 50.0],
       color=[C_BASELINE, C_FAIL, C_REAL], edgecolor="white")
ax.axhline(50, color=C_REAL, linestyle="--", linewidth=1.5)
ax.set_title("Pro/Sup ROM ✗\n(Hypothesis WRONG — got worse!)", fontweight="bold", color=C_FAIL)
ax.set_ylabel("Degrees")
for x, v in zip([0, 1, 2], [vals_baseline[0], vals_ps[0], 50.0]):
    ax.text(x, v + 0.5, f"{v:.1f}°", ha="center", fontsize=9, fontweight="bold")
ax.set_ylim(0, 110)

# 5. Corrected narrative box
ax = fig3.add_subplot(gs[1, 1:])
ax.axis("off")
text = (
    "WHAT THE SAGITTAL CONSTRAINT DOES AND DOES NOT DO\n\n"
    "✓  Eliminates lateral (X-axis) wrist drift:  34mm → 0.29mm  (100× reduction)\n"
    "✓  Improves FMA gradient:  wrist_rho 0.443 → 0.665  (+50%)\n"
    "     Why: constraining X makes Y-axis the dominant motion direction,\n"
    "     so the wrist-Y range metric more accurately captures reach amplitude.\n\n"
    "✗  Does NOT fix pro/supination ROM overestimation\n"
    "     D1 pro/sup = 86.2°  (baseline 79.3°, target ~50°)  — actually worse!\n"
    "     Root cause: D1 generates wrist trajectories with large Z-axis excursions\n"
    "     from the reference pose.  The IK solver independently routes Z-axis motion\n"
    "     into pro/sup regardless of lateral constraint.  This is an IK solver\n"
    "     distributional bias — not fixable via CVAE training."
)
ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=9.5,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#F4F6F7", edgecolor="#7F8C8D"))

out = os.path.join(FIG_DIR, "sagittal_finding_corrected.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ sagittal_finding_corrected.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Complete ablation overview (all 16 experiments, now with context)
# ══════════════════════════════════════════════════════════════════════════════
fig4, axes = plt.subplots(2, 2, figsize=(16, 10))
fig4.suptitle("All 16 Experiments — Full Ablation Overview\n(test2/ systematic study)",
              fontsize=14, fontweight="bold")

# colour by phase
phase_colors = {
    "A": "#3498DB", "B": "#9B59B6", "C": "#E67E22", "D": "#E74C3C"
}
phase_labels = {"A": "Phase A (Architecture)", "B": "Phase B (Loss)",
                "C": "Phase C (Constraints)", "D": "Phase D (Stage3+Sag)"}

def phase_color(exp_name):
    letter = exp_name[0]
    return phase_colors.get(letter, C_NEUTRAL)

df_plot = df_all[df_all["experiment"].str.match(r"[ABCD]\d")].copy()
df_plot = df_plot.sort_values("wrist_rho", ascending=True)

exp_names = df_plot["experiment"].values
colors_exp = [phase_color(e) for e in exp_names]
y_pos = np.arange(len(exp_names))

# Panel: wrist_rho ranked
ax = axes[0, 0]
bars = ax.barh(y_pos, df_plot["wrist_rho"].values, color=colors_exp, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(exp_names, fontsize=8)
ax.set_title("FMA Gradient (wrist_rho) — ranked", fontweight="bold")
ax.set_xlabel("Spearman ρ")
# Highlight D1
idx_d1 = list(exp_names).index("D1_stage3_sag") if "D1_stage3_sag" in exp_names else None
if idx_d1 is not None:
    ax.axhline(idx_d1, color=C_D1, linewidth=2, alpha=0.4)
    ax.text(df_plot["wrist_rho"].values[idx_d1] + 0.005, idx_d1, " D1 (best w/ sag)", va="center", fontsize=7.5, color=C_D1, fontweight="bold")
legend_patches = [mpatches.Patch(color=v, label=phase_labels[k]) for k, v in phase_colors.items()]
ax.legend(handles=legend_patches, fontsize=7, loc="lower right")

# Panel: sag_dev ranked
ax = axes[0, 1]
df_sag = df_plot.sort_values("sag_dev_mean", ascending=False)
exp_names_sag = df_sag["experiment"].values
colors_sag_all = [phase_color(e) for e in exp_names_sag]
bars = ax.barh(np.arange(len(exp_names_sag)), df_sag["sag_dev_mean"].values, color=colors_sag_all, edgecolor="white")
ax.set_yticks(np.arange(len(exp_names_sag)))
ax.set_yticklabels(exp_names_sag, fontsize=8)
ax.set_title("Lateral Deviation (sag_dev) — ranked", fontweight="bold")
ax.set_xlabel("mm (lower = more sagittal)")
ax.legend(handles=legend_patches, fontsize=7, loc="lower right")

# Panel: segment_std scatter
ax = axes[1, 0]
for i, row in df_plot.iterrows():
    ax.scatter(row["wrist_rho"], row["segment_std_mean"],
               color=phase_color(row["experiment"]), s=60, zorder=3, alpha=0.85)
    ax.annotate(row["experiment"].replace("_stage3","").replace("_no_","−"),
                (row["wrist_rho"], row["segment_std_mean"]),
                fontsize=6.5, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
ax.axhline(6, color=C_REAL, linewidth=1.5, linestyle="--", label="Real MoCap ~6mm")
ax.set_xlabel("FMA Gradient (wrist_rho)"); ax.set_ylabel("Segment std (mm)")
ax.set_title("FMA Gradient vs Segment Consistency", fontweight="bold")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Panel: sag_dev vs wrist_rho scatter — key trade-off
ax = axes[1, 1]
for i, row in df_plot.iterrows():
    ax.scatter(row["sag_dev_mean"], row["wrist_rho"],
               color=phase_color(row["experiment"]), s=80, zorder=3, alpha=0.85)
    ax.annotate(row["experiment"].replace("_stage3","").replace("_no_","−"),
                (row["sag_dev_mean"], row["wrist_rho"]),
                fontsize=6.5, ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")
# D1 callout
if D1 is not None:
    ax.scatter(D1["sag_dev_mean"], D1["wrist_rho"], color=C_D1, s=160, zorder=5, marker="*")
ax.set_xlabel("Lateral Deviation (sag_dev mm)"); ax.set_ylabel("FMA Gradient (wrist_rho)")
ax.set_title("Key Trade-off: Lateral Constraint vs FMA Gradient\n★ = D1 (best combined)", fontweight="bold")
ax.grid(True, alpha=0.3); ax.legend(handles=legend_patches, fontsize=7)
# Ideal region annotation
ax.annotate("Ideal region\n(low sag, high rho)", xy=(1, 0.62), xytext=(10, 0.35),
            arrowprops=dict(arrowstyle="->", color="grey"), color="grey", fontsize=8, style="italic")

plt.tight_layout()
out = os.path.join(FIG_DIR, "ablation_overview_full.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ ablation_overview_full.png")

print(f"\nAll figures saved to: {FIG_DIR}")
