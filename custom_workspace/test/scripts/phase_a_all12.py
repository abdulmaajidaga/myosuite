"""
phase_a_all12.py — All-12-configs versions of rom_trends, trunk_wrist, joint_rom.

Saves 3 figures alongside the existing ones in figures_phase_a/:
  rom_trends_all12.png
  trunk_wrist_ratio_all12.png
  joint_rom_vs_fma_all12.png

Color  = architectural stage (4 colours)
Linestyle = augmentation method (solid / dashed / dotted)

Run from custom_workspace/:
  python test/scripts/phase_a_all12.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
FINAL_DIR    = os.path.join(BASE_DIR, "output", "final")
OUT_DIR      = os.path.join(BASE_DIR, "output", "figures_phase_a")
os.makedirs(OUT_DIR, exist_ok=True)

STAGES  = [0, 1, 2, 3]
AUGS    = ["dtw", "smote", "linear"]
FMA_ALL = list(range(16, 67))

AUG_LABELS   = {"dtw": "DTW", "smote": "SMOTE", "linear": "Linear"}
STAGE_LABELS = {0: "Stage 0 (Baseline)", 1: "Stage 1 (+CFG)",
                2: "Stage 2 (+FiLM)",    3: "Stage 3 (+Residual)"}
REAL_COLOR   = "#2ca02c"

# 12 distinct colours — one per (stage, aug) combo
# Stage 0: blues, Stage 1: oranges, Stage 2: greens, Stage 3: reds
CONFIG_COLORS = {
    (0, "dtw"):    "#4e79a7",
    (0, "smote"):  "#76b7e8",
    (0, "linear"): "#1b3f6e",
    (1, "dtw"):    "#f28e2b",
    (1, "smote"):  "#ffc06e",
    (1, "linear"): "#a05a00",
    (2, "dtw"):    "#59a14f",
    (2, "smote"):  "#8fd180",
    (2, "linear"): "#2a6120",
    (3, "dtw"):    "#e15759",
    (3, "smote"):  "#ff9ea0",
    (3, "linear"): "#8b1a1c",
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "figure.dpi": 150,
})

# ── Shared legend handles ─────────────────────────────────────────────────────

def make_legend_handles():
    handles = []
    for s in STAGES:
        for a in AUGS:
            handles.append(mlines.Line2D([], [], color=CONFIG_COLORS[(s, a)],
                                         linewidth=2,
                                         label=f"S{s} {AUG_LABELS[a]}"))
    return handles


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_csv(stage, aug, fma):
    p = os.path.join(FINAL_DIR, f"stage{stage}_{aug}", "csv", f"FMA_{fma}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def load_mot(stage, aug, fma):
    p = os.path.join(FINAL_DIR, f"stage{stage}_{aug}", "mot", f"FMA_{fma}.mot")
    return pd.read_csv(p, sep="\t", skiprows=6) if os.path.exists(p) else None


def build_csv_table():
    rows = []
    for stage in STAGES:
        for aug in AUGS:
            for fma in FMA_ALL:
                df = load_csv(stage, aug, fma)
                if df is None:
                    continue
                wr = df[["Wr_x","Wr_y","Wr_z"]].values
                tr = df[["Trunk_x","Trunk_y","Trunk_z"]].values
                dt = 1/200.0
                wd = np.linalg.norm(wr - wr[0], axis=1).max()
                td = np.linalg.norm(tr - tr[0], axis=1).max()
                rows.append({
                    "stage": stage, "aug": aug, "fma": fma,
                    "wrist_range": wr[:,1].max() - wr[:,1].min(),
                    "peak_vel":    np.linalg.norm(
                                     np.gradient(wr, dt, axis=0), axis=1).max(),
                    "trunk_disp":  td,
                    "trunk_wrist": td / wd if wd > 1 else np.nan,
                })
    return pd.DataFrame(rows)


def build_mot_table():
    rows = []
    for stage in STAGES:
        for aug in AUGS:
            for fma in FMA_ALL:
                mot = load_mot(stage, aug, fma)
                if mot is None:
                    continue
                r = {"stage": stage, "aug": aug, "fma": fma}
                for col, lbl in [("elbow_flexion", "elbow_rom"),
                                  ("pro_sup",       "pro_sup_rom")]:
                    if col in mot.columns:
                        v = mot[col].values * (180 / np.pi)
                        r[lbl] = v.max() - v.min()
                    else:
                        r[lbl] = np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def plot_all12(ax, data, metric, group_col="fma"):
    """Plot one line per (stage, aug) combo on ax."""
    for stage in STAGES:
        for aug in AUGS:
            sub = data[(data.stage==stage) & (data.aug==aug)].sort_values(group_col)
            valid = sub.dropna(subset=[metric])
            ax.plot(valid[group_col], valid[metric],
                    color=CONFIG_COLORS[(stage, aug)],
                    linewidth=1.5,
                    alpha=0.9)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — ROM trends all 12
# ══════════════════════════════════════════════════════════════════════════════

def fig_rom_all12(data):
    print("Generating: rom_trends_all12.png")

    metrics = [
        ("wrist_range", "Wrist Range (mm)",      "↑ with FMA"),
        ("peak_vel",    "Peak Velocity (mm/s)",   "↑ with FMA"),
        ("trunk_disp",  "Trunk Displacement (mm)","↓ with FMA"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    fig.suptitle("ROM Trends vs FMA Score — All 12 Configurations\n"
                 "Colour = Stage  |  3 lines per stage = DTW / SMOTE / Linear",
                 fontweight="bold")

    for ax, (mc, ml, note) in zip(axes, metrics):
        plot_all12(ax, data, mc)
        ax.set_title(ml)
        ax.set_xlabel("FMA Score")
        ax.set_ylabel(ml if ax == axes[0] else "")
        ax.grid(True, alpha=0.25)
        ax.annotate(note, xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", fontsize=8.5, color="gray", style="italic")

    fig.legend(handles=make_legend_handles(), loc="center right",
               bbox_to_anchor=(1.13, 0.5), frameon=True,
               title="Config key", title_fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "rom_trends_all12.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Trunk/wrist ratio all 12
# ══════════════════════════════════════════════════════════════════════════════

def fig_trunk_all12(data):
    print("Generating: trunk_wrist_ratio_all12.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.suptitle("Trunk Compensation Ratio vs FMA Score — All 12 Configurations\n"
                 "Colour = Stage  |  3 lines per stage = DTW / SMOTE / Linear  |  "
                 "Clinical expectation: ratio ↓ as FMA ↑",
                 fontweight="bold")

    plot_all12(ax, data, "trunk_wrist")
    ax.set_xlabel("FMA Score")
    ax.set_ylabel("Trunk Disp / Wrist Disp")
    ax.grid(True, alpha=0.25)

    fig.legend(handles=make_legend_handles(), loc="center right",
               bbox_to_anchor=(1.18, 0.5), frameon=True,
               title="Config key", title_fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "trunk_wrist_ratio_all12.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Joint ROM all 12
# ══════════════════════════════════════════════════════════════════════════════

def fig_joint_all12(mot_data):
    print("Generating: joint_rom_vs_fma_all12.png")

    joints = [
        ("elbow_rom",   "Elbow Flexion ROM (°)",          "Real mean 78° | generated ~58–63°"),
        ("pro_sup_rom", "Pronation/Supination ROM (°)",   "Real mean 50° | generated ~77–80°"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Joint ROM vs FMA Score — All 12 Configurations\n"
                 "Colour = Stage  |  3 lines per stage = DTW / SMOTE / Linear",
                 fontweight="bold")

    for ax, (jc, jl, note) in zip(axes, joints):
        plot_all12(ax, mot_data, jc)
        ax.set_title(jl)
        ax.set_xlabel("FMA Score")
        ax.set_ylabel(jl if ax == axes[0] else "")
        ax.grid(True, alpha=0.25)
        ax.annotate(note, xy=(0.97, 0.05), xycoords="axes fraction",
                    ha="right", fontsize=8.5, color="gray", style="italic")

    fig.legend(handles=make_legend_handles(), loc="center right",
               bbox_to_anchor=(1.15, 0.5), frameon=True,
               title="Config key", title_fontsize=9)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "joint_rom_vs_fma_all12.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output: {OUT_DIR}\n")

    print("Loading CSV metrics...")
    csv_data = build_csv_table()
    print(f"  {len(csv_data)} records\n")

    fig_rom_all12(csv_data)
    fig_trunk_all12(csv_data)

    print("\nLoading MOT joint angles...")
    mot_data = build_mot_table()
    print(f"  {len(mot_data)} records\n")

    fig_joint_all12(mot_data)

    print("\nDone:")
    for f in ["rom_trends_all12.png", "trunk_wrist_ratio_all12.png",
               "joint_rom_vs_fma_all12.png"]:
        p = os.path.join(OUT_DIR, f)
        print(f"  {'✓' if os.path.exists(p) else '✗'}  {f}")
