"""
phase_a_refined.py — Replaces and adds to phase_a_extended figures.

Overwrites 4 existing figures and adds 1 new one:
  segment_consistency.png       — bar chart per config + real reference
  rom_trends.png                — stage-avg vs aug-avg, with bands
  trunk_wrist_ratio.png         — stage-avg vs aug-avg, with real reference
  joint_rom_vs_fma.png          — elbow + shoulder, stage-avg vs aug-avg
  velocity_profiles_vs_real.png — NEW: generated vs original at FMA 16-20 and 66

Run from custom_workspace/:
  python test/scripts/phase_a_refined.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(BASE_DIR)
FINAL_DIR   = os.path.join(BASE_DIR,    "output", "final")
OUT_DIR     = os.path.join(BASE_DIR,    "output", "figures_phase_a")
ORIG_DIR    = os.path.join(PROJECT_ROOT, "data", "kinematic", "cutoff", "original")
SCORES_PATH = os.path.join(BASE_DIR,    "output", "scores.csv")
os.makedirs(OUT_DIR, exist_ok=True)

STAGES  = [0, 1, 2, 3]
AUGS    = ["dtw", "smote", "linear"]
FMA_ALL = list(range(16, 67))

STAGE_COLORS = {0: "#4e79a7", 1: "#f28e2b", 2: "#59a14f", 3: "#e15759"}
AUG_COLORS   = {"dtw": "#9467bd", "smote": "#8c564b", "linear": "#17becf"}
AUG_LABELS   = {"dtw": "DTW", "smote": "SMOTE", "linear": "Linear"}
STAGE_LABELS = {0: "Stage 0 (Baseline)", 1: "Stage 1 (+CFG)",
                2: "Stage 2 (+FiLM)",   3: "Stage 3 (+Residual)"}
REAL_COLOR   = "#2ca02c"

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "figure.dpi": 150,
})

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_csv(stage, aug, fma):
    p = os.path.join(FINAL_DIR, f"stage{stage}_{aug}", "csv", f"FMA_{fma}.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def load_mot(stage, aug, fma):
    p = os.path.join(FINAL_DIR, f"stage{stage}_{aug}", "mot", f"FMA_{fma}.mot")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, sep="\t", skiprows=6)


def wrist_speed(df_csv, dt=1/200.0):
    wr = df_csv[["Wr_x", "Wr_y", "Wr_z"]].values
    v  = np.linalg.norm(np.gradient(wr, dt, axis=0), axis=1)
    return v


def load_real_scores():
    """Return dict: base_filename (no ext) -> fma_score. Healthy files -> 66."""
    scores = {}
    if os.path.exists(SCORES_PATH):
        df = pd.read_csv(SCORES_PATH)
        for _, row in df.iterrows():
            base = os.path.splitext(row["filename"])[0]
            scores[base] = int(row["fma_score"])
    for fname in os.listdir(ORIG_DIR):
        if not fname.endswith(".csv"):
            continue
        base = os.path.splitext(fname)[0]
        if not base.startswith("S") and base not in scores:
            scores[base] = 66
    return scores


def load_real_profiles(fma_list):
    """Return list of normalised velocity profiles from original data."""
    score_map = load_real_scores()
    N = 100
    t_norm = np.linspace(0, 1, N)
    profiles = []
    for fname in os.listdir(ORIG_DIR):
        if not fname.endswith(".csv"):
            continue
        base = os.path.splitext(fname)[0]
        fma  = score_map.get(base)
        if fma not in fma_list:
            continue
        try:
            df = pd.read_csv(os.path.join(ORIG_DIR, fname))
            if "Wr_x" not in df.columns:
                continue
            v = wrist_speed(df)
            if len(v) < 10:
                continue
            t_src = np.linspace(0, 1, len(v))
            v_res = np.interp(t_norm, t_src, v)
            rng   = v_res.max() - v_res.min()
            if rng > 0:
                profiles.append((v_res - v_res.min()) / rng)
        except Exception:
            continue
    return profiles


# ── Pre-compute master table ───────────────────────────────────────────────────

def build_table():
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
                ua = np.linalg.norm(el - sh, axis=1)
                fa = np.linalg.norm(wr - el, axis=1)
                td = np.linalg.norm(tr - tr[0], axis=1).max()
                wd = np.linalg.norm(wr - wr[0], axis=1).max()
                rows.append({
                    "stage": stage, "aug": aug, "fma": fma,
                    "wrist_range": wr[:,1].max() - wr[:,1].min(),
                    "peak_vel":    np.linalg.norm(
                                     np.gradient(wr, dt, axis=0), axis=1).max(),
                    "trunk_disp":  td,
                    "wrist_disp":  wd,
                    "trunk_wrist": td / wd if wd > 1 else np.nan,
                    "ua_std": ua.std(), "fa_std": fa.std(),
                    "ua_mean": ua.mean(), "fa_mean": fa.mean(),
                })
    return pd.DataFrame(rows)


def agg_by_stage(data, metric, fma_vals=None):
    """For each (stage, fma): mean and std averaged over 3 augs."""
    sub = data if fma_vals is None else data[data.fma.isin(fma_vals)]
    g = sub.groupby(["stage","fma"])[metric].agg(["mean","std"]).reset_index()
    return g


def agg_by_aug(data, metric, fma_vals=None):
    """For each (aug, fma): mean and std averaged over 4 stages."""
    sub = data if fma_vals is None else data[data.fma.isin(fma_vals)]
    g = sub.groupby(["aug","fma"])[metric].agg(["mean","std"]).reset_index()
    return g


def band_plot(ax, x, mean, std, color, label, ls="-", lw=2.0, alpha=0.15):
    ax.plot(x, mean, color=color, linestyle=ls, linewidth=lw, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Segment consistency (bar chart)
# ══════════════════════════════════════════════════════════════════════════════

def fig_segment_consistency(data):
    print("Generating: segment_consistency.png")

    # Compute real data reference segment lengths
    score_map = load_real_scores()
    ua_real, fa_real = [], []
    for fname in os.listdir(ORIG_DIR):
        if not fname.endswith(".csv"): continue
        try:
            df = pd.read_csv(os.path.join(ORIG_DIR, fname))
            sh = df[["Sh_x","Sh_y","Sh_z"]].values
            el = df[["El_x","El_y","El_z"]].values
            wr = df[["Wr_x","Wr_y","Wr_z"]].values
            ua_real.append(np.linalg.norm(el - sh, axis=1).std())
            fa_real.append(np.linalg.norm(wr - el, axis=1).std())
        except Exception:
            continue
    ref_ua = np.mean(ua_real) if ua_real else None
    ref_fa = np.mean(fa_real) if fa_real else None

    # Mean std per config (averaged across all FMA scores)
    summary = data.groupby(["stage","aug"])[["ua_std","fa_std"]].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Segment Length Consistency — Lower = More Physically Plausible\n"
                 "Each bar = mean std across all 51 FMA scores  |  dashed line = real MoCap reference",
                 fontweight="bold")

    for ax, col, label, ref in zip(axes,
                                    ["ua_std","fa_std"],
                                    ["Upper Arm Length Std Dev (mm)",
                                     "Forearm Length Std Dev (mm)"],
                                    [ref_ua, ref_fa]):
        n_stages = len(STAGES)
        n_augs   = len(AUGS)
        group_w  = 0.8
        bar_w    = group_w / n_augs
        x_base   = np.arange(n_stages)

        for j, aug in enumerate(AUGS):
            vals   = []
            for stage in STAGES:
                row = summary[(summary.stage==stage) & (summary.aug==aug)]
                vals.append(row[col].values[0] if len(row) else 0)
            offset = (j - 1) * bar_w
            bars = ax.bar(x_base + offset, vals, bar_w,
                          color=AUG_COLORS[aug], label=AUG_LABELS[aug],
                          alpha=0.88, edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.03,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

        if ref is not None:
            ax.axhline(ref, color=REAL_COLOR, linestyle="--", linewidth=1.8,
                       label=f"Real MoCap ({ref:.1f} mm)")

        ax.set_xticks(x_base)
        ax.set_xticklabels([STAGE_LABELS[s] for s in STAGES], fontsize=8.5)
        ax.set_ylabel(label)
        ax.set_xlabel("Architectural Stage")
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "segment_consistency.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — ROM trends (stage-avg left, aug-avg right per metric)
# ══════════════════════════════════════════════════════════════════════════════

def fig_rom_trends(data):
    print("Generating: rom_trends.png")

    metrics = [
        ("wrist_range", "Wrist Range (mm)",      "Higher FMA → greater reach"),
        ("peak_vel",    "Peak Velocity (mm/s)",   "Higher FMA → faster movement"),
        ("trunk_disp",  "Trunk Displacement (mm)","Higher FMA → less trunk lean"),
    ]

    fig, axes = plt.subplots(len(metrics), 2,
                             figsize=(13, 11), sharex=True)
    fig.suptitle("Kinematic Metric Trends vs FMA Score\n"
                 "Left: effect of Stage (averaged across augmentations)  |  "
                 "Right: effect of Augmentation (averaged across stages)",
                 fontweight="bold", y=1.01)

    for row, (mc, ml, note) in enumerate(metrics):
        # Left — by stage
        ax_l = axes[row][0]
        ax_l.set_ylabel(ml)
        if row == 0:
            ax_l.set_title("Stage Comparison\n(shaded = ±1 std across aug methods)",
                           fontsize=10)
        g = agg_by_stage(data, mc)
        for stage in STAGES:
            s = g[g.stage==stage].sort_values("fma")
            band_plot(ax_l, s["fma"], s["mean"], s["std"],
                      STAGE_COLORS[stage], STAGE_LABELS[stage])
        ax_l.grid(True, alpha=0.25)
        ax_l.annotate(note, xy=(0.97, 0.05), xycoords="axes fraction",
                      ha="right", fontsize=8, color="gray",
                      style="italic")
        ax_l.legend(fontsize=8)
        ax_l.set_xlabel("FMA Score" if row == len(metrics)-1 else "")

        # Right — by augmentation
        ax_r = axes[row][1]
        if row == 0:
            ax_r.set_title("Augmentation Comparison\n(shaded = ±1 std across stages)",
                           fontsize=10)
        g = agg_by_aug(data, mc)
        for aug in AUGS:
            s = g[g.aug==aug].sort_values("fma")
            band_plot(ax_r, s["fma"], s["mean"], s["std"],
                      AUG_COLORS[aug], AUG_LABELS[aug])
        ax_r.grid(True, alpha=0.25)
        ax_r.legend(fontsize=8)
        ax_r.set_xlabel("FMA Score" if row == len(metrics)-1 else "")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "rom_trends.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Trunk/wrist ratio (with real data reference)
# ══════════════════════════════════════════════════════════════════════════════

def fig_trunk_wrist(data):
    print("Generating: trunk_wrist_ratio.png")

    # Real data reference
    score_map = load_real_scores()
    real_rows = []
    for fname in os.listdir(ORIG_DIR):
        if not fname.endswith(".csv"): continue
        base = os.path.splitext(fname)[0]
        fma  = score_map.get(base)
        if fma is None: continue
        try:
            df = pd.read_csv(os.path.join(ORIG_DIR, fname))
            tr = df[["Trunk_x","Trunk_y","Trunk_z"]].values
            wr = df[["Wr_x","Wr_y","Wr_z"]].values
            td = np.linalg.norm(tr - tr[0], axis=1).max()
            wd = np.linalg.norm(wr - wr[0], axis=1).max()
            if wd > 1:
                real_rows.append({"fma": fma, "ratio": td / wd})
        except Exception:
            continue
    real_df = pd.DataFrame(real_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Trunk Compensation Ratio (Trunk Disp / Wrist Disp) vs FMA Score\n"
                 "Clinical expectation: ratio decreases as FMA increases (less compensation in healthier patients)\n"
                 "Green = real patient data reference",
                 fontweight="bold")

    for ax, title, agg_fn, items, color_map, label_map in [
        (axes[0], "Stage Comparison (avg across augmentations)",
         agg_by_stage, STAGES, STAGE_COLORS, STAGE_LABELS),
        (axes[1], "Augmentation Comparison (avg across stages)",
         agg_by_aug, AUGS, AUG_COLORS, AUG_LABELS),
    ]:
        ax.set_title(title, fontsize=10)
        g = agg_fn(data, "trunk_wrist")
        group_col = "stage" if agg_fn == agg_by_stage else "aug"
        for item in items:
            s = g[g[group_col]==item].sort_values("fma").dropna(subset=["mean"])
            band_plot(ax, s["fma"], s["mean"], s["std"],
                      color_map[item], label_map[item])

        # Real data scatter
        if not real_df.empty:
            for fma_val, grp in real_df.groupby("fma"):
                ax.scatter([fma_val]*len(grp), grp["ratio"],
                           color=REAL_COLOR, s=40, zorder=5,
                           alpha=0.8, edgecolors="white", linewidths=0.5)
            # Add legend entry for real data
            ax.scatter([], [], color=REAL_COLOR, s=40,
                       label="Real MoCap data", zorder=5)

        ax.set_xlabel("FMA Score"); ax.set_ylabel("Trunk / Wrist Ratio" if ax==axes[0] else "")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

        # Annotate trend direction
        ax.annotate("← more compensation\n(impaired)", xy=(18, ax.get_ylim()[1]*0.92),
                    fontsize=8, color="gray", style="italic")
        ax.annotate("less compensation →\n(healthier)", xy=(56, ax.get_ylim()[1]*0.92),
                    fontsize=8, color="gray", style="italic", ha="right")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "trunk_wrist_ratio.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Joint ROM vs FMA (elbow + shoulder, stage vs aug)
# ══════════════════════════════════════════════════════════════════════════════

def fig_joint_rom():
    print("Generating: joint_rom_vs_fma.png  (reading MOT files...)")

    # Load from MOT files
    rows = []
    for stage in STAGES:
        for aug in AUGS:
            for fma in FMA_ALL:
                mot = load_mot(stage, aug, fma)
                if mot is None: continue
                r = {"stage": stage, "aug": aug, "fma": fma}
                for col, lbl in [("elbow_flexion","elbow_rom"),
                                  ("pro_sup","pro_sup_rom")]:
                    if col in mot.columns:
                        v = mot[col].values * (180/np.pi)
                        r[lbl] = v.max() - v.min()
                    else:
                        r[lbl] = np.nan
                rows.append(r)
    data = pd.DataFrame(rows)

    # Real data reference from MOT files in workspace
    real_mot_dir = os.path.join(PROJECT_ROOT, "output", "originals", "mot")
    scores_ws    = os.path.join(PROJECT_ROOT, "output", "scores.csv")
    real_joints  = []
    score_map    = {}
    if os.path.exists(scores_ws):
        df_s = pd.read_csv(scores_ws)
        for _, row in df_s.iterrows():
            score_map[row["filename"]] = int(row["fma_score"])
    if os.path.exists(real_mot_dir):
        for fname in os.listdir(real_mot_dir):
            if not fname.endswith(".mot"): continue
            fma = score_map.get(fname, 66 if not fname.startswith("S") else None)
            if fma is None: continue
            try:
                mot = pd.read_csv(os.path.join(real_mot_dir, fname),
                                  sep="\t", skiprows=6)
                r = {"fma": fma}
                for col, lbl in [("elbow_flexion","elbow_rom"),
                                   ("pro_sup","pro_sup_rom")]:
                    if col in mot.columns:
                        v = mot[col].values * (180/np.pi)
                        r[lbl] = v.max() - v.min()
                real_joints.append(r)
            except Exception:
                continue
    real_jdf = pd.DataFrame(real_joints) if real_joints else pd.DataFrame()

    joints = [
        ("elbow_rom",   "Elbow Flexion ROM (°)",         "Real MoCap mean 78° — generated ~58–63°"),
        ("pro_sup_rom", "Pronation/Supination ROM (°)",   "Real MoCap mean 50° — generated ~77–80°"),
    ]

    fig, axes = plt.subplots(len(joints), 2, figsize=(13, 9), sharex=True)
    fig.suptitle("Joint Range of Motion vs FMA Score\n"
                 "Left: Stage effect  |  Right: Augmentation effect  |  "
                 "Green dots = real patient/healthy MoCap",
                 fontweight="bold", y=1.01)

    for row, (jc, jl, note) in enumerate(joints):
        for col, (title, agg_fn, items, color_map, label_map) in enumerate([
            ("Stage Comparison\n(avg across augmentations)",
             agg_by_stage, STAGES, STAGE_COLORS, STAGE_LABELS),
            ("Augmentation Comparison\n(avg across stages)",
             agg_by_aug, AUGS, AUG_COLORS, AUG_LABELS),
        ]):
            ax = axes[row][col]
            if row == 0: ax.set_title(title, fontsize=10)
            g_col = "stage" if col == 0 else "aug"
            g = agg_fn(data, jc)
            for item in items:
                s = g[g[g_col]==item].sort_values("fma").dropna(subset=["mean"])
                band_plot(ax, s["fma"], s["mean"], s["std"],
                          color_map[item], label_map[item])

            # Real data scatter
            if not real_jdf.empty and jc in real_jdf.columns:
                for fma_val, grp in real_jdf.groupby("fma"):
                    ax.scatter([fma_val]*len(grp), grp[jc].dropna(),
                               color=REAL_COLOR, s=40, zorder=5,
                               alpha=0.8, edgecolors="white", linewidths=0.5)
                ax.scatter([], [], color=REAL_COLOR, s=40, label="Real MoCap")

            ax.set_ylabel(jl if col == 0 else "")
            ax.set_xlabel("FMA Score" if row == len(joints)-1 else "")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            ax.annotate(note, xy=(0.97, 0.05), xycoords="axes fraction",
                        ha="right", fontsize=8, color="gray", style="italic")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "joint_rom_vs_fma.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Velocity profiles (improved, cleaner)
# ══════════════════════════════════════════════════════════════════════════════

def fig_velocity_profiles_improved(data):
    print("Generating: velocity_profiles.png  (improved)")

    fma_groups = {
        "FMA 16–20\n(Severe stroke)":      list(range(16, 21)),
        "FMA 21–40\n(Moderate)":           list(range(21, 41)),
        "FMA 41–66\n(Mild / Healthy)":     list(range(41, 67)),
    }
    N = 100
    t = np.linspace(0, 1, N)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle("Mean Wrist Speed Profile (Normalised Time & Amplitude)\n"
                 "Stage effect only — 4 lines (avg across all 3 augmentations) with ±1 std band\n"
                 "Bell-shaped = smooth healthy reach  |  flat/multi-peaked = impaired",
                 fontweight="bold")

    for col, (grp_label, fma_list) in enumerate(fma_groups.items()):
        ax = axes[col]
        ax.set_title(grp_label, fontsize=11)

        for stage in STAGES:
            all_profiles = []
            for aug in AUGS:
                sub = data[(data.stage==stage) & (data.aug==aug) &
                           (data.fma.isin(fma_list))]
                for fma in fma_list:
                    df = load_csv(stage, aug, fma)
                    if df is None: continue
                    v = wrist_speed(df)
                    if len(v) < 10: continue
                    v_res = np.interp(t, np.linspace(0,1,len(v)), v)
                    rng = v_res.max() - v_res.min()
                    if rng > 0:
                        all_profiles.append((v_res - v_res.min()) / rng)

            if all_profiles:
                arr  = np.array(all_profiles)
                mean = arr.mean(axis=0)
                std  = arr.std(axis=0)
                band_plot(ax, t, mean, std, STAGE_COLORS[stage],
                          STAGE_LABELS[stage], lw=2.0)

        ax.set_xlabel("Normalised Time")
        ax.set_ylabel("Normalised Speed" if col == 0 else "")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "velocity_profiles.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 (NEW) — Generated vs Original velocity profiles
# ══════════════════════════════════════════════════════════════════════════════

def fig_velocity_vs_real():
    print("Generating: velocity_profiles_vs_real.png  (loading original data...)")

    N = 100
    t = np.linspace(0, 1, N)

    # Groups: real FMA 16-20, real FMA 66, generated FMA 16-20, generated FMA 66
    real_stroke  = load_real_profiles(list(range(16, 21)))
    real_healthy = load_real_profiles([66])

    # Generated: best (stage3_smote) and overall mean (all 12 configs)
    def gen_profiles_for(fma_list, stage=None, aug=None):
        profiles = []
        s_iter = [stage] if stage is not None else STAGES
        a_iter = [aug]   if aug   is not None else AUGS
        for s in s_iter:
            for a in a_iter:
                for fma in fma_list:
                    df = load_csv(s, a, fma)
                    if df is None: continue
                    v = wrist_speed(df)
                    if len(v) < 10: continue
                    v_res = np.interp(t, np.linspace(0,1,len(v)), v)
                    rng = v_res.max() - v_res.min()
                    if rng > 0:
                        profiles.append((v_res - v_res.min()) / rng)
        return profiles

    gen_stroke_all    = gen_profiles_for(list(range(16,21)))
    gen_healthy_all   = gen_profiles_for([66])
    gen_stroke_best   = gen_profiles_for(list(range(16,21)),  stage=3, aug="smote")
    gen_healthy_best  = gen_profiles_for([66],                stage=3, aug="smote")

    def plot_group(ax, profiles_list, colors, labels, alphas):
        for profiles, color, label, alpha in zip(profiles_list, colors, labels, alphas):
            if not profiles:
                continue
            arr  = np.array(profiles)
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            ax.plot(t, mean, color=color, linewidth=2.2, label=f"{label} (n={len(profiles)})")
            ax.fill_between(t, mean-std, mean+std, color=color, alpha=alpha)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.suptitle("Generated vs Real Wrist Speed Profiles\n"
                 "Shaded = ±1 std  |  'All configs' = mean across all 12 CVAE configurations",
                 fontweight="bold")

    panel_data = [
        (axes[0], "Stroke Patients — FMA 16–20\n(severe impairment)",
         [real_stroke,       gen_stroke_all,      gen_stroke_best],
         [REAL_COLOR,        "#4e79a7",            "#e15759"],
         ["Real MoCap",      "Generated (all 12)", "Generated (Stage 3 + SMOTE)"],
         [0.15,              0.12,                 0.12]),
        (axes[1], "Healthy Subjects — FMA 66\n(no impairment)",
         [real_healthy,      gen_healthy_all,      gen_healthy_best],
         [REAL_COLOR,        "#4e79a7",            "#e15759"],
         ["Real MoCap",      "Generated (all 12)", "Generated (Stage 3 + SMOTE)"],
         [0.15,              0.12,                 0.12]),
    ]

    for ax, title, prof_list, colors, labels, alphas in panel_data:
        ax.set_title(title, fontsize=11)
        plot_group(ax, prof_list, colors, labels, alphas)
        ax.set_xlabel("Normalised Time")
        ax.set_ylabel("Normalised Speed" if ax == axes[0] else "")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.3)

        # Annotate expected shape
        ax.annotate("ideal: single\nbell shape ↑", xy=(0.48, 0.92),
                    xycoords="axes fraction", ha="center",
                    fontsize=8, color="gray", style="italic")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "velocity_profiles_vs_real.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"  saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output: {OUT_DIR}\n")
    print("Pre-computing kinematic metrics for all 612 files...")
    data = build_table()
    print(f"  loaded {len(data)} records\n")

    fig_segment_consistency(data)
    fig_rom_trends(data)
    fig_trunk_wrist(data)
    fig_joint_rom()
    fig_velocity_profiles_improved(data)
    fig_velocity_vs_real()

    print("\nDone. Updated figures in figures_phase_a/:")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")
