"""
Plot trunk compensation — forward lean only, Stroke vs Healthy.
Requires compute_trunk_comp.py to have been run first.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_project_root

ROOT = get_project_root()
CSV  = os.path.join(ROOT, "output", "trunk_lean_validation", "trunk_comp_table.csv")
OUT  = os.path.join(ROOT, "output", "trunk_lean_validation", "trunk_comp_plot.png")

df      = pd.read_csv(CSV)
stroke  = df[df.group == "Stroke"]
healthy = df[df.group == "Healthy"]
SC = "#C0392B"; HC = "#2980B9"
rng = np.random.default_rng(0)

plt.rcParams.update({"font.family": "sans-serif",
                     "axes.spines.top": False, "axes.spines.right": False})

fig = plt.figure(figsize=(18, 11), facecolor="#F7F9FC")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)
fig.suptitle("Forward Trunk Compensation at Peak Reach  —  Stroke vs Healthy",
             fontsize=13, fontweight="bold", y=0.99)


def subject_strip(ax, data, comp_col, ylabel, title):
    ax.set_facecolor("#F7F9FC")
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
    stroke_d  = data[data.group == "Stroke"]
    healthy_d = data[data.group == "Healthy"]
    offset = 0
    for grp, color in [(stroke_d, SC), (healthy_d, HC)]:
        subjects = sorted(grp.subject.unique())
        for j, subj in enumerate(subjects):
            vals = grp[grp.subject == subj][comp_col].values
            xs   = np.full(len(vals), offset + j, dtype=float)
            xs  += rng.uniform(-0.2, 0.2, len(vals))
            ax.scatter(xs, vals, color=color, alpha=0.75, s=55, zorder=3)
            ax.plot([offset+j-0.3, offset+j+0.3],
                    [vals.mean(), vals.mean()], color=color, lw=2.5, zorder=4)
        offset += len(subjects) + 2

    n_s = len(stroke_d.subject.unique()); n_h = len(healthy_d.subject.unique())
    ax.set_xticks(list(range(n_s)) + list(range(n_s+2, n_s+2+n_h)))
    ax.set_xticklabels(sorted(stroke_d.subject.unique()) + sorted(healthy_d.subject.unique()),
                       rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ylim = ax.get_ylim()
    top = ylim[1]
    ax.text(n_s/2 - 0.5,     top * 0.93, "STROKE",  color=SC, fontsize=11, fontweight="bold", ha="center")
    ax.text(n_s+2+n_h/2-0.5, top * 0.93, "HEALTHY", color=HC, fontsize=11, fontweight="bold", ha="center")
    ax.grid(True, axis="y", alpha=0.25, ls="--")


def strip_box(ax, s_vals, h_vals, ylabel, title):
    ax.set_facecolor("#F7F9FC")
    ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)
    for xi, (vals, color) in enumerate([(s_vals, SC), (h_vals, HC)]):
        bp = ax.boxplot(vals, positions=[xi], widths=0.4, patch_artist=True,
                        medianprops=dict(color="white", lw=2),
                        whiskerprops=dict(color=color), capprops=dict(color=color),
                        flierprops=dict(marker="o", color=color, alpha=0.4, ms=4))
        bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_alpha(0.35)
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(xi + jitter, vals, color=color, alpha=0.7, s=45, zorder=3)
        ax.text(xi, vals.max() + abs(vals.max()) * 0.08 + 0.3,
                f"μ={vals.mean():.1f}°\nσ={vals.std():.1f}°",
                ha="center", fontsize=9, color=color, fontweight="bold")
    t, p = stats.ttest_ind(s_vals, h_vals)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Stroke", "Healthy"], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f"{title}\np={p:.3f}", fontsize=10)
    ax.grid(True, axis="y", alpha=0.25, ls="--")


# ── Row 0: Per-subject strip + group boxplot ──────────────────────────────────
ax00 = fig.add_subplot(gs[0, :2])
subject_strip(ax00, df, "fwd_comp", "Forward comp (°)",
              "Per-subject forward trunk compensation  (dots = trials, bar = mean)")

ax01 = fig.add_subplot(gs[0, 2])
strip_box(ax01, stroke.fwd_comp, healthy.fwd_comp,
          "Forward comp (°)", "Group comparison")

# ── Row 1: Scatter vs wrist exc | start vs peak | histogram ──────────────────
ax10 = fig.add_subplot(gs[1, 0])
ax10.set_facecolor("#F7F9FC")
for data, color, label in [(stroke, SC, "Stroke"), (healthy, HC, "Healthy")]:
    ax10.scatter(data.wrist_exc_mm, data.fwd_comp,
                 color=color, alpha=0.65, s=55, label=label)
ax10.axhline(0, color="gray", lw=0.8, ls="--")
ax10.set_xlabel("Wrist excursion (mm)", fontsize=10)
ax10.set_ylabel("Forward comp (°)", fontsize=10)
ax10.set_title("Trunk comp vs Reach distance", fontsize=10)
ax10.legend(fontsize=9); ax10.grid(True, alpha=0.25, ls="--")

ax11 = fig.add_subplot(gs[1, 1])
ax11.set_facecolor("#F7F9FC")
for data, color, label in [(stroke, SC, "Stroke"), (healthy, HC, "Healthy")]:
    ax11.scatter(data.fwd_tilt_start, data.fwd_tilt_peak,
                 color=color, alpha=0.65, s=55, label=label)
lim = max(df.fwd_tilt_start.abs().max(), df.fwd_tilt_peak.abs().max()) * 1.1
ax11.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.5, label="No change")
ax11.set_xlabel("Forward tilt at start (°)", fontsize=10)
ax11.set_ylabel("Forward tilt at peak reach (°)", fontsize=10)
ax11.set_title("Start vs Peak tilt\n(above line = leaned more during reach)", fontsize=10)
ax11.legend(fontsize=9); ax11.grid(True, alpha=0.25, ls="--")

ax12 = fig.add_subplot(gs[1, 2])
ax12.set_facecolor("#F7F9FC")
bins = np.linspace(df.fwd_comp.min()-1, df.fwd_comp.max()+1, 20)
ax12.hist(healthy.fwd_comp, bins=bins, color=HC, alpha=0.55, label="Healthy", edgecolor="white")
ax12.hist(stroke.fwd_comp,  bins=bins, color=SC, alpha=0.55, label="Stroke",  edgecolor="white")
ax12.axvline(stroke.fwd_comp.mean(),  color=SC, lw=2, ls="--",
             label=f"Stroke μ={stroke.fwd_comp.mean():.1f}°")
ax12.axvline(healthy.fwd_comp.mean(), color=HC, lw=2, ls="--",
             label=f"Healthy μ={healthy.fwd_comp.mean():.1f}°")
ax12.axvline(0, color="gray", lw=0.8, ls=":")
ax12.set_xlabel("Forward comp (°)", fontsize=10)
ax12.set_ylabel("Count", fontsize=10)
ax12.set_title("Distribution", fontsize=10)
ax12.legend(fontsize=8); ax12.grid(True, axis="y", alpha=0.25, ls="--")

plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {OUT}")
plt.show()
