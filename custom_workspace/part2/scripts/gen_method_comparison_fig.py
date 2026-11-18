"""
Generate method comparison figure: Standard IK (no weighting) vs SVD Two-Pass IK.
Reads from output/experiments/exp1_standard_vs_svd/results.csv.
Output: paper_publication_assets/figures_and_tables/damping_comparison.png
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_project_root

ROOT    = get_project_root()
CSV     = os.path.join(ROOT, 'output', 'experiments', 'exp1_standard_vs_svd', 'results.csv')
OUT     = os.path.join(ROOT, 'paper_publication_assets', 'figures_and_tables', 'damping_comparison.png')

STROKE_COL  = '#d62728'
HEALTHY_COL = '#4472c4'

METHOD_LABELS = {
    'Standard IK': 'Standard IK\n(no weighting)',
    'SVD Best':    'SVD Two-Pass IK\n(SC hierarchy)',
}

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'DejaVu Sans'],
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
})

df = pd.read_csv(CSV)
df = df[df['status'] == 'OK']

methods = ['Standard IK', 'SVD Best']
trials  = df['trial'].unique().tolist()
# Stroke trials first, then healthy
stroke_trials  = [t for t in trials if df[df['trial']==t]['group'].iloc[0] == 'stroke']
healthy_trials = [t for t in trials if df[df['trial']==t]['group'].iloc[0] == 'healthy']
ordered_trials = stroke_trials + healthy_trials

n      = len(ordered_trials)
bar_w  = 0.35
x_pos  = np.arange(n)

METHOD_COLS = {
    'Standard IK': '#888888',
    'SVD Best':    '#4472c4',
}

fig, ax = plt.subplots(figsize=(9, 4), facecolor='white')

for mi, method in enumerate(methods):
    vals = [df[(df['method']==method) & (df['trial']==t)]['ik_error_mm'].values[0]
            if len(df[(df['method']==method) & (df['trial']==t)]) > 0 else np.nan
            for t in ordered_trials]
    offset = (mi - 0.5) * bar_w
    ax.bar(x_pos + offset, vals, bar_w,
           color=METHOD_COLS[method], alpha=0.82,
           label=METHOD_LABELS[method].replace('\n', ' '),
           zorder=3)

# Divider between stroke and healthy
div = len(stroke_trials) - 0.5
ax.axvline(div, color='#cccccc', lw=1.0, linestyle='--', zorder=2)
ax.text(div - len(stroke_trials)/2,  ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 400,
        'Stroke', ha='center', va='bottom', fontsize=7.5, color=STROKE_COL, fontstyle='italic')
ax.text(div + len(healthy_trials)/2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 400,
        'Healthy', ha='center', va='bottom', fontsize=7.5, color=HEALTHY_COL, fontstyle='italic')

ax.set_xticks(x_pos)
ax.set_xticklabels(ordered_trials, rotation=35, ha='right', fontsize=8, color='#333333')
ax.set_ylabel('IK reconstruction error (mm)', fontsize=8.5, color='#444444', labelpad=4)
ax.tick_params(labelsize=7.5, colors='#555555')
ax.spines['left'].set_color('#888888')
ax.spines['bottom'].set_color('#888888')
ax.tick_params(color='#888888')
ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.set_xlim(-0.6, n - 0.4)

ax.legend(fontsize=8, frameon=False, handlelength=1.2,
          handletextpad=0.5, loc='upper right')

plt.tight_layout(pad=0.8)
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved {OUT}")
