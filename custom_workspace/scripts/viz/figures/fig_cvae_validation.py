#!/usr/bin/env python3
"""
Figure 7: CVAE validation — D_base generated vs SMOTE training data.

Grid: 6 FMA rows × 4 joint columns (Sh_y, El_y, Wr_y, Trunk_y).
Both datasets are delta-normalised (first frame subtracted) for comparison.

Output:
    figures/cvae_validation.png  (300 DPI)
    figures/cvae_validation.pdf  (vector)
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'font.size':          8,
    'axes.unicode_minus': False,
    'axes.linewidth':     0.7,
})

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
SMOTE_DIR   = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'augmented_smote_test')
GEN_DIR     = os.path.join(ROOT, 'test', 'output', 'D_base', 'csv_n10')
OUT_DIR     = os.path.join(ROOT, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

FMA_LEVELS  = [18, 20, 30, 40, 50, 66]
PLOT_COLS   = [('Sh\_y',  'Sh_y'),
               ('El\_y',  'El_y'),
               ('Wr\_y',  'Wr_y'),
               ('Trunk\_y','Trunk_y')]
SMOTE_N     = 60
RNG         = np.random.default_rng(42)

BLUE  = '#1F77B4'
RED   = '#D62728'


def load_smote(fma, n=SMOTE_N):
    files = glob.glob(os.path.join(SMOTE_DIR, f'*FMA{fma}.csv'))
    if not files:
        return None
    chosen = list(RNG.choice(files, size=min(n, len(files)), replace=False))
    trajs = []
    for f in chosen:
        try:
            df = pd.read_csv(f)
            if df.shape[1] >= 15 and len(df) >= 10:
                arr = df.values.astype(float)
                trajs.append(arr - arr[0])   # delta-normalise
        except Exception:
            continue
    return np.array(trajs) if trajs else None


def load_generated(fma):
    path = os.path.join(GEN_DIR, f'FMA_{fma}.csv')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        if df.shape[1] < 15 or len(df) < 10:
            return None
        arr = df.values.astype(float)
        return arr - arr[0]   # delta-normalise
    except Exception:
        return None


def col_idx(name):
    cols = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
            'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z',
            'Trunk_x','Trunk_y','Trunk_z']
    return cols.index(name)


def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(labelsize=6, length=2.5, width=0.6)
    ax.grid(linestyle='--', alpha=0.18, linewidth=0.4)


# ── Figure ────────────────────────────────────────────────────────────────────
nrows, ncols = len(FMA_LEVELS), len(PLOT_COLS)
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(6.30, 7.00),
                         gridspec_kw={'hspace': 0.55, 'wspace': 0.38})
fig.patch.set_facecolor('white')

x = np.arange(100)

for r, fma in enumerate(FMA_LEVELS):
    smote = load_smote(fma)
    gen   = load_generated(fma)

    for c, (label, colname) in enumerate(PLOT_COLS):
        ax = axes[r, c]
        ci = col_idx(colname)

        # SMOTE band
        if smote is not None and smote.shape[2] > ci:
            mn = smote[:, :, ci].mean(0)
            sd = smote[:, :, ci].std(0)
            ax.fill_between(x, mn - sd, mn + sd, alpha=0.18, color=BLUE)
            ax.plot(x, mn, color=BLUE, lw=1.1, label='SMOTE' if (r == 0 and c == ncols-1) else '_')

        # Generated line
        if gen is not None and gen.shape[1] > ci:
            ax.plot(x, gen[:, ci], color=RED, lw=1.1, ls='--',
                    label='Generated' if (r == 0 and c == ncols-1) else '_')

        style_ax(ax)

        # Column headers (top row only)
        if r == 0:
            ax.set_title(label, fontsize=7.5, pad=3)

        # Row FMA labels (left column only)
        if c == 0:
            ax.set_ylabel(f'FMA {fma}', fontsize=7, labelpad=3)

        # x-axis label (bottom row only)
        if r == nrows - 1:
            ax.set_xlabel('Frame', fontsize=6.5)

# ── Shared legend ─────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=BLUE,  lw=1.3,              label='SMOTE (mean $\\pm$1 SD)'),
    Line2D([0], [0], color=RED,   lw=1.3, ls='--',     label='Generated ($N=10$)'),
]
fig.legend(handles=handles, loc='upper center', ncol=2,
           fontsize=7, framealpha=0.7, handlelength=1.8,
           borderpad=0.4, bbox_to_anchor=(0.5, 0.995))

# ── Save ─────────────────────────────────────────────────────────────────────
for fmt in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT_DIR, f'cvae_validation.{fmt}'),
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved cvae_validation.{png,pdf}')
