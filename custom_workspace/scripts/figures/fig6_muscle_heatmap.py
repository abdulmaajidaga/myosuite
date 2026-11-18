#!/usr/bin/env python3
"""Figure 6: Muscle activation heatmaps at FMA 18, 40, 66.

Data: output/generated/id/FMA_{N}/activations.csv  (100 frames × 32 muscles)
Output: figures/muscle_activation_heatmap.{png,pdf}
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'font.size':          8,
    'axes.unicode_minus': False,
    'axes.linewidth':     0.6,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
ID_DIR     = os.path.join(ROOT, 'output', 'generated', 'id')
OUT_DIR    = os.path.join(ROOT, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

FMA_LEVELS = [18, 40, 66]
PANEL_TITLES = ['FMA 18  (severe)', 'FMA 40  (moderate)', 'FMA 66  (healthy)']

# Muscle order matching Table 11 (tab:muscle_joint_mask), grouped by function
MUSCLE_GROUPS = {
    'Shoulder': ['DELT1','DELT2','DELT3','SUPSP','INFSP','SUBSC',
                 'TMIN','TMAJ','PECM1','PECM2','PECM3','LAT1','LAT2','LAT3','CORB'],
    'Biartic.': ['TRIlong','BIClong','BICshort'],
    'Elbow':    ['TRIlat','TRImed','ANC','BRA'],
    'El/Fore':  ['BRD','SUP','PT'],
    'Forearm':  ['ECRL','ECRB','ECU','FCR','FCU','PL','PQ'],
}
MUSCLE_ORDER = [m for grp in MUSCLE_GROUPS.values() for m in grp]
GROUP_SIZES  = {g: len(ms) for g, ms in MUSCLE_GROUPS.items()}

# ── Load activations ──────────────────────────────────────────────────────────
def resample_100(arr):
    """Resample (n_frames, n_muscles) → (100, n_muscles) via linear interpolation."""
    n = arr.shape[0]
    if n == 100:
        return arr
    x_old = np.linspace(0, 1, n)
    x_new = np.linspace(0, 1, 100)
    return np.stack([np.interp(x_new, x_old, arr[:, i]) for i in range(arr.shape[1])], axis=1)

def load_act(fma):
    path = os.path.join(ID_DIR, f'FMA_{fma}', 'activations.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Missing: {path}')
    df = pd.read_csv(path)
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
    available = [m for m in MUSCLE_ORDER if m in df.columns]
    missing   = [m for m in MUSCLE_ORDER if m not in df.columns]
    if missing:
        print(f'  FMA {fma}: muscles not found: {missing}')
    df = df[available]
    arr = resample_100(df.values)   # (100, n_muscles)
    return arr.T                    # (n_muscles, 100)

data = {}
for fma in FMA_LEVELS:
    data[fma] = load_act(fma)
    print(f'FMA {fma}: activations shape {data[fma].shape}  '
          f'  file: {ID_DIR}/FMA_{fma}/activations.csv')

n_muscles = data[FMA_LEVELS[0]].shape[0]

# ── Figure layout ─────────────────────────────────────────────────────────────
# 3 heatmap panels + narrow colorbar column
fig = plt.figure(figsize=(7.87, 5.51), facecolor='white')   # ~20 cm × 14 cm
gs  = gridspec.GridSpec(1, 4,
                        left=0.13, right=0.90, top=0.93, bottom=0.08,
                        wspace=0.08,
                        width_ratios=[1, 1, 1, 0.045])

axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax  = fig.add_subplot(gs[0, 3])

norm = Normalize(vmin=0, vmax=0.5)   # data peaks ~0.3; cap at 0.5 to show variation

for col, (fma, title) in enumerate(zip(FMA_LEVELS, PANEL_TITLES)):
    ax  = axes[col]
    mat = data[fma]                   # (n_muscles, 100)
    im  = ax.imshow(mat, aspect='auto', cmap='viridis', norm=norm,
                    origin='upper', interpolation='nearest')

    ax.set_title(title, fontsize=7.5, pad=4, fontweight='bold')
    ax.set_xlabel('Frame', fontsize=7)
    ax.set_xticks([0, 25, 50, 75, 99])
    ax.set_xticklabels(['0', '25', '50', '75', '100'], fontsize=6)
    ax.tick_params(axis='x', length=2, width=0.5)

    # Draw thin white lines between muscle groups
    boundary = 0
    for grp, size in GROUP_SIZES.items():
        boundary += size
        if boundary < n_muscles:
            ax.axhline(boundary - 0.5, color='white', lw=0.6, alpha=0.8)

    # Y-axis: muscle names on leftmost panel only
    if col == 0:
        ax.set_yticks(range(n_muscles))
        ax.set_yticklabels(MUSCLE_ORDER, fontsize=5.0)
        ax.tick_params(axis='y', length=0)

        # Group labels on far left
        boundary = 0
        for grp, size in GROUP_SIZES.items():
            mid = boundary + size / 2 - 0.5
            ax.text(-0.26, mid, grp, transform=ax.get_yaxis_transform(),
                    fontsize=5, ha='center', va='center', rotation=90,
                    color='#444444')
            boundary += size
    else:
        ax.set_yticks([])

    # box outline
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
        sp.set_color('#444444')

# ── Colorbar ──────────────────────────────────────────────────────────────────
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Activation', fontsize=7, labelpad=4)
cbar.ax.tick_params(labelsize=6, length=2, width=0.5)
cbar.outline.set_linewidth(0.5)

for fmt in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT_DIR, f'muscle_activation_heatmap.{fmt}'),
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved muscle_activation_heatmap.{png,pdf}')
print(f'Sessions used:')
for fma in FMA_LEVELS:
    print(f'  FMA {fma}: {ID_DIR}/FMA_{fma}/activations.csv')
