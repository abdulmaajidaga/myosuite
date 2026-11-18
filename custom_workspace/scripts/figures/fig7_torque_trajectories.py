#!/usr/bin/env python3
"""Figure 7: Joint torque trajectories across all 51 sessions.

Data: output/generated/id/FMA_{N}/torques.csv  (100 frames × 7 DOFs)
Output: figures/torque_trajectories.{png,pdf}
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.cm as cm

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'font.size':          8,
    'axes.unicode_minus': False,
    'axes.linewidth':     0.7,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
ID_DIR     = os.path.join(ROOT, 'output', 'generated', 'id')
OUT_DIR    = os.path.join(ROOT, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# 5 primary solved DOFs (deviation and flexion have zero ROM and are excluded)
DOFS = ['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion', 'pro_sup']
DOF_LABELS = [
    'Elevation angle ($\\tau$)',
    'Shoulder elevation ($\\tau$)',
    'Shoulder rotation ($\\tau$)',
    'Elbow flexion ($\\tau$)',
    'Pro/supination ($\\tau$)',
]

CMAP = cm.RdYlGn
FMA_MIN, FMA_MAX = 16, 66
norm = mcolors.Normalize(vmin=FMA_MIN, vmax=FMA_MAX)

# ── Load all torques ──────────────────────────────────────────────────────────
sessions = []
for fma in range(16, 67):
    path = os.path.join(ID_DIR, f'FMA_{fma}', 'torques.csv')
    if not os.path.exists(path):
        print(f'MISSING: {path}')
        continue
    df = pd.read_csv(path)
    if 'time' in df.columns:
        df = df.drop(columns=['time'])
    sessions.append((fma, df))

print(f'Loaded {len(sessions)} sessions')

frames = np.arange(100)

# ── Figure: 2×3 grid (5 DOFs + empty bottom-right) ───────────────────────────
fig = plt.figure(figsize=(6.30, 4.20), facecolor='white')
gs  = gridspec.GridSpec(2, 3,
                        left=0.09, right=0.86, top=0.95, bottom=0.10,
                        hspace=0.52, wspace=0.40)
axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

for idx, (dof, label) in enumerate(zip(DOFS, DOF_LABELS)):
    ax = axes[idx]
    for fma, df in sessions:
        if dof not in df.columns:
            continue
        raw = df[dof].values
        x_old = np.linspace(0, 99, len(raw))
        x_new = np.arange(100)
        vals = np.interp(x_new, x_old, raw)
        ax.plot(x_new, vals, color=CMAP(norm(fma)), lw=0.7, alpha=0.65)

    ax.set_title(label, fontsize=7, pad=3)
    ax.set_xlabel('Frame', fontsize=6.5)
    ax.set_ylabel('Torque (Nm)', fontsize=6.5)
    ax.set_xlim(0, 99)
    ax.set_xticks([0, 50, 99])
    ax.set_xticklabels(['0', '50', '100'], fontsize=6)
    ax.tick_params(labelsize=6, length=2, width=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.grid(linestyle='--', alpha=0.25, linewidth=0.35, color='#888888')

# hide unused 6th subplot
axes[5].set_visible(False)

# ── Shared colorbar ───────────────────────────────────────────────────────────
cbar_ax = fig.add_axes([0.88, 0.10, 0.022, 0.85])
sm = cm.ScalarMappable(cmap=CMAP, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('FMA score', fontsize=7, labelpad=4)
cbar.set_ticks([16, 25, 35, 45, 55, 66])
cbar.ax.tick_params(labelsize=6, length=2, width=0.5)
cbar.outline.set_linewidth(0.5)

for fmt in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT_DIR, f'torque_trajectories.{fmt}'),
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved torque_trajectories.{png,pdf}')
print(f'Data source: {ID_DIR}/FMA_{{16..66}}/torques.csv')
