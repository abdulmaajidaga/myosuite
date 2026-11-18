#!/usr/bin/env python3
"""
Figure 4: SMOTE augmentation pipeline (publication quality).

Panel (a): Within-class k-NN expansion — original vs synthetic at FMA 20 and FMA 66.
Panel (b): Cross-class FMA-proportional blending — mean ± 1 SD per FMA level.

Uses augmented_smote_test dataset.
Originals: filenames without 'smote_' prefix.
Synthetics: filenames starting with 'smote_'.

Output:
    figures/aug_smote_phases.png  (300 DPI)
    figures/aug_smote_phases.pdf  (vector)
"""

import os
import re
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
    'axes.linewidth':     0.8,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
SMOTE_DIR  = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'augmented_smote_test')
OUT_DIR    = os.path.join(ROOT, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

FMA_PATTERN = re.compile(r'FMA(\d+)', re.IGNORECASE)

def files_for_fma(fma):
    all_f = [f for f in os.listdir(SMOTE_DIR)
             if f.endswith('.csv') and f'FMA{fma}' in f]
    orig  = [f for f in all_f if not f.startswith('smote_')]
    synth = [f for f in all_f if f.startswith('smote_')]
    return orig, synth

def load_wy(fname):
    df = pd.read_csv(os.path.join(SMOTE_DIR, fname))
    if 'Wr_y' not in df.columns:
        return None
    arr = df['Wr_y'].values[:100]
    return arr if len(arr) >= 10 else None

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    ax.tick_params(labelsize=7, length=3, width=0.7)
    ax.grid(linestyle='--', alpha=0.22, linewidth=0.45)

# ── colours ───────────────────────────────────────────────────────────────────
FMA_SHOW   = [20, 30, 40, 50, 66]
CMAP_VALS  = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(FMA_SHOW)))
COLOR_20   = CMAP_VALS[0]
COLOR_66   = CMAP_VALS[-1]

rng = np.random.default_rng(42)

fig, axes = plt.subplots(1, 2, figsize=(6.30, 2.90),
                         gridspec_kw={'wspace': 0.38})
fig.patch.set_facecolor('white')

PANEL_LABELS = ['(a)', '(b)']
for ax, lbl in zip(axes, PANEL_LABELS):
    ax.text(-0.14, 1.06, lbl, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top')

# ── (a) Within-class k-NN expansion ──────────────────────────────────────────
ax = axes[0]
for fma, color in zip([20, 66], [COLOR_20, COLOR_66]):
    orig, synth = files_for_fma(fma)
    # up to 3 originals (solid) and 6 synthetics (dashed)
    orig_sel  = list(rng.choice(orig,  size=min(3,  len(orig)),  replace=False))
    synth_sel = list(rng.choice(synth, size=min(6,  len(synth)), replace=False))
    first = True
    for fn in orig_sel:
        wy = load_wy(fn)
        if wy is None: continue
        ax.plot(wy, color=color, lw=1.4, alpha=0.9,
                label=f'FMA {fma}' if first else '_')
        first = False
    for fn in synth_sel:
        wy = load_wy(fn)
        if wy is None: continue
        ax.plot(wy, color=color, lw=0.7, ls='--', alpha=0.40)

ax.set_title('Within-class $k$-NN expansion', fontsize=7.5, pad=3)
ax.set_xlabel('Frame', fontsize=7.5)
ax.set_ylabel('Wrist $Y$ displacement (mm)', fontsize=7.5)
ax.legend(fontsize=6.5, framealpha=0.6, handlelength=1.4,
          borderpad=0.4, loc='best',
          title='solid = original\ndashed = synthetic',
          title_fontsize=5.5)
style_ax(ax)

# ── (b) Cross-class blending (mean ± 1 SD) ───────────────────────────────────
ax = axes[1]
for fma, color in zip(FMA_SHOW, CMAP_VALS):
    _, synth = files_for_fma(fma)
    chosen = list(rng.choice(synth, size=min(30, len(synth)), replace=False))
    trajs = [wy for fn in chosen if (wy := load_wy(fn)) is not None and len(wy) == 100]
    if len(trajs) < 3:
        continue
    arr = np.array(trajs)
    mn, sd = arr.mean(0), arr.std(0)
    x = np.arange(100)
    ax.fill_between(x, mn - sd, mn + sd, alpha=0.15, color=color)
    ax.plot(x, mn, color=color, lw=1.4, label=f'FMA {fma}')

ax.set_title('Cross-class FMA-proportional blending\n(mean $\\pm$ 1 SD)', fontsize=7.5, pad=3)
ax.set_xlabel('Frame', fontsize=7.5)
ax.set_ylabel('Wrist $Y$ displacement (mm)', fontsize=7.5)
ax.legend(fontsize=6.5, framealpha=0.6, handlelength=1.4,
          borderpad=0.4, ncol=2, loc='best')
style_ax(ax)

# ── Save ─────────────────────────────────────────────────────────────────────
for fmt in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT_DIR, f'aug_smote_phases.{fmt}'),
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved aug_smote_phases.{png,pdf}')
