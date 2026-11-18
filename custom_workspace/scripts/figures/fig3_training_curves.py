#!/usr/bin/env python3
"""Figure 10: D_base training curves — two publication-quality figures.

Fig 10a: 1×3 — Total loss (log), Position MSE, Velocity MSE
Fig 10b: 1×3 — Acceleration MSE, KL divergence, Dynamics corr. loss

Uses test/output/D_base_rerun/history.csv (per-component logging).

Outputs:
    figures/training_curves_a.{png,pdf}
    figures/training_curves_b.{png,pdf}
"""

import os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'font.size':          8,
    'axes.unicode_minus': False,
    'axes.linewidth':     0.7,
})

BLUE   = '#1F77B4'
ORANGE = '#FF7F0E'

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
WORKSPACE   = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DEFAULT_CSV = os.path.join(WORKSPACE, 'test', 'output', 'D_base_rerun', 'history.csv')
OUT_DIR     = os.path.join(WORKSPACE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
df = pd.read_csv(csv_path)
if 'train_loss' not in df.columns and 'train' in df.columns:
    df = df.rename(columns={'train': 'train_loss', 'val': 'val_loss'})

print(f'Loaded: {csv_path}  ({len(df)} rows, cols: {list(df.columns)})')

epochs    = df['epoch'].values
train_tot = df['train_loss'].values
val_tot   = df['val_loss'].values
best_idx  = int(np.argmin(val_tot))
best_ep   = int(epochs[best_idx])
best_val  = float(val_tot[best_idx])


def style_ax(ax, log_y=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(labelsize=6.5, length=2.5, width=0.6)
    ax.grid(linestyle='--', alpha=0.22, linewidth=0.4)
    if log_y:
        ax.set_yscale('log')


def plot_pair(ax, y_tr, y_va, title, ylabel, log_y=False):
    style_ax(ax, log_y)
    ax.plot(epochs, y_tr, color=BLUE,   lw=1.2, label='Train')
    ax.plot(epochs, y_va, color=ORANGE, lw=1.2, linestyle='--', label='Val')
    bi = int(np.argmin(y_va))
    ax.axvline(epochs[bi], color='#888888', lw=0.7, linestyle=':', zorder=2)
    ax.annotate(f'ep {int(epochs[bi])}',
                xy=(epochs[bi], y_va[bi]), xytext=(4, 4),
                textcoords='offset points', fontsize=5.5, color='#555555')
    ax.set_title(title, fontsize=7.5, pad=4)
    ax.set_xlabel('Epoch', fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.legend(fontsize=6, framealpha=0.6, loc='upper right',
              handlelength=1.4, borderpad=0.35)
    # keep y-axis ticks to 4 max to avoid crowding
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune='both'))


COMPONENT_COLS = ['train_pos', 'train_vel', 'train_acc', 'train_kl', 'train_dyn']
HAS_COMPONENTS = all(c in df.columns for c in COMPONENT_COLS)

if not HAS_COMPONENTS:
    print('WARNING: per-component columns not found — single-panel fallback')
    fig, ax = plt.subplots(figsize=(4.72, 2.76))
    fig.patch.set_facecolor('white')
    ax.plot(epochs, train_tot, color=BLUE,   lw=1.3, label='Train')
    ax.plot(epochs, val_tot,   color=ORANGE, lw=1.3, linestyle='--', label='Val')
    ax.axvline(best_ep, color='#777777', lw=0.8, linestyle=':', zorder=2)
    ax.legend(fontsize=7, framealpha=0.6, loc='upper right', handlelength=1.5, borderpad=0.4)
    style_ax(ax, log_y=True)
    ax.set_xlabel('Epoch', fontsize=7)
    ax.set_ylabel('Loss', fontsize=7)
    ax.set_title(f'Best val: {best_val:.4f} at ep {best_ep}', fontsize=7, pad=3)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.88, bottom=0.18)
    for fmt in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'training_curves_a.{fmt}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved training_curves_a.{png,pdf}  [single-panel fallback]')
    raise SystemExit


def make_figure(panels, filename):
    """
    panels: list of (y_train, y_val, title, ylabel, log_y)
    """
    n = len(panels)
    fig = plt.figure(figsize=(6.30, 2.70), facecolor='white')
    gs = gridspec.GridSpec(
        1, n,
        left=0.10, right=0.97,
        top=0.82, bottom=0.20,
        wspace=0.48,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]

    panel_labels = ['(a)', '(b)', '(c)']
    for ax, lbl, (y_tr, y_va, title, ylabel, log_y) in zip(axes, panel_labels, panels):
        plot_pair(ax, y_tr, y_va, title, ylabel, log_y)
        ax.text(-0.18, 1.10, lbl, transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top')

    for fmt in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f'{filename}.{fmt}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved {filename}.{{png,pdf}}')


# ── Fig 10a ───────────────────────────────────────────────────────────────────
make_figure([
    (train_tot,              val_tot,                  'Total loss',       'Loss',      True),
    (df['train_pos'].values, df['val_pos'].values,     'Position loss',    'MSE (pos)', False),
    (df['train_vel'].values, df['val_vel'].values,     'Velocity loss',    'MSE (vel)', False),
], 'training_curves_a')

# ── Fig 10b ───────────────────────────────────────────────────────────────────
make_figure([
    (df['train_acc'].values, df['val_acc'].values,     'Acceleration loss', 'MSE (acc)',   False),
    (df['train_kl'].values,  df['val_kl'].values,      'KL divergence',     'KLD',         False),
    (df['train_dyn'].values, df['val_dyn'].values,     'Dynamics corr.',    'Corr. loss',  False),
], 'training_curves_b')
