#!/usr/bin/env python3
"""
Publication-quality trunk compensation figure (2×2).

Panels:
  (a) Total trunk displacement      — boxplot, Healthy vs Stroke
  (b) Trunk/Wrist ratio             — boxplot, Healthy vs Stroke
  (c) Total trunk displacement vs FMA — scatter + OLS trend
  (d) Trunk/Wrist ratio vs FMA      — scatter + OLS trend

Output:
  figures/trunk_compensation.png  (300 DPI)
  figures/trunk_compensation.pdf  (vector)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE  = os.path.join(SCRIPT_DIR, '..', '..', '..')
CSV_PATH   = os.path.join(WORKSPACE, 'output', 'generated', 'plots', 'trunk_compensation_data.csv')
OUT_DIR    = os.path.join(WORKSPACE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size':        8,
    'axes.unicode_minus': False,
    'axes.linewidth':   0.8,
})

GREEN  = '#2CA02C'   # healthy
RED    = '#D62728'   # stroke
GREY   = '#888888'

def sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.7)
    ax.spines['bottom'].set_linewidth(0.7)
    ax.tick_params(labelsize=7, length=3, width=0.7)
    ax.grid(axis='y', linestyle='--', alpha=0.25, linewidth=0.5)

def add_sig_bar(ax, x1, x2, y, p, dy=0.03):
    """Draw a significance bracket between two box positions."""
    ymax = ax.get_ylim()[1]
    h = ymax * dy
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.8, color='#333333')
    label = sig_label(p)
    ax.text((x1+x2)/2, y+h*1.1, label, ha='center', va='bottom',
            fontsize=7, color='#333333')

# ── Load data ─────────────────────────────────────────────────────────────────
df      = pd.read_csv(CSV_PATH)
healthy = df[df['group'] == 'Healthy']
stroke  = df[df['group'] == 'Stroke']

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(6.30, 5.20),
                         gridspec_kw={'hspace': 0.48, 'wspace': 0.38})
fig.patch.set_facecolor('white')

PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)']
for ax, lbl in zip(axes.flat, PANEL_LABELS):
    ax.text(-0.14, 1.04, lbl, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top')

# ── (a) Total trunk displacement — boxplot ────────────────────────────────────
ax = axes[0, 0]
bdata = [healthy['trunk_total_disp'].values, stroke['trunk_total_disp'].values]
bp = ax.boxplot(bdata, patch_artist=True, widths=0.45,
                medianprops=dict(color='white', linewidth=1.5),
                whiskerprops=dict(linewidth=0.8, color=GREY),
                capprops=dict(linewidth=0.8, color=GREY),
                flierprops=dict(marker='o', markersize=3,
                                markerfacecolor=GREY, markeredgewidth=0.5,
                                alpha=0.6))
bp['boxes'][0].set_facecolor(GREEN); bp['boxes'][0].set_alpha(0.75)
bp['boxes'][1].set_facecolor(RED);   bp['boxes'][1].set_alpha(0.75)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Healthy', 'Stroke'], fontsize=7.5)
ax.set_ylabel('Displacement (mm)', fontsize=7.5)
ax.set_title('Total Trunk Displacement', fontsize=8, pad=4)
style_ax(ax)
_, p = stats.ttest_ind(healthy['trunk_total_disp'], stroke['trunk_total_disp'])
ybar = max(bdata[0].max(), bdata[1].max()) * 1.05
add_sig_bar(ax, 1, 2, ybar, p)
ax.set_ylim(top=ybar * 1.18)

# ── (b) Trunk/Wrist ratio — boxplot ──────────────────────────────────────────
ax = axes[0, 1]
bdata = [healthy['trunk_wrist_ratio'].values, stroke['trunk_wrist_ratio'].values]
bp = ax.boxplot(bdata, patch_artist=True, widths=0.45,
                medianprops=dict(color='white', linewidth=1.5),
                whiskerprops=dict(linewidth=0.8, color=GREY),
                capprops=dict(linewidth=0.8, color=GREY),
                flierprops=dict(marker='o', markersize=3,
                                markerfacecolor=GREY, markeredgewidth=0.5,
                                alpha=0.6))
bp['boxes'][0].set_facecolor(GREEN); bp['boxes'][0].set_alpha(0.75)
bp['boxes'][1].set_facecolor(RED);   bp['boxes'][1].set_alpha(0.75)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Healthy', 'Stroke'], fontsize=7.5)
ax.set_ylabel('Trunk / Wrist ratio', fontsize=7.5)
ax.set_title('Trunk–Wrist Compensation Ratio', fontsize=8, pad=4)
style_ax(ax)
_, p = stats.ttest_ind(healthy['trunk_wrist_ratio'], stroke['trunk_wrist_ratio'])
ybar = max(bdata[0].max(), bdata[1].max()) * 1.08
add_sig_bar(ax, 1, 2, ybar, p)
ax.set_ylim(top=ybar * 1.28)

# ── (c) Total trunk displacement vs FMA ───────────────────────────────────────
ax = axes[1, 0]
ax.scatter(healthy['fma_score'], healthy['trunk_total_disp'],
           color=GREEN, s=18, alpha=0.75, zorder=3, linewidths=0,
           label='Healthy')
ax.scatter(stroke['fma_score'], stroke['trunk_total_disp'],
           color=RED, s=18, alpha=0.75, zorder=3, linewidths=0,
           label='Stroke')
r, p = stats.pearsonr(df['fma_score'], df['trunk_total_disp'])
z = np.polyfit(df['fma_score'], df['trunk_total_disp'], 1)
xs = np.linspace(df['fma_score'].min()-1, df['fma_score'].max()+1, 100)
ax.plot(xs, np.polyval(z, xs), '--', color='#444444', lw=0.9, zorder=2)
ax.set_xlabel('FMA-UE score', fontsize=7.5)
ax.set_ylabel('Displacement (mm)', fontsize=7.5)
ax.set_title('Total Trunk Displacement vs FMA', fontsize=8, pad=4)
ax.text(0.97, 0.05, f'Pearson $r = {r:.2f}$',
        transform=ax.transAxes, fontsize=6.5,
        ha='right', va='bottom', color='#333333',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='none', alpha=0.8))
ax.legend(fontsize=6.5, framealpha=0.6, handlelength=1.2,
          borderpad=0.4, loc='upper right')
style_ax(ax)

# ── (d) Trunk/Wrist ratio vs FMA ──────────────────────────────────────────────
ax = axes[1, 1]
ax.scatter(healthy['fma_score'], healthy['trunk_wrist_ratio'],
           color=GREEN, s=18, alpha=0.75, zorder=3, linewidths=0,
           label='Healthy')
ax.scatter(stroke['fma_score'], stroke['trunk_wrist_ratio'],
           color=RED, s=18, alpha=0.75, zorder=3, linewidths=0,
           label='Stroke')
r, p = stats.pearsonr(df['fma_score'], df['trunk_wrist_ratio'])
z = np.polyfit(df['fma_score'], df['trunk_wrist_ratio'], 1)
xs = np.linspace(df['fma_score'].min()-1, df['fma_score'].max()+1, 100)
ax.plot(xs, np.polyval(z, xs), '--', color='#444444', lw=0.9, zorder=2)
ax.set_xlabel('FMA-UE score', fontsize=7.5)
ax.set_ylabel('Trunk / Wrist ratio', fontsize=7.5)
ax.set_title('Trunk–Wrist Ratio vs FMA', fontsize=8, pad=4)
ax.text(0.97, 0.05, f'Pearson $r = {r:.2f}$',
        transform=ax.transAxes, fontsize=6.5,
        ha='right', va='bottom', color='#333333',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                  edgecolor='none', alpha=0.8))
ax.legend(fontsize=6.5, framealpha=0.6, handlelength=1.2,
          borderpad=0.4, loc='upper right')
ylo, _ = ax.get_ylim()
ax.set_ylim(ylo, 0.285)
style_ax(ax)

# ── Save ─────────────────────────────────────────────────────────────────────
for fmt in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT_DIR, f'trunk_compensation.{fmt}'),
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved trunk_compensation.{png,pdf}')
