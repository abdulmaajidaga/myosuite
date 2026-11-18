#!/usr/bin/env python3
"""Figure 5: CCI gradient scatter plot — CCI vs FMA for all 51 generated sessions.

Data: output/generated/id/FMA_{N}/effort_metrics.json  (field: CCI)
Output: figures/cci_gradient.{png,pdf}
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

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

# ── Severity group colours ────────────────────────────────────────────────────
GROUP_COLOURS = {
    'Severe (16\u201320)':   '#D62728',
    'Moderate (21\u201340)': '#FF9800',
    'Mild (41\u201355)':     '#FDD835',
    'Healthy (56\u201366)':  '#2CA02C',
}

def severity_group(fma):
    if fma <= 20:  return 'Severe (16\u201320)'
    if fma <= 40:  return 'Moderate (21\u201340)'
    if fma <= 55:  return 'Mild (41\u201355)'
    return 'Healthy (56\u201366)'

# ── Load data ─────────────────────────────────────────────────────────────────
records = []
for fma in range(16, 67):
    path = os.path.join(ID_DIR, f'FMA_{fma}', 'effort_metrics.json')
    if not os.path.exists(path):
        print(f'MISSING: {path}')
        continue
    with open(path) as f:
        m = json.load(f)
    records.append({'fma': fma, 'cci': m['CCI'], 'group': severity_group(fma)})

df = pd.DataFrame(records)
print(f'Loaded {len(df)} sessions, FMA {df.fma.min()}\u2013{df.fma.max()}')

# ── Regression ────────────────────────────────────────────────────────────────
slope, intercept, r, p, se = stats.linregress(df['fma'], df['cci'])
x_fit = np.linspace(16, 66, 200)
y_fit = slope * x_fit + intercept

# 95% CI
n   = len(df)
x_m = df['fma'].mean()
s_e = np.sqrt(np.sum((df['cci'] - (slope * df['fma'] + intercept))**2) / (n - 2))
t95 = stats.t.ppf(0.975, df=n - 2)
se_fit = s_e * np.sqrt(1/n + (x_fit - x_m)**2 / np.sum((df['fma'] - x_m)**2))
ci_lo = y_fit - t95 * se_fit
ci_hi = y_fit + t95 * se_fit

spearman_r, spearman_p = stats.spearmanr(df['fma'], df['cci'])
print(f'Spearman r={spearman_r:.3f}, p={spearman_p:.2e}')

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.72, 3.15))   # ~12 cm × 8 cm
fig.patch.set_facecolor('white')

ax.fill_between(x_fit, ci_lo, ci_hi, color='#BDBDBD', alpha=0.35, zorder=1)
ax.plot(x_fit, y_fit, color='#424242', lw=1.2, zorder=2)

for grp, col in GROUP_COLOURS.items():
    sub = df[df['group'] == grp]
    ax.scatter(sub['fma'], sub['cci'], s=32, color=col, alpha=0.85,
               zorder=3, label=grp, edgecolors='none')

ax.set_xlabel('FMA-UE score', fontsize=8)
ax.set_ylabel('Co-Contraction Index (CCI)', fontsize=8)
ax.set_xlim(14, 68)
y_pad = 0.015
ax.set_ylim(df['cci'].min() - y_pad, df['cci'].max() + y_pad)
ax.set_xticks(range(16, 67, 10))
ax.tick_params(labelsize=7, length=2.5, width=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.6)
ax.spines['bottom'].set_linewidth(0.6)
ax.grid(linestyle='--', alpha=0.28, linewidth=0.4, color='#888888')

# annotation box
p_str = '$p < 0.0001$' if spearman_p < 1e-4 else f'$p = {spearman_p:.4f}$'
ann = f'$\\rho = {spearman_r:.3f}$\n{p_str}'
ax.text(0.97, 0.97, ann, transform=ax.transAxes,
        fontsize=7, va='top', ha='right', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#BDBDBD', linewidth=0.6))

ax.legend(fontsize=6.5, framealpha=0.7, handlelength=1.2,
          borderpad=0.4, loc='lower right',
          title='Severity group', title_fontsize=6.5)

fig.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.14)
for fmt in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT_DIR, f'cci_gradient.{fmt}'),
                dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('Saved cci_gradient.{png,pdf}')
print(f'Data source: {ID_DIR}/FMA_{{16..66}}/effort_metrics.json')
