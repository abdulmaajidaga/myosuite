#!/usr/bin/env python3
"""
Fig 8a: Timeseries + boxplots  (2×4, saved as training_verification_a.png/pdf)
Fig 8b: FMA scatter plots      (1×2, saved as training_verification_b.png/pdf)

Sources:
  processed/         - real recordings (absolute -> delta)
  augmented_smote_test/ - SMOTE (already delta)
  test/output/D_base/csv_n10/ - generated (absolute -> delta)
  output/scores.csv
"""

import os, sys, glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.cm as mpl_cm
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'font.size':          8,
    'axes.unicode_minus': False,
    'axes.linewidth':     0.7,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

PROC_DIR   = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'processed')
AUG_DIR    = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'augmented_smote_test')
GEN_DIR    = os.path.join(ROOT, 'test', 'output', 'D_base', 'csv_n10')
SCORES_F   = os.path.join(ROOT, 'output', 'scores.csv')
OUT_DIR    = os.path.join(ROOT, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

ALL_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
            'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z',
            'Trunk_x','Trunk_y','Trunk_z']

C_HEALTHY = '#2CA02C'
C_STROKE  = '#D62728'
C_GEN_MK  = '#1F77B4'
C_AUG     = '#9467BD'
RNG = np.random.default_rng(42)

GEN_BINS = [
    ('16--20', 16, 20, '#e6550d'),
    ('21--30', 21, 30, '#fd8d3c'),
    ('31--40', 31, 40, '#fdae6b'),
    ('41--50', 41, 50, '#a1d99b'),
    ('51--60', 51, 60, '#74c476'),
    ('61--66', 61, 66, '#31a354'),
]


# ── helpers ───────────────────────────────────────────────────────────────────
def to_delta(arr): return arr - arr[0:1]

def load_scores():
    df = pd.read_csv(SCORES_F)
    return {r['filename'].replace('.mot','').replace('.csv',''): int(r['fma_score'])
            for _, r in df.iterrows()}

def ensure_cols(df):
    for c in ALL_COLS:
        if c not in df.columns: df[c] = 0.0
    return df

def load_real(scores_map):
    H, S, hs, ss = [], [], [], []
    for f in sorted(glob.glob(os.path.join(PROC_DIR, '*.csv'))):
        stem = os.path.splitext(os.path.basename(f))[0]
        try:
            df = ensure_cols(pd.read_csv(f))
            arr = to_delta(df[ALL_COLS].values.astype(float))
            if stem.startswith('S'):
                S.append(arr); ss.append(scores_map.get(stem, 19))
            else:
                H.append(arr); hs.append(66)
        except Exception: continue
    return np.array(H), hs, np.array(S), ss

def load_generated():
    gen = {}
    for f in sorted(glob.glob(os.path.join(GEN_DIR, 'FMA_*.csv'))):
        fma = int(os.path.basename(f).split('_')[1].replace('.csv',''))
        try:
            df = ensure_cols(pd.read_csv(f))
            gen[fma] = to_delta(df[ALL_COLS].values.astype(float))
        except Exception: continue
    return gen

def load_smote(n=40):
    sampled = {}
    all_f = glob.glob(os.path.join(AUG_DIR, '*.csv'))
    by_fma = {}
    for f in all_f:
        bn = os.path.basename(f)
        if 'FMA' not in bn: continue
        try:
            fma = int(bn.split('FMA')[1].replace('.csv',''))
            by_fma.setdefault(fma, []).append(f)
        except: continue
    for fma, files in sorted(by_fma.items()):
        chosen = list(RNG.choice(files, min(n, len(files)), replace=False))
        for f in chosen:
            try:
                df = ensure_cols(pd.read_csv(f))
                arr = df[ALL_COLS].values.astype(float)
                sampled.setdefault(fma, []).append(arr)  # already delta
            except: continue
    return sampled

def metrics(arr):
    wy    = arr[:, 7]
    wr3   = arr[:, 6:9]
    trunk = arr[:, 12:15]
    vel   = np.linalg.norm(np.diff(wr3, axis=0), axis=1)
    acc   = np.linalg.norm(np.diff(vel))
    T     = max(len(wy) - 1, 1)
    return {
        'wrist_range_y': np.ptp(wy),
        'peak_velocity': np.max(vel),
        'trunk_disp':    np.max(np.linalg.norm(trunk - trunk[0:1], axis=1)),
        'norm_jerk':     float(np.sum(np.diff(vel)**2)) / T,
    }

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(labelsize=6.5, length=2.5, width=0.6)
    ax.grid(linestyle='--', alpha=0.20, linewidth=0.4)


# ── load data ─────────────────────────────────────────────────────────────────
print('Loading data...')
scores_map = load_scores()
H, hs, S, ss = load_real(scores_map)
gen         = load_generated()
smote       = load_smote()
print(f'  Healthy: {len(H)}, Stroke: {len(S)}, Generated FMA levels: {len(gen)}, SMOTE bins: {len(smote)}')

fma_norm = mpl_colors.Normalize(vmin=16, vmax=66)
fma_cmap = mpl_cm.RdYlGn
t100     = np.linspace(0, 100, 100)


# ═════════════════════════════════════════════════════════════════════════════
# FIG 8a: 2×4 timeseries + boxplots
# ═════════════════════════════════════════════════════════════════════════════
fig_a, axes_a = plt.subplots(2, 4, figsize=(6.30*1.6, 4.80),
                              gridspec_kw={'hspace': 0.52, 'wspace': 0.38})
fig_a.patch.set_facecolor('white')

PANEL_LABELS = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
for ax, lbl in zip(axes_a.flat, PANEL_LABELS):
    ax.text(-0.16, 1.07, lbl, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top')

# ── row 1: timeseries ────────────────────────────────────────────────────────
TS_SPECS = [
    ('Wrist $Y$ (reach)',    'Position (mm)',      7),
    ('Wrist $Z$ (height)',   'Position (mm)',      8),
    ('Trunk displacement',   'Displacement (mm)',  None),
    ('Wrist velocity',       'Speed (mm/frame)',   None),
]

def trunk_disp_curve(arr):
    t = arr[:, 12:15]
    return np.linalg.norm(t - t[0:1], axis=1)

def vel_curve(arr):
    return np.concatenate([[0], np.linalg.norm(np.diff(arr[:, 6:9], axis=0), axis=1)])

for col, (title, ylabel, ci) in enumerate(TS_SPECS):
    ax = axes_a[0, col]
    extract = (lambda a, ci=ci: a[:, ci]) if ci is not None else \
              (trunk_disp_curve if col == 2 else vel_curve)

    h_curves = np.array([extract(a) for a in H])
    s_curves = np.array([extract(a) for a in S])
    h_mn, h_sd = h_curves.mean(0), h_curves.std(0)
    s_mn, s_sd = s_curves.mean(0), s_curves.std(0)

    ax.fill_between(t100, h_mn-h_sd, h_mn+h_sd, color=C_HEALTHY, alpha=0.18)
    ax.plot(t100, h_mn, color=C_HEALTHY, lw=1.3, label='Healthy')
    ax.fill_between(t100, s_mn-s_sd, s_mn+s_sd, color=C_STROKE,  alpha=0.18)
    ax.plot(t100, s_mn, color=C_STROKE,  lw=1.3, label='Stroke')

    for fma in sorted(gen.keys()):
        curve = extract(gen[fma])
        ax.plot(t100[:len(curve)], curve[:100],
                color=fma_cmap(fma_norm(fma)), lw=0.8, alpha=0.7, ls='--')

    ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_xlabel('% Motion', fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    style_ax(ax)

# legend + colorbar on first panel
leg_handles = [
    Line2D([0],[0], color=C_HEALTHY, lw=1.3, label='Healthy (mean $\\pm$1 SD)'),
    Line2D([0],[0], color=C_STROKE,  lw=1.3, label='Stroke (mean $\\pm$1 SD)'),
    Line2D([0],[0], color='gray',    lw=0.8, ls='--', label='Generated'),
]
axes_a[0, 0].legend(handles=leg_handles, fontsize=5.5, framealpha=0.6,
                    loc='lower right', handlelength=1.4, borderpad=0.4)

# ── row 2: boxplots ──────────────────────────────────────────────────────────
BP_SPECS = [
    ('Wrist range $Y$',    'Range (mm)',       'wrist_range_y'),
    ('Peak velocity',      'Speed (mm/frame)', 'peak_velocity'),
    ('Trunk displacement', 'Max disp (mm)',    'trunk_disp'),
    ('Normalised jerk',    'Jerk',             'norm_jerk'),
]

for col, (title, ylabel, key) in enumerate(BP_SPECS):
    ax = axes_a[1, col]
    h_vals = [metrics(a)[key] for a in H]
    s_vals = [metrics(a)[key] for a in S]
    data, colors, labels = [h_vals, s_vals], [C_HEALTHY, C_STROKE], ['Healthy', 'Stroke']

    for lbl, lo, hi, color in GEN_BINS:
        vals = [metrics(a)[key] for fma, arrays in smote.items()
                if lo <= fma <= hi for a in arrays]
        if vals:
            data.append(vals); colors.append(color); labels.append(f'G {lbl}')

    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color='white', lw=1.2),
                    whiskerprops=dict(lw=0.7, color='#555'),
                    capprops=dict(lw=0.7, color='#555'),
                    flierprops=dict(marker='o', ms=2, alpha=0.4, markeredgewidth=0))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.72)

    ax.set_xticks(range(1, len(labels)+1))
    ax.set_xticklabels(labels, fontsize=5.5, rotation=30, ha='right')
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_ylabel(ylabel, fontsize=7)
    style_ax(ax)
    ax.grid(axis='y', linestyle='--', alpha=0.20, linewidth=0.4)
    ax.grid(axis='x', visible=False)

for fmt in ('png', 'pdf'):
    fig_a.savefig(os.path.join(OUT_DIR, f'training_verification_a.{fmt}'),
                  dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig_a)
print('Saved training_verification_a.{png,pdf}')


# ═════════════════════════════════════════════════════════════════════════════
# FIG 8b: 1×2 FMA scatter plots
# ═════════════════════════════════════════════════════════════════════════════
fig_b, axes_b = plt.subplots(1, 2, figsize=(6.30, 2.90),
                              gridspec_kw={'wspace': 0.38})
fig_b.patch.set_facecolor('white')

for ax, lbl in zip(axes_b, ['(a)', '(b)']):
    ax.text(-0.14, 1.07, lbl, transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top')

SCATTER_SPECS = [
    ('FMA vs wrist range', 'Wrist range $Y$ (mm)', 'wrist_range_y'),
    ('FMA vs trunk compensation', 'Trunk displacement (mm)', 'trunk_disp'),
]

for ax, (title, ylabel, key) in zip(axes_b, SCATTER_SPECS):
    aug_fmas, aug_vals = [], []
    for fma, arrays in smote.items():
        for arr in arrays:
            aug_fmas.append(fma); aug_vals.append(metrics(arr)[key])

    ax.scatter(aug_fmas, aug_vals, c=C_AUG, s=6, alpha=0.25,
               linewidths=0, label='SMOTE', rasterized=True)

    for arr, fma in zip(H, hs):
        ax.scatter(fma, metrics(arr)[key], c=C_HEALTHY, s=35,
                   edgecolors='white', linewidths=0.5, zorder=5)
    for arr, fma in zip(S, ss):
        ax.scatter(fma, metrics(arr)[key], c=C_STROKE, s=35,
                   edgecolors='white', linewidths=0.5, zorder=5)
    for fma, arr in gen.items():
        ax.scatter(fma, metrics(arr)[key], c=fma_cmap(fma_norm(fma)),
                   s=55, marker='D', edgecolors='#333', linewidths=0.6, zorder=6)

    if aug_fmas:
        slope, intercept, r, *_ = stats.linregress(aug_fmas, aug_vals)
        xs = np.linspace(min(aug_fmas), max(aug_fmas), 100)
        ax.plot(xs, slope*xs + intercept, '--', color='#333', lw=0.9, zorder=4)
        ax.text(0.97, 0.05, f'Pearson $r = {r:.2f}$',
                transform=ax.transAxes, fontsize=6.5,
                ha='right', va='bottom', color='#333',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

    ax.set_title(title, fontsize=7.5, pad=3)
    ax.set_xlabel('FMA-UE score', fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    style_ax(ax)

# shared legend
leg_b = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_AUG,    ms=5,  label='SMOTE'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_HEALTHY, ms=5,  label='Healthy'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_STROKE,  ms=5,  label='Stroke'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='gray',    ms=5,  label='Generated'),
]
fig_b.legend(handles=leg_b, loc='upper center', ncol=4,
             fontsize=6.5, framealpha=0.7, handlelength=1.0,
             borderpad=0.4, bbox_to_anchor=(0.5, 1.01))

for fmt in ('png', 'pdf'):
    fig_b.savefig(os.path.join(OUT_DIR, f'training_verification_b.{fmt}'),
                  dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig_b)
print('Saved training_verification_b.{png,pdf}')
