"""
Publication-ready 5-metric distribution plot.
Box + swarm, Mann-Whitney U p-values, Cohen's d. No title.
Row 1: IK-dependent (trunk, pronation, shoulder)
Row 2: solver-independent (RPR, wrist reach) + empty
"""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from scipy import stats

# ── global style ──────────────────────────────────────────────────────────────
rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'DejaVu Sans'],
    'font.size':         9,
    'axes.titlesize':    9.5,
    'axes.titleweight':  'semibold',
    'axes.titlepad':     6,
    'axes.labelsize':    8.5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.labelsize':   8.5,
    'ytick.labelsize':   8,
})

RAD2DEG  = 57.2958
BASE     = '/home/abdul/Desktop/myosuite/custom_workspace/part2/output/neutral_start'
EXCLUDE  = {'01_12_1', '01_12_2', '01_12_3', '15_12_1'}
OUT      = ('/home/abdul/Desktop/myosuite/custom_workspace/part2'
            '/paper_publication_assets/figures_and_tables'
            '/final_5_metric_distribution.png')

# ── loaders ───────────────────────────────────────────────────────────────────
def load_mot(path):
    with open(path) as f: lines = f.readlines()
    hend = next(i for i,l in enumerate(lines) if 'endheader' in l)
    cols = lines[hend+1].strip().split('\t')
    arr  = np.array([l.strip().split('\t') for l in lines[hend+2:] if l.strip()], dtype=float)
    return {c: arr[:,i] for i,c in enumerate(cols)}

def load_trc_wrist(path):
    with open(path) as f: lines = f.readlines()
    marker_line = lines[3].strip().split('\t')
    data_lines  = [l for l in lines[6:] if l.strip()]
    if not data_lines: return {}
    data = np.array([l.strip().split('\t') for l in data_lines], dtype=float)
    markers = {}; col = 2
    for m in marker_line[2:]:
        if m and col+2 < data.shape[1]:
            markers[m] = data[:, col:col+3]; col += 3
    return markers

# ── data ──────────────────────────────────────────────────────────────────────
rows = []
for group in ['stroke', 'healthy']:
    mot_dir = f'{BASE}/{group}/mot'
    trc_dir = f'{BASE}/{group}/trc'
    for fn in sorted(os.listdir(mot_dir)):
        if not fn.endswith('.mot'): continue
        fid = fn.replace('.mot','')
        if group == 'healthy' and fid in EXCLUDE: continue
        m = load_mot(os.path.join(mot_dir, fn))

        trunk_fwd    = float(np.max(np.abs(m['trunk_forward_lean']))) * RAD2DEG if 'trunk_forward_lean' in m else np.nan
        pronation    = float(np.max(np.abs(m['pro_sup'])))            * RAD2DEG if 'pro_sup'            in m else np.nan
        shoulder_elv = float(np.max(np.abs(m['shoulder_elv'])))       * RAD2DEG if 'shoulder_elv'       in m else np.nan

        rpr = np.nan; wrist_cm = np.nan
        trc_path = os.path.join(trc_dir, f'{fid}.trc')
        if os.path.exists(trc_path):
            try:
                trc = load_trc_wrist(trc_path)
                if 'V_Wrist' in trc:
                    w = trc['V_Wrist']; valid = ~np.any(np.isnan(w), axis=1)
                    if valid.sum() > 2:
                        w = w[valid]; peak_f = int(np.argmax(w[:,1]))
                        if peak_f > 1:
                            reach    = w[:peak_f+1]
                            path_len = float(np.sum(np.linalg.norm(np.diff(reach, axis=0), axis=1)))
                            straight = float(np.linalg.norm(reach[-1] - reach[0]))
                            wrist_cm = straight / 10.0
                            if straight > 10.0: rpr = path_len / straight
            except Exception: pass

        elbow_rom = np.nan
        if 'elbow_flexion' in m:
            e = m['elbow_flexion'] * RAD2DEG
            elbow_rom = float(np.max(e) - np.min(e))

        rows.append({'subject': fid, 'group': group,
                     'trunk_fwd': trunk_fwd, 'pronation': pronation,
                     'shoulder_elv': shoulder_elv, 'elbow_rom': elbow_rom,
                     'reach_path_ratio': rpr, 'wrist_reach_cm': wrist_cm})

df = pd.DataFrame(rows)
S  = df[df.group=='stroke']
H  = df[df.group=='healthy']

# ── statistics helpers ────────────────────────────────────────────────────────
def cohens_d(a, b):
    a, b   = np.array(a), np.array(b)
    pooled = np.sqrt((np.std(a,ddof=1)**2 + np.std(b,ddof=1)**2) / 2)
    return (np.mean(a)-np.mean(b)) / pooled if pooled > 0 else 0.0

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'

def pval_str(p):
    if p < 0.001: return 'p < 0.001'
    if p < 0.01:  return f'p = {p:.3f}'
    return f'p = {p:.3f}'

# ── panel spec ────────────────────────────────────────────────────────────────
# (metric_key, panel_title, y_label, type)
PANELS = [
    ('trunk_fwd',        'Trunk Forward Lean',    'Degrees (°)',         'ik'),
    ('pronation',        'Peak Pronation',         'Degrees (°)',         'ik'),
    ('shoulder_elv',     'Peak Shoulder Elevation','Degrees (°)',         'ik'),
    ('elbow_rom',        'Elbow ROM',              'Degrees (°)',         'ik'),
    ('reach_path_ratio', 'Reach-Path-Ratio',       'Path / Straight-line','si'),
    ('wrist_reach_cm',   'Wrist Reach Distance',   'Centimetres (cm)',    'si'),
]

# Stroke: deep brick red / light salmon
SC      = '#B03A2E'
SC_BOX  = '#F1948A'
# Healthy: deep navy / light blue
HC      = '#1A5276'
HC_BOX  = '#85C1E9'

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(13, 9))
fig.patch.set_facecolor('white')
axes_flat = axes.flatten()

rng = np.random.default_rng(0)

for idx, (metric, title, ylabel, mtype) in enumerate(PANELS):
    ax  = axes_flat[idx]
    ax.set_facecolor('white')
    ax.yaxis.grid(True, linestyle=':', linewidth=0.7, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    sv = S[metric].dropna().values
    hv = H[metric].dropna().values

    # ── boxplots ──
    bp = ax.boxplot(
        [sv, hv], positions=[0, 1], widths=0.38, patch_artist=True,
        notch=False,
        medianprops  =dict(color='#111',    linewidth=2.0),
        whiskerprops =dict(color='#444',    linewidth=1.2),
        capprops     =dict(color='#444',    linewidth=1.2),
        flierprops   =dict(marker='',       linestyle='none'),
        boxprops     =dict(linewidth=0.9),
    )
    for patch, col in zip(bp['boxes'], [SC_BOX, HC_BOX]):
        patch.set_facecolor(col); patch.set_alpha(0.50)

    # ── swarm ──
    js = rng.uniform(-0.14, 0.14, len(sv))
    jh = rng.uniform(-0.14, 0.14, len(hv))
    ax.scatter(js,       sv, color=SC, alpha=0.72, s=28, zorder=4,
               linewidths=0.3, edgecolors='white')
    ax.scatter(jh + 1.0, hv, color=HC, alpha=0.60, s=22, zorder=4,
               linewidths=0.3, edgecolors='white')

    # ── stats ──
    _, p  = stats.mannwhitneyu(sv, hv, alternative='two-sided')
    d     = cohens_d(sv, hv)
    stars = sig_stars(p)
    stat_line = f'{pval_str(p)},  d = {d:.2f}  ({stars})'

    ymax   = max(np.max(sv), np.max(hv))
    ymin   = min(np.min(sv), np.min(hv))
    yrng   = ymax - ymin
    bh     = ymax + yrng * 0.10

    # bracket
    tick_h = yrng * 0.025
    ax.plot([0, 0, 1, 1],
            [bh, bh+tick_h, bh+tick_h, bh],
            color='#333', linewidth=1.0, solid_capstyle='round')
    ax.text(0.5, bh + tick_h + yrng*0.01, stat_line,
            ha='center', va='bottom', fontsize=7.8,
            color='#111', fontweight='bold')

    # ── type badge (top-left inside panel) ──
    badge_txt = 'IK-dependent' if mtype == 'ik' else 'Solver-independent'
    badge_col = '#7B241C'       if mtype == 'ik' else '#1A5276'
    ax.text(0.03, 0.97, badge_txt, transform=ax.transAxes,
            ha='left', va='top', fontsize=6.8, fontweight='bold',
            color='white',
            bbox=dict(facecolor=badge_col, edgecolor='none',
                      boxstyle='round,pad=0.22', alpha=0.80))

    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Stroke\n(n={len(sv)})', f'Healthy\n(n={len(hv)})'])
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(ymin - yrng*0.07,
                bh + tick_h + yrng*0.18)


# ── legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=SC, edgecolor='none', label=f'Stroke  (n = 24)'),
    mpatches.Patch(facecolor=HC, edgecolor='none', label=f'Healthy (n = 49)'),
]
fig.legend(handles=handles, loc='lower right',
           bbox_to_anchor=(0.95, 0.05), ncol=1, fontsize=9,
           frameon=True, framealpha=0.95,
           edgecolor='#ccc', fancybox=False)

plt.tight_layout(rect=[0, 0, 1, 1], h_pad=3.0, w_pad=2.5)
plt.savefig(OUT, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {OUT}')
