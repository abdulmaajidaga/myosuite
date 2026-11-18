"""
Clinical validation: Standard IK (no weighting) vs SVD Two-Pass IK (SC hierarchy)
Metrics: trunk forward lean, pronation, reach-path-ratio, shoulder elevation,
         wrist reach distance, elbow ROM.
Subject 01 and 15_12_1 excluded from healthy cohort.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import get_project_root

ROOT = get_project_root()
OUT_DIR = os.path.join(ROOT, 'paper_publication_assets', 'figures_and_tables')

RUNS = {
    'Standard IK': os.path.join(ROOT, 'output', 'originals'),
    'SVD Two-Pass IK': os.path.join(ROOT, 'output', 'neutral_start'),
}

EXCLUDE_HEALTHY = {'01_12_1', '01_12_2', '01_12_3', '15_12_1'}
RAD2DEG = 57.2958

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_mot(path):
    with open(path) as f:
        lines = f.readlines()
    hend = next(i for i, l in enumerate(lines) if 'endheader' in l)
    cols = lines[hend + 1].strip().split('\t')
    arr  = np.array([l.strip().split('\t') for l in lines[hend + 2:] if l.strip()],
                    dtype=float)
    return {c: arr[:, i] for i, c in enumerate(cols)}


def load_trc(path):
    with open(path) as f:
        lines = f.readlines()
    marker_line = lines[3].strip().split('\t')
    data_lines  = [l for l in lines[6:] if l.strip()]
    if not data_lines:
        return {}
    data = np.array([l.strip().split('\t') for l in data_lines], dtype=float)
    markers, col = {}, 2
    for m in marker_line[2:]:
        if m and col + 2 < data.shape[1]:
            markers[m] = data[:, col:col + 3]
            col += 3
    return markers


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def file_metrics(mot_path, trc_path):
    mot = load_mot(mot_path)
    trc = load_trc(trc_path) if trc_path and os.path.exists(trc_path) else {}

    def peak(key):
        if key not in mot:
            return np.nan
        return float(np.max(np.abs(mot[key]))) * RAD2DEG

    elbow_rom = np.nan
    if 'elbow_flexion' in mot:
        e = mot['elbow_flexion'] * RAD2DEG
        elbow_rom = float(np.max(e) - np.min(e))

    kin_eff = np.nan
    wrist_reach_cm = np.nan
    if 'V_Wrist' in trc:
        w = trc['V_Wrist']
        valid = ~np.any(np.isnan(w), axis=1)
        if valid.sum() > 2:
            w = w[valid]
            peak_frame = int(np.argmax(w[:, 1]))
            if peak_frame > 1:
                reach     = w[:peak_frame + 1]
                path_len  = float(np.sum(np.linalg.norm(np.diff(reach, axis=0), axis=1)))
                straight  = float(np.linalg.norm(reach[-1] - reach[0]))
                wrist_reach_cm = straight / 10.0
                if straight > 10.0:
                    kin_eff = path_len / straight

    return {
        'trunk_fwd':      peak('trunk_forward_lean'),
        'pronation':      peak('pro_sup'),
        'shoulder_elv':   peak('shoulder_elv'),
        'elbow_rom':      elbow_rom,
        'wrist_reach_cm': wrist_reach_cm,
        'kin_eff':        kin_eff,
    }


# ---------------------------------------------------------------------------
# Load all data — falls back to directory scan when batch_report is empty
# ---------------------------------------------------------------------------

def load_run(run_dir, exclude_healthy=EXCLUDE_HEALTHY):
    records = []
    for group in ['stroke', 'healthy']:
        mot_dir = os.path.join(run_dir, group, 'mot')
        trc_dir = os.path.join(run_dir, group, 'trc')
        if not os.path.exists(mot_dir):
            continue
        for fname in sorted(os.listdir(mot_dir)):
            if not fname.endswith('.mot'):
                continue
            fid = os.path.splitext(fname)[0]
            if group == 'healthy' and fid in exclude_healthy:
                continue
            mot_path = os.path.join(mot_dir, fname)
            trc_path = os.path.join(trc_dir, f'{fid}.trc')
            try:
                m = file_metrics(mot_path, trc_path)
            except Exception:
                continue
            records.append({'file': fid, 'group': group, **m})
    return pd.DataFrame(records)


print("Loading data…")
dfs = {label: load_run(path) for label, path in RUNS.items()}
for label, df in dfs.items():
    for g in ['stroke', 'healthy']:
        print(f"  {label} / {g}: n={len(df[df['group']==g])}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

METRICS = ['trunk_fwd', 'pronation', 'kin_eff', 'shoulder_elv', 'wrist_reach_cm', 'elbow_rom']

PANEL_TITLES = {
    'trunk_fwd':      'Trunk Forward Lean',
    'pronation':      'Peak Pronation',
    'kin_eff':        'Reach-Path-Ratio',
    'shoulder_elv':   'Peak Shoulder Elevation',
    'wrist_reach_cm': 'Wrist Reach Distance',
    'elbow_rom':      'Elbow ROM',
}
YLABELS = {
    'trunk_fwd':      'Degrees (°)',
    'pronation':      'Degrees (°)',
    'kin_eff':        'Path / Straight-line',
    'shoulder_elv':   'Degrees (°)',
    'wrist_reach_cm': 'Centimetres (cm)',
    'elbow_rom':      'Degrees (°)',
}
DIRECTION = {
    'trunk_fwd':      '↑ Stroke > Healthy',
    'pronation':      '↑ Stroke > Healthy',
    'kin_eff':        '↑ Stroke > Healthy',
    'shoulder_elv':   '↓ Stroke < Healthy',
    'wrist_reach_cm': '↓ Stroke < Healthy',
    'elbow_rom':      '↓ Stroke < Healthy',
}

STROKE_COL  = '#d62728'
HEALTHY_COL = '#4472c4'

C = {
    ('healthy', 'Standard IK'):    '#b8cce4',
    ('healthy', 'SVD Two-Pass IK'): HEALTHY_COL,
    ('stroke',  'Standard IK'):    '#f4b8b8',
    ('stroke',  'SVD Two-Pass IK'): STROKE_COL,
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


def agg(df, group, metric):
    sub = df[df['group'] == group][metric].dropna()
    return float(sub.mean()), float(sub.std() / np.sqrt(max(len(sub), 1)))


run_labels = list(RUNS.keys())
x = np.arange(len(run_labels))
w = 0.30

fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor='white')
axes = axes.flatten()

for ai, metric in enumerate(METRICS):
    ax = axes[ai]
    ax.set_facecolor('white')

    all_tops = []

    for gi, group in enumerate(['healthy', 'stroke']):
        offset = (gi - 0.5) * (w + 0.05)
        means, sems, cols = [], [], []
        for rl in run_labels:
            mn, se = agg(dfs[rl], group, metric)
            means.append(mn); sems.append(se); cols.append(C[(group, rl)])
            all_tops.append(mn + se)

        bars = ax.bar(x + offset, means, w, yerr=sems, color=cols,
                      capsize=4, linewidth=0,
                      error_kw=dict(elinewidth=1.0, ecolor='#555', capthick=1.0),
                      zorder=3)
        for bar, mn, se in zip(bars, means, sems):
            if not np.isnan(mn):
                fmt = f'{mn:.2f}' if metric in ('kin_eff',) else f'{mn:.1f}'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        mn + se + max(all_tops + [1]) * 0.01,
                        fmt, ha='center', va='bottom', fontsize=7.5,
                        fontweight='semibold',
                        color=STROKE_COL if group == 'stroke' else HEALTHY_COL)

    # Delta bracket between SVD bars
    w_s, w_s_se = agg(dfs['SVD Two-Pass IK'], 'stroke',  metric)
    w_h, w_h_se = agg(dfs['SVD Two-Pass IK'], 'healthy', metric)
    gap = w_s - w_h
    ymax     = max(all_tops) * 1.10
    brack_y  = ymax
    xs = x[1] + (1 - 0.5) * (w + 0.05)
    xh = x[1] + (0 - 0.5) * (w + 0.05)
    ax.annotate('', xy=(xs, brack_y), xytext=(xh, brack_y),
                arrowprops=dict(arrowstyle='<->', color='#444', lw=1.2,
                                mutation_scale=8))
    gap_txt = f'Δ={gap:+.2f}' if metric == 'kin_eff' else f'Δ={gap:+.1f}'
    ax.text((xs + xh) / 2, brack_y * 1.01, gap_txt,
            ha='center', va='bottom', fontsize=8, color='#222', fontweight='semibold')

    ax.text(0.03, 0.97, DIRECTION[metric], transform=ax.transAxes,
            ha='left', va='top', fontsize=7, color='#666', fontstyle='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, fontsize=8.5, color='#333')
    ax.set_ylabel(YLABELS[metric], fontsize=8.5, color='#444', labelpad=3)
    ax.set_title(PANEL_TITLES[metric], fontsize=9.5, fontweight='semibold',
                 pad=5, color='#1a1a1a')
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.tick_params(colors='#555555')
    ax.yaxis.grid(True, linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0, top=brack_y * 1.15)

legend_patches = [
    mpatches.Patch(color=C[('healthy', 'Standard IK')],    label='Healthy — Standard IK'),
    mpatches.Patch(color=C[('healthy', 'SVD Two-Pass IK')], label='Healthy — SVD Two-Pass IK'),
    mpatches.Patch(color=C[('stroke',  'Standard IK')],    label='Stroke — Standard IK'),
    mpatches.Patch(color=C[('stroke',  'SVD Two-Pass IK')], label='Stroke — SVD Two-Pass IK'),
]
fig.legend(handles=legend_patches, loc='lower center', ncol=4,
           fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01),
           handlelength=1.2, handletextpad=0.5, columnspacing=1.2)

plt.tight_layout(rect=[0, 0.05, 1, 1], h_pad=2.5, w_pad=2.0)

out_png = os.path.join(OUT_DIR, 'clinical_validation_results.png')
os.makedirs(OUT_DIR, exist_ok=True)
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\nSaved: {out_png}")
