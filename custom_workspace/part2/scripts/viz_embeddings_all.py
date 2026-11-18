"""
Three separate embedding figures, one per feature strategy.
  A — MOT summary stats (mean + std + peak per joint, 33 features)
  B — Clinical metrics (7 validated peak/range features)
  C — Frame-level (11 joint angles per frame, ~1460 points)

Run from part2/:
    /home/abdul/miniconda3/envs/MyoSuite/bin/python scripts/viz_embeddings_all.py
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

warnings.filterwarnings('ignore')

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(ROOT, 'paper_publication_assets', 'figures_and_tables')
N_FRAMES = 100
EXCLUDE  = {'01_12_1', '01_12_2', '01_12_3', '15_12_1'}

MOT_JOINTS = [
    'trunk_forward_lean', 'trunk_lateral_lean',
    'sternoclavicular_r2', 'sternoclavicular_r3',
    'elv_angle', 'shoulder_elv', 'shoulder_rot',
    'elbow_flexion', 'pro_sup', 'deviation', 'flexion',
]

C_STROKE  = '#E84C4C'
C_HEALTHY = '#4C9BE8'

# ── data loading ───────────────────────────────────────────────────────────

def load_mot(mot_path):
    try:
        with open(mot_path) as f:
            lines = f.readlines()
        hend   = next(i for i, l in enumerate(lines) if 'endheader' in l)
        header = lines[hend + 1].strip().split('\t')
        arr    = np.array([l.strip().split('\t') for l in lines[hend + 2:] if l.strip()], dtype=float)
        idx    = [header.index(j) for j in MOT_JOINTS if j in header]
        return arr[:, idx] if idx else None
    except Exception:
        return None

def resample(arr):
    return interp1d(np.linspace(0, 1, len(arr)), arr, axis=0)(np.linspace(0, 1, N_FRAMES))

def collect_mot(mot_dir, group_label):
    X, labels = [], []
    for mot_path in sorted(glob.glob(os.path.join(mot_dir, '*.mot'))):
        fid = os.path.splitext(os.path.basename(mot_path))[0]
        if fid in EXCLUDE:
            continue
        arr = load_mot(mot_path)
        if arr is None or len(arr) < 10:
            continue
        X.append(resample(arr))
        labels.append(group_label)
    return X, labels

print("Loading MOT data...")
Xs, ls = collect_mot(os.path.join(ROOT, 'output/neutral_start/stroke/mot'),  'stroke')
Xh, lh = collect_mot(os.path.join(ROOT, 'output/neutral_start/healthy/mot'), 'healthy')

sessions  = Xs + Xh
labels    = ls + lh
n_stroke  = len(ls)
n_healthy = len(lh)
colours   = [C_STROKE if l == 'stroke' else C_HEALTHY for l in labels]
print(f"  {len(labels)} sessions ({n_stroke} stroke, {n_healthy} healthy)")

# ── feature sets ───────────────────────────────────────────────────────────

ji = {j: i for i, j in enumerate(MOT_JOINTS)}

X_A = np.array([
    np.concatenate([s.mean(0), s.std(0), np.abs(s).max(0)])
    for s in sessions
])  # (73, 33)

X_B = np.array([
    [
        s[:, ji['trunk_forward_lean']].max(),
        s[:, ji['trunk_lateral_lean']].max(),
        s[:, ji['shoulder_elv']].max(),
        s[:, ji['elv_angle']].max(),
        s[:, ji['shoulder_rot']].min(),
        s[:, ji['elbow_flexion']].max() - s[:, ji['elbow_flexion']].min(),
        s[:, ji['pro_sup']].max(),
    ]
    for s in sessions
])  # (73, 7)

N_SUB     = 20
idx_sub   = np.linspace(0, N_FRAMES - 1, N_SUB, dtype=int)
X_C       = np.array([s[fi] for s in sessions for fi in idx_sub])   # (1460, 11)
colours_C = [c for c in colours for _ in idx_sub]

print(f"Feature shapes — A: {X_A.shape}  B: {X_B.shape}  C: {X_C.shape}")

# ── reduction ──────────────────────────────────────────────────────────────

def reduce2(X, method):
    Xs = StandardScaler().fit_transform(X)
    if method == 'pca':
        return PCA(n_components=2).fit_transform(Xs)
    elif method == 'tsne':
        return TSNE(n_components=2, perplexity=min(30, len(X)-1), random_state=42).fit_transform(Xs)
    elif method == 'umap':
        return umap.UMAP(n_components=2, n_neighbors=min(15, len(X)-2), random_state=42).fit_transform(Xs)

# ── plot helper ────────────────────────────────────────────────────────────

from matplotlib.lines import Line2D

def make_figure(X, cols, title, filename, pt_size=55, alpha=0.85):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    for ax, method, ml in zip(axes, ['pca', 'tsne', 'umap'], ['PCA', 't-SNE', 'UMAP']):
        print(f"  {ml}...")
        emb = reduce2(X, method)
        ax.scatter(emb[:, 0], emb[:, 1], c=cols, s=pt_size, alpha=alpha,
                   edgecolors='white', linewidths=0.4, zorder=3)
        ax.set_title(ml, fontsize=13, fontweight='bold', pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.set_facecolor('#F8F8F8')
    fig.legend(handles=[
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_STROKE,
               markersize=10, label=f'Stroke (n={n_stroke})'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=C_HEALTHY,
               markersize=10, label=f'Healthy (n={n_healthy})'),
    ], loc='lower center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0.06, 1, 1.0])
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {path}")

os.makedirs(OUT_DIR, exist_ok=True)

print("\nOption A — Summary stats (33 features, 73 pts)")
make_figure(X_A, colours,
            'Option A: MOT Summary Stats — mean + std + peak of 11 joints (33 features)',
            'embed_A_summary_stats.png')

print("\nOption B — Clinical metrics (7 features, 73 pts)")
make_figure(X_B, colours,
            'Option B: Clinical Metrics — 7 validated peak/range features',
            'embed_B_clinical_metrics.png')

print("\nOption C — Frame-level (11 features, 1460 pts)")
make_figure(X_C, colours_C,
            'Option C: Frame-level — 11 joint angles per frame (20 frames × 73 sessions)',
            'embed_C_frame_level.png', pt_size=12, alpha=0.4)

print("\nDone.")


# ── Option D: single joint — trunk_forward_lean full trajectory ────────────

print("\nOption D — trunk_forward_lean trajectory (100 features, 73 pts)")

X_D = np.array([s[:, ji['trunk_forward_lean']] for s in sessions])   # (73, 100)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.patch.set_facecolor('white')

# first 3 panels: PCA / t-SNE / UMAP
for ax, method, ml in zip(axes[:3], ['pca', 'tsne', 'umap'], ['PCA', 't-SNE', 'UMAP']):
    print(f"  {ml}...")
    emb = reduce2(X_D, method)
    ax.scatter(emb[:, 0], emb[:, 1], c=colours, s=60, alpha=0.85,
               edgecolors='white', linewidths=0.4, zorder=3)
    ax.set_title(ml, fontsize=13, fontweight='bold', pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.set_facecolor('#F8F8F8')

# 4th panel: mean ± 1 SD trajectory per group
ax = axes[3]
t = np.linspace(0, 100, N_FRAMES)
for label, color, idxs in [
    ('Stroke',  C_STROKE,  [i for i,l in enumerate(labels) if l=='stroke']),
    ('Healthy', C_HEALTHY, [i for i,l in enumerate(labels) if l=='healthy']),
]:
    traj = X_D[idxs]
    mu, sd = traj.mean(0), traj.std(0)
    ax.plot(t, mu, color=color, lw=2, label=label)
    ax.fill_between(t, mu - sd, mu + sd, color=color, alpha=0.18)

ax.set_xlabel('% of movement', fontsize=10)
ax.set_ylabel('Trunk forward lean (°)', fontsize=10)
ax.set_title('Mean ± 1 SD trajectory', fontsize=13, fontweight='bold', pad=8)
ax.legend(fontsize=10, frameon=False)
ax.spines[['top','right']].set_visible(False)
ax.set_facecolor('#F8F8F8')

fig.legend(handles=[
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_STROKE,
           markersize=10, label=f'Stroke (n={n_stroke})'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_HEALTHY,
           markersize=10, label=f'Healthy (n={n_healthy})'),
], loc='lower center', ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.38, 0.01))

fig.suptitle('Option D: trunk_forward_lean — full trajectory embedding (100 features)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.06, 1, 1.0])
path = os.path.join(OUT_DIR, 'embed_D_trunk_lean.png')
plt.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {path}")
