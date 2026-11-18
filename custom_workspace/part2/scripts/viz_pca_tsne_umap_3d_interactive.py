"""
Interactive 3D PCA, t-SNE, UMAP — rotate with mouse.
Run from part2/:
    /home/abdul/miniconda3/envs/MyoSuite/bin/python scripts/viz_pca_tsne_umap_3d_interactive.py
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

warnings.filterwarnings('ignore')

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def resample(arr, n=N_FRAMES):
    return interp1d(np.linspace(0, 1, len(arr)), arr, axis=0)(np.linspace(0, 1, n))


def load_csv_features(csv_path):
    try:
        with open(csv_path) as f:
            row0 = f.readline().rstrip('\n').split(',')
            row1 = f.readline().rstrip('\n').split(',')
        current = ''
        markers = []
        for name in row0:
            name = name.strip()
            if name: current = name
            markers.append(current)
        marker_cols = {}
        for i, (m, ax) in enumerate(zip(markers, row1)):
            ax = ax.strip()
            if m not in marker_cols: marker_cols[m] = {}
            marker_cols[m][ax] = i
        arr = pd.read_csv(csv_path, header=None, skiprows=2).values.astype(float)

        def centre(*names):
            pts = [arr[:, [mc['X'], mc['Y'], mc['Z']]]
                   for n in names
                   for mc in [marker_cols.get(n, {})]
                   if all(k in mc for k in ('X', 'Y', 'Z'))]
            return np.nanmean(pts, axis=0) if pts else None

        shoulder = centre('SA_1', 'SA_2', 'SA_3')
        elbow    = centre('ELB_L', 'ELB_M')
        wrist    = centre('WRA', 'WRB')
        trunk    = centre('CS_1', 'CS_2', 'CS_3', 'CS_4')
        if any(x is None for x in [shoulder, elbow, wrist, trunk]):
            return None
        wra_mc, wrb_mc = marker_cols.get('WRA', {}), marker_cols.get('WRB', {})
        wra   = arr[:, [wra_mc['X'], wra_mc['Y'], wra_mc['Z']]]
        wrb   = arr[:, [wrb_mc['X'], wrb_mc['Y'], wrb_mc['Z']]]
        wrvec = wrb - wra
        data  = np.concatenate([shoulder, elbow, wrist, wrvec, trunk], axis=1)
        for col in range(0, 9, 3):
            data[:, col:col+3] -= trunk
        data = data[~np.any(np.isnan(data), axis=1)]
        return resample(data).flatten() if len(data) >= 10 else None
    except Exception:
        return None


def load_mot_features(mot_path):
    try:
        with open(mot_path) as f:
            lines = f.readlines()
        hend   = next(i for i, l in enumerate(lines) if 'endheader' in l)
        header = lines[hend + 1].strip().split('\t')
        arr    = np.array([l.strip().split('\t') for l in lines[hend + 2:] if l.strip()], dtype=float)
        idx    = [header.index(j) for j in MOT_JOINTS if j in header]
        return resample(arr[:, idx]).flatten() if idx else None
    except Exception:
        return None


def collect(csv_dir, mot_dir, group_label):
    X_csv, X_mot, labels = [], [], []
    for csv_path in sorted(glob.glob(os.path.join(csv_dir, '*.csv'))):
        fid = os.path.splitext(os.path.basename(csv_path))[0]
        if fid in EXCLUDE: continue
        mot_path = os.path.join(mot_dir, f'{fid}.mot')
        if not os.path.exists(mot_path): continue
        fc, fm = load_csv_features(csv_path), load_mot_features(mot_path)
        if fc is None or fm is None: continue
        X_csv.append(fc); X_mot.append(fm); labels.append(group_label)
    return X_csv, X_mot, labels


print("Loading data...")
cs, ms, ls = collect(os.path.join(ROOT, 'data/kinematic/stroke'),
                     os.path.join(ROOT, 'output/neutral_start/stroke/mot'), 'stroke')
ch, mh, lh = collect(os.path.join(ROOT, 'data/kinematic/healthy'),
                     os.path.join(ROOT, 'output/neutral_start/healthy/mot'), 'healthy')

X_csv_all = np.array(cs + ch)
X_mot_all = np.array(ms + mh)
labels    = ls + lh
colours   = [C_STROKE if l == 'stroke' else C_HEALTHY for l in labels]
n_stroke, n_healthy = len(ls), len(lh)
print(f"Loaded: {len(labels)} sessions ({n_stroke} stroke, {n_healthy} healthy)")


def reduce3(X, method):
    Xs = StandardScaler().fit_transform(X)
    if method == 'pca':
        return PCA(n_components=3).fit_transform(Xs)
    elif method == 'tsne':
        return TSNE(n_components=3, perplexity=min(30, len(X)-1), random_state=42).fit_transform(Xs)
    elif method == 'umap':
        return umap.UMAP(n_components=3, n_neighbors=min(15, len(X)-2), random_state=42).fit_transform(Xs)


print("Running reductions (this takes ~30s)...")
embeddings = [
    [reduce3(X_csv_all, 'pca'),  reduce3(X_csv_all, 'tsne'),  reduce3(X_csv_all, 'umap')],
    [reduce3(X_mot_all, 'pca'),  reduce3(X_mot_all, 'tsne'),  reduce3(X_mot_all, 'umap')],
]
print("Done. Opening window — click and drag to rotate each subplot.")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')
row_labels = ['Raw Mocap (CSV)', 'IK Output (MOT)']
col_labels = ['PCA', 't-SNE', 'UMAP']

for ri in range(2):
    for ci in range(3):
        ax  = fig.add_subplot(2, 3, ri * 3 + ci + 1, projection='3d')
        emb = embeddings[ri][ci]
        ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2],
                   c=colours, s=45, alpha=0.85,
                   edgecolors='white', linewidths=0.3)
        ax.set_title(f'{col_labels[ci]}\n{row_labels[ri]}', fontsize=10, fontweight='bold', pad=4)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_facecolor('#F8F8F8')
        ax.view_init(elev=25, azim=45)

from matplotlib.lines import Line2D
fig.legend(handles=[
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_STROKE,  markersize=9, label=f'Stroke (n={n_stroke})'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=C_HEALTHY, markersize=9, label=f'Healthy (n={n_healthy})'),
], loc='lower center', ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.01))

fig.suptitle('Manifold Structure (3D — interactive): Raw Mocap vs IK-Derived Joint Angles',
             fontsize=12, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0, 0.05, 1, 0.97])
plt.show()
