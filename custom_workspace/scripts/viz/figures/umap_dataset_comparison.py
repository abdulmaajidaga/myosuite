"""
UMAP dataset comparison — original vs SMOTE vs DTW vs Linear augmentation.

What it does:
  Fits a UMAP on the original 77-subject dataset, then uses transform() to
  project each augmentation method into the same embedding space (UMAP supports
  this, unlike t-SNE). Plots a 2×2 grid showing how each method fills the gap.

Input:
  - data/kinematic/cutoff/processed/*.csv        (original 77 subjects, 15-col, 100 frames)
  - data/kinematic/cutoff/augmented_smote/*.csv   (SMOTE-augmented, 56k files)
  - data/kinematic/cutoff/augmented_dtw/*.csv     (DTW-augmented)
  - data/kinematic/cutoff/augmented_linear/*.csv  (linear-interpolated)
  - output/scores.csv                             (FMA score per subject)

Output:
  - figures/umap_dataset_comparison.png           (2×2 grid; LaTeX-referenced)

Usage:
  python scripts/viz/figures/umap_dataset_comparison.py
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import umap
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)

# ── paths ──────────────────────────────────────────────────────────────────
PROCESSED_DIR  = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'processed')
SMOTE_DIR      = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'augmented_smote')
DTW_DIR        = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'augmented_dtw')
LINEAR_DIR     = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'augmented_linear')
SCORES_FILE    = os.path.join(ROOT, 'output', 'scores.csv')
OUTPUT_PATH    = os.path.join(ROOT, 'figures', 'umap_dataset_comparison.png')

# ── subsample size per FMA level ───────────────────────────────────────────
AUG_SAMPLES_PER_LEVEL = 20   # × 51 levels = 1020 points per panel


def load_scores():
    df = pd.read_csv(SCORES_FILE)
    scores = {}
    for _, row in df.iterrows():
        stem = os.path.splitext(row['filename'])[0]
        scores[stem] = int(row['fma_score'])
    return scores


def flatten(df_100x15):
    arr = df_100x15.values.astype(float)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0)])  # 30-d summary


def load_original(scores_map):
    X, labels = [], []
    for fname in sorted(os.listdir(PROCESSED_DIR)):
        if not fname.endswith('.csv'):
            continue
        stem = os.path.splitext(fname)[0]
        if stem in scores_map:
            fma = scores_map[stem]
        elif stem.startswith('S'):
            fma = 18
        else:
            fma = 66
        try:
            df = pd.read_csv(os.path.join(PROCESSED_DIR, fname))
            if len(df) < 10:
                continue
            X.append(flatten(df))
            labels.append(fma)
        except Exception:
            continue
    return np.array(X), np.array(labels)


def load_augmented(aug_dir, samples_per_level=AUG_SAMPLES_PER_LEVEL):
    fma_pattern = re.compile(r'FMA(\d+)', re.IGNORECASE)
    by_level = {}
    for fname in os.listdir(aug_dir):
        if not fname.endswith('.csv'):
            continue
        m = fma_pattern.search(fname)
        if not m:
            continue
        fma = int(m.group(1))
        by_level.setdefault(fma, []).append(fname)

    X, labels = [], []
    rng = np.random.default_rng(42)
    for fma in sorted(by_level):
        files = by_level[fma]
        chosen = rng.choice(files, size=min(samples_per_level, len(files)), replace=False)
        for fname in chosen:
            try:
                df = pd.read_csv(os.path.join(aug_dir, fname), header=None
                                 if not any(c.isalpha() for c in
                                            open(os.path.join(aug_dir, fname)).readline())
                                 else 0)
                if df.shape[1] != 15 or len(df) < 10:
                    continue
                X.append(flatten(df))
                labels.append(fma)
            except Exception:
                continue
    return np.array(X), np.array(labels)


def fit_shared_umap(X_orig, X_smote, X_dtw, X_linear):
    """Fit UMAP on the union of all datasets so panels share the same embedding."""
    all_X = np.vstack([X_orig, X_smote, X_dtw, X_linear])
    scaler = StandardScaler()
    all_X_s = scaler.fit_transform(all_X)
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                        metric='euclidean', random_state=42)
    reducer.fit(all_X_s)
    return reducer, scaler


def project(reducer, scaler, X):
    return reducer.transform(scaler.transform(X))


# ── plotting ────────────────────────────────────────────────────────────────
CMAP = 'RdYlGn'
FMA_MIN, FMA_MAX = 16, 66


def main():
    print("Loading scores...")
    scores_map = load_scores()

    print("Loading original data...")
    X_orig, y_orig = load_original(scores_map)
    print(f"  {len(X_orig)} original sessions")

    print("Loading SMOTE augmented data...")
    X_smote, y_smote = load_augmented(SMOTE_DIR)
    print(f"  {len(X_smote)} SMOTE samples")

    print("Loading DTW augmented data...")
    X_dtw, y_dtw = load_augmented(DTW_DIR)
    print(f"  {len(X_dtw)} DTW samples")

    print("Loading Linear augmented data...")
    X_linear, y_linear = load_augmented(LINEAR_DIR)
    print(f"  {len(X_linear)} Linear samples")

    print("\nFitting shared UMAP on all data...")
    reducer, scaler = fit_shared_umap(X_orig, X_smote, X_dtw, X_linear)
    print("  UMAP done.")

    proj_orig   = project(reducer, scaler, X_orig)
    proj_smote  = project(reducer, scaler, X_smote)
    proj_dtw    = project(reducer, scaler, X_dtw)
    proj_linear = project(reducer, scaler, X_linear)

    # ── figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.87, top=0.91, bottom=0.07,
                          hspace=0.38, wspace=0.35)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]

    fig.suptitle('UMAP of Kinematic Feature Space — Original Data vs Augmentation Methods',
                 fontsize=13, fontweight='bold', y=0.97)

    norm = mcolors.Normalize(vmin=FMA_MIN, vmax=FMA_MAX)

    panels = [
        (axes[0][0], proj_orig,   y_orig,
         f'(a) Original Data  [n={len(X_orig)}]\nFMA 16–20 (stroke)  +  FMA 66 (healthy)'),
        (axes[0][1], proj_smote,  y_smote,
         f'(b) SMOTE Augmentation  [n={len(X_smote):,}]\nFMA 16–66 continuous'),
        (axes[1][0], proj_dtw,    y_dtw,
         f'(c) DTW Morphing  [n={len(X_dtw):,}]\nFMA 16–66 continuous'),
        (axes[1][1], proj_linear, y_linear,
         f'(d) Linear Interpolation  [n={len(X_linear):,}]\nFMA 16–66 continuous'),
    ]

    sc = None
    for ax, proj, labels, title in panels:
        sc = ax.scatter(proj[:, 0], proj[:, 1],
                        c=labels, cmap=CMAP, norm=norm,
                        s=16, alpha=0.75, linewidths=0)
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=6, loc='left')
        ax.set_xlabel('UMAP 1', fontsize=8)
        ax.set_ylabel('UMAP 2', fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.spines[['top', 'right']].set_visible(False)

    # ── annotations on original panel ─────────────────────────────────────
    ax0 = axes[0][0]
    stroke_cx = proj_orig[y_orig <= 20, 0].mean()
    stroke_cy = proj_orig[y_orig <= 20, 1].mean()
    health_cx = proj_orig[y_orig == 66, 0].mean()
    health_cy = proj_orig[y_orig == 66, 1].mean()
    ax0_xl, ax0_xr = ax0.get_xlim()
    ax0_yb, ax0_yt = ax0.get_ylim()

    ax0.annotate('Stroke cluster\n(FMA 16–20)',
                 xy=(stroke_cx, stroke_cy),
                 xytext=(stroke_cx - 0.28*(ax0_xr-ax0_xl),
                         stroke_cy + 0.22*(ax0_yt-ax0_yb)),
                 fontsize=7.5, color='#c0392b', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))
    ax0.annotate('Healthy cluster\n(FMA 66)',
                 xy=(health_cx, health_cy),
                 xytext=(health_cx + 0.05*(ax0_xr-ax0_xl),
                         health_cy - 0.28*(ax0_yt-ax0_yb)),
                 fontsize=7.5, color='#27ae60', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.2))

    ax0.annotate('', xy=(health_cx - 0.05*(ax0_xr-ax0_xl), health_cy),
                 xytext=(stroke_cx + 0.05*(ax0_xr-ax0_xl), stroke_cy),
                 arrowprops=dict(arrowstyle='<->', color='#7f8c8d', lw=1.5,
                                 connectionstyle='arc3,rad=0.0'))
    gap_x = (stroke_cx + health_cx) / 2
    gap_y = (stroke_cy + health_cy) / 2
    ax0.text(gap_x, gap_y + 0.07*(ax0_yt-ax0_yb), 'No data\nFMA 21–65',
             ha='center', va='bottom', fontsize=7, color='#7f8c8d', style='italic')

    # ── shared colourbar ──────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.89, 0.07, 0.022, 0.84])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('FMA-UE Score', fontsize=10, labelpad=8)
    cbar.set_ticks([16, 25, 35, 45, 55, 66])
    cbar.ax.tick_params(labelsize=8.5)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {OUTPUT_PATH}")
    plt.close()


if __name__ == '__main__':
    main()
