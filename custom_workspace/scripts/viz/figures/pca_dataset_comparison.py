"""
PCA dataset comparison — original vs SMOTE vs DTW vs Linear augmentation.

What it does:
  Fits a PCA on all 15-DOF delta trajectories, flattened to feature vectors,
  then plots a 2×2 grid showing how each augmentation method fills the FMA
  coverage gap visible in the original 77-subject dataset.

Input:
  - data/kinematic/cutoff/processed/*.csv       (original 77 subjects, 15-col, 100 frames)
  - data/kinematic/cutoff/augmented_smote/*.csv  (SMOTE-augmented, 56k files)
  - data/kinematic/cutoff/augmented_dtw/*.csv    (DTW-augmented)
  - data/kinematic/cutoff/augmented_linear/*.csv (linear-interpolated)
  - output/scores.csv                            (FMA score per subject)

Output:
  - figures/pca_dataset_comparison.png           (2×2 grid; LaTeX-referenced)

Usage:
  python scripts/viz/figures/pca_dataset_comparison.py
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'Liberation Serif', 'DejaVu Serif'],
    'mathtext.fontset':   'stix',
    'font.size':          8,
    'axes.unicode_minus': False,
    'axes.linewidth':     0.8,
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, ROOT)

# ── paths (test datasets) ──────────────────────────────────────────────────
PROCESSED_DIR  = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'processed')
SMOTE_DIR      = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'augmented_smote_test')
DTW_DIR        = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'augmented_dtw_test')
LINEAR_DIR     = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'augmented_linear_test')
SCORES_FILE    = os.path.join(ROOT, 'output', 'scores.csv')
OUTPUT_PATH    = os.path.join(ROOT, 'figures', 'pca_dataset_comparison_test.png')

# ── how many augmented samples to show (per FMA level) ─────────────────────
AUG_SAMPLES_PER_LEVEL = 20   # × 51 levels = 1020 points per panel


def load_scores():
    df = pd.read_csv(SCORES_FILE)
    # strip extension, build {stem: fma} dict
    scores = {}
    for _, row in df.iterrows():
        stem = os.path.splitext(row['filename'])[0]
        scores[stem] = int(row['fma_score'])
    return scores


def flatten(df_100x15):
    """Flatten 100×15 to 30-d feature vector.
    Subtracts first frame so absolute-position and delta-form files are comparable.
    """
    arr = df_100x15.values.astype(float)
    arr = arr - arr[0]   # delta from first frame
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0)])


def load_original(scores_map):
    """Load all 77 processed sessions. Returns (X, fma_labels)."""
    X, labels = [], []
    for fname in sorted(os.listdir(PROCESSED_DIR)):
        if not fname.endswith('.csv'):
            continue
        stem = os.path.splitext(fname)[0]
        # determine FMA: stroke files in scores_map, healthy = 66
        if stem in scores_map:
            fma = scores_map[stem]
        elif stem.startswith('S'):
            fma = 18   # fallback (shouldn't happen if scores.csv complete)
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
    """
    Load a balanced subsample from an augmented directory.
    Filenames encode FMA as:
      smote_XXXXX_FMA{N}.csv   (SMOTE)
      FMA_{N}.csv              (DTW / linear)
    """
    fma_pattern = re.compile(r'FMA(\d+)', re.IGNORECASE)

    # group files by FMA level
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


def fit_shared_pca(X_orig, X_smote, X_dtw, X_linear):
    """Fit PCA on the union of all datasets so panels share the same projection."""
    all_X = np.vstack([X_orig, X_smote, X_dtw, X_linear])
    scaler = StandardScaler()
    all_X_s = scaler.fit_transform(all_X)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_X_s)
    return pca, scaler


def project(pca, scaler, X):
    return pca.transform(scaler.transform(X))


# ── plotting ────────────────────────────────────────────────────────────────
CMAP = 'RdYlGn'   # red=severe(16) → yellow=moderate → green=healthy(66)
FMA_MIN, FMA_MAX = 16, 66


def scatter_panel(ax, proj, labels, title, n_total, pca):
    labels = np.array(labels)
    norm = mcolors.Normalize(vmin=FMA_MIN, vmax=FMA_MAX)
    sc = ax.scatter(proj[:, 0], proj[:, 1],
                    c=labels, cmap=CMAP, norm=norm,
                    s=18, alpha=0.7, linewidths=0)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)', fontsize=9)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)', fontsize=9)
    ax.tick_params(labelsize=8)
    # annotate cluster labels for original panel
    ax.text(0.02, 0.97, f'n = {n_total}', transform=ax.transAxes,
            fontsize=8, va='top', color='#444444')
    return sc


def main():
    print("Loading scores...")
    scores_map = load_scores()

    print("Loading original data (77 sessions)...")
    X_orig, y_orig = load_original(scores_map)
    print(f"  Loaded {len(X_orig)} original sessions")

    print("Loading SMOTE augmented data...")
    X_smote, y_smote = load_augmented(SMOTE_DIR)
    print(f"  Loaded {len(X_smote)} SMOTE samples")

    print("Loading DTW augmented data...")
    X_dtw, y_dtw = load_augmented(DTW_DIR)
    print(f"  Loaded {len(X_dtw)} DTW samples")

    print("Loading Linear augmented data...")
    X_linear, y_linear = load_augmented(LINEAR_DIR)
    print(f"  Loaded {len(X_linear)} Linear samples")

    print("Fitting shared PCA...")
    pca, scaler = fit_shared_pca(X_orig, X_smote, X_dtw, X_linear)
    ev = pca.explained_variance_ratio_
    print(f"  PC1: {ev[0]:.1%}, PC2: {ev[1]:.1%}, total: {sum(ev):.1%}")

    proj_orig   = project(pca, scaler, X_orig)
    proj_smote  = project(pca, scaler, X_smote)
    proj_dtw    = project(pca, scaler, X_dtw)
    proj_linear = project(pca, scaler, X_linear)

    # ── figure ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(6.30, 5.40))
    fig.patch.set_facecolor('white')
    # leave right margin for colourbar
    gs = fig.add_gridspec(2, 2, left=0.09, right=0.86, top=0.93, bottom=0.09,
                          hspace=0.46, wspace=0.36)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]

    norm = mcolors.Normalize(vmin=FMA_MIN, vmax=FMA_MAX)

    PANEL_LABELS = ['(a)', '(b)', '(c)', '(d)']
    SUBTITLES = [
        'Original data',
        'SMOTE augmentation',
        'DTW morphing',
        'Linear interpolation',
    ]
    COUNTS = [len(X_orig), len(X_smote), len(X_dtw), len(X_linear)]
    DATA   = [(proj_orig, y_orig), (proj_smote, y_smote),
              (proj_dtw, y_dtw), (proj_linear, y_linear)]

    pc1_lbl = f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)'
    pc2_lbl = f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)'

    sc = None
    for idx, ((r, c), (proj, labels), lbl, sub, n) in enumerate(
            zip([(0,0),(0,1),(1,0),(1,1)], DATA, PANEL_LABELS, SUBTITLES, COUNTS)):
        ax = axes[r][c]
        sc = ax.scatter(proj[:, 0], proj[:, 1],
                        c=labels, cmap=CMAP, norm=norm,
                        s=10, alpha=0.65, linewidths=0, rasterized=True)
        # panel label + subtitle
        ax.text(-0.14, 1.06, lbl, transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top')
        ax.set_title(sub, fontsize=7.5, pad=3)
        ax.text(0.97, 0.97, f'$n={n:,}$', transform=ax.transAxes,
                fontsize=6.5, ha='right', va='top', color='#555555')
        ax.set_xlabel(pc1_lbl, fontsize=7)
        ax.set_ylabel(pc2_lbl, fontsize=7)
        ax.tick_params(labelsize=6.5, length=2.5, width=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.grid(linestyle='--', alpha=0.20, linewidth=0.4)

    # ── shared colourbar ───────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.88, 0.09, 0.020, 0.84])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('FMA-UE score', fontsize=7.5, labelpad=6)
    cbar.set_ticks([16, 25, 35, 45, 55, 66])
    cbar.ax.tick_params(labelsize=7, length=2.5, width=0.6)
    cbar.outline.set_linewidth(0.6)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    for fmt in ('png', 'pdf'):
        path = OUTPUT_PATH.replace('.png', f'.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {path}")
    plt.close()


if __name__ == '__main__':
    main()
