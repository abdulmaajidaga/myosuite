"""
Generate small test augmented datasets (3 stroke × 3 healthy) for all three
augmentation methods and regenerate the PCA comparison figure.

Outputs:
  data/kinematic/cutoff/augmented_linear_test/
  data/kinematic/cutoff/augmented_smote_test/
  data/kinematic/cutoff/augmented_dtw_test/
  figures/pca_dataset_comparison_test.png

Usage:
  python scripts/generate_test_datasets.py
  python scripts/generate_test_datasets.py --n-stroke 5 --n-healthy 5
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.spatial.distance import cdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.utils.config import get_path

# ── paths ──────────────────────────────────────────────────────────────────
PROCESSED_DIR   = get_path("data_cutoff_processed")
RAW_HEALTHY_DIR = get_path("data_healthy")
RAW_STROKE_DIR  = get_path("data_stroke")
SCORES_FILE     = get_path("scores_file")

LINEAR_TEST_DIR = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'augmented_linear_test')
SMOTE_TEST_DIR  = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'augmented_smote_test')
DTW_TEST_DIR    = os.path.join(ROOT, 'data', 'kinematic', 'cutoff', 'augmented_dtw_test')

HEALTHY_FMA  = 66
TARGET_LEN   = 100
ARM_COLS     = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
                'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS   = ['Trunk_x','Trunk_y','Trunk_z']
ALL_COLS     = ARM_COLS + TRUNK_COLS


# ── shared utilities ────────────────────────────────────────────────────────

def load_scores():
    df = pd.read_csv(SCORES_FILE)
    m = {}
    for _, row in df.iterrows():
        name = str(row.iloc[0]).replace('.mot','').replace('.csv','').strip()
        m[name] = int(row.iloc[1])
    return m


def load_raw_trunk(raw_path):
    try:
        df_raw = pd.read_csv(raw_path, header=[0, 1])
        new_cols, cur = [], None
        for c0, c1 in df_raw.columns:
            c0 = c0.strip() if isinstance(c0, str) else c0
            if not str(c0).startswith('Unnamed'):
                cur = c0
            new_cols.append((cur, c1.strip() if isinstance(c1, str) else c1))
        df_raw.columns = pd.MultiIndex.from_tuples(new_cols)
        cs = [df_raw[m][['X','Y','Z']].values.astype(float)
              for m in ['CS_1','CS_2','CS_3','CS_4']
              if m in df_raw.columns.get_level_values(0)]
        return np.mean(cs, axis=0) if len(cs) >= 2 else None
    except Exception:
        return None


def load_motion(processed_path, raw_dir):
    df = pd.read_csv(processed_path)
    for col in ARM_COLS:
        if col not in df.columns:
            df[col] = 0.0
    fname    = os.path.basename(processed_path)
    trunk    = load_raw_trunk(os.path.join(raw_dir, fname))
    if trunk is not None:
        if len(trunk) != len(df):
            trunk = resample(trunk, len(df))
        df['Trunk_x'], df['Trunk_y'], df['Trunk_z'] = trunk[:,0], trunk[:,1], trunk[:,2]
    else:
        df['Trunk_x'] = df['Trunk_y'] = df['Trunk_z'] = 0.0
    return df[ALL_COLS]


def to_delta(df):
    d = df.copy()
    first = df.iloc[0]
    for col in ALL_COLS:
        d[col] = df[col] - first[col]
    return d


def resample_df(df, n=TARGET_LEN):
    return pd.DataFrame(
        {c: resample(df[c].values, n) for c in df.columns},
        columns=df.columns)


def save(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# ── DTW helpers ─────────────────────────────────────────────────────────────

def dtw_align(a, b):
    n, m   = len(a), len(b)
    D      = cdist(a, b)
    acc    = np.full((n, m), np.inf)
    acc[0, 0] = D[0, 0]
    for i in range(1, n): acc[i, 0] = acc[i-1, 0] + D[i, 0]
    for j in range(1, m): acc[0, j] = acc[0, j-1] + D[0, j]
    for i in range(1, n):
        for j in range(1, m):
            acc[i, j] = D[i, j] + min(acc[i-1,j-1], acc[i-1,j], acc[i,j-1])
    path, i, j = [], n-1, m-1
    path.append((i, j))
    while i > 0 or j > 0:
        if   i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            idx = np.argmin([acc[i-1,j-1], acc[i-1,j], acc[i,j-1]])
            if   idx == 0: i -= 1; j -= 1
            elif idx == 1: i -= 1
            else:          j -= 1
        path.append((i, j))
    path.reverse()
    return path, float(acc[n-1, m-1])


def dtw_warp(df_s, df_h):
    a = resample_df(df_s)[ALL_COLS].values
    b = resample_df(df_h)[ALL_COLS].values
    path, cost = dtw_align(a, b)
    wa = np.array([a[i] for i,j in path])
    wb = np.array([b[j] for i,j in path])
    def _rs(arr):
        return np.column_stack([resample(arr[:,k], TARGET_LEN) for k in range(arr.shape[1])])
    return (pd.DataFrame(_rs(wa), columns=ALL_COLS),
            pd.DataFrame(_rs(wb), columns=ALL_COLS),
            cost)


# ── LINEAR augmentation ─────────────────────────────────────────────────────

def gen_linear(stroke_data, healthy_data, out_dir):
    print(f"\n[LINEAR] → {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for s_name, (s_fma, df_s) in stroke_data.items():
        for h_name, df_h in healthy_data.items():
            for tgt in range(s_fma + 1, HEALTHY_FMA):
                alpha  = (tgt - s_fma) / (HEALTHY_FMA - s_fma)
                ds     = resample_df(df_s)
                dh     = resample_df(df_h)
                morphed = pd.DataFrame(
                    {c: (1-alpha)*ds[c].values + alpha*dh[c].values for c in ALL_COLS},
                    columns=ALL_COLS)
                save(morphed, os.path.join(out_dir, f"{s_name}_x_{h_name}_FMA{tgt}.csv"))
                count += 1
        # save raw stroke endpoint
        save(resample_df(df_s), os.path.join(out_dir, f"{s_name}_FMA{s_fma}.csv"))
    # save healthy endpoints
    for h_name, df_h in healthy_data.items():
        save(resample_df(df_h), os.path.join(out_dir, f"{h_name}_FMA{HEALTHY_FMA}.csv"))
    print(f"  Generated {count} interpolated files")


# ── SMOTE augmentation ──────────────────────────────────────────────────────

def gen_smote(stroke_data, healthy_data, out_dir):
    """
    Simplified SMOTE: cross-class interpolation only (no k-NN expansion).
    For a full k-NN expansion, use the main generate_augmented_smote.py.
    """
    print(f"\n[SMOTE]  → {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    count = 0

    # Flatten to feature vectors for k-NN-style blending
    all_healthy_vecs = {n: resample_df(df)[ALL_COLS].values.flatten()
                        for n, df in healthy_data.items()}
    all_stroke_data  = list(stroke_data.items())

    for s_name, (s_fma, df_s) in stroke_data.items():
        s_vec = resample_df(df_s)[ALL_COLS].values.flatten()
        for h_name, df_h in healthy_data.items():
            h_vec = all_healthy_vecs[h_name]
            for tgt in range(s_fma + 1, HEALTHY_FMA):
                # SMOTE-style: random perturbation of linear blend
                alpha = (tgt - s_fma) / (HEALTHY_FMA - s_fma)
                rng   = np.random.default_rng(seed=hash((s_name, h_name, tgt)) % (2**31))
                # pick a random neighbour from the healthy pool to add diversity
                neighbour_name = rng.choice(list(all_healthy_vecs.keys()))
                n_vec = all_healthy_vecs[neighbour_name]
                lam   = rng.uniform(0.0, 0.3)   # small SMOTE perturbation
                blend_vec = (1-alpha)*s_vec + alpha*((1-lam)*h_vec + lam*n_vec)
                arr   = blend_vec.reshape(TARGET_LEN, len(ALL_COLS))
                morphed = pd.DataFrame(arr, columns=ALL_COLS)
                save(morphed, os.path.join(out_dir, f"smote_{s_name}_x_{h_name}_FMA{tgt}.csv"))
                count += 1
        save(resample_df(df_s), os.path.join(out_dir, f"{s_name}_FMA{s_fma}.csv"))
    for h_name, df_h in healthy_data.items():
        save(resample_df(df_h), os.path.join(out_dir, f"{h_name}_FMA{HEALTHY_FMA}.csv"))
    print(f"  Generated {count} SMOTE files")


# ── DTW augmentation ────────────────────────────────────────────────────────

def gen_dtw(stroke_data, healthy_data, out_dir):
    print(f"\n[DTW]    → {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for s_name, (s_fma, df_s) in stroke_data.items():
        for h_name, df_h in healthy_data.items():
            print(f"  DTW aligning {s_name} × {h_name} …", end=' ', flush=True)
            df_sa, df_ha, _ = dtw_warp(df_s, df_h)
            print("done")
            for tgt in range(s_fma + 1, HEALTHY_FMA):
                alpha   = (tgt - s_fma) / (HEALTHY_FMA - s_fma)
                morphed = pd.DataFrame(
                    {c: (1-alpha)*df_sa[c].values + alpha*df_ha[c].values for c in ALL_COLS},
                    columns=ALL_COLS)
                save(morphed, os.path.join(out_dir, f"{s_name}_x_{h_name}_FMA{tgt}.csv"))
                count += 1
        save(resample_df(df_s), os.path.join(out_dir, f"{s_name}_FMA{s_fma}.csv"))
    for h_name, df_h in healthy_data.items():
        save(resample_df(df_h), os.path.join(out_dir, f"{h_name}_FMA{HEALTHY_FMA}.csv"))
    print(f"  Generated {count} DTW files")


# ── PCA figure ──────────────────────────────────────────────────────────────

def regen_figure():
    import re
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    ORIG_DIR = PROCESSED_DIR
    SCORES   = load_scores()
    OUT_PNG  = os.path.join(ROOT, 'figures', 'pca_dataset_comparison_test.png')

    fma_pat = re.compile(r'FMA(\d+)', re.IGNORECASE)

    def flatten(df):
        a = df.values.astype(float)
        return np.concatenate([a.mean(0), a.std(0)])

    def load_orig():
        X, y = [], []
        for f in sorted(os.listdir(ORIG_DIR)):
            if not f.endswith('.csv'): continue
            stem = f[:-4]
            fma  = SCORES.get(stem, 66 if not stem.startswith('S') else 18)
            try:
                df = pd.read_csv(os.path.join(ORIG_DIR, f))
                if len(df) < 10: continue
                X.append(flatten(df)); y.append(fma)
            except Exception: continue
        return np.array(X), np.array(y)

    def load_aug(d):
        X, y = [], []
        for f in os.listdir(d):
            if not f.endswith('.csv'): continue
            m = fma_pat.search(f)
            if not m: continue
            fma = int(m.group(1))
            try:
                first = open(os.path.join(d, f)).readline()
                hdr   = 0 if any(c.isalpha() for c in first) else None
                df    = pd.read_csv(os.path.join(d, f), header=hdr)
                if df.shape[1] != 15 or len(df) < 10: continue
                X.append(flatten(df)); y.append(fma)
            except Exception: continue
        return np.array(X), np.array(y)

    print("\nBuilding PCA figure …")
    X_orig,   y_orig   = load_orig()
    X_linear, y_linear = load_aug(LINEAR_TEST_DIR)
    X_smote,  y_smote  = load_aug(SMOTE_TEST_DIR)
    X_dtw,    y_dtw    = load_aug(DTW_TEST_DIR)
    print(f"  orig={len(X_orig)}  linear={len(X_linear)}  smote={len(X_smote)}  dtw={len(X_dtw)}")

    all_X  = np.vstack([X_orig, X_linear, X_smote, X_dtw])
    sc     = StandardScaler().fit(all_X)
    pca    = PCA(n_components=2, random_state=42).fit(sc.transform(all_X))
    ev     = pca.explained_variance_ratio_
    print(f"  PC1={ev[0]:.1%}  PC2={ev[1]:.1%}")

    def proj(X): return pca.transform(sc.transform(X))

    CMAP = 'RdYlGn'
    norm = mcolors.Normalize(vmin=16, vmax=66)
    fig  = plt.figure(figsize=(12, 10))
    gs   = fig.add_gridspec(2, 2, left=0.07, right=0.87, top=0.91, bottom=0.07,
                             hspace=0.38, wspace=0.35)
    fig.suptitle('PCA — Original vs Augmented (test datasets)',
                 fontsize=13, fontweight='bold', y=0.97)

    panels = [
        (fig.add_subplot(gs[0,0]), proj(X_orig),   y_orig,
         f'(a) Original  [n={len(X_orig)}]\nFMA 16–20 stroke + FMA 66 healthy'),
        (fig.add_subplot(gs[0,1]), proj(X_smote),  y_smote,
         f'(b) SMOTE  [n={len(X_smote):,}]'),
        (fig.add_subplot(gs[1,0]), proj(X_dtw),    y_dtw,
         f'(c) DTW  [n={len(X_dtw):,}]'),
        (fig.add_subplot(gs[1,1]), proj(X_linear), y_linear,
         f'(d) Linear  [n={len(X_linear):,}]'),
    ]

    sc_plot = None
    for ax, p, labels, title in panels:
        sc_plot = ax.scatter(p[:,0], p[:,1], c=labels, cmap=CMAP, norm=norm,
                             s=16, alpha=0.75, linewidths=0)
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=6, loc='left')
        ax.set_xlabel(f'PC1 ({ev[0]:.1%})', fontsize=8)
        ax.set_ylabel(f'PC2 ({ev[1]:.1%})', fontsize=8)
        ax.tick_params(labelsize=7.5)
        ax.spines[['top','right']].set_visible(False)

    cbar_ax = fig.add_axes([0.89, 0.07, 0.022, 0.84])
    cb = fig.colorbar(sc_plot, cax=cbar_ax)
    cb.set_label('FMA-UE Score', fontsize=10, labelpad=8)
    cb.set_ticks([16, 25, 35, 45, 55, 66])
    cb.ax.tick_params(labelsize=8.5)

    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUT_PNG}")
    return OUT_PNG


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-stroke',  type=int, default=3,
                    help='Number of stroke patients to use (default 3)')
    ap.add_argument('--n-healthy', type=int, default=3,
                    help='Number of healthy subjects to use (default 3)')
    ap.add_argument('--figure-only', action='store_true',
                    help='Skip generation, just rebuild the figure from existing test dirs')
    args = ap.parse_args()

    if not args.figure_only:
        scores = load_scores()

        stroke_files  = sorted(glob.glob(os.path.join(PROCESSED_DIR, "S*.csv")))[:args.n_stroke]
        healthy_files = sorted([f for f in glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))
                                 if not os.path.basename(f).startswith('S')])[:args.n_healthy]

        print(f"Using {len(stroke_files)} stroke + {len(healthy_files)} healthy files")

        stroke_data = {}
        for f in stroke_files:
            name = os.path.basename(f)[:-4]
            fma  = scores.get(name)
            if fma is None:
                print(f"  Skipping {name}: no FMA score"); continue
            stroke_data[name] = (fma, to_delta(load_motion(f, RAW_STROKE_DIR)))

        healthy_data = {}
        for f in healthy_files:
            name = os.path.basename(f)[:-4]
            healthy_data[name] = to_delta(load_motion(f, RAW_HEALTHY_DIR))

        gen_linear(stroke_data, healthy_data, LINEAR_TEST_DIR)
        gen_smote (stroke_data, healthy_data, SMOTE_TEST_DIR)
        gen_dtw   (stroke_data, healthy_data, DTW_TEST_DIR)

    regen_figure()


if __name__ == '__main__':
    main()
