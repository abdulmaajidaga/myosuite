"""
CVAE Distribution Validation.

Produces two figures:

  Figure A — SMOTE training data vs generated (cvae_validation.png)
    Six FMA levels (18, 20, 30, 40, 50, 66).
    Blue  = mean ±1 SD of 100 randomly-sampled SMOTE training trajectories.
    Red   = single CVAE output (N=10 averaged draws).

  Figure B — Original recordings vs generated (cvae_validation_original.png)
    Only FMA levels with real recordings: 16, 17, 18, 19, 20 (stroke) + 66 (healthy).
    Light blue traces = individual real recordings; bold blue = mean.
    Red dashed = CVAE output.

Usage:
  python scripts/cvae_validation.py
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.utils.config import get_path, get_project_root

COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z',
        'WrVec_x','WrVec_y','WrVec_z','Trunk_x','Trunk_y','Trunk_z']

PLOT_COLS = [('Sh_y', 1), ('El_y', 4), ('Wr_y', 7), ('Trunk_y', 13)]

# FMA levels to compare against SMOTE data
SMOTE_FMA_LEVELS = [18, 20, 30, 40, 50, 66]
# FMA levels with real original recordings (stroke 16-20, healthy 66)
ORIG_FMA_LEVELS  = [16, 17, 18, 19, 20, 66]

REAL_SAMPLE_SIZE = 100


# ─── helpers ──────────────────────────────────────────────────────────────────

def _to_delta(arr):
    return arr - arr[0:1]


def _resample(t, target_len=100):
    if t.shape[0] == target_len:
        return t
    x_old = np.linspace(0, 1, t.shape[0])
    x_new = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, t.shape[1]))
    for c in range(t.shape[1]):
        out[:, c] = interp1d(x_old, t[:, c], kind='linear')(x_new)
    return out


def _mean_std(trajs, target_len=100):
    resampled = [_resample(t, target_len) for t in trajs]
    arr = np.stack(resampled, axis=0)
    return arr.mean(axis=0), arr.std(axis=0)


# ─── data loaders ─────────────────────────────────────────────────────────────

def load_smote_data(augmented_dir, fma_score, max_files=REAL_SAMPLE_SIZE):
    """Return up to max_files SMOTE trajectories (already in delta format)."""
    pattern = os.path.join(augmented_dir, f'*_FMA{fma_score}.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    rng = np.random.RandomState(42)
    if len(files) > max_files:
        files = [files[i] for i in sorted(rng.choice(len(files), max_files, replace=False))]
    trajs = []
    for f in files:
        try:
            arr = pd.read_csv(f).values
            if arr.shape[1] >= 15:
                trajs.append(arr.astype(float))
        except Exception:
            pass
    return trajs if trajs else None


def load_original_data(proc_dir, scores_csv):
    """Load processed recordings grouped by FMA score.

    Returns dict: fma_score -> list of delta-format arrays (100, 15).
    """
    scores_df = pd.read_csv(scores_csv)
    score_map = {row['filename'].replace('.mot', ''): int(row['fma_score'])
                 for _, row in scores_df.iterrows()}

    grouped = {}
    for f in sorted(glob.glob(os.path.join(proc_dir, '*.csv'))):
        stem = os.path.splitext(os.path.basename(f))[0]
        fma  = score_map.get(stem)
        if fma is None:
            continue
        try:
            df = pd.read_csv(f)
            for col in COLS:
                if col not in df.columns:
                    df[col] = 0.0
            arr = _to_delta(df[COLS].values.astype(float))
            grouped.setdefault(fma, []).append(arr)
        except Exception:
            pass
    return grouped


def load_generated(csv_dir, fma_score):
    """Load CVAE output for one FMA level, converted to delta."""
    trajs = []
    single = os.path.join(csv_dir, f'FMA_{fma_score}.csv')
    if os.path.exists(single):
        try:
            arr = pd.read_csv(single).values.astype(float)
            trajs.append(_to_delta(arr))
        except Exception:
            pass
    for f in sorted(glob.glob(os.path.join(csv_dir, f'FMA_{fma_score}_s*.csv'))):
        try:
            arr = pd.read_csv(f).values.astype(float)
            trajs.append(_to_delta(arr))
        except Exception:
            pass
    return trajs if trajs else None


# ─── figure A: SMOTE vs generated ─────────────────────────────────────────────

def create_smote_figure(augmented_dir, csv_dir, figures_dir, output_dir):
    n_fma = len(SMOTE_FMA_LEVELS)
    n_cols = len(PLOT_COLS)

    fig, axes = plt.subplots(n_fma, n_cols, figsize=(4 * n_cols, 3 * n_fma),
                             squeeze=False)
    fig.suptitle('CVAE Validation: SMOTE Training Data vs Generated',
                 fontsize=14, fontweight='bold', y=0.995)

    for row, fma in enumerate(SMOTE_FMA_LEVELS):
        smote_trajs = load_smote_data(augmented_dir, fma)
        gen_trajs   = load_generated(csv_dir, fma)

        for col_i, (col_name, col_idx) in enumerate(PLOT_COLS):
            ax = axes[row, col_i]

            if smote_trajs:
                m, s = _mean_std(smote_trajs)
                t = np.arange(m.shape[0])
                ax.plot(t, m[:, col_idx], color='steelblue', lw=1.5,
                        label='SMOTE training data', alpha=0.9)
                ax.fill_between(t, m[:, col_idx] - s[:, col_idx],
                                m[:, col_idx] + s[:, col_idx],
                                color='steelblue', alpha=0.15)

            if gen_trajs:
                m, s = _mean_std(gen_trajs)
                t = np.arange(m.shape[0])
                ax.plot(t, m[:, col_idx], color='tomato', lw=1.5,
                        ls='--', label='CVAE generated', alpha=0.9)

            if row == 0:
                ax.set_title(col_name, fontweight='bold', fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f'FMA {fma}', fontsize=10, fontweight='bold')
            if row == n_fma - 1:
                ax.set_xlabel('Frame', fontsize=8)
            if row == 0 and col_i == n_cols - 1:
                ax.legend(fontsize=8, loc='upper right')
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for d, name in [(output_dir, 'cvae_validation.png'),
                    (figures_dir, 'cvae_validation.png')]:
        fig.savefig(os.path.join(d, name), dpi=150, bbox_inches='tight',
                    facecolor='white')
        print(f'Saved: {os.path.join(d, name)}')
    plt.close(fig)


# ─── figure B: original recordings vs generated ───────────────────────────────

def create_original_figure(proc_dir, scores_csv, csv_dir, figures_dir, output_dir):
    orig_grouped = load_original_data(proc_dir, scores_csv)

    # Only show FMA levels present in both real and generated data
    fma_levels = [f for f in ORIG_FMA_LEVELS
                  if f in orig_grouped and load_generated(csv_dir, f) is not None]
    if not fma_levels:
        print('No matching original recordings found — skipping figure B.')
        return

    n_fma = len(fma_levels)
    n_cols = len(PLOT_COLS)

    fig, axes = plt.subplots(n_fma, n_cols, figsize=(4 * n_cols, 3 * n_fma),
                             squeeze=False)
    fig.suptitle('CVAE Validation: Original Recordings vs Generated',
                 fontsize=14, fontweight='bold', y=0.995)

    for row, fma in enumerate(fma_levels):
        real_trajs = orig_grouped[fma]
        gen_trajs  = load_generated(csv_dir, fma)

        for col_i, (col_name, col_idx) in enumerate(PLOT_COLS):
            ax = axes[row, col_i]

            # Individual real recordings (light traces)
            for arr in real_trajs:
                r = _resample(arr, 100)
                ax.plot(np.arange(100), r[:, col_idx],
                        color='steelblue', lw=0.8, alpha=0.35)

            # Mean of real recordings (bold)
            m, _ = _mean_std(real_trajs)
            ax.plot(np.arange(m.shape[0]), m[:, col_idx],
                    color='steelblue', lw=2.0,
                    label=f'Real recordings (n={len(real_trajs)})')

            # CVAE generated
            if gen_trajs:
                gm, _ = _mean_std(gen_trajs)
                ax.plot(np.arange(gm.shape[0]), gm[:, col_idx],
                        color='tomato', lw=1.5, ls='--',
                        label='CVAE generated')

            if row == 0:
                ax.set_title(col_name, fontweight='bold', fontsize=10)
            if col_i == 0:
                ax.set_ylabel(f'FMA {fma}', fontsize=10, fontweight='bold')
            if row == n_fma - 1:
                ax.set_xlabel('Frame', fontsize=8)
            if row == 0 and col_i == n_cols - 1:
                ax.legend(fontsize=8, loc='upper right')
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.15)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    for d, name in [(output_dir, 'cvae_validation_original.png'),
                    (figures_dir, 'cvae_validation_original.png')]:
        fig.savefig(os.path.join(d, name), dpi=150, bbox_inches='tight',
                    facecolor='white')
        print(f'Saved: {os.path.join(d, name)}')
    plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    augmented_dir = get_path('data_cutoff_augmented_smote')
    proc_dir      = get_path('data_cutoff_processed')
    scores_csv    = get_path('scores_file')
    csv_dir       = get_path('output_generated_csv')
    output_dir    = get_path('output_generated_plots')
    figures_dir   = os.path.join(get_project_root(), 'figures')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print('Figure A: SMOTE training data vs generated...')
    create_smote_figure(augmented_dir, csv_dir, figures_dir, output_dir)

    print('\nFigure B: original recordings vs generated...')
    create_original_figure(proc_dir, scores_csv, csv_dir, figures_dir, output_dir)


if __name__ == '__main__':
    main()
