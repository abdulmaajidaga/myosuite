"""
CVAE Distribution Validation: Compare generated motions to real training data.

For each FMA level:
  - KS test (2-sample) per joint column
  - Wasserstein distance per joint column
  - Trajectory overlay plots (mean real vs mean generated)

Data sources:
  - Real: data/kinematic/cutoff/augmented/ (56k files, sample ~100 per FMA)
  - Generated: output/generated/csv/ (FMA_{score}.csv and FMA_{score}_s{idx}.csv)

Output:
  - output/generated/plots/cvae_validation.md
  - output/generated/plots/cvae_validation.png

Usage:
  python scripts/cvae_validation.py
"""
import os
import sys
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.utils.config import get_path

COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z',
        'WrVec_x','WrVec_y','WrVec_z','Trunk_x','Trunk_y','Trunk_z']

# FMA levels to validate (must match what was generated)
FMA_LEVELS = [18, 20, 30, 40, 50, 66]

# How many real training files to sample per FMA level
REAL_SAMPLE_SIZE = 100


def load_real_data(augmented_dir, fma_score, max_files=REAL_SAMPLE_SIZE):
    """Load up to max_files from augmented data for a given FMA score."""
    pattern = os.path.join(augmented_dir, f'*_FMA{fma_score}.csv')
    files = sorted(glob.glob(pattern))

    if not files:
        return None

    # Random sample if too many
    rng = np.random.RandomState(42)
    if len(files) > max_files:
        indices = rng.choice(len(files), max_files, replace=False)
        files = [files[i] for i in sorted(indices)]

    trajectories = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df.columns) >= 15:
                trajectories.append(df.values)
        except Exception:
            continue

    if not trajectories:
        return None

    return trajectories


def load_generated_data(csv_dir, fma_score):
    """Load all generated CSVs for a given FMA score (FMA_{score}.csv + FMA_{score}_s*.csv)."""
    trajectories = []

    # Single sample
    single = os.path.join(csv_dir, f'FMA_{fma_score}.csv')
    if os.path.exists(single):
        try:
            df = pd.read_csv(single)
            trajectories.append(df.values)
        except Exception:
            pass

    # Multi-sample
    pattern = os.path.join(csv_dir, f'FMA_{fma_score}_s*.csv')
    for f in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(f)
            trajectories.append(df.values)
        except Exception:
            continue

    return trajectories if trajectories else None


def compute_distribution_stats(real_trajs, gen_trajs):
    """Compute KS test and Wasserstein distance per column.

    Compares the pooled frame-level distributions (flatten all frames from all trajectories).
    """
    results = []

    for col_idx, col_name in enumerate(COLS):
        # Pool all frames across all trajectories
        real_vals = np.concatenate([t[:, col_idx] for t in real_trajs if t.shape[1] > col_idx])
        gen_vals = np.concatenate([t[:, col_idx] for t in gen_trajs if t.shape[1] > col_idx])

        if len(real_vals) == 0 or len(gen_vals) == 0:
            results.append({'column': col_name, 'ks_stat': np.nan, 'ks_pvalue': np.nan,
                            'wasserstein': np.nan})
            continue

        ks_stat, ks_p = stats.ks_2samp(real_vals, gen_vals)
        w_dist = stats.wasserstein_distance(real_vals, gen_vals)

        results.append({
            'column': col_name,
            'ks_stat': ks_stat,
            'ks_pvalue': ks_p,
            'wasserstein': w_dist,
        })

    return results


def compute_mean_trajectories(trajs, target_len=100):
    """Compute mean trajectory (resampled to target_len if needed)."""
    resampled = []
    for t in trajs:
        if t.shape[0] == target_len:
            resampled.append(t)
        else:
            # Simple linear interpolation to target length
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, t.shape[0])
            x_new = np.linspace(0, 1, target_len)
            new_t = np.zeros((target_len, t.shape[1]))
            for c in range(t.shape[1]):
                f = interp1d(x_old, t[:, c], kind='linear')
                new_t[:, c] = f(x_new)
            resampled.append(new_t)

    return np.mean(resampled, axis=0), np.std(resampled, axis=0)


def create_validation_figure(all_results, augmented_dir, csv_dir, output_dir):
    """Create overlay plots of real vs generated mean trajectories."""
    # Key joints to plot: Shoulder Y, Elbow Y, Wrist Y (reach), Trunk Y
    plot_cols = [('Sh_y', 1), ('El_y', 4), ('Wr_y', 7), ('Trunk_y', 13)]

    n_fma = len(FMA_LEVELS)
    n_cols = len(plot_cols)

    fig, axes = plt.subplots(n_fma, n_cols, figsize=(4 * n_cols, 3 * n_fma))
    fig.suptitle('CVAE Validation: Real vs Generated Trajectories', fontsize=14, fontweight='bold', y=0.99)

    for row, fma in enumerate(FMA_LEVELS):
        real_trajs = load_real_data(augmented_dir, fma)
        gen_trajs = load_generated_data(csv_dir, fma)

        for col_plot, (col_name, col_idx) in enumerate(plot_cols):
            ax = axes[row, col_plot] if n_fma > 1 else axes[col_plot]

            if real_trajs is not None:
                real_mean, real_std = compute_mean_trajectories(real_trajs)
                if col_idx < real_mean.shape[1]:
                    frames = np.arange(real_mean.shape[0])
                    ax.plot(frames, real_mean[:, col_idx], 'b-', lw=1.5, label='Real', alpha=0.8)
                    ax.fill_between(frames,
                                    real_mean[:, col_idx] - real_std[:, col_idx],
                                    real_mean[:, col_idx] + real_std[:, col_idx],
                                    alpha=0.15, color='blue')

            if gen_trajs is not None:
                gen_mean, gen_std = compute_mean_trajectories(gen_trajs)
                if col_idx < gen_mean.shape[1]:
                    frames = np.arange(gen_mean.shape[0])
                    ax.plot(frames, gen_mean[:, col_idx], 'r--', lw=1.5, label='Generated', alpha=0.8)
                    ax.fill_between(frames,
                                    gen_mean[:, col_idx] - gen_std[:, col_idx],
                                    gen_mean[:, col_idx] + gen_std[:, col_idx],
                                    alpha=0.15, color='red')

            if row == 0:
                ax.set_title(col_name, fontweight='bold')
            if col_plot == 0:
                ax.set_ylabel(f'FMA {fma}', fontsize=10, fontweight='bold')
            if row == n_fma - 1:
                ax.set_xlabel('Frame')
            if row == 0 and col_plot == n_cols - 1:
                ax.legend(fontsize=7, loc='upper right')

            ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(output_dir, 'cvae_validation.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Validation figure saved to: {out_path}')


def main():
    augmented_dir = get_path("data_cutoff_augmented")
    csv_dir = get_path("output_generated_csv")
    output_dir = get_path("output_generated_plots")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("CVAE Distribution Validation")
    print("=" * 60)

    report_lines = ['# CVAE Distribution Validation Report\n']
    all_results = {}

    for fma in FMA_LEVELS:
        print(f"\nFMA {fma}:")

        real_trajs = load_real_data(augmented_dir, fma)
        gen_trajs = load_generated_data(csv_dir, fma)

        if real_trajs is None:
            print(f"  WARNING: No real data found for FMA {fma}")
            report_lines.append(f'\n## FMA {fma}\n\nNo real training data found.\n')
            continue
        if gen_trajs is None:
            print(f"  WARNING: No generated data found for FMA {fma}")
            report_lines.append(f'\n## FMA {fma}\n\nNo generated data found.\n')
            continue

        print(f"  Real trajectories: {len(real_trajs)}, Generated: {len(gen_trajs)}")

        results = compute_distribution_stats(real_trajs, gen_trajs)
        all_results[fma] = results

        report_lines.append(f'\n## FMA {fma} (Real N={len(real_trajs)}, Gen N={len(gen_trajs)})\n')
        report_lines.append('| Column | KS Statistic | KS p-value | Wasserstein | Result |')
        report_lines.append('|--------|-------------|------------|-------------|--------|')

        n_pass = 0
        for r in results:
            if np.isnan(r['ks_pvalue']):
                result = 'N/A'
            elif r['ks_pvalue'] > 0.05:
                result = 'PASS (distributions match)'
                n_pass += 1
            else:
                result = 'DIFF (distributions differ)'

            report_lines.append(
                f"| {r['column']} | {r['ks_stat']:.4f} | {r['ks_pvalue']:.4f} "
                f"| {r['wasserstein']:.4f} | {result} |"
            )

        print(f"  KS test: {n_pass}/{len(results)} columns pass (p>0.05)")

    # Write report
    report_path = os.path.join(output_dir, 'cvae_validation.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f'\nReport saved to: {report_path}')

    # Create trajectory overlay figure
    print('Creating trajectory overlay figure...')
    create_validation_figure(all_results, augmented_dir, csv_dir, output_dir)


if __name__ == '__main__':
    main()
