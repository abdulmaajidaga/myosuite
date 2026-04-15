"""
Visualize SMOTE augmentation quality:
  1. PCA of all samples colored by FMA score — confirms smooth gradient
  2. Wrist trajectories across FMA spectrum — confirms motion realism
  3. Kinematic metrics across FMA — confirms monotonic trends
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.decomposition import PCA
from scipy.signal import resample
import os
import sys
import glob
import re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path

SMOTE_DIR = get_path("data_cutoff_augmented_smote")
OUTPUT_DIR = get_path("output_generated_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
            'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z',
            'Trunk_x','Trunk_y','Trunk_z']


def load_samples(data_dir, max_per_fma=50):
    """Load a balanced subset of augmented files."""
    all_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))

    # Group by FMA
    fma_groups = {}
    for fpath in all_files:
        match = re.search(r'_FMA(\d+)\.csv$', os.path.basename(fpath))
        if match:
            score = int(match.group(1))
            fma_groups.setdefault(score, []).append(fpath)

    # Sample up to max_per_fma per score
    np.random.seed(42)
    flat_data = []   # (N, 1500)
    fma_scores = []  # (N,)
    raw_data = []    # (N, 100, 15) for trajectory vis

    for score in sorted(fma_groups.keys()):
        files = fma_groups[score]
        if len(files) > max_per_fma:
            files = list(np.random.choice(files, max_per_fma, replace=False))

        for fpath in files:
            try:
                df = pd.read_csv(fpath)
                arr = df[ALL_COLS].values  # (100, 15)
                flat_data.append(arr.flatten())
                fma_scores.append(score)
                raw_data.append(arr)
            except Exception:
                pass

    return np.array(flat_data), np.array(fma_scores), np.array(raw_data)


def plot_pca(flat_data, fma_scores, output_path):
    """PCA projection colored by FMA score."""
    pca = PCA(n_components=2)
    proj = pca.fit_transform(flat_data)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: scatter plot
    ax = axes[0]
    norm = Normalize(vmin=16, vmax=66)
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=fma_scores, cmap='RdYlGn',
                    norm=norm, s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='FMA Score')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title('PCA of SMOTE Augmented Data')

    # Right: mean PC1 per FMA (should be monotonic)
    ax = axes[1]
    unique_fma = sorted(set(fma_scores))
    mean_pc1 = [proj[fma_scores == f, 0].mean() for f in unique_fma]
    mean_pc2 = [proj[fma_scores == f, 1].mean() for f in unique_fma]
    ax.plot(unique_fma, mean_pc1, 'o-', color='tab:blue', markersize=3, label='Mean PC1')
    ax.plot(unique_fma, mean_pc2, 's-', color='tab:orange', markersize=3, label='Mean PC2')
    ax.set_xlabel('FMA Score')
    ax.set_ylabel('Mean PC Value')
    ax.set_title('Mean PCA Components vs FMA')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA plot: {output_path}")


def plot_trajectories(raw_data, fma_scores, output_path):
    """Wrist XYZ trajectories at selected FMA levels."""
    target_fmas = [16, 25, 35, 45, 55, 66]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=16, vmax=66)

    for i, fma in enumerate(target_fmas):
        ax = axes[i]
        mask = fma_scores == fma
        if not mask.any():
            # Find nearest
            fma = min(set(fma_scores), key=lambda x: abs(x - fma))
            mask = fma_scores == fma

        samples = raw_data[mask]
        # Plot up to 10 samples
        n_plot = min(10, len(samples))
        color = cmap(norm(fma))

        for j in range(n_plot):
            arr = samples[j]
            # Wrist Y (col 7) — main reaching axis
            ax.plot(arr[:, 7], color=color, alpha=0.4, linewidth=1)

        # Plot mean
        mean_traj = samples[:n_plot].mean(axis=0)
        ax.plot(mean_traj[:, 7], color='black', linewidth=2, label='Mean')

        ax.set_title(f'FMA {fma} (n={mask.sum()})', fontsize=12)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Wrist Y (mm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Wrist Y Trajectories Across FMA Spectrum', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot: {output_path}")


def plot_metrics(raw_data, fma_scores, output_path):
    """Kinematic metrics across FMA spectrum."""
    unique_fma = sorted(set(fma_scores))

    metrics = {fma: {'peak_vel': [], 'wrist_range': [], 'trunk_disp': [], 'norm_jerk': []}
               for fma in unique_fma}

    for idx in range(len(raw_data)):
        arr = raw_data[idx]  # (100, 15)
        fma = fma_scores[idx]

        # Wrist velocity (cols 6:9)
        wrist = arr[:, 6:9]
        vel = np.diff(wrist, axis=0)
        speed = np.linalg.norm(vel, axis=1)
        metrics[fma]['peak_vel'].append(speed.max())

        # Wrist range Y
        metrics[fma]['wrist_range'].append(np.ptp(arr[:, 7]))

        # Trunk displacement
        trunk = arr[:, 12:15]
        metrics[fma]['trunk_disp'].append(np.max(np.linalg.norm(trunk, axis=1)))

        # Normalized jerk
        acc = np.diff(vel, axis=0)
        jerk = np.diff(acc, axis=0)
        jerk_sq = np.sum(jerk ** 2)
        path_len = speed.sum() + 1e-6
        T = float(len(arr))
        metrics[fma]['norm_jerk'].append((T ** 5) / (2 * path_len ** 2) * jerk_sq)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    titles = ['Peak Velocity', 'Wrist Range Y', 'Trunk Displacement', 'Normalized Jerk']
    keys = ['peak_vel', 'wrist_range', 'trunk_disp', 'norm_jerk']
    ylabels = ['mm/frame', 'mm', 'mm', 'a.u.']

    for i, (ax, title, key, ylabel) in enumerate(zip(axes.flatten(), titles, keys, ylabels)):
        means = [np.mean(metrics[f][key]) for f in unique_fma]
        stds = [np.std(metrics[f][key]) for f in unique_fma]

        ax.fill_between(unique_fma,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, color='tab:blue')
        ax.plot(unique_fma, means, 'o-', markersize=2, color='tab:blue')
        ax.set_xlabel('FMA Score')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Kinematic Metrics of SMOTE Augmented Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics plot: {output_path}")


def main():
    print("Loading SMOTE augmented data...")
    flat_data, fma_scores, raw_data = load_samples(SMOTE_DIR, max_per_fma=50)
    print(f"Loaded {len(flat_data)} samples, FMA range {fma_scores.min()}-{fma_scores.max()}")

    plot_pca(flat_data, fma_scores,
             os.path.join(OUTPUT_DIR, "smote_pca.png"))
    plot_trajectories(raw_data, fma_scores,
                      os.path.join(OUTPUT_DIR, "smote_trajectories.png"))
    plot_metrics(raw_data, fma_scores,
                 os.path.join(OUTPUT_DIR, "smote_metrics.png"))

    print("\nAll plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
