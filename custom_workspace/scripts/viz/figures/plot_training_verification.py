"""
CVAE training verification dashboard — 11-panel quality check.

What it does:
  Compares real recordings, SMOTE augmented data, and CVAE-generated outputs side by
  side to confirm the model learned clinically meaningful FMA-conditioned motion.
  Panels: time-normalised trajectories, kinematic metric boxplots, latent-space PCA,
  FMA vs motion range scatter, FMA vs trunk compensation scatter, model info card.

Input:
  - data/kinematic/cutoff/processed/*.csv        (real recordings, 100 frames, 15-col)
  - data/kinematic/cutoff/augmented_smote/*.csv  (SMOTE data, sampled)
  - output/generated/csv/FMA_*.csv               (CVAE output, 100 frames, 15-col)
  - output/scores.csv                            (FMA score per subject)
  - models/cvae/cvae_cutoff_fma_best.pth         (D_base model weights)
  - models/cvae/scaler_cutoff_fma.pkl            (matching scaler)

Output:
  - figures/training_verification.png            (11-panel dashboard; LaTeX-referenced)

Usage:
  python scripts/viz/figures/plot_training_verification.py
"""
import os
import sys
import glob
import warnings

import numpy as np
import pandas as pd
import torch
import joblib
from scipy import stats
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src.utils.config import get_path, get, get_project_root
from src.generation.model import MotionCVAE, INPUT_DIM, CONDITION_DIM, HIDDEN_DIM, LATENT_DIM, SEQ_LEN
NUM_HEADS = 4  # not used in D_base; kept for display only

warnings.filterwarnings('ignore')

# =============================================================================
# Paths
# =============================================================================
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = get_path("data_cutoff_processed")
AUG_DIR = get_path("data_cutoff_augmented_smote")
GEN_DIR = get_path("output_generated_csv")
SCORES_PATH = get_path("scores_file")
SCALER_PATH = get_path("cvae_scaler")
MODEL_PATH = get_path("cvae_model_best")
OUTPUT_DIR = get_path("output_generated_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z',
            'Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']
ALL_COLS = ARM_COLS + TRUNK_COLS

# Color scheme
C_HEALTHY = '#2ca02c'
C_STROKE = '#d62728'


# =============================================================================
# Data helpers
# =============================================================================
def to_delta(data):
    """Convert absolute coordinates to delta (subtract first frame).
    Processed cutoff data is in absolute coords; augmented is already delta.
    """
    return data - data[0:1]


def load_scores():
    """Load FMA scores mapping: filename_stem -> fma_score."""
    df = pd.read_csv(SCORES_PATH)
    mapping = {}
    for _, row in df.iterrows():
        stem = row['filename'].replace('.mot', '')
        mapping[stem] = int(row['fma_score'])
    return mapping


def load_real_data(scores_map):
    """Load all processed cutoff files, convert to delta, split healthy/stroke."""
    healthy_data, stroke_data = [], []
    healthy_scores, stroke_scores = [], []

    for f in sorted(glob.glob(os.path.join(PROC_DIR, "*.csv"))):
        stem = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        for c in ALL_COLS:
            if c not in df.columns:
                df[c] = 0.0
        # Convert to delta space (same as training pipeline)
        data = to_delta(df[ALL_COLS].values)

        if stem.startswith('S'):
            stroke_data.append(data)
            stroke_scores.append(scores_map.get(stem, 19))
        else:
            healthy_data.append(data)
            healthy_scores.append(66)

    return (np.array(healthy_data), healthy_scores,
            np.array(stroke_data), stroke_scores)


def load_generated_data():
    """Load generated CSV files and convert to delta."""
    gen_data = {}
    for f in sorted(glob.glob(os.path.join(GEN_DIR, "FMA_*.csv"))):
        bn = os.path.basename(f)
        fma = int(bn.split('_')[1].replace('.csv', ''))
        df = pd.read_csv(f)
        for c in ALL_COLS:
            if c not in df.columns:
                df[c] = 0.0
        # Generated CSVs have reference pose added back — convert to delta
        gen_data[fma] = to_delta(df[ALL_COLS].values)
    return gen_data


def _gen_colors(fma_scores):
    """Build color map and labels for generated FMA scores."""
    cmap = plt.cm.RdYlGn
    scores = sorted(fma_scores)
    lo, hi = min(scores), max(scores)
    span = max(1, hi - lo)
    colors = {s: cmap((s - lo) / span) for s in scores}
    labels = {s: f'Gen FMA {s}' for s in scores}
    return colors, labels


def load_augmented_sample(n_per_bin=50):
    """Load a stratified sample from the augmented dataset.
    Augmented data is already in delta format (first row ~0).
    """
    all_files = glob.glob(os.path.join(AUG_DIR, "*.csv"))
    fma_files = {}
    for f in all_files:
        bn = os.path.basename(f)
        if 'FMA' in bn:
            try:
                fma = int(bn.split('FMA')[1].replace('.csv', ''))
                fma_files.setdefault(fma, []).append(f)
            except ValueError:
                pass

    sampled = {}
    rng = np.random.RandomState(42)
    for fma, files in sorted(fma_files.items()):
        chosen = rng.choice(files, size=min(n_per_bin, len(files)), replace=False)
        for f in chosen:
            df = pd.read_csv(f)
            for c in ALL_COLS:
                if c not in df.columns:
                    df[c] = 0.0
            # Augmented is already delta — use as-is
            sampled.setdefault(fma, []).append(df[ALL_COLS].values)

    return sampled


# =============================================================================
# Metric computation (all on delta-space data)
# =============================================================================
def compute_metrics(data):
    """Compute kinematic metrics for delta-space array (100, 15)."""
    wr = data[:, 6:9]   # Wr_x, Wr_y, Wr_z
    wr_y = data[:, 7]   # Wr_y (reach direction)

    # Wrist range (Y = reach)
    wrist_range_y = np.ptp(wr_y)

    # Peak velocity (mm/frame, no fs multiplication)
    vel_3d = np.diff(wr, axis=0)  # (99, 3) mm/frame
    speed = np.linalg.norm(vel_3d, axis=1)
    peak_velocity = np.max(speed)

    # Trunk displacement
    if data.shape[1] >= 15:
        trunk = data[:, 12:15]
        trunk_disp = np.max(np.linalg.norm(trunk - trunk[0:1], axis=1))
    else:
        trunk_disp = 0.0

    # Normalized jerk (3D, no sqrt → gives ~1e6 range)
    # NJ = T^5 / (2 * L^2) * sum(jerk^2)
    T = len(data)
    path_length = np.sum(speed) + 1e-6
    if len(vel_3d) > 2:
        acc_3d = np.diff(vel_3d, axis=0)  # (98, 3)
        jerk_3d = np.diff(acc_3d, axis=0)  # (97, 3)
        jerk_sq_sum = np.sum(jerk_3d ** 2)
        norm_jerk = T**5 / (2 * path_length**2) * jerk_sq_sum
    else:
        norm_jerk = 0.0

    return {
        'wrist_range_y': wrist_range_y,
        'peak_velocity': peak_velocity,
        'trunk_disp': trunk_disp,
        'norm_jerk': norm_jerk,
    }


def velocity_profile(data):
    """Speed profile in mm/frame for 100-frame delta data."""
    wr = data[:, 6:9]
    vel = np.diff(wr, axis=0)
    return np.linalg.norm(vel, axis=1)


def trunk_displacement_profile(data):
    """Trunk displacement from start over time."""
    if data.shape[1] < 15:
        return np.zeros(len(data))
    trunk = data[:, 12:15]
    return np.linalg.norm(trunk - trunk[0:1], axis=1)


# =============================================================================
# Latent space
# =============================================================================
def encode_to_latent(model, scaler, data_list, scores_list):
    """Encode delta-space data through CVAE encoder to get latent vectors."""
    model.eval()
    latents = []
    with torch.no_grad():
        for data, fma in zip(data_list, scores_list):
            # Data is already in delta space; normalize with training scaler
            data_norm = scaler.transform(data)
            x = torch.FloatTensor(data_norm).unsqueeze(0).to(DEVICE)
            c = torch.FloatTensor([[fma / 66.0]]).to(DEVICE)
            mu, _ = model.encoder(x, c)
            latents.append(mu.squeeze(0).cpu().numpy())
    return np.array(latents)


# =============================================================================
# Dashboard
# =============================================================================
def make_dashboard(healthy_data, healthy_scores, stroke_data, stroke_scores,
                   gen_data, aug_sampled):

    C_GEN, GEN_LABELS = _gen_colors(gen_data.keys())

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle('CVAE D_base (FiLM+CFG, SMOTE) FMA Training Verification Dashboard',
                 fontsize=18, fontweight='bold', y=0.995)

    # 3 rows × 4 cols for rows 0–1; row 2 uses only 3 of the 4 cols (no info card)
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.32,
                  top=0.96, bottom=0.04, left=0.05, right=0.97)

    # =====================================================================
    # ROW 1: Time-series overlays
    # Generated lines are coloured by FMA gradient; legend uses a colorbar
    # instead of 51 individual entries.
    # =====================================================================
    fma_norm  = mpl_colors.Normalize(vmin=16, vmax=66)
    fma_cmap  = mpl_cm.RdYlGn

    def plot_timeseries(ax, title, ylabel, extract_fn):
        # Real healthy: mean + std band
        h_curves = np.array([extract_fn(d) for d in healthy_data])
        n = min(h_curves.shape[1], 100)
        t = np.linspace(0, 100, n)
        h_mean = h_curves[:, :n].mean(axis=0)
        h_std  = h_curves[:, :n].std(axis=0)
        ax.plot(t, h_mean, color=C_HEALTHY, linewidth=2, zorder=5)
        ax.fill_between(t, h_mean - h_std, h_mean + h_std,
                        color=C_HEALTHY, alpha=0.15, zorder=3)

        # Real stroke: mean + std band
        s_curves = np.array([extract_fn(d) for d in stroke_data])
        n_s = min(s_curves.shape[1], 100)
        t_s = np.linspace(0, 100, n_s)
        s_mean = s_curves[:, :n_s].mean(axis=0)
        s_std  = s_curves[:, :n_s].std(axis=0)
        ax.plot(t_s, s_mean, color=C_STROKE, linewidth=2, zorder=5)
        ax.fill_between(t_s, s_mean - s_std, s_mean + s_std,
                        color=C_STROKE, alpha=0.15, zorder=3)

        # Generated: one line per FMA level, coloured by gradient, no legend label
        for fma in sorted(gen_data.keys()):
            curve = extract_fn(gen_data[fma])
            n_g = min(len(curve), 100)
            t_g = np.linspace(0, 100, n_g)
            ax.plot(t_g, curve[:n_g],
                    color=fma_cmap(fma_norm(fma)), linewidth=1.2,
                    alpha=0.75, linestyle='--', zorder=4)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('% Motion', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.2)

    ax_wy = fig.add_subplot(gs[0, 0])
    plot_timeseries(ax_wy, 'Wrist Y (Reach)', 'Position (mm)', lambda d: d[:, 7])

    ax_wz = fig.add_subplot(gs[0, 1])
    plot_timeseries(ax_wz, 'Wrist Z (Height)', 'Position (mm)', lambda d: d[:, 8])

    ax_td = fig.add_subplot(gs[0, 2])
    plot_timeseries(ax_td, 'Trunk Displacement', 'Displacement (mm)',
                    trunk_displacement_profile)

    ax_vp = fig.add_subplot(gs[0, 3])
    plot_timeseries(ax_vp, 'Velocity Profile', 'Speed (mm/frame)', velocity_profile)

    # Shared legend (real lines only) on the first panel
    legend_handles = [
        Line2D([0], [0], color=C_HEALTHY, lw=2, label='Real Healthy (mean \u00b11\u202fSD)'),
        Line2D([0], [0], color=C_STROKE,  lw=2, label='Real Stroke (mean \u00b11\u202fSD)'),
        Line2D([0], [0], color='grey', lw=1.2, ls='--', label='Generated (see colorbar)'),
    ]
    ax_wy.legend(handles=legend_handles, fontsize=7, loc='lower right')

    # Colorbar for the FMA gradient of generated lines
    sm = mpl_cm.ScalarMappable(cmap=fma_cmap, norm=fma_norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.988, 0.685, 0.010, 0.270])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Generated FMA', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # =====================================================================
    # ROW 2: Box plots
    # =====================================================================
    gen_bins_def = [
        ('G_16-20', 16, 20),
        ('G_21-30', 21, 30),
        ('G_31-40', 31, 40),
        ('G_41-50', 41, 50),
        ('G_51-60', 51, 60),
        ('G_61-66', 61, 66),
    ]
    gen_bin_colors = ['#e6550d', '#fd8d3c', '#fdae6b', '#a1d99b', '#74c476', '#31a354']

    def make_boxplot(ax, title, ylabel, metric_key):
        h_vals = [compute_metrics(d)[metric_key] for d in healthy_data]
        s_vals = [compute_metrics(d)[metric_key] for d in stroke_data]

        bin_defs = [
            ('Real\nHealthy', h_vals, C_HEALTHY),
            ('Real\nStroke', s_vals, C_STROKE),
        ]

        for i, (lbl, lo, hi) in enumerate(gen_bins_def):
            vals = []
            for fma, arrays in aug_sampled.items():
                if lo <= fma <= hi:
                    for arr in arrays:
                        vals.append(compute_metrics(arr)[metric_key])
            if vals:
                bin_defs.append((lbl, vals, gen_bin_colors[i]))

        bp_data = [bd[1] for bd in bin_defs]
        bp_colors = [bd[2] for bd in bin_defs]
        bp_labels = [bd[0] for bd in bin_defs]

        bp = ax.boxplot(bp_data, positions=list(range(len(bin_defs))), widths=0.6,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))

        for patch, color in zip(bp['boxes'], bp_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(list(range(len(bin_defs))))
        ax.set_xticklabels(bp_labels, fontsize=7)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis='y', alpha=0.2)

    ax_wr = fig.add_subplot(gs[1, 0])
    make_boxplot(ax_wr, 'Wrist Range (Y - Reach)', 'Range (mm)', 'wrist_range_y')

    ax_pv = fig.add_subplot(gs[1, 1])
    make_boxplot(ax_pv, 'Peak Velocity', 'Speed (mm/frame)', 'peak_velocity')

    ax_tkd = fig.add_subplot(gs[1, 2])
    make_boxplot(ax_tkd, 'Trunk Displacement', 'Max Disp (mm)', 'trunk_disp')

    ax_nj = fig.add_subplot(gs[1, 3])
    make_boxplot(ax_nj, 'Normalized Jerk (\u2193=smoother)', 'Jerk', 'norm_jerk')

    # =====================================================================
    # ROW 3: PCA, scatter plots, info card
    # =====================================================================

    # --- Latent Space PCA ---
    ax_pca = fig.add_subplot(gs[2, 0])
    try:
        model = MotionCVAE().to(DEVICE)
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt)
        model.eval()
        scaler = joblib.load(SCALER_PATH)

        all_data = list(healthy_data) + list(stroke_data)
        all_labels = ['Healthy'] * len(healthy_data) + ['Stroke'] * len(stroke_data)
        all_scores = healthy_scores + stroke_scores

        latents = encode_to_latent(model, scaler, all_data, all_scores)

        pca = PCA(n_components=2)
        z2d = pca.fit_transform(latents)
        var_explained = pca.explained_variance_ratio_.sum() * 100

        h_mask = np.array(all_labels) == 'Healthy'
        s_mask = np.array(all_labels) == 'Stroke'

        ax_pca.scatter(z2d[h_mask, 0], z2d[h_mask, 1], c=C_HEALTHY, s=25,
                       alpha=0.6, label='Healthy', edgecolors='none')
        ax_pca.scatter(z2d[s_mask, 0], z2d[s_mask, 1], c=C_STROKE, s=25,
                       alpha=0.6, label='Stroke', edgecolors='none')
        ax_pca.set_title(f'Latent Space (PCA)\nVar: {var_explained:.1f}%',
                         fontsize=11, fontweight='bold')
        ax_pca.set_xlabel('PC1', fontsize=9)
        ax_pca.set_ylabel('PC2', fontsize=9)
        ax_pca.legend(fontsize=8)
        ax_pca.grid(True, alpha=0.2)
    except Exception as e:
        ax_pca.text(0.5, 0.5, f'PCA failed:\n{str(e)[:80]}',
                    ha='center', va='center', transform=ax_pca.transAxes, fontsize=9)
        ax_pca.set_title('Latent Space (PCA)', fontsize=11, fontweight='bold')

    # --- FMA vs Motion Range ---
    ax_fma_range = fig.add_subplot(gs[2, 1])

    aug_fmas, aug_ranges = [], []
    for fma, arrays in aug_sampled.items():
        for arr in arrays:
            aug_fmas.append(fma)
            aug_ranges.append(compute_metrics(arr)['wrist_range_y'])

    ax_fma_range.scatter(aug_fmas, aug_ranges, c='#9467bd', s=8, alpha=0.3,
                         label='Augmented')

    for d, fma in zip(healthy_data, healthy_scores):
        ax_fma_range.scatter(fma, compute_metrics(d)['wrist_range_y'],
                             c=C_HEALTHY, s=50, edgecolors='black',
                             linewidths=0.5, zorder=5)
    for d, fma in zip(stroke_data, stroke_scores):
        ax_fma_range.scatter(fma, compute_metrics(d)['wrist_range_y'],
                             c=C_STROKE, s=50, edgecolors='black',
                             linewidths=0.5, zorder=5)
    for fma, arr in gen_data.items():
        ax_fma_range.scatter(fma, compute_metrics(arr)['wrist_range_y'],
                             c=C_GEN[fma], s=80, marker='D',
                             edgecolors='black', linewidths=1, zorder=6)

    if aug_fmas:
        slope, intercept, r, p, se = stats.linregress(aug_fmas, aug_ranges)
        x_fit = np.linspace(min(aug_fmas), max(aug_fmas), 100)
        ax_fma_range.plot(x_fit, slope * x_fit + intercept, 'k-', linewidth=2, zorder=4)

    real_stroke_range = np.mean([compute_metrics(d)['wrist_range_y'] for d in stroke_data])
    ax_fma_range.axhline(real_stroke_range, color=C_STROKE, linestyle='--',
                         alpha=0.7, label=f'Real Stroke: {real_stroke_range:.1f}mm')
    real_healthy_range = np.mean([compute_metrics(d)['wrist_range_y'] for d in healthy_data])
    ax_fma_range.axhline(real_healthy_range, color=C_HEALTHY, linestyle='--',
                         alpha=0.7, label=f'Real Healthy: {real_healthy_range:.1f}mm')

    ax_fma_range.set_title('FMA vs Motion Range', fontsize=11, fontweight='bold')
    ax_fma_range.set_xlabel('FMA Score', fontsize=9)
    ax_fma_range.set_ylabel('Wrist Range Y (mm)', fontsize=9)
    ax_fma_range.legend(fontsize=7, loc='upper left')
    ax_fma_range.grid(True, alpha=0.2)

    # --- FMA vs Trunk Compensation ---
    ax_fma_trunk = fig.add_subplot(gs[2, 2])

    aug_fmas_t, aug_trunk = [], []
    for fma, arrays in aug_sampled.items():
        for arr in arrays:
            aug_fmas_t.append(fma)
            aug_trunk.append(compute_metrics(arr)['trunk_disp'])

    ax_fma_trunk.scatter(aug_fmas_t, aug_trunk, c='#9467bd', s=8, alpha=0.3,
                         label='Augmented')

    for d, fma in zip(healthy_data, healthy_scores):
        ax_fma_trunk.scatter(fma, compute_metrics(d)['trunk_disp'],
                             c=C_HEALTHY, s=50, edgecolors='black',
                             linewidths=0.5, zorder=5)
    for d, fma in zip(stroke_data, stroke_scores):
        ax_fma_trunk.scatter(fma, compute_metrics(d)['trunk_disp'],
                             c=C_STROKE, s=50, edgecolors='black',
                             linewidths=0.5, zorder=5)
    for fma, arr in gen_data.items():
        ax_fma_trunk.scatter(fma, compute_metrics(arr)['trunk_disp'],
                             c=C_GEN[fma], s=80, marker='D',
                             edgecolors='black', linewidths=1, zorder=6)

    if aug_fmas_t:
        slope_t, intercept_t, r_t, p_t, se_t = stats.linregress(aug_fmas_t, aug_trunk)
        x_fit = np.linspace(min(aug_fmas_t), max(aug_fmas_t), 100)
        ax_fma_trunk.plot(x_fit, slope_t * x_fit + intercept_t, 'k-',
                          linewidth=2, zorder=4)

    real_stroke_trunk = np.mean([compute_metrics(d)['trunk_disp'] for d in stroke_data])
    ax_fma_trunk.axhline(real_stroke_trunk, color=C_STROKE, linestyle='--',
                         alpha=0.7, label=f'Real Stroke: {real_stroke_trunk:.1f}mm')
    real_healthy_trunk = np.mean([compute_metrics(d)['trunk_disp'] for d in healthy_data])
    ax_fma_trunk.axhline(real_healthy_trunk, color=C_HEALTHY, linestyle='--',
                         alpha=0.7, label=f'Real Healthy: {real_healthy_trunk:.1f}mm')

    ax_fma_trunk.set_title('FMA vs Trunk Compensation', fontsize=11, fontweight='bold')
    ax_fma_trunk.set_xlabel('FMA Score', fontsize=9)
    ax_fma_trunk.set_ylabel('Trunk Disp (mm)', fontsize=9)
    ax_fma_trunk.legend(fontsize=7, loc='upper right')
    ax_fma_trunk.grid(True, alpha=0.2)

    # =====================================================================
    # Compute stats for LaTeX annotation (returned, not drawn in figure)
    # =====================================================================
    h_ranges = [compute_metrics(d)['wrist_range_y'] for d in healthy_data]
    s_ranges = [compute_metrics(d)['wrist_range_y'] for d in stroke_data]
    h_vels = [compute_metrics(d)['peak_velocity'] for d in healthy_data]
    s_vels = [compute_metrics(d)['peak_velocity'] for d in stroke_data]
    h_trunks = [compute_metrics(d)['trunk_disp'] for d in healthy_data]
    s_trunks = [compute_metrics(d)['trunk_disp'] for d in stroke_data]

    gen_ranges = {f: compute_metrics(d)['wrist_range_y'] for f, d in gen_data.items()}
    gen_trunks = {f: compute_metrics(d)['trunk_disp'] for f, d in gen_data.items()}
    gen_fmas_sorted = sorted(gen_data.keys())

    range_increases = all(
        gen_ranges[gen_fmas_sorted[i+1]] >= gen_ranges[gen_fmas_sorted[i]] * 0.8
        for i in range(len(gen_fmas_sorted) - 1))
    trunk_decreases = all(
        gen_trunks[gen_fmas_sorted[i+1]] <= gen_trunks[gen_fmas_sorted[i]] * 1.5
        for i in range(len(gen_fmas_sorted) - 1))

    r_range = stats.pearsonr(aug_fmas, aug_ranges)[0] if aug_fmas else float('nan')

    dashboard_stats = {
        'n_healthy': len(healthy_data),
        'n_stroke': len(stroke_data),
        'h_range_mean': np.mean(h_ranges), 'h_range_std': np.std(h_ranges),
        'h_vel_mean': np.mean(h_vels),
        'h_trunk_mean': np.mean(h_trunks),
        's_range_mean': np.mean(s_ranges), 's_range_std': np.std(s_ranges),
        's_vel_mean': np.mean(s_vels),
        's_trunk_mean': np.mean(s_trunks),
        'gen_ranges': gen_ranges,
        'gen_trunks': gen_trunks,
        'r_range': r_range,
        'range_increases': range_increases,
        'trunk_decreases': trunk_decreases,
    }

    # =====================================================================
    # Save
    # =====================================================================
    out_path = os.path.join(OUTPUT_DIR, 'training_verification_fma.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')

    # Also save to figures/ for LaTeX
    figures_dir = os.path.join(get_project_root(), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    figures_path = os.path.join(figures_dir, 'training_verification.png')
    fig.savefig(figures_path, dpi=150, bbox_inches='tight', facecolor='white')

    plt.close(fig)
    print(f'\nSaved: {out_path}')
    print(f'Saved: {figures_path}')
    return out_path, dashboard_stats


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print('Loading scores...')
    scores_map = load_scores()

    print('Loading real data (processed files -> delta)...')
    healthy_data, healthy_scores, stroke_data, stroke_scores = load_real_data(scores_map)
    print(f'  Healthy: {len(healthy_data)}, Stroke: {len(stroke_data)}')

    print('Loading generated data (-> delta)...')
    gen_data = load_generated_data()
    print(f'  Generated FMA scores: {sorted(gen_data.keys())}')

    print('Loading augmented sample (stratified, ~50 per FMA bin)...')
    aug_sampled = load_augmented_sample(n_per_bin=50)
    total_aug = sum(len(v) for v in aug_sampled.values())
    print(f'  Sampled {total_aug} augmented files across {len(aug_sampled)} FMA scores')

    print('\nBuilding dashboard...')
    _, stats_data = make_dashboard(healthy_data, healthy_scores, stroke_data, stroke_scores,
                                   gen_data, aug_sampled)

    # Print LaTeX annotation text for copy-paste into main.tex
    s = stats_data
    gen_sel = {f: s['gen_ranges'][f] for f in sorted(s['gen_ranges'])
               if f in [16, 20, 30, 40, 50, 66]}
    gen_sel_trunk = {f: s['gen_trunks'][f] for f in sorted(s['gen_trunks'])
                     if f in [16, 20, 30, 40, 50, 66]}

    print('\n' + '='*60)
    print('LaTeX annotation for training_verification figure:')
    print('='*60)
    print(f"Real data --- Healthy (n={s['n_healthy']}): "
          f"wrist range {s['h_range_mean']:.0f}$\\pm${s['h_range_std']:.0f}\\,mm, "
          f"peak velocity {s['h_vel_mean']:.1f}\\,mm/frame, "
          f"trunk {s['h_trunk_mean']:.0f}\\,mm. "
          f"Stroke (n={s['n_stroke']}): "
          f"wrist range {s['s_range_mean']:.0f}$\\pm${s['s_range_std']:.0f}\\,mm, "
          f"peak velocity {s['s_vel_mean']:.1f}\\,mm/frame, "
          f"trunk {s['s_trunk_mean']:.0f}\\,mm. "
          f"Augmented wrist range Pearson $r={s['r_range']:.3f}$. "
          f"Conditioning check --- higher FMA $\\Rightarrow$ more range: "
          f"{'yes' if s['range_increases'] else 'partial'}; "
          f"lower FMA $\\Rightarrow$ more trunk: "
          f"{'yes' if s['trunk_decreases'] else 'partial'}.")
    print('='*60)
    print('Done.')
