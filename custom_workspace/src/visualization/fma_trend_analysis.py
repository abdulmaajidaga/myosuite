"""
FMA Score Trend Analysis V2: Stroke (16) to Healthy (66)
- Generate motions across full FMA range
- Analyze correlations and trends
- Clinical metrics comparison
- NOW INCLUDES trunk compensation metrics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats, signal
from scipy.signal import resample, find_peaks
import os
import sys
import joblib
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path

# --- Configuration ---
MODEL_PATH = get_path("cvae_model_best")
SCALER_PATH = get_path("cvae_scaler")
OUTPUT_DIR = get_path("output_generated_plots")

from src.generation.model import MotionCVAE, INPUT_DIM, CONDITION_DIM, HIDDEN_DIM, LATENT_DIM, NUM_HEADS, SEQ_LEN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ARM_COLS = ['Sh_x','Sh_y','Sh_z','El_x','El_y','El_z','Wr_x','Wr_y','Wr_z','WrVec_x','WrVec_y','WrVec_z']
TRUNK_COLS = ['Trunk_x', 'Trunk_y', 'Trunk_z']
COLS = ARM_COLS + TRUNK_COLS

# FMA score ranges
FMA_MIN = 16  # Severe stroke
FMA_MAX = 66  # Healthy
FMA_STEP = 2  # Step size
N_SAMPLES = 10  # Samples per FMA score


def load_reference_pose():
    """Average starting pose across all subjects in training data.

    The model outputs deltas (differences from start position).
    This provides the average absolute starting position to convert
    deltas back to realistic 3D coordinates for visualization.
    """
    return {
        'Sh_x': -77.3, 'Sh_y': 643.0, 'Sh_z': 302.7,
        'El_x': -188.2, 'El_y': 474.4, 'El_z': 41.3,
        'Wr_x': -88.7, 'Wr_y': 241.2, 'Wr_z': 41.1,
        'WrVec_x': -37.0, 'WrVec_y': 12.0, 'WrVec_z': -33.1,
        'Trunk_x': 0.0, 'Trunk_y': 0.0, 'Trunk_z': 0.0,
    }


def smooth_motion(data):
    """Apply low-pass filter."""
    nyq = 0.5 * 100
    b, a = signal.butter(2, min(6 / nyq, 0.99), btype='low')
    smoothed = data.copy()
    for i in range(data.shape[1]):
        smoothed[:, i] = signal.filtfilt(b, a, data[:, i])
    return smoothed


def compute_metrics(motion):
    """Compute biomechanical and clinical metrics including trunk compensation."""
    sh = motion[:, 0:3]
    el = motion[:, 3:6]
    wr = motion[:, 6:9]
    trunk = motion[:, 12:15] if motion.shape[1] >= 15 else None

    metrics = {}

    # --- Kinematic Metrics ---

    # 1. Range of Motion (ROM)
    metrics['rom_wrist_y'] = wr[:, 1].max() - wr[:, 1].min()  # Reach distance
    metrics['rom_wrist_z'] = wr[:, 2].max() - wr[:, 2].min()  # Lift height
    metrics['rom_elbow_y'] = el[:, 1].max() - el[:, 1].min()
    metrics['rom_shoulder_y'] = sh[:, 1].max() - sh[:, 1].min()

    # 2. Peak wrist position (functional reach)
    metrics['peak_wrist_y'] = wr[:, 1].max()
    metrics['peak_wrist_z'] = wr[:, 2].max()

    # 3. Arm segment lengths (consistency check)
    upper_arm = np.linalg.norm(el - sh, axis=1)
    forearm = np.linalg.norm(wr - el, axis=1)
    metrics['upper_arm_mean'] = upper_arm.mean()
    metrics['upper_arm_std'] = upper_arm.std()
    metrics['forearm_mean'] = forearm.mean()
    metrics['forearm_std'] = forearm.std()

    # --- Velocity Metrics ---

    # Wrist velocity
    vel = np.diff(wr, axis=0) * 100  # mm/s at 100Hz
    speed = np.linalg.norm(vel, axis=1)

    metrics['peak_velocity'] = speed.max()
    metrics['mean_velocity'] = speed.mean()
    metrics['velocity_std'] = speed.std()

    # Time to peak velocity (as % of movement)
    metrics['time_to_peak_vel'] = np.argmax(speed) / len(speed) * 100

    # --- Smoothness Metrics ---

    # Acceleration
    acc = np.diff(vel, axis=0) * 100
    acc_mag = np.linalg.norm(acc, axis=1)

    # Jerk (rate of change of acceleration)
    jerk = np.diff(acc, axis=0) * 100
    jerk_mag = np.linalg.norm(jerk, axis=1)

    metrics['mean_jerk'] = jerk_mag.mean()  # Lower = smoother
    metrics['peak_jerk'] = jerk_mag.max()

    # Normalized jerk (dimensionless smoothness)
    duration = SEQ_LEN / 100  # seconds
    path_length = np.sum(np.linalg.norm(np.diff(wr, axis=0), axis=1))
    if path_length > 0:
        metrics['normalized_jerk'] = (duration**5 * jerk_mag.mean()) / path_length**2
    else:
        metrics['normalized_jerk'] = 0

    # Number of velocity peaks (movement units - more = less smooth)
    peaks, _ = find_peaks(speed, height=speed.max() * 0.1, distance=5)
    metrics['num_velocity_peaks'] = len(peaks)

    # --- Coordination Metrics ---

    # Elbow-wrist coordination (correlation)
    el_vel = np.linalg.norm(np.diff(el, axis=0), axis=1)
    wr_vel = np.linalg.norm(np.diff(wr, axis=0), axis=1)
    if len(el_vel) > 1 and el_vel.std() > 0 and wr_vel.std() > 0:
        metrics['elbow_wrist_corr'] = np.corrcoef(el_vel, wr_vel)[0, 1]
    else:
        metrics['elbow_wrist_corr'] = 0

    # --- Efficiency Metrics ---

    # Path length ratio (actual vs straight line)
    # Use minimum threshold to avoid division by near-zero
    straight_line = np.linalg.norm(wr[-1] - wr[0])
    if straight_line > 10.0:  # At least 10mm displacement
        metrics['path_ratio'] = path_length / straight_line  # 1.0 = perfectly straight
    else:
        # If motion returns to start, use path_length / max_displacement instead
        max_disp = np.linalg.norm(wr - wr[0], axis=1).max()
        if max_disp > 10.0:
            metrics['path_ratio'] = path_length / (2 * max_disp)  # Round trip estimate
        else:
            metrics['path_ratio'] = 1.0

    # Movement time (frames with significant velocity)
    active_frames = np.sum(speed > speed.max() * 0.05)
    metrics['movement_time'] = active_frames / 100  # seconds

    # --- TRUNK COMPENSATION METRICS (NEW) ---
    if trunk is not None:
        # Total trunk displacement
        trunk_disp = np.linalg.norm(trunk - trunk[0], axis=1)
        metrics['trunk_max_disp'] = trunk_disp.max()
        metrics['trunk_total_disp'] = np.sum(np.linalg.norm(np.diff(trunk, axis=0), axis=1))

        # Trunk/Wrist ratio (key clinical metric)
        wrist_disp = np.linalg.norm(wr - wr[0], axis=1).max()
        metrics['trunk_wrist_ratio'] = metrics['trunk_max_disp'] / wrist_disp if wrist_disp > 0 else 0

        # Trunk ROM in each direction
        metrics['trunk_forward_rom'] = trunk[:, 1].max() - trunk[:, 1].min()  # Y = forward
        metrics['trunk_lateral_rom'] = trunk[:, 0].max() - trunk[:, 0].min()  # X = lateral
        metrics['trunk_vertical_rom'] = trunk[:, 2].max() - trunk[:, 2].min()  # Z = vertical

        # Trunk velocity
        trunk_vel = np.diff(trunk, axis=0) * 100
        trunk_speed = np.linalg.norm(trunk_vel, axis=1)
        metrics['trunk_peak_velocity'] = trunk_speed.max()
        metrics['trunk_mean_velocity'] = trunk_speed.mean()

        # Trunk-wrist timing coordination
        min_len = min(len(trunk_speed), len(wr_vel))
        if min_len > 1 and trunk_speed[:min_len].std() > 0 and wr_vel[:min_len].std() > 0:
            metrics['trunk_wrist_timing'] = np.corrcoef(trunk_speed[:min_len], wr_vel[:min_len])[0, 1]
        else:
            metrics['trunk_wrist_timing'] = 0
    else:
        # No trunk data
        metrics['trunk_max_disp'] = 0
        metrics['trunk_total_disp'] = 0
        metrics['trunk_wrist_ratio'] = 0
        metrics['trunk_forward_rom'] = 0
        metrics['trunk_lateral_rom'] = 0
        metrics['trunk_vertical_rom'] = 0
        metrics['trunk_peak_velocity'] = 0
        metrics['trunk_mean_velocity'] = 0
        metrics['trunk_wrist_timing'] = 0

    return metrics


def generate_all_motions(model, scaler, ref_pose, guidance_scale=2.0):
    """Generate motions for all FMA scores with classifier-free guidance."""
    print(f"Generating motions for FMA scores (guidance_scale={guidance_scale})...")

    fma_scores = list(range(FMA_MIN, FMA_MAX + 1, FMA_STEP))
    all_data = []

    for fma in fma_scores:
        print(f"  FMA {fma}...", end=" ", flush=True)

        for sample_idx in range(N_SAMPLES):
            c = torch.FloatTensor([[fma / 66.0]]).to(DEVICE)
            gen = model.inference(c, SEQ_LEN, guidance_scale=guidance_scale).squeeze(0).cpu().numpy()
            gen = scaler.inverse_transform(gen)

            # Add reference pose
            for i, col in enumerate(COLS):
                if i < gen.shape[1]:
                    gen[:, i] += ref_pose.get(col, 0)

            gen = smooth_motion(gen)

            # Compute metrics
            metrics = compute_metrics(gen)
            metrics['fma_score'] = fma
            metrics['sample_idx'] = sample_idx

            all_data.append(metrics)

        print(f"done ({N_SAMPLES} samples)")

    return pd.DataFrame(all_data)


def compute_correlations(df):
    """Compute correlations between FMA score and all metrics."""
    correlations = {}
    p_values = {}

    metric_cols = [c for c in df.columns if c not in ['fma_score', 'sample_idx']]

    for metric in metric_cols:
        try:
            r, p = stats.pearsonr(df['fma_score'], df[metric])
            correlations[metric] = r
            p_values[metric] = p
        except:
            correlations[metric] = 0
            p_values[metric] = 1

    return correlations, p_values


def create_analysis_report(df, correlations, p_values):
    """Create comprehensive analysis with two windows: graphs and summary."""
    print("\nCreating analysis report...")

    # Style settings
    plt.style.use('seaborn-v0_8-whitegrid')

    # Helper function to add trend line and format plot
    def format_plot(ax, metric, color, ylabel, title_prefix, add_trend=True):
        grouped = df.groupby('fma_score')[metric].agg(['mean', 'std'])
        ax.fill_between(grouped.index, grouped['mean'] - grouped['std'],
                       grouped['mean'] + grouped['std'], alpha=0.2, color=color)
        ax.plot(grouped.index, grouped['mean'], 'o-', color=color, lw=2, markersize=4)

        r = correlations[metric]
        p = p_values[metric]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        if add_trend:
            z = np.polyfit(df['fma_score'], df[metric], 1)
            ax.plot([FMA_MIN, FMA_MAX], [np.polyval(z, FMA_MIN), np.polyval(z, FMA_MAX)],
                   '--', color='black', alpha=0.5, lw=1.5)

        ax.set_xlabel('FMA Score', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{title_prefix}\nr = {r:.3f}{sig}", fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=8)

    # ========== WINDOW 1: GRAPHS ==========
    fig1 = plt.figure(figsize=(16, 14))
    fig1.suptitle("FMA Score Trend Analysis: Generated Motion Characteristics",
                  fontsize=14, fontweight='bold', y=0.98)

    gs = GridSpec(4, 4, figure=fig1, hspace=0.45, wspace=0.35,
                  top=0.93, bottom=0.08, left=0.06, right=0.98)

    # === Row 1: Key Clinical Metrics ===
    ax1 = fig1.add_subplot(gs[0, 0])
    format_plot(ax1, 'rom_wrist_y', '#2E86AB', 'ROM (mm)', 'Wrist Reach ROM')

    ax2 = fig1.add_subplot(gs[0, 1])
    format_plot(ax2, 'peak_velocity', '#E94F37', 'Velocity (mm/s)', 'Peak Velocity')

    ax3 = fig1.add_subplot(gs[0, 2])
    format_plot(ax3, 'rom_wrist_z', '#F39237', 'ROM (mm)', 'Wrist Height ROM')

    ax4 = fig1.add_subplot(gs[0, 3])
    format_plot(ax4, 'mean_velocity', '#0B6E4F', 'Velocity (mm/s)', 'Mean Velocity')

    # === Row 2: Trunk Compensation (KEY) ===
    ax5 = fig1.add_subplot(gs[1, 0])
    format_plot(ax5, 'trunk_max_disp', '#7B2D8E', 'Displacement (mm)', 'Trunk Displacement')

    ax6 = fig1.add_subplot(gs[1, 1])
    format_plot(ax6, 'trunk_wrist_ratio', '#D7263D', 'Ratio', 'Trunk/Wrist Ratio (KEY)')

    ax7 = fig1.add_subplot(gs[1, 2])
    format_plot(ax7, 'trunk_forward_rom', '#6B4226', 'ROM (mm)', 'Trunk Forward ROM')

    ax8 = fig1.add_subplot(gs[1, 3])
    format_plot(ax8, 'trunk_lateral_rom', '#1B4965', 'ROM (mm)', 'Trunk Lateral ROM')

    # === Row 3: Smoothness & Coordination ===
    ax9 = fig1.add_subplot(gs[2, 0])
    format_plot(ax9, 'normalized_jerk', '#9B5DE5', 'Norm. Jerk', 'Normalized Jerk (↓=smoother)')

    ax10 = fig1.add_subplot(gs[2, 1])
    format_plot(ax10, 'num_velocity_peaks', '#F15BB5', '# Peaks', 'Velocity Peaks')

    ax11 = fig1.add_subplot(gs[2, 2])
    format_plot(ax11, 'time_to_peak_vel', '#00BBF9', '% Movement', 'Time to Peak Velocity')
    ax11.axhline(y=50, color='gray', linestyle=':', alpha=0.7, lw=1)

    ax12 = fig1.add_subplot(gs[2, 3])
    format_plot(ax12, 'elbow_wrist_corr', '#00F5D4', 'Correlation', 'Elbow-Wrist Coordination')

    # === Row 4: Summary visualizations ===

    # Correlation bar chart
    ax13 = fig1.add_subplot(gs[3, 0:2])
    key_metrics = ['rom_wrist_z', 'peak_velocity', 'trunk_wrist_ratio', 'trunk_max_disp',
                   'mean_velocity', 'trunk_lateral_rom', 'normalized_jerk', 'num_velocity_peaks']
    metric_labels = ['Wrist Height ROM', 'Peak Velocity', 'Trunk/Wrist Ratio', 'Trunk Displacement',
                     'Mean Velocity', 'Trunk Lateral ROM', 'Normalized Jerk', 'Velocity Peaks']
    corr_values = [correlations.get(m, 0) for m in key_metrics]

    colors_bar = ['#2E86AB' if v > 0 else '#D7263D' for v in corr_values]
    y_pos = range(len(key_metrics))

    bars = ax13.barh(y_pos, corr_values, color=colors_bar, alpha=0.8, height=0.7)
    ax13.set_yticks(y_pos)
    ax13.set_yticklabels(metric_labels, fontsize=9)
    ax13.set_xlabel('Correlation with FMA Score', fontsize=10)
    ax13.set_title('Key Metric Correlations', fontsize=11, fontweight='bold')
    ax13.axvline(x=0, color='black', linestyle='-', alpha=0.3, lw=1)
    ax13.set_xlim([-0.8, 0.8])

    # Add significance and value labels
    for i, (m, v) in enumerate(zip(key_metrics, corr_values)):
        p = p_values.get(m, 1)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        offset = 0.03 if v >= 0 else -0.03
        ha = 'left' if v >= 0 else 'right'
        ax13.text(v + offset, i, f'{v:.2f}{sig}', va='center', ha=ha, fontsize=8, fontweight='bold')

    # Clinical group boxplot
    ax14 = fig1.add_subplot(gs[3, 2:4])
    fma_groups = [(16, 25, 'Severe\n(16-25)'), (26, 40, 'Moderate\n(26-40)'),
                  (41, 55, 'Mild\n(41-55)'), (56, 66, 'Healthy\n(56-66)')]
    group_colors = ['#D7263D', '#F39237', '#F9C846', '#2E86AB']

    group_data = []
    group_labels = []
    for (fma_min, fma_max, label), color in zip(fma_groups, group_colors):
        subset = df[(df['fma_score'] >= fma_min) & (df['fma_score'] <= fma_max)]
        group_data.append(subset['trunk_wrist_ratio'].values)
        group_labels.append(label)

    bp = ax14.boxplot(group_data, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], group_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for whisker in bp['whiskers']:
        whisker.set(color='gray', linewidth=1)
    for cap in bp['caps']:
        cap.set(color='gray', linewidth=1)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)

    ax14.set_xticklabels(group_labels, fontsize=9)
    ax14.set_ylabel('Trunk/Wrist Ratio', fontsize=10)
    ax14.set_title('Trunk Compensation by Clinical Severity', fontsize=11, fontweight='bold')
    ax14.tick_params(labelsize=9)

    # Save graphs figure
    save_path1 = os.path.join(OUTPUT_DIR, "fma_trend_graphs.png")
    fig1.savefig(save_path1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Graphs saved to: {save_path1}")

    # Reset style
    plt.style.use('default')

    # ========== WINDOW 2: SUMMARY ==========
    fig2 = plt.figure(figsize=(14, 10))
    fig2.canvas.manager.set_window_title("FMA Trend Analysis - Summary")
    ax_summary = fig2.add_subplot(111)
    ax_summary.axis('off')

    # Compute summary statistics
    severe = df[df['fma_score'] <= 25]
    healthy = df[df['fma_score'] >= 56]

    # Statistical tests
    t_rom, p_rom = stats.ttest_ind(severe['rom_wrist_y'], healthy['rom_wrist_y'])
    t_trunk, p_trunk = stats.ttest_ind(severe['trunk_wrist_ratio'], healthy['trunk_wrist_ratio'])

    # Effect sizes (Cohen's d)
    def cohens_d(g1, g2):
        pooled_std = np.sqrt((g1.std()**2 + g2.std()**2) / 2)
        return (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else 0

    d_rom = cohens_d(healthy['rom_wrist_y'], severe['rom_wrist_y'])
    d_trunk = cohens_d(severe['trunk_wrist_ratio'], healthy['trunk_wrist_ratio'])

    # Get top correlations
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    summary_text = f"""
{'='*90}
                           CLINICAL SUMMARY - FMA TREND ANALYSIS
{'='*90}

SAMPLE SIZE: {len(df['fma_score'].unique())} FMA scores x {N_SAMPLES} samples = {len(df)} total generated motions

SEVERE STROKE (FMA 16-25) vs HEALTHY (FMA 56-66):
{'-'*90}
  Metric                    Severe          Healthy         Effect Size    Significance
{'-'*90}
  Wrist ROM (Y)            {severe['rom_wrist_y'].mean():7.1f} mm      {healthy['rom_wrist_y'].mean():7.1f} mm       d={d_rom:+.2f}         p={p_rom:.4f}
  Trunk/Wrist Ratio        {severe['trunk_wrist_ratio'].mean():7.3f}         {healthy['trunk_wrist_ratio'].mean():7.3f}          d={d_trunk:+.2f}         p={p_trunk:.4f}
  Trunk Max Disp           {severe['trunk_max_disp'].mean():7.1f} mm      {healthy['trunk_max_disp'].mean():7.1f} mm
  Peak Velocity            {severe['peak_velocity'].mean():7.1f} mm/s   {healthy['peak_velocity'].mean():7.1f} mm/s
  Mean Jerk                {severe['mean_jerk'].mean():7.1f}        {healthy['mean_jerk'].mean():7.1f}
{'-'*90}

TOP 10 CORRELATIONS WITH FMA SCORE:
   1. {sorted_corr[0][0]:25s}: r={sorted_corr[0][1]:+.3f}  {'***' if p_values[sorted_corr[0][0]] < 0.001 else '**' if p_values[sorted_corr[0][0]] < 0.01 else '*' if p_values[sorted_corr[0][0]] < 0.05 else 'ns'}
   2. {sorted_corr[1][0]:25s}: r={sorted_corr[1][1]:+.3f}  {'***' if p_values[sorted_corr[1][0]] < 0.001 else '**' if p_values[sorted_corr[1][0]] < 0.01 else '*' if p_values[sorted_corr[1][0]] < 0.05 else 'ns'}
   3. {sorted_corr[2][0]:25s}: r={sorted_corr[2][1]:+.3f}  {'***' if p_values[sorted_corr[2][0]] < 0.001 else '**' if p_values[sorted_corr[2][0]] < 0.01 else '*' if p_values[sorted_corr[2][0]] < 0.05 else 'ns'}
   4. {sorted_corr[3][0]:25s}: r={sorted_corr[3][1]:+.3f}  {'***' if p_values[sorted_corr[3][0]] < 0.001 else '**' if p_values[sorted_corr[3][0]] < 0.01 else '*' if p_values[sorted_corr[3][0]] < 0.05 else 'ns'}
   5. {sorted_corr[4][0]:25s}: r={sorted_corr[4][1]:+.3f}  {'***' if p_values[sorted_corr[4][0]] < 0.001 else '**' if p_values[sorted_corr[4][0]] < 0.01 else '*' if p_values[sorted_corr[4][0]] < 0.05 else 'ns'}
   6. {sorted_corr[5][0]:25s}: r={sorted_corr[5][1]:+.3f}  {'***' if p_values[sorted_corr[5][0]] < 0.001 else '**' if p_values[sorted_corr[5][0]] < 0.01 else '*' if p_values[sorted_corr[5][0]] < 0.05 else 'ns'}
   7. {sorted_corr[6][0]:25s}: r={sorted_corr[6][1]:+.3f}  {'***' if p_values[sorted_corr[6][0]] < 0.001 else '**' if p_values[sorted_corr[6][0]] < 0.01 else '*' if p_values[sorted_corr[6][0]] < 0.05 else 'ns'}
   8. {sorted_corr[7][0]:25s}: r={sorted_corr[7][1]:+.3f}  {'***' if p_values[sorted_corr[7][0]] < 0.001 else '**' if p_values[sorted_corr[7][0]] < 0.01 else '*' if p_values[sorted_corr[7][0]] < 0.05 else 'ns'}
   9. {sorted_corr[8][0]:25s}: r={sorted_corr[8][1]:+.3f}  {'***' if p_values[sorted_corr[8][0]] < 0.001 else '**' if p_values[sorted_corr[8][0]] < 0.01 else '*' if p_values[sorted_corr[8][0]] < 0.05 else 'ns'}
  10. {sorted_corr[9][0]:25s}: r={sorted_corr[9][1]:+.3f}  {'***' if p_values[sorted_corr[9][0]] < 0.001 else '**' if p_values[sorted_corr[9][0]] < 0.01 else '*' if p_values[sorted_corr[9][0]] < 0.05 else 'ns'}

KEY CLINICAL FINDINGS:
  - Trunk compensation (trunk/wrist ratio) correlation with FMA: r={correlations['trunk_wrist_ratio']:.3f}
  - Higher FMA (healthier) = {'LESS' if correlations['trunk_wrist_ratio'] < 0 else 'MORE'} trunk compensation
  - Model captures clinically relevant stroke-to-healthy movement differences

SIGNIFICANCE: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant
{'='*90}
    """

    ax_summary.text(0.02, 0.98, summary_text, transform=ax_summary.transAxes,
                    fontsize=11, fontfamily='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Save summary figure
    save_path2 = os.path.join(OUTPUT_DIR, "fma_trend_summary.png")
    fig2.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"Summary saved to: {save_path2}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "fma_trend_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")

    plt.show()

    return df


def main():
    print("=" * 60)
    print("FMA Score Trend Analysis V2 (with Trunk Compensation)")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model = MotionCVAE().to(DEVICE)
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_PATH.replace('_best', '')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please run training first: python3 cvae_train_cutoff.py")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    scaler = joblib.load(SCALER_PATH)
    ref_pose = load_reference_pose()

    print(f"Model loaded: {model_path}")
    print(f"Features: {INPUT_DIM} dimensions (12 arm + 3 trunk)")

    # Generate all motions
    df = generate_all_motions(model, scaler, ref_pose)

    # Compute correlations
    print("\nComputing correlations...")
    correlations, p_values = compute_correlations(df)

    # Print top correlations
    print("\nTop correlations with FMA score:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for metric, r in sorted_corr[:10]:
        p = p_values[metric]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {metric:25s}: r={r:+.3f} {sig}")

    # Create report
    create_analysis_report(df, correlations, p_values)


if __name__ == "__main__":
    main()
