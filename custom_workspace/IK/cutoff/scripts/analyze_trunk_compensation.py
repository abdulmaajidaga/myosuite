"""
Trunk Compensation Analysis
- Extract trunk movement from CS (chest/sternum) markers
- Compare trunk compensation between stroke and healthy
- Add trunk features to the dataset for improved modeling
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats, signal
from scipy.signal import resample
import os
import glob

# --- Configuration ---
BASE_DIR = "/home/abdul/Desktop/myosuite/custom_workspace"
HEALTHY_DIR = os.path.join(BASE_DIR, "data/kinematic/Healthy/filtered")
STROKE_DIR = os.path.join(BASE_DIR, "data/kinematic/Stroke/filtered")
OUTPUT_DIR = os.path.join(BASE_DIR, "IK/cutoff/output")

TARGET_FRAMES = 100


def load_raw_mocap(filepath):
    """Load raw motion capture data with all markers."""
    try:
        header = pd.read_csv(filepath, header=None, nrows=2, sep=',')
        marker_names = header.iloc[0].str.strip().ffill()
        axis_names = header.iloc[1].str.strip()
        mi = pd.MultiIndex.from_arrays([marker_names, axis_names])

        df = pd.read_csv(filepath, header=None, skiprows=2, names=mi, sep=',')
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.interpolate(method='linear', limit_direction='both').fillna(0)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def compute_trunk_metrics(df):
    """
    Compute trunk compensation metrics from chest markers.

    CS markers = Chest/Sternum markers for trunk tracking
    """
    metrics = {}

    try:
        # Trunk center (average of CS markers)
        trunk_x = (df['CS_1']['X'] + df['CS_2']['X'] + df['CS_3']['X'] + df['CS_4']['X']) / 4
        trunk_y = (df['CS_1']['Y'] + df['CS_2']['Y'] + df['CS_3']['Y'] + df['CS_4']['Y']) / 4
        trunk_z = (df['CS_1']['Z'] + df['CS_2']['Z'] + df['CS_3']['Z'] + df['CS_4']['Z']) / 4

        trunk = np.column_stack([trunk_x, trunk_y, trunk_z])

        # Shoulder center
        sh_x = (df['SA_1']['X'] + df['SA_2']['X'] + df['SA_3']['X']) / 3
        sh_y = (df['SA_1']['Y'] + df['SA_2']['Y'] + df['SA_3']['Y']) / 3
        sh_z = (df['SA_1']['Z'] + df['SA_2']['Z'] + df['SA_3']['Z']) / 3
        shoulder = np.column_stack([sh_x, sh_y, sh_z])

        # Wrist center
        wr_x = (df['WRA']['X'] + df['WRB']['X']) / 2
        wr_y = (df['WRA']['Y'] + df['WRB']['Y']) / 2
        wr_z = (df['WRA']['Z'] + df['WRB']['Z']) / 2
        wrist = np.column_stack([wr_x, wr_y, wr_z])

        # === TRUNK COMPENSATION METRICS ===

        # 1. Trunk Forward Displacement (Y-axis typically forward)
        trunk_forward = trunk[:, 1] - trunk[0, 1]  # Relative to start
        metrics['trunk_forward_max'] = trunk_forward.max()
        metrics['trunk_forward_rom'] = trunk_forward.max() - trunk_forward.min()

        # 2. Trunk Lateral Displacement (X-axis)
        trunk_lateral = trunk[:, 0] - trunk[0, 0]
        metrics['trunk_lateral_max'] = np.abs(trunk_lateral).max()
        metrics['trunk_lateral_rom'] = trunk_lateral.max() - trunk_lateral.min()

        # 3. Trunk Vertical Displacement (Z-axis)
        trunk_vertical = trunk[:, 2] - trunk[0, 2]
        metrics['trunk_vertical_max'] = trunk_vertical.max()
        metrics['trunk_vertical_rom'] = trunk_vertical.max() - trunk_vertical.min()

        # 4. Total Trunk Displacement (3D)
        trunk_disp = np.linalg.norm(trunk - trunk[0], axis=1)
        metrics['trunk_total_disp'] = trunk_disp.max()

        # 5. Trunk-to-Wrist Ratio (key compensation metric)
        # How much trunk moves relative to wrist movement
        wrist_disp = np.linalg.norm(wrist - wrist[0], axis=1).max()
        if wrist_disp > 0:
            metrics['trunk_wrist_ratio'] = trunk_disp.max() / wrist_disp
        else:
            metrics['trunk_wrist_ratio'] = 0

        # 6. Shoulder Protraction (shoulder moving forward relative to trunk)
        shoulder_rel_trunk = shoulder[:, 1] - trunk[:, 1]
        metrics['shoulder_protraction_rom'] = shoulder_rel_trunk.max() - shoulder_rel_trunk.min()

        # 7. Trunk Rotation (using front-back CS markers)
        # Angle between CS_1-CS_2 line and initial orientation
        front_back_x = df['CS_1']['X'] - df['CS_3']['X']
        front_back_y = df['CS_1']['Y'] - df['CS_3']['Y']
        trunk_angle = np.arctan2(front_back_y, front_back_x) * 180 / np.pi
        metrics['trunk_rotation_rom'] = trunk_angle.max() - trunk_angle.min()

        # 8. Trunk velocity (movement speed)
        trunk_vel = np.diff(trunk, axis=0) * 200  # Assuming 200Hz
        trunk_speed = np.linalg.norm(trunk_vel, axis=1)
        metrics['trunk_peak_velocity'] = trunk_speed.max()
        metrics['trunk_mean_velocity'] = trunk_speed.mean()

        # 9. Timing of trunk compensation
        # When does max trunk displacement occur relative to movement?
        metrics['trunk_peak_timing'] = np.argmax(trunk_disp) / len(trunk_disp) * 100

        # 10. Arm-Trunk Coordination
        # Correlation between wrist movement and trunk movement
        wrist_vel = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
        if len(trunk_speed) == len(wrist_vel) and trunk_speed.std() > 0 and wrist_vel.std() > 0:
            metrics['arm_trunk_correlation'] = np.corrcoef(trunk_speed, wrist_vel)[0, 1]
        else:
            metrics['arm_trunk_correlation'] = 0

        # Store trajectories for visualization
        metrics['_trunk_trajectory'] = trunk
        metrics['_wrist_trajectory'] = wrist
        metrics['_shoulder_trajectory'] = shoulder

    except KeyError as e:
        print(f"Missing marker: {e}")
        return None

    return metrics


def analyze_all_files():
    """Analyze trunk compensation in all healthy and stroke files."""
    print("=" * 60)
    print("Trunk Compensation Analysis")
    print("=" * 60)

    results = []

    # Process healthy files
    print("\nProcessing Healthy subjects...")
    healthy_files = sorted(glob.glob(os.path.join(HEALTHY_DIR, "*.csv")))
    for fpath in healthy_files:
        fname = os.path.basename(fpath)
        df = load_raw_mocap(fpath)
        if df is None:
            continue

        metrics = compute_trunk_metrics(df)
        if metrics is None:
            continue

        # Remove trajectory data for DataFrame
        metrics_clean = {k: v for k, v in metrics.items() if not k.startswith('_')}
        metrics_clean['filename'] = fname
        metrics_clean['group'] = 'Healthy'
        metrics_clean['fma_score'] = 66
        results.append(metrics_clean)

    print(f"  Processed {len([r for r in results if r['group'] == 'Healthy'])} healthy files")

    # Process stroke files
    print("\nProcessing Stroke subjects...")
    stroke_files = sorted(glob.glob(os.path.join(STROKE_DIR, "*.csv")))

    # Load FMA scores
    scores_path = os.path.join(BASE_DIR, "IK/output/scores.csv")
    if os.path.exists(scores_path):
        scores_df = pd.read_csv(scores_path)
        score_map = {
            str(row['filename']).strip().replace('.mot', ''): row['fma_score']
            for _, row in scores_df.iterrows()
        }
    else:
        score_map = {}

    for fpath in stroke_files:
        fname = os.path.basename(fpath)
        df = load_raw_mocap(fpath)
        if df is None:
            continue

        metrics = compute_trunk_metrics(df)
        if metrics is None:
            continue

        metrics_clean = {k: v for k, v in metrics.items() if not k.startswith('_')}
        metrics_clean['filename'] = fname
        metrics_clean['group'] = 'Stroke'

        # Get FMA score
        fname_base = fname.replace('.csv', '')
        metrics_clean['fma_score'] = score_map.get(fname_base, 30)  # Default 30 if not found

        results.append(metrics_clean)

    print(f"  Processed {len([r for r in results if r['group'] == 'Stroke'])} stroke files")

    return pd.DataFrame(results)


def create_trunk_report(df):
    """Create comprehensive trunk compensation report."""
    print("\nCreating report...")

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Trunk Compensation Analysis: Stroke vs Healthy", fontsize=16, fontweight='bold')

    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

    healthy = df[df['group'] == 'Healthy']
    stroke = df[df['group'] == 'Stroke']

    # === Row 1: Key Trunk Metrics Comparison ===

    # 1.1: Trunk Forward Displacement
    ax1 = fig.add_subplot(gs[0, 0])
    data = [healthy['trunk_forward_max'].values, stroke['trunk_forward_max'].values]
    bp = ax1.boxplot(data, tick_labels=['Healthy', 'Stroke'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    t, p = stats.ttest_ind(healthy['trunk_forward_max'], stroke['trunk_forward_max'])
    ax1.set_title(f"Trunk Forward Disp.\np={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
    ax1.set_ylabel('Displacement (mm)')
    ax1.grid(True, alpha=0.3)

    # 1.2: Total Trunk Displacement
    ax2 = fig.add_subplot(gs[0, 1])
    data = [healthy['trunk_total_disp'].values, stroke['trunk_total_disp'].values]
    bp = ax2.boxplot(data, tick_labels=['Healthy', 'Stroke'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    t, p = stats.ttest_ind(healthy['trunk_total_disp'], stroke['trunk_total_disp'])
    ax2.set_title(f"Total Trunk Disp.\np={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
    ax2.set_ylabel('Displacement (mm)')
    ax2.grid(True, alpha=0.3)

    # 1.3: Trunk-to-Wrist Ratio (KEY METRIC)
    ax3 = fig.add_subplot(gs[0, 2])
    data = [healthy['trunk_wrist_ratio'].values, stroke['trunk_wrist_ratio'].values]
    bp = ax3.boxplot(data, tick_labels=['Healthy', 'Stroke'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    t, p = stats.ttest_ind(healthy['trunk_wrist_ratio'], stroke['trunk_wrist_ratio'])
    ax3.set_title(f"Trunk/Wrist Ratio (KEY)\np={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
    ax3.set_ylabel('Ratio')
    ax3.grid(True, alpha=0.3)

    # 1.4: Shoulder Protraction
    ax4 = fig.add_subplot(gs[0, 3])
    data = [healthy['shoulder_protraction_rom'].values, stroke['shoulder_protraction_rom'].values]
    bp = ax4.boxplot(data, tick_labels=['Healthy', 'Stroke'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    t, p = stats.ttest_ind(healthy['shoulder_protraction_rom'], stroke['shoulder_protraction_rom'])
    ax4.set_title(f"Shoulder Protraction\np={p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
    ax4.set_ylabel('ROM (mm)')
    ax4.grid(True, alpha=0.3)

    # === Row 2: Correlations with FMA ===

    # 2.1: Trunk Displacement vs FMA
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.scatter(healthy['fma_score'], healthy['trunk_total_disp'], c='green', alpha=0.6, label='Healthy', s=50)
    ax5.scatter(stroke['fma_score'], stroke['trunk_total_disp'], c='red', alpha=0.6, label='Stroke', s=50)
    r, p = stats.pearsonr(df['fma_score'], df['trunk_total_disp'])
    z = np.polyfit(df['fma_score'], df['trunk_total_disp'], 1)
    ax5.plot([15, 70], [np.polyval(z, 15), np.polyval(z, 70)], 'k--', alpha=0.5)
    ax5.set_xlabel('FMA Score')
    ax5.set_ylabel('Trunk Displacement (mm)')
    ax5.set_title(f"Trunk Disp. vs FMA\nr={r:.3f}, p={p:.4f}")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 2.2: Trunk-Wrist Ratio vs FMA
    ax6 = fig.add_subplot(gs[1, 1])
    ax6.scatter(healthy['fma_score'], healthy['trunk_wrist_ratio'], c='green', alpha=0.6, label='Healthy', s=50)
    ax6.scatter(stroke['fma_score'], stroke['trunk_wrist_ratio'], c='red', alpha=0.6, label='Stroke', s=50)
    r, p = stats.pearsonr(df['fma_score'], df['trunk_wrist_ratio'])
    z = np.polyfit(df['fma_score'], df['trunk_wrist_ratio'], 1)
    ax6.plot([15, 70], [np.polyval(z, 15), np.polyval(z, 70)], 'k--', alpha=0.5)
    ax6.set_xlabel('FMA Score')
    ax6.set_ylabel('Trunk/Wrist Ratio')
    ax6.set_title(f"Trunk/Wrist Ratio vs FMA\nr={r:.3f}, p={p:.4f}")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 2.3: Trunk Rotation vs FMA
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.scatter(healthy['fma_score'], healthy['trunk_rotation_rom'], c='green', alpha=0.6, label='Healthy', s=50)
    ax7.scatter(stroke['fma_score'], stroke['trunk_rotation_rom'], c='red', alpha=0.6, label='Stroke', s=50)
    r, p = stats.pearsonr(df['fma_score'], df['trunk_rotation_rom'])
    z = np.polyfit(df['fma_score'], df['trunk_rotation_rom'], 1)
    ax7.plot([15, 70], [np.polyval(z, 15), np.polyval(z, 70)], 'k--', alpha=0.5, label='Trend')
    ax7.set_xlabel('FMA Score')
    ax7.set_ylabel('Rotation (degrees)')
    ax7.set_title(f"Trunk Rotation vs FMA\nr={r:.3f}, p={p:.4f}")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 2.4: Arm-Trunk Coordination vs FMA
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.scatter(healthy['fma_score'], healthy['arm_trunk_correlation'], c='green', alpha=0.6, label='Healthy', s=50)
    ax8.scatter(stroke['fma_score'], stroke['arm_trunk_correlation'], c='red', alpha=0.6, label='Stroke', s=50)
    r, p = stats.pearsonr(df['fma_score'], df['arm_trunk_correlation'])
    z = np.polyfit(df['fma_score'], df['arm_trunk_correlation'], 1)
    ax8.plot([15, 70], [np.polyval(z, 15), np.polyval(z, 70)], 'k--', alpha=0.5, label='Trend')
    ax8.set_xlabel('FMA Score')
    ax8.set_ylabel('Correlation')
    ax8.set_title(f"Arm-Trunk Coordination vs FMA\nr={r:.3f}, p={p:.4f}")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # === Row 3: Correlation Heatmap & Summary ===

    # 3.1-3.2: Correlation heatmap
    ax9 = fig.add_subplot(gs[2, 0:2])
    trunk_metrics = ['trunk_forward_max', 'trunk_total_disp', 'trunk_wrist_ratio',
                     'shoulder_protraction_rom', 'trunk_rotation_rom', 'arm_trunk_correlation']
    correlations = []
    for m in trunk_metrics:
        r, p = stats.pearsonr(df['fma_score'], df[m])
        correlations.append(r)

    colors = ['green' if v < 0 else 'red' for v in correlations]  # Negative = less compensation = better
    bars = ax9.barh(range(len(trunk_metrics)), correlations, color=colors, alpha=0.7)
    ax9.set_yticks(range(len(trunk_metrics)))
    ax9.set_yticklabels([m.replace('_', ' ').title() for m in trunk_metrics])
    ax9.set_xlabel('Correlation with FMA Score')
    ax9.set_title('Trunk Metrics Correlation with FMA\n(Negative = Less compensation in healthier patients)')
    ax9.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax9.set_xlim([-1, 1])
    ax9.grid(True, alpha=0.3, axis='x')

    # 3.3-3.4: Summary text
    ax10 = fig.add_subplot(gs[2, 2:4])
    ax10.axis('off')

    # Compute summary statistics
    h_trunk = healthy['trunk_total_disp'].mean()
    s_trunk = stroke['trunk_total_disp'].mean()
    h_ratio = healthy['trunk_wrist_ratio'].mean()
    s_ratio = stroke['trunk_wrist_ratio'].mean()

    t_trunk, p_trunk = stats.ttest_ind(healthy['trunk_total_disp'], stroke['trunk_total_disp'])
    t_ratio, p_ratio = stats.ttest_ind(healthy['trunk_wrist_ratio'], stroke['trunk_wrist_ratio'])

    summary = f"""
══════════════════════════════════════════════════════════════
                   TRUNK COMPENSATION SUMMARY
══════════════════════════════════════════════════════════════

SAMPLE SIZE:
  Healthy: {len(healthy)} trials
  Stroke:  {len(stroke)} trials

KEY FINDINGS:
──────────────────────────────────────────────────────────────
                        Healthy       Stroke       Difference
──────────────────────────────────────────────────────────────
Trunk Displacement     {h_trunk:6.1f} mm    {s_trunk:6.1f} mm    {((s_trunk/h_trunk)-1)*100:+.0f}%
Trunk/Wrist Ratio      {h_ratio:6.3f}       {s_ratio:6.3f}       {((s_ratio/h_ratio)-1)*100:+.0f}%
──────────────────────────────────────────────────────────────

CLINICAL INTERPRETATION:
  • Stroke patients show {((s_trunk/h_trunk)-1)*100:.0f}% MORE trunk displacement
  • Trunk/Wrist ratio is {((s_ratio/h_ratio)-1)*100:.0f}% HIGHER in stroke
    (More trunk movement per unit of wrist movement)

STATISTICAL SIGNIFICANCE:
  • Trunk displacement: p={p_trunk:.4f} {'***' if p_trunk < 0.001 else '**' if p_trunk < 0.01 else '*' if p_trunk < 0.05 else 'ns'}
  • Trunk/Wrist ratio:  p={p_ratio:.4f} {'***' if p_ratio < 0.001 else '**' if p_ratio < 0.01 else '*' if p_ratio < 0.05 else 'ns'}

RECOMMENDATION FOR MODEL IMPROVEMENT:
  • ADD trunk metrics to CVAE training data
  • Trunk/Wrist ratio is a KEY discriminator
  • Include: trunk_forward, trunk_lateral, trunk_rotation

══════════════════════════════════════════════════════════════
"""

    ax10.text(0.02, 0.98, summary, transform=ax10.transAxes,
              fontsize=9, fontfamily='monospace', verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Save
    save_path = os.path.join(OUTPUT_DIR, "trunk_compensation_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to: {save_path}")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "trunk_compensation_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")

    plt.show()

    return df


def main():
    df = analyze_all_files()
    if len(df) > 0:
        create_trunk_report(df)
    else:
        print("No data to analyze!")


if __name__ == "__main__":
    main()
