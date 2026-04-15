"""
Cross-FMA inverse dynamics comparison plots.

Generates multi-panel figures comparing torques, muscle activations,
synergy coefficients, and summary bar charts across all generated FMA scores.

Can be called from the pipeline via generate_all() or run standalone.
"""
import os
import re
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d

from src.utils.config import get_path

# =============================================================================
# Constants
# =============================================================================
KEY_JOINTS = ['elv_angle', 'shoulder_elv', 'shoulder_rot', 'elbow_flexion', 'pro_sup', 'deviation', 'flexion']
JOINT_LABELS = {
    'elv_angle': 'Elevation Angle', 'shoulder_elv': 'Shoulder Elevation',
    'shoulder_rot': 'Shoulder Rotation',
    'elbow_flexion': 'Elbow Flexion', 'pro_sup': 'Pro/Supination',
    'deviation': 'Wrist Deviation', 'flexion': 'Wrist Flexion'
}

SHOULDER_MUSCLES = ['DELT1', 'DELT2', 'DELT3', 'SUPSP', 'INFSP', 'SUBSC', 'TMIN', 'TMAJ']
ELBOW_MUSCLES = ['TRIlong', 'TRIlat', 'TRImed', 'BIClong', 'BICshort', 'BRA', 'BRD', 'ANC']
WRIST_MUSCLES = ['ECRL', 'ECRB', 'ECU', 'FCR', 'FCU', 'PL', 'PT', 'PQ']
PEC_LAT = ['PECM1', 'PECM2', 'PECM3', 'LAT1', 'LAT2', 'LAT3', 'CORB']

MUSCLE_GROUPS = [
    ('Shoulder\n(Delt/Cuff)', SHOULDER_MUSCLES),
    ('Pec/Lat/Corb', PEC_LAT),
    ('Elbow\n(Tri/Bic/Bra)', ELBOW_MUSCLES),
    ('Wrist\n(ECR/FCR/etc)', WRIST_MUSCLES),
]


# =============================================================================
# Helpers
# =============================================================================
def _make_color_map(fma_scores):
    """Generate a color for each FMA score using a red->blue colormap."""
    cmap = plt.cm.RdYlGn
    normed = [(s - min(fma_scores)) / max(1, max(fma_scores) - min(fma_scores))
              for s in fma_scores]
    return {s: cmap(n) for s, n in zip(fma_scores, normed)}


def _make_labels(fma_scores):
    """Generate display labels for each FMA score with clinical categories."""
    labels = {}
    for s in fma_scores:
        if s <= 20:
            labels[s] = f'FMA {s} (severe)'
        elif s <= 40:
            labels[s] = f'FMA {s} (moderate)'
        elif s <= 55:
            labels[s] = f'FMA {s} (mild)'
        elif s >= 56:
            labels[s] = f'FMA {s} (healthy)'
        else:
            labels[s] = f'FMA {s}'
    return labels


def _adaptive_bar_width(n_scores):
    """Compute bar width that works for both 6 and 13+ FMA scores."""
    if n_scores <= 6:
        return 0.6
    elif n_scores <= 10:
        return 0.5
    else:
        return 0.4


def normalize_time(df, time_col='time'):
    """Normalize time to 0-100% of motion."""
    t = df[time_col].values
    t_range = t[-1] - t[0]
    if t_range == 0:
        return np.linspace(0, 100, len(t))
    return (t - t[0]) / t_range * 100


def resample_to_percent(values, t_norm, n_points=200):
    """Resample a signal to uniform 0-100% grid."""
    t_uniform = np.linspace(0, 100, n_points)
    f = interp1d(t_norm, values, kind='linear', fill_value='extrapolate')
    return t_uniform, f(t_uniform)


def detect_fma_scores(id_base_dir):
    """Auto-detect available FMA scores from directory names like FMA_50/."""
    scores = []
    if not os.path.isdir(id_base_dir):
        return scores
    for name in os.listdir(id_base_dir):
        m = re.match(r'^FMA_(\d+)$', name)
        if m and os.path.isdir(os.path.join(id_base_dir, name)):
            d = os.path.join(id_base_dir, name)
            # Only include if it has the required CSVs
            if os.path.exists(os.path.join(d, 'torques.csv')):
                scores.append(int(m.group(1)))
    return sorted(scores)


def load_all(id_base_dir, fma_scores):
    """Load all ID results into dicts keyed by FMA score."""
    torques, activations, synergy_coefs, phase_labels, effort_metrics = {}, {}, {}, {}, {}
    for fma in fma_scores:
        d = os.path.join(id_base_dir, f'FMA_{fma}')
        torques[fma] = pd.read_csv(os.path.join(d, 'torques.csv'))
        activations[fma] = pd.read_csv(os.path.join(d, 'activations.csv'))
        syn_path = os.path.join(d, 'synergy_coefficients.csv')
        if os.path.exists(syn_path):
            synergy_coefs[fma] = pd.read_csv(syn_path)
        phase_labels[fma] = pd.read_csv(os.path.join(d, 'phase_labels.csv'))
        metrics_path = os.path.join(d, 'effort_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                effort_metrics[fma] = json.load(f)
    return torques, activations, synergy_coefs, phase_labels, effort_metrics


# =============================================================================
# FIGURE 1: Joint Torques Overlay
# =============================================================================
def plot_torques_comparison(torques, fma_scores, colors, labels, output_dir):
    n_joints = len(KEY_JOINTS)
    n_cols = 4
    n_rows = (n_joints + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    fig.suptitle('Joint Torques Across FMA Scores', fontsize=16, fontweight='bold', y=0.98)
    axes = axes.flatten()

    t_pct = np.linspace(0, 100, 200)
    # Adaptive alpha for many overlapping curves
    line_alpha = max(0.4, 0.85 - 0.03 * len(fma_scores))
    line_width = max(1.0, 2.0 - 0.08 * len(fma_scores))

    for idx, joint in enumerate(KEY_JOINTS):
        ax = axes[idx]
        for fma in fma_scores:
            df = torques[fma]
            if joint not in df.columns:
                continue
            t_norm = normalize_time(df)
            _, vals = resample_to_percent(df[joint].values, t_norm)
            ax.plot(t_pct, vals, color=colors[fma], linewidth=line_width, alpha=line_alpha, label=labels[fma])

        ax.set_title(JOINT_LABELS.get(joint, joint), fontsize=13, fontweight='bold')
        ax.set_xlabel('Motion Progress (%)', fontsize=10)
        ax.set_ylabel('Torque (Nm)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

        ax.axvspan(0, 40, alpha=0.04, color='blue')
        ax.axvspan(40, 70, alpha=0.04, color='green')
        ax.axvspan(70, 100, alpha=0.04, color='orange')

    # Hide extra axes
    for idx in range(n_joints, len(axes)):
        axes[idx].set_visible(False)

    axes[0].legend(fontsize=9, loc='best')

    for ax in axes[:min(n_cols, n_joints)]:
        ylim = ax.get_ylim()
        y_top = ylim[1] - (ylim[1] - ylim[0]) * 0.05
        for pct, label in [(20, 'Pick'), (55, 'Drink'), (85, 'Place')]:
            ax.text(pct, y_top, label, ha='center', fontsize=8, color='gray', style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'id_comparison_torques.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# =============================================================================
# FIGURE 2: Muscle Group Activations
# =============================================================================
def plot_muscle_group_activations(activations, fma_scores, colors, labels, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Muscle Group Activations Across FMA Scores', fontsize=16, fontweight='bold', y=0.98)
    axes = axes.flatten()

    t_pct = np.linspace(0, 100, 200)

    for idx, (group_name, muscles) in enumerate(MUSCLE_GROUPS):
        ax = axes[idx]
        for fma in fma_scores:
            df = activations[fma]
            present = [m for m in muscles if m in df.columns]
            if not present:
                continue
            t_norm = normalize_time(df)
            group_mean = df[present].mean(axis=1).values
            _, vals = resample_to_percent(group_mean, t_norm)
            ax.plot(t_pct, vals, color=colors[fma], linewidth=2, alpha=0.85, label=labels[fma])

            group_std = df[present].std(axis=1).values
            _, std_vals = resample_to_percent(group_std, t_norm)
            ax.fill_between(t_pct, vals - std_vals, vals + std_vals, color=colors[fma], alpha=0.08)

        ax.set_title(group_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Motion Progress (%)', fontsize=10)
        ax.set_ylabel('Activation (0-1)', fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        ax.axvspan(0, 40, alpha=0.04, color='blue')
        ax.axvspan(40, 70, alpha=0.04, color='green')
        ax.axvspan(70, 100, alpha=0.04, color='orange')

    axes[0].legend(fontsize=9, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'id_comparison_activations.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# =============================================================================
# FIGURE 3: Synergy Coefficients
# =============================================================================
def plot_synergy_comparison(synergy_coefs, fma_scores, colors, labels, output_dir):
    if not synergy_coefs:
        print('  Skipping synergy comparison (no synergy data)')
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Muscle Synergy Activation Across FMA Scores', fontsize=16, fontweight='bold', y=0.98)
    axes = axes.flatten()

    t_pct = np.linspace(0, 100, 200)

    for s_idx in range(4):
        ax = axes[s_idx]
        col = f'Synergy_{s_idx+1}'
        for fma in fma_scores:
            if fma not in synergy_coefs:
                continue
            df = synergy_coefs[fma]
            if col not in df.columns:
                continue
            t_norm = normalize_time(df)
            _, vals = resample_to_percent(df[col].values, t_norm)
            ax.plot(t_pct, vals, color=colors[fma], linewidth=2, alpha=0.85, label=labels[fma])

        ax.set_title(f'Synergy {s_idx+1}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Motion Progress (%)', fontsize=10)
        ax.set_ylabel('Coefficient', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.05)

        ax.axvspan(0, 40, alpha=0.04, color='blue')
        ax.axvspan(40, 70, alpha=0.04, color='green')
        ax.axvspan(70, 100, alpha=0.04, color='orange')

    axes[0].legend(fontsize=9, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'id_comparison_synergies.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# =============================================================================
# FIGURE 4: Summary Bar Charts
# =============================================================================
def plot_summary_bars(torques, activations, synergy_coefs, phase_labels,
                      fma_scores, colors, labels, output_dir):
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    fig.suptitle('Inverse Dynamics Summary: FMA Score Comparison', fontsize=18, fontweight='bold', y=0.98)

    x = np.arange(len(fma_scores))
    width = _adaptive_bar_width(len(fma_scores))
    bar_colors = [colors[f] for f in fma_scores]
    xlabels = [f'FMA {f}' for f in fma_scores]

    # --- Panel 1: Mean absolute torque per key joint ---
    ax1 = fig.add_subplot(gs[0, :2])
    joints_for_bar = ['shoulder_elv', 'elbow_flexion', 'pro_sup', 'deviation', 'flexion']
    n_joints = len(joints_for_bar)
    bar_w = min(0.15, 0.8 / max(n_joints * len(fma_scores), 1))
    for j_idx, joint in enumerate(joints_for_bar):
        vals = []
        for fma in fma_scores:
            df = torques[fma]
            vals.append(np.mean(np.abs(df[joint].values)) if joint in df.columns else 0)
        offset = (j_idx - n_joints/2 + 0.5) * bar_w
        ax1.bar(x + offset, vals, bar_w * 0.9, label=JOINT_LABELS.get(joint, joint), alpha=0.85)
    ax1.set_xticks(x)
    tick_rotation = 45 if len(fma_scores) > 8 else 0
    tick_fontsize = max(7, 11 - len(fma_scores) // 4)
    ax1.set_xticklabels(xlabels, fontsize=tick_fontsize, rotation=tick_rotation, ha='right' if tick_rotation else 'center')
    ax1.set_ylabel('Mean |Torque| (Nm)', fontsize=11)
    ax1.set_title('Mean Absolute Joint Torques', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=8, ncol=3, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)

    # --- Panel 2: Peak torques ---
    ax2 = fig.add_subplot(gs[0, 2])
    elbow_peaks = [np.max(np.abs(torques[f]['elbow_flexion'].values)) for f in fma_scores]
    shoulder_peaks = [np.max(np.abs(torques[f]['shoulder_elv'].values)) for f in fma_scores]
    ax2.bar(x - 0.2, elbow_peaks, 0.35, label='Elbow Flex Peak', color='#2ca02c', alpha=0.8)
    ax2.bar(x + 0.2, shoulder_peaks, 0.35, label='Shoulder Elv Peak', color='#d62728', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, fontsize=10)
    ax2.set_ylabel('Peak |Torque| (Nm)', fontsize=11)
    ax2.set_title('Peak Joint Torques', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # --- Panel 3: Mean muscle group activations ---
    ax3 = fig.add_subplot(gs[1, :2])
    n_groups = len(MUSCLE_GROUPS)
    bar_w = 0.15
    for g_idx, (group_name, muscles) in enumerate(MUSCLE_GROUPS):
        vals = []
        for fma in fma_scores:
            df = activations[fma]
            present = [m for m in muscles if m in df.columns]
            vals.append(np.mean(df[present].values) if present else 0)
        offset = (g_idx - n_groups/2 + 0.5) * bar_w
        ax3.bar(x + offset, vals, bar_w * 0.9, label=group_name.replace('\n', ' '), alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(xlabels, fontsize=11)
    ax3.set_ylabel('Mean Activation', fontsize=11)
    ax3.set_title('Mean Muscle Group Activation', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8, ncol=2)
    ax3.grid(axis='y', alpha=0.3)

    # --- Panel 4: Overall effort ---
    ax4 = fig.add_subplot(gs[1, 2])
    overall_act = []
    total_effort = []
    for fma in fma_scores:
        act_df = activations[fma]
        act_cols = [c for c in act_df.columns if c != 'time']
        overall_act.append(np.mean(act_df[act_cols].values))
        tq_df = torques[fma]
        avail_joints = [j for j in KEY_JOINTS if j in tq_df.columns]
        total_effort.append(np.mean(np.abs(tq_df[avail_joints].values)) if avail_joints else 0)

    ax4_twin = ax4.twinx()
    bars1 = ax4.bar(x - 0.2, overall_act, 0.35, color='#1f77b4', alpha=0.8, label='Mean Activation')
    bars2 = ax4_twin.bar(x + 0.2, total_effort, 0.35, color='#ff7f0e', alpha=0.8, label='Mean |Torque|')
    ax4.set_xticks(x)
    ax4.set_xticklabels(xlabels, fontsize=10)
    ax4.set_ylabel('Mean Activation', fontsize=11, color='#1f77b4')
    ax4_twin.set_ylabel('Mean |Torque| (Nm)', fontsize=11, color='#ff7f0e')
    ax4.set_title('Overall Effort', fontsize=13, fontweight='bold')
    ax4.legend([bars1, bars2], ['Mean Activation', 'Mean |Torque|'], fontsize=9, loc='upper left')
    ax4.grid(axis='y', alpha=0.3)

    # --- Panel 5: Phase proportions ---
    ax5 = fig.add_subplot(gs[2, 0])
    phase_colors = {'Pick': '#4c72b0', 'Drink': '#55a868', 'Place': '#c44e52'}
    phase_order = ['Pick', 'Drink', 'Place']
    bottom = np.zeros(len(fma_scores))
    for phase_name in phase_order:
        proportions = []
        for fma in fma_scores:
            pdf = phase_labels[fma]
            total = len(pdf)
            count = len(pdf[pdf['phase'] == phase_name])
            proportions.append(count / total * 100)
        ax5.bar(x, proportions, width, bottom=bottom, label=phase_name,
                color=phase_colors.get(phase_name, 'gray'), alpha=0.85)
        bottom += np.array(proportions)
    ax5.set_xticks(x)
    ax5.set_xticklabels(xlabels, fontsize=10)
    ax5.set_ylabel('% of Motion', fontsize=11)
    ax5.set_title('Phase Proportions', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.set_ylim(0, 105)

    # --- Panel 6: Synergy dominance ---
    ax6 = fig.add_subplot(gs[2, 1])
    if synergy_coefs:
        syn_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
        bar_w = 0.18
        for s_idx in range(4):
            col = f'Synergy_{s_idx+1}'
            vals = []
            for fma in fma_scores:
                if fma in synergy_coefs and col in synergy_coefs[fma].columns:
                    vals.append(np.mean(synergy_coefs[fma][col].values))
                else:
                    vals.append(0)
            offset = (s_idx - 2 + 0.5) * bar_w
            ax6.bar(x + offset, vals, bar_w * 0.9, label=f'Syn {s_idx+1}',
                    color=syn_colors[s_idx], alpha=0.8)
        ax6.legend(fontsize=9, ncol=2)
    ax6.set_xticks(x)
    ax6.set_xticklabels(xlabels, fontsize=10)
    ax6.set_ylabel('Mean Coefficient', fontsize=11)
    ax6.set_title('Synergy Usage', fontsize=13, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)

    # --- Panel 7: Co-contraction index ---
    ax7 = fig.add_subplot(gs[2, 2])
    cocontraction = []
    for fma in fma_scores:
        df = activations[fma]
        bic_cols = [c for c in ['BIClong', 'BICshort'] if c in df.columns]
        tri_cols = [c for c in ['TRIlong', 'TRIlat', 'TRImed'] if c in df.columns]
        if bic_cols and tri_cols:
            biceps = df[bic_cols].mean(axis=1).values
            triceps = df[tri_cols].mean(axis=1).values
            numer = 2 * np.minimum(biceps, triceps)
            denom = biceps + triceps + 1e-8
            cocontraction.append(np.mean(numer / denom))
        else:
            cocontraction.append(0)
    ax7.bar(x, cocontraction, width, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax7.set_xticks(x)
    ax7.set_xticklabels(xlabels, fontsize=10)
    ax7.set_ylabel('Co-contraction Index', fontsize=11)
    ax7.set_title('Elbow Co-contraction\n(Biceps vs Triceps)', fontsize=13, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    if len(fma_scores) >= 2:
        z = np.polyfit(fma_scores, cocontraction, 1)
        ax7.plot(x, np.polyval(z, fma_scores), 'k--', linewidth=1.5, alpha=0.6)

    plt.savefig(os.path.join(output_dir, 'id_comparison_summary.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {os.path.join(output_dir, "id_comparison_summary.png")}')


# =============================================================================
# FIGURE 5: Effort Metrics Comparison (ATI, CCI, TRR)
# =============================================================================
def plot_effort_comparison(effort_metrics, fma_scores, colors, labels, output_dir):
    if not effort_metrics:
        print('  Skipping effort comparison (no effort_metrics.json data)')
        return

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Effort Metrics Across FMA Scores', fontsize=16, fontweight='bold', y=0.98)

    x = np.arange(len(fma_scores))
    bar_colors = [colors[f] for f in fma_scores]
    xlabels = [f'FMA {f}' for f in fma_scores]
    width = _adaptive_bar_width(len(fma_scores))

    # --- Panel 1: ATI (Activation-Time Integral) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ati_vals = [effort_metrics[f]['ATI'] for f in fma_scores]
    ax1.bar(x, ati_vals, width, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=10)
    ax1.set_ylabel('ATI (sum a_i^2)', fontsize=11)
    ax1.set_title('Activation-Time Integral\n(Total Muscular Effort)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    if len(fma_scores) >= 2:
        z = np.polyfit(fma_scores, ati_vals, 1)
        ax1.plot(x, np.polyval(z, fma_scores), 'k--', linewidth=1.5, alpha=0.6)

    # --- Panel 2: CCI (Co-Contraction Index) ---
    ax2 = fig.add_subplot(gs[0, 1])
    cci_vals = [effort_metrics[f]['CCI'] for f in fma_scores]
    ax2.bar(x, cci_vals, width, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, fontsize=10)
    ax2.set_ylabel('CCI', fontsize=11)
    ax2.set_title('Co-Contraction Index\n(Biceps vs Triceps)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    # Reference ranges from literature
    ax2.axhspan(0.2, 0.5, alpha=0.08, color='green', label='Healthy range')
    ax2.axhspan(0.4, 0.8, alpha=0.08, color='red', label='Impaired range')
    ax2.legend(fontsize=8, loc='upper right')
    if len(fma_scores) >= 2:
        z = np.polyfit(fma_scores, cci_vals, 1)
        ax2.plot(x, np.polyval(z, fma_scores), 'k--', linewidth=1.5, alpha=0.6)

    # --- Panel 3: ATI vs FMA scatter ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(fma_scores, ati_vals, c=bar_colors, s=100, edgecolors='black', zorder=3)
    if len(fma_scores) >= 2:
        z = np.polyfit(fma_scores, ati_vals, 1)
        x_fit = np.linspace(min(fma_scores), max(fma_scores), 100)
        ax3.plot(x_fit, np.polyval(z, x_fit), 'r--', linewidth=1.5, alpha=0.7)
        corr = np.corrcoef(fma_scores, ati_vals)[0, 1]
        ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes,
                 fontsize=11, verticalalignment='top')
    ax3.set_xlabel('FMA Score', fontsize=11)
    ax3.set_ylabel('ATI', fontsize=11)
    ax3.set_title('ATI vs FMA Score', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Per-joint TRR bar chart ---
    ax4 = fig.add_subplot(gs[1, :2])
    trr_joints = list(effort_metrics[fma_scores[0]]['TRR'].keys())
    n_j = len(trr_joints)
    bar_w = 0.8 / max(len(fma_scores), 1)
    for f_idx, fma in enumerate(fma_scores):
        vals = [effort_metrics[fma]['TRR'].get(j, 0) for j in trr_joints]
        offset = (f_idx - len(fma_scores)/2 + 0.5) * bar_w
        ax4.bar(np.arange(n_j) + offset, vals, bar_w * 0.9,
                color=colors[fma], alpha=0.85, label=labels[fma])
    ax4.set_xticks(np.arange(n_j))
    ax4.set_xticklabels([JOINT_LABELS.get(j, j) for j in trr_joints], fontsize=9, rotation=30, ha='right')
    ax4.set_ylabel('TRR (Torque / ROM)', fontsize=11)
    ax4.set_title('Torque-ROM Ratio per Joint', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=8, ncol=min(len(fma_scores), 4))
    ax4.grid(axis='y', alpha=0.3)

    # --- Panel 5: CCI vs FMA scatter ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(fma_scores, cci_vals, c=bar_colors, s=100, edgecolors='black', zorder=3)
    if len(fma_scores) >= 2:
        z = np.polyfit(fma_scores, cci_vals, 1)
        x_fit = np.linspace(min(fma_scores), max(fma_scores), 100)
        ax5.plot(x_fit, np.polyval(z, x_fit), 'r--', linewidth=1.5, alpha=0.7)
        corr = np.corrcoef(fma_scores, cci_vals)[0, 1]
        ax5.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax5.transAxes,
                 fontsize=11, verticalalignment='top')
    ax5.set_xlabel('FMA Score', fontsize=11)
    ax5.set_ylabel('CCI', fontsize=11)
    ax5.set_title('CCI vs FMA Score', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'id_comparison_effort.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# =============================================================================
# Entry point (called by pipeline or standalone)
# =============================================================================
def generate_all(id_base_dir=None, output_dir=None):
    """
    Generate all cross-FMA comparison plots.

    Args:
        id_base_dir: Directory containing FMA_*/  subdirs with ID CSVs.
        output_dir:  Where to save the 4 comparison PNGs.
    """
    if id_base_dir is None:
        id_base_dir = get_path("output_generated") + "/id"
    if output_dir is None:
        output_dir = get_path("output_generated") + "/plots"

    os.makedirs(output_dir, exist_ok=True)

    fma_scores = detect_fma_scores(id_base_dir)
    if len(fma_scores) < 2:
        print(f'  Skipping comparison plots (need >= 2 FMA scores, found {len(fma_scores)})')
        return

    colors = _make_color_map(fma_scores)
    labels = _make_labels(fma_scores)

    print(f'\n  Generating ID comparison plots for FMA scores: {fma_scores}')
    torques, activations, synergy_coefs, phase_labels, effort_metrics = load_all(id_base_dir, fma_scores)

    plot_torques_comparison(torques, fma_scores, colors, labels, output_dir)
    plot_muscle_group_activations(activations, fma_scores, colors, labels, output_dir)
    plot_synergy_comparison(synergy_coefs, fma_scores, colors, labels, output_dir)
    plot_summary_bars(torques, activations, synergy_coefs, phase_labels,
                      fma_scores, colors, labels, output_dir)
    plot_effort_comparison(effort_metrics, fma_scores, colors, labels, output_dir)

    print(f'  All comparison plots saved to: {output_dir}')


if __name__ == '__main__':
    generate_all()
