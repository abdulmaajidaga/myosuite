"""
Publication-quality figures for stroke rehabilitation inverse dynamics analysis.

Generates 5 figures:
  1. Muscle Activation Heatmap (32 muscles x time x FMA)
  2. Phase-Segmented Torque Comparison (3x2 grid)
  3. Synergy Weight Matrix (bar charts per FMA)
  4. FMA Gradient Waterfall (ATI/CCI/TRR scatter with trends)
  5. Healthy Original vs Generated Overlay (mean+std)

Usage:
  python -m src.visualization.plot_publication_figures
  python src/visualization/plot_publication_figures.py
"""
import os
import sys
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path

# =============================================================================
# CONSTANTS
# =============================================================================

# Muscle groupings for heatmap ordering
MUSCLE_ORDER = [
    # Shoulder
    'DELT1', 'DELT2', 'DELT3', 'SUPSP', 'INFSP', 'SUBSC', 'TMIN', 'TMAJ',
    # Pec/Lat
    'PECM1', 'PECM2', 'PECM3', 'LAT1', 'LAT2', 'LAT3', 'CORB',
    # Biarticular
    'BIClong', 'BICshort', 'TRIlong',
    # Elbow
    'TRIlat', 'TRImed', 'ANC', 'BRA', 'BRD', 'SUP',
    # Forearm
    'ECRL', 'ECRB', 'ECU', 'FCR', 'FCU', 'PL', 'PT', 'PQ',
]

MUSCLE_GROUP_BOUNDARIES = {
    'Shoulder': (0, 8),
    'Pec/Lat': (8, 15),
    'Biarticular': (15, 18),
    'Elbow': (18, 24),
    'Forearm': (24, 32),
}

MUSCLE_GROUPS_4 = {
    'Deltoid': ['DELT1', 'DELT2', 'DELT3'],
    'Biceps': ['BIClong', 'BICshort'],
    'Triceps': ['TRIlong', 'TRIlat', 'TRImed'],
    'Forearm': ['ECRL', 'ECRB', 'ECU', 'FCR', 'FCU'],
}

CLINICAL_GROUPS = [
    ('Severe',   0, 25, '#D7263D'),
    ('Moderate', 26, 40, '#F39237'),
    ('Mild',     41, 55, '#F9C846'),
    ('Healthy',  56, 66, '#2E86AB'),
]

# Colormap for FMA gradient
FMA_CMAP = plt.cm.RdYlGn


def _assign_group(fma):
    for name, lo, hi, _ in CLINICAL_GROUPS:
        if lo <= fma <= hi:
            return name
    return 'Unknown'


def _group_color(fma):
    for name, lo, hi, color in CLINICAL_GROUPS:
        if lo <= fma <= hi:
            return color
    return 'gray'


# =============================================================================
# DATA LOADING
# =============================================================================

def _detect_sessions(id_dir):
    """Detect all FMA sessions with ID data."""
    sessions = []
    if not os.path.isdir(id_dir):
        return sessions
    for name in sorted(os.listdir(id_dir)):
        m = re.match(r'FMA_(\d+)(?:_s\d+)?$', name)
        if not m:
            continue
        d = os.path.join(id_dir, name)
        if os.path.isfile(os.path.join(d, 'activations.csv')):
            sessions.append((int(m.group(1)), name, d))
    return sessions


def _load_activations(session_dir):
    df = pd.read_csv(os.path.join(session_dir, 'activations.csv'))
    return df


def _load_phases(session_dir):
    return pd.read_csv(os.path.join(session_dir, 'phase_labels.csv'))


def _load_torques(session_dir):
    return pd.read_csv(os.path.join(session_dir, 'torques.csv'))


def _load_synergy_weights(session_dir):
    path = os.path.join(session_dir, 'synergy_weights.csv')
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, index_col=0)


def _load_effort(session_dir):
    path = os.path.join(session_dir, 'effort_metrics.json')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _resample(values, n_points=200):
    """Resample a 1D array to uniform n_points."""
    x_old = np.linspace(0, 1, len(values))
    x_new = np.linspace(0, 1, n_points)
    f = interp1d(x_old, values, kind='linear', fill_value='extrapolate')
    return f(x_new)


# =============================================================================
# FIGURE 1: Muscle Activation Heatmap
# =============================================================================

def plot_activation_heatmap(sessions, output_dir):
    """32 muscles x time, averaged per FMA level, phase boundary lines."""
    fma_levels = sorted(set(fma for fma, _, _ in sessions))
    n_fma = len(fma_levels)
    n_time = 200

    fig, axes = plt.subplots(1, n_fma, figsize=(3 * n_fma, 8), sharey=True)
    if n_fma == 1:
        axes = [axes]
    fig.suptitle('Muscle Activation Heatmaps by FMA Score', fontsize=14, fontweight='bold', y=1.02)

    for ax_idx, fma in enumerate(fma_levels):
        # Collect all sessions for this FMA
        fma_sessions = [(f, n, d) for f, n, d in sessions if f == fma]
        heatmaps = []

        for _, _, d in fma_sessions:
            act_df = _load_activations(d)
            act_cols = [m for m in MUSCLE_ORDER if m in act_df.columns]
            act_vals = act_df[act_cols].values  # (frames, muscles)

            # Resample to uniform time
            resampled = np.zeros((n_time, len(act_cols)))
            for mi, m in enumerate(act_cols):
                resampled[:, mi] = _resample(act_vals[:, mi], n_time)
            heatmaps.append(resampled)

        if not heatmaps:
            continue

        # Average across samples
        avg_heatmap = np.mean(heatmaps, axis=0).T  # (muscles, time)

        ax = axes[ax_idx]
        im = ax.imshow(avg_heatmap, aspect='auto', cmap='hot',
                        extent=[0, 100, len(act_cols) - 0.5, -0.5],
                        vmin=0, vmax=0.6, interpolation='bilinear')

        # Phase boundary lines (approximate)
        ax.axvline(40, color='cyan', linewidth=1, linestyle='--', alpha=0.7)
        ax.axvline(70, color='cyan', linewidth=1, linestyle='--', alpha=0.7)

        # Muscle group boundaries
        for group_name, (start, end) in MUSCLE_GROUP_BOUNDARIES.items():
            if start > 0:
                ax.axhline(start - 0.5, color='white', linewidth=0.5, alpha=0.5)

        ax.set_xlabel('Motion %', fontsize=9)
        ax.set_title(f'FMA {fma}', fontsize=11, fontweight='bold',
                     color=_group_color(fma))

        if ax_idx == 0:
            ax.set_yticks(range(len(act_cols)))
            ax.set_yticklabels(act_cols, fontsize=7)
        else:
            ax.set_yticks([])

    fig.colorbar(im, ax=axes, label='Activation', shrink=0.6, pad=0.02)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pub_fig1_activation_heatmap.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Figure 1 saved: {path}')


# =============================================================================
# FIGURE 2: Phase-Segmented Torque Comparison
# =============================================================================

def plot_phase_torques(sessions, output_dir):
    """3x2 grid: rows = Pick/Drink/Place, cols = shoulder_elv/elbow_flexion."""
    fma_levels = sorted(set(fma for fma, _, _ in sessions))
    joints = ['shoulder_elv', 'elbow_flexion']
    phases = ['Pick', 'Drink', 'Place']

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle('Phase-Segmented Torques by FMA Score',
                 fontsize=14, fontweight='bold', y=0.98)

    for row, phase in enumerate(phases):
        for col, joint in enumerate(joints):
            ax = axes[row, col]

            # Collect per-FMA boxplot data
            data_per_fma = []
            fma_labels = []

            for fma in fma_levels:
                fma_sessions = [(f, n, d) for f, n, d in sessions if f == fma]
                phase_torques = []

                for _, _, d in fma_sessions:
                    torque_df = _load_torques(d)
                    phase_df = _load_phases(d)
                    if joint not in torque_df.columns:
                        continue
                    mask = phase_df['phase'] == phase
                    if mask.sum() == 0:
                        continue
                    vals = torque_df.loc[mask, joint].values
                    phase_torques.append(np.mean(np.abs(vals)))

                if phase_torques:
                    data_per_fma.append(phase_torques)
                    fma_labels.append(f'{fma}')

            if not data_per_fma:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue

            bp = ax.boxplot(data_per_fma, patch_artist=True, widths=0.6)
            for i, (patch, fma) in enumerate(zip(bp['boxes'], fma_levels)):
                patch.set_facecolor(_group_color(fma))
                patch.set_alpha(0.7)
            for median in bp['medians']:
                median.set(color='black', linewidth=2)

            ax.set_xticklabels(fma_labels, fontsize=8,
                               rotation=45 if len(fma_labels) > 8 else 0)
            joint_label = 'Shoulder Elevation' if joint == 'shoulder_elv' else 'Elbow Flexion'
            ax.set_ylabel(f'Mean |Torque| (Nm)', fontsize=9)

            if row == 0:
                ax.set_title(joint_label, fontsize=12, fontweight='bold')
            if col == 0:
                ax.annotate(phase, xy=(-0.2, 0.5), xycoords='axes fraction',
                           fontsize=13, fontweight='bold', rotation=90, va='center',
                           color=_group_color(fma_levels[0]))
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0.05, 0, 1, 0.95])
    path = os.path.join(output_dir, 'pub_fig2_phase_torques.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Figure 2 saved: {path}')


# =============================================================================
# FIGURE 3: Synergy Weight Matrix
# =============================================================================

def plot_synergy_weights(sessions, output_dir):
    """Bar charts of 4 synergy weight vectors per FMA level."""
    fma_levels = sorted(set(fma for fma, _, _ in sessions))

    # Select representative FMA levels (max 6 for readability)
    if len(fma_levels) > 6:
        # Pick evenly spaced
        indices = np.linspace(0, len(fma_levels) - 1, 6, dtype=int)
        selected_fma = [fma_levels[i] for i in indices]
    else:
        selected_fma = fma_levels

    n_fma = len(selected_fma)
    fig, axes = plt.subplots(4, n_fma, figsize=(3 * n_fma, 10), sharey='row')
    fig.suptitle('Synergy Weight Vectors by FMA Score',
                 fontsize=14, fontweight='bold', y=1.01)

    syn_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    for col_idx, fma in enumerate(selected_fma):
        fma_sessions = [(f, n, d) for f, n, d in sessions if f == fma]

        # Average synergy weights across samples
        all_weights = []
        for _, _, d in fma_sessions:
            sw = _load_synergy_weights(d)
            if sw is not None:
                all_weights.append(sw.values)

        if not all_weights:
            for s in range(4):
                axes[s, col_idx].text(0.5, 0.5, 'No data',
                                       transform=axes[s, col_idx].transAxes, ha='center')
            continue

        avg_weights = np.mean(all_weights, axis=0)  # (4, n_muscles)
        muscle_names = list(_load_synergy_weights(fma_sessions[0][2]).columns)

        for s_idx in range(min(4, avg_weights.shape[0])):
            ax = axes[s_idx, col_idx]
            w = avg_weights[s_idx]

            # Color by magnitude
            colors_arr = [syn_colors[s_idx] if v > 0.1 else '#cccccc' for v in w]
            ax.bar(range(len(w)), w, color=colors_arr, width=0.8)

            if col_idx == 0:
                ax.set_ylabel(f'Synergy {s_idx + 1}', fontsize=10, fontweight='bold')
            if s_idx == 0:
                ax.set_title(f'FMA {fma}', fontsize=11, fontweight='bold',
                             color=_group_color(fma))
            if s_idx == 3:
                # Bottom row: show muscle labels
                ax.set_xticks(range(len(muscle_names)))
                ax.set_xticklabels(muscle_names, fontsize=5, rotation=90)
            else:
                ax.set_xticks([])

            ax.set_xlim(-0.5, len(w) - 0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pub_fig3_synergy_weights.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Figure 3 saved: {path}')


# =============================================================================
# FIGURE 4: FMA Gradient Waterfall (ATI/CCI/TRR scatter)
# =============================================================================

def plot_fma_gradient(sessions, orig_id_dir, output_dir):
    """ATI, CCI, TRR scatter with median trend line and percentile bands."""
    # Collect per-session metrics
    rows = []
    for fma, name, d in sessions:
        effort = _load_effort(d)
        if effort is None:
            continue
        rows.append({
            'fma': fma, 'session': name,
            'ATI': effort.get('ATI', np.nan),
            'CCI': effort.get('CCI', np.nan),
            'TRR_elbow': effort.get('TRR', {}).get('elbow_flexion', np.nan),
        })
    df = pd.DataFrame(rows)

    # Original healthy baseline
    orig_ati, orig_cci = None, None
    if os.path.isdir(orig_id_dir):
        orig_vals = []
        for name in os.listdir(orig_id_dir):
            path = os.path.join(orig_id_dir, name, 'effort_metrics.json')
            if os.path.isfile(path):
                with open(path) as f:
                    m = json.load(f)
                orig_vals.append(m)
        if orig_vals:
            orig_ati = np.mean([m['ATI'] for m in orig_vals])
            orig_cci = np.mean([m['CCI'] for m in orig_vals])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('FMA Gradient: Effort Metrics with Trend Lines',
                 fontsize=14, fontweight='bold', y=1.02)

    metrics = [('ATI', 'Activation-Time Integral', orig_ati),
               ('CCI', 'Co-Contraction Index', orig_cci),
               ('TRR_elbow', 'Elbow TRR', None)]

    for ax, (metric, title, orig_ref) in zip(axes, metrics):
        valid = df[['fma', metric]].dropna()
        if valid.empty:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        x = valid['fma'].values
        y = valid[metric].values

        # Scatter with group colors
        colors = [_group_color(f) for f in x]
        ax.scatter(x, y, c=colors, s=40, alpha=0.6, edgecolors='white', linewidth=0.5, zorder=3)

        # Median trend line
        fma_unique = sorted(valid['fma'].unique())
        medians = [valid.loc[valid['fma'] == f, metric].median() for f in fma_unique]
        ax.plot(fma_unique, medians, 'k-', linewidth=2, alpha=0.8, label='Median trend')

        # 25th-75th percentile band
        q25 = [valid.loc[valid['fma'] == f, metric].quantile(0.25) for f in fma_unique]
        q75 = [valid.loc[valid['fma'] == f, metric].quantile(0.75) for f in fma_unique]
        ax.fill_between(fma_unique, q25, q75, alpha=0.15, color='gray', label='IQR')

        # Original healthy baseline
        if orig_ref is not None:
            ax.axhline(orig_ref, color='orange', linewidth=2, linestyle='--',
                       alpha=0.8, label=f'Original healthy ({orig_ref:.3f})')

        # Spearman correlation
        if len(x) >= 4:
            rho, p = stats.spearmanr(x, y)
            ax.text(0.05, 0.95, f'rho={rho:.3f}, p={p:.4f}',
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('FMA Score', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(output_dir, 'pub_fig4_fma_gradient.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Figure 4 saved: {path}')


# =============================================================================
# FIGURE 5: Healthy Original vs Generated Overlay
# =============================================================================

def plot_healthy_overlay(sessions, orig_id_dir, output_dir):
    """Mean+std activation comparison for 4 muscle groups, time-normalized."""
    n_time = 200
    t_pct = np.linspace(0, 100, n_time)

    # Collect generated healthy activations
    gen_healthy = [(f, n, d) for f, n, d in sessions if f >= 56]

    # Collect original healthy activations
    orig_sessions = []
    if os.path.isdir(orig_id_dir):
        for name in sorted(os.listdir(orig_id_dir)):
            act_path = os.path.join(orig_id_dir, name, 'activations.csv')
            if os.path.isfile(act_path):
                orig_sessions.append(os.path.join(orig_id_dir, name))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Generated Healthy vs Original Healthy: Muscle Group Activations',
                 fontsize=14, fontweight='bold', y=0.98)
    axes = axes.flatten()

    for idx, (group_name, muscles) in enumerate(MUSCLE_GROUPS_4.items()):
        ax = axes[idx]

        # Generated healthy
        gen_traces = []
        for _, _, d in gen_healthy:
            act_df = _load_activations(d)
            present = [m for m in muscles if m in act_df.columns]
            if not present:
                continue
            group_mean = act_df[present].mean(axis=1).values
            gen_traces.append(_resample(group_mean, n_time))

        # Original healthy
        orig_traces = []
        for d in orig_sessions[:50]:  # cap at 50 for speed
            act_path = os.path.join(d, 'activations.csv')
            act_df = pd.read_csv(act_path)
            present = [m for m in muscles if m in act_df.columns]
            if not present:
                continue
            group_mean = act_df[present].mean(axis=1).values
            orig_traces.append(_resample(group_mean, n_time))

        if gen_traces:
            gen_arr = np.array(gen_traces)
            gen_mean = np.mean(gen_arr, axis=0)
            gen_std = np.std(gen_arr, axis=0)
            ax.plot(t_pct, gen_mean, color='#2E86AB', linewidth=2, label='Generated Healthy')
            ax.fill_between(t_pct, gen_mean - gen_std, gen_mean + gen_std,
                           color='#2E86AB', alpha=0.2)

        if orig_traces:
            orig_arr = np.array(orig_traces)
            orig_mean = np.mean(orig_arr, axis=0)
            orig_std = np.std(orig_arr, axis=0)
            ax.plot(t_pct, orig_mean, color='#F39237', linewidth=2,
                    linestyle='--', label='Original Healthy')
            ax.fill_between(t_pct, orig_mean - orig_std, orig_mean + orig_std,
                           color='#F39237', alpha=0.2)

        ax.set_title(group_name, fontsize=13, fontweight='bold')
        ax.set_xlabel('Motion Progress (%)', fontsize=10)
        ax.set_ylabel('Mean Activation', fontsize=10)
        ax.set_ylim(-0.05, 0.8)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)

        # Phase regions
        ax.axvspan(0, 40, alpha=0.03, color='blue')
        ax.axvspan(40, 70, alpha=0.03, color='green')
        ax.axvspan(70, 100, alpha=0.03, color='orange')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(output_dir, 'pub_fig5_healthy_overlay.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Figure 5 saved: {path}')


# =============================================================================
# MAIN
# =============================================================================

def generate_all(gen_id_dir=None, orig_id_dir=None, output_dir=None):
    """Generate all 5 publication figures."""
    if gen_id_dir is None:
        gen_id_dir = os.path.join(get_path("output_generated"), "id")
    if orig_id_dir is None:
        orig_id_dir = get_path("output_originals_id")
    if output_dir is None:
        output_dir = get_path("output_generated_plots")

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Publication Figures")
    print("=" * 60)

    sessions = _detect_sessions(gen_id_dir)
    if not sessions:
        print(f"No generated ID sessions found in {gen_id_dir}")
        return

    fma_levels = sorted(set(fma for fma, _, _ in sessions))
    print(f"Found {len(sessions)} sessions across {len(fma_levels)} FMA levels: {fma_levels}")

    print("\n  Generating Figure 1: Muscle Activation Heatmap...")
    plot_activation_heatmap(sessions, output_dir)

    print("  Generating Figure 2: Phase-Segmented Torques...")
    plot_phase_torques(sessions, output_dir)

    print("  Generating Figure 3: Synergy Weight Matrix...")
    plot_synergy_weights(sessions, output_dir)

    print("  Generating Figure 4: FMA Gradient Waterfall...")
    plot_fma_gradient(sessions, orig_id_dir, output_dir)

    print("  Generating Figure 5: Healthy Original vs Generated Overlay...")
    plot_healthy_overlay(sessions, orig_id_dir, output_dir)

    print(f"\n  All publication figures saved to: {output_dir}")


if __name__ == '__main__':
    generate_all()
