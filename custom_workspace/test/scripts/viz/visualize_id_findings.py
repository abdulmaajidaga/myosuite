"""
Publication-quality visualizations of inverse dynamics findings across FMA scores.

Generates 6 figures in output/generated/plots/findings/:
  1. muscle_dominance.png      — Top-10 muscles ranked by activation, per FMA
  2. activation_profiles.png   — Muscle group time-series with phase shading
  3. phase_proportions.png     — Stacked bar + pie charts of Pick/Drink/Place
  4. torque_profiles.png       — Key joint torques with severity gradient
  5. synergy_timing.png        — Synergy coefficient heatmaps + temporal peaks
  6. cocontraction.png         — Biceps/triceps co-contraction over motion
  7. summary_dashboard.png     — Single-page overview of all key metrics

Usage:
  python scripts/viz/visualize_id_findings.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os, sys, glob

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path

ID_DIR = os.path.join(get_path("output_dir"), "generated/id")
OUT_DIR = os.path.join(get_path("output_dir"), "generated/plots/findings")
os.makedirs(OUT_DIR, exist_ok=True)

# FMA score ordering (severe → healthy) and color map
FMA_SCORES = [18, 20, 30, 40, 50, 66]
FMA_LABELS = {18: 'FMA 18\n(severe)', 20: 'FMA 20\n(severe)', 30: 'FMA 30\n(moderate)',
              40: 'FMA 40\n(moderate)', 50: 'FMA 50\n(mild)', 66: 'FMA 66\n(healthy)'}
FMA_SHORT = {18: 'FMA 18', 20: 'FMA 20', 30: 'FMA 30', 40: 'FMA 40', 50: 'FMA 50', 66: 'FMA 66'}

# Red (severe) → Green (healthy) colormap
CMAP = plt.cm.RdYlGn
FMA_COLORS = {s: CMAP(i / (len(FMA_SCORES) - 1)) for i, s in enumerate(FMA_SCORES)}

PHASE_COLORS = {'Pick': '#3498db', 'Drink': '#e74c3c', 'Place': '#2ecc71'}

# Muscle groups for organized display
MUSCLE_GROUPS = {
    'Shoulder (Delt/Cuff)': ['DELT1', 'DELT2', 'DELT3', 'SUPSP', 'INFSP', 'SUBSC', 'TMIN', 'TMAJ'],
    'Pec/Lat/Corb': ['PECM1', 'PECM2', 'PECM3', 'LAT1', 'LAT2', 'LAT3', 'CORB'],
    'Elbow (Tri/Bic/Bra)': ['TRIlong', 'TRIlat', 'TRImed', 'BIClong', 'BICshort', 'BRA'],
    'Forearm/Wrist': ['PT', 'PQ', 'SUP', 'ECR_L', 'ECR_B', 'ECU', 'FCR', 'FCU', 'PL'],
}


def load_all_data():
    """Load activations, torques, phases, synergies for all FMA scores."""
    data = {}
    for fma in FMA_SCORES:
        d = os.path.join(ID_DIR, f"FMA_{fma}")
        if not os.path.isdir(d):
            continue
        entry = {}
        entry['act'] = pd.read_csv(os.path.join(d, 'activations.csv'))
        entry['torques'] = pd.read_csv(os.path.join(d, 'torques.csv'))
        entry['phases'] = pd.read_csv(os.path.join(d, 'phase_labels.csv'))
        syn_coeff_path = os.path.join(d, 'synergy_coefficients.csv')
        if os.path.exists(syn_coeff_path):
            entry['syn_coeff'] = pd.read_csv(syn_coeff_path)
        syn_w_path = os.path.join(d, 'synergy_weights.csv')
        if os.path.exists(syn_w_path):
            entry['syn_weights'] = pd.read_csv(syn_w_path, index_col=0)
        data[fma] = entry
    return data


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Muscle Dominance — Top 10 muscles per FMA
# ═══════════════════════════════════════════════════════════════════════
def plot_muscle_dominance(data):
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, fma in enumerate(FMA_SCORES):
        ax = axes[idx]
        act = data[fma]['act']
        muscle_cols = [c for c in act.columns if c != 'time']
        means = act[muscle_cols].mean().sort_values(ascending=True)
        top10 = means.tail(10)

        colors = ['#e74c3c' if 'BIC' in m else '#3498db' if 'TRI' in m
                  else '#f39c12' if 'DELT' in m else '#95a5a6' for m in top10.index]

        bars = ax.barh(range(len(top10)), top10.values, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(top10.index, fontsize=9)
        ax.set_xlabel('Mean Activation', fontsize=10)
        ax.set_title(FMA_SHORT[fma], fontsize=13, fontweight='bold', color=FMA_COLORS[fma])
        ax.set_xlim(0, 0.75)
        ax.grid(True, alpha=0.2, axis='x')
        ax.axvline(0.5, color='red', ls='--', alpha=0.3, lw=1)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='Biceps'),
                       Patch(facecolor='#3498db', label='Triceps'),
                       Patch(facecolor='#f39c12', label='Deltoid'),
                       Patch(facecolor='#95a5a6', label='Other')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Top 10 Most Active Muscles by FMA Score\n'
                 'Biceps dominance confirms valid drinking-task biomechanics',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    path = os.path.join(OUT_DIR, 'muscle_dominance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Activation Profiles — Muscle groups over time with phases
# ═══════════════════════════════════════════════════════════════════════
def plot_activation_profiles(data):
    groups = list(MUSCLE_GROUPS.keys())
    fig, axes = plt.subplots(len(groups), 1, figsize=(16, 4 * len(groups)), sharex=True)

    for row, (group_name, muscles) in enumerate(MUSCLE_GROUPS.items()):
        ax = axes[row]

        for fma in FMA_SCORES:
            act = data[fma]['act']
            phases = data[fma]['phases']
            available = [m for m in muscles if m in act.columns]
            if not available:
                continue

            pct = np.linspace(0, 100, len(act))
            group_mean = act[available].mean(axis=1).values
            group_std = act[available].std(axis=1).values

            ax.plot(pct, group_mean, color=FMA_COLORS[fma], lw=2, label=FMA_SHORT[fma])
            ax.fill_between(pct, group_mean - group_std, group_mean + group_std,
                           color=FMA_COLORS[fma], alpha=0.08)

        # Phase shading (use FMA_50 as reference for phase boundaries)
        phases_ref = data[50]['phases']
        n_ref = len(phases_ref)
        phase_vals = phases_ref['phase'].values
        for phase_name, color in PHASE_COLORS.items():
            mask = phase_vals == phase_name
            if mask.any():
                start_pct = np.where(mask)[0][0] / n_ref * 100
                end_pct = (np.where(mask)[0][-1] + 1) / n_ref * 100
                ax.axvspan(start_pct, end_pct, alpha=0.06, color=color)
                mid = (start_pct + end_pct) / 2
                if row == 0:
                    ax.text(mid, ax.get_ylim()[1] * 0.95, phase_name,
                            ha='center', fontsize=10, color=color, fontweight='bold', alpha=0.7)

        ax.set_ylabel('Activation', fontsize=11)
        ax.set_title(group_name, fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.2)
        if row == 0:
            ax.legend(fontsize=9, loc='upper right', ncol=3)

    axes[-1].set_xlabel('Motion Progress (%)', fontsize=12)
    plt.suptitle('Muscle Group Activation Profiles Across FMA Scores\n'
                 'Phase-shaded: Pick (blue) | Drink (red) | Place (green)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, 'activation_profiles.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Phase Proportions
# ═══════════════════════════════════════════════════════════════════════
def plot_phase_proportions(data):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [2, 1]})

    # Left: stacked bar chart
    ax = axes[0]
    proportions = {}
    for fma in FMA_SCORES:
        phases = data[fma]['phases']['phase'].value_counts(normalize=True)
        proportions[fma] = {p: phases.get(p, 0) * 100 for p in ['Pick', 'Drink', 'Place']}

    x = np.arange(len(FMA_SCORES))
    bottom = np.zeros(len(FMA_SCORES))
    for phase in ['Pick', 'Drink', 'Place']:
        vals = [proportions[f][phase] for f in FMA_SCORES]
        bars = ax.bar(x, vals, bottom=bottom, color=PHASE_COLORS[phase],
                      label=phase, edgecolor='white', linewidth=1.5, width=0.6)
        # Label percentages
        for i, v in enumerate(vals):
            if v > 5:
                ax.text(x[i], bottom[i] + v/2, f'{v:.0f}%',
                        ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([FMA_SHORT[f] for f in FMA_SCORES], fontsize=11)
    ax.set_ylabel('Proportion of Motion (%)', fontsize=12)
    ax.set_title('Movement Phase Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis='y')

    # Right: Drink phase duration comparison
    ax2 = axes[1]
    drink_pcts = [proportions[f]['Drink'] for f in FMA_SCORES]
    drink_times = [len(data[f]['phases']) / 200 * proportions[f]['Drink'] / 100 for f in FMA_SCORES]
    colors = [FMA_COLORS[f] for f in FMA_SCORES]

    bars = ax2.barh(range(len(FMA_SCORES)), drink_times, color=colors,
                     edgecolor='white', linewidth=1)
    ax2.set_yticks(range(len(FMA_SCORES)))
    ax2.set_yticklabels([FMA_SHORT[f] for f in FMA_SCORES], fontsize=11)
    ax2.set_xlabel('Drink Phase Duration (seconds)', fontsize=12)
    ax2.set_title('Time Spent Drinking', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='x')

    for i, (t, pct) in enumerate(zip(drink_times, drink_pcts)):
        ax2.text(t + 0.02, i, f'{t:.2f}s ({pct:.0f}%)', va='center', fontsize=10)

    plt.suptitle('Phase Analysis: Impaired Patients Spend More Time in Drink Phase\n'
                 'Healthy subjects (FMA 66) execute a quick, efficient sip',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUT_DIR, 'phase_proportions.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Key Joint Torque Profiles
# ═══════════════════════════════════════════════════════════════════════
def plot_torque_profiles(data):
    joints = ['elv_angle', 'shoulder_elv', 'elbow_flexion', 'pro_sup']
    joint_labels = ['Elevation Angle', 'Shoulder Elevation', 'Elbow Flexion', 'Pro/Supination']

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (joint, label) in enumerate(zip(joints, joint_labels)):
        ax = axes[idx]

        for fma in FMA_SCORES:
            torques = data[fma]['torques']
            if joint not in torques.columns:
                continue
            pct = np.linspace(0, 100, len(torques))
            ax.plot(pct, torques[joint].values, color=FMA_COLORS[fma], lw=2.5,
                    label=FMA_SHORT[fma], alpha=0.85)

        # Phase shading
        phases_ref = data[50]['phases']
        n_ref = len(phases_ref)
        for phase_name, color in PHASE_COLORS.items():
            mask = phases_ref['phase'].values == phase_name
            if mask.any():
                s = np.where(mask)[0][0] / n_ref * 100
                e = (np.where(mask)[0][-1] + 1) / n_ref * 100
                ax.axvspan(s, e, alpha=0.06, color=color)

        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xlabel('Motion Progress (%)', fontsize=11)
        ax.set_ylabel('Torque (Nm)', fontsize=11)
        ax.grid(True, alpha=0.2)
        ax.axhline(0, color='black', lw=0.5, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9, loc='best', ncol=2)

    plt.suptitle('Joint Torque Profiles: Severity Gradient Visible in Shoulder & Elbow\n'
                 'Elbow flexion bell-curve confirms drinking-task biomechanics',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, 'torque_profiles.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: Synergy Timing
# ═══════════════════════════════════════════════════════════════════════
def plot_synergy_timing(data):
    n_syn = 4
    fig, axes = plt.subplots(n_syn, 1, figsize=(16, 3.5 * n_syn), sharex=True)

    for syn_idx in range(n_syn):
        ax = axes[syn_idx]
        syn_name = f'Synergy_{syn_idx + 1}'

        for fma in FMA_SCORES:
            if 'syn_coeff' not in data[fma]:
                continue
            sc = data[fma]['syn_coeff']
            if syn_name not in sc.columns:
                continue
            pct = np.linspace(0, 100, len(sc))
            vals = sc[syn_name].values
            ax.plot(pct, vals, color=FMA_COLORS[fma], lw=2.5, label=FMA_SHORT[fma], alpha=0.85)

            # Mark peak
            peak_idx = np.argmax(vals)
            peak_pct = pct[peak_idx]
            ax.plot(peak_pct, vals[peak_idx], 'o', color=FMA_COLORS[fma],
                    markersize=8, markeredgecolor='white', markeredgewidth=1.5, zorder=5)

        # Phase shading
        phases_ref = data[50]['phases']
        n_ref = len(phases_ref)
        for phase_name, color in PHASE_COLORS.items():
            mask = phases_ref['phase'].values == phase_name
            if mask.any():
                s = np.where(mask)[0][0] / n_ref * 100
                e = (np.where(mask)[0][-1] + 1) / n_ref * 100
                ax.axvspan(s, e, alpha=0.06, color=color)

        ax.set_title(f'Synergy {syn_idx + 1}', fontsize=13, fontweight='bold')
        ax.set_ylabel('Coefficient', fontsize=11)
        ax.grid(True, alpha=0.2)
        if syn_idx == 0:
            ax.legend(fontsize=9, loc='upper right', ncol=3)

    axes[-1].set_xlabel('Motion Progress (%)', fontsize=12)
    plt.suptitle('Muscle Synergy Timing: Peak Shifts with Impairment Severity\n'
                 'Dots mark peak activation — impaired patients show delayed recruitment',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, 'synergy_timing.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: Co-contraction (Biceps vs Triceps)
# ═══════════════════════════════════════════════════════════════════════
def plot_cocontraction(data):
    bicep_muscles = ['BIClong', 'BICshort']
    tricep_muscles = ['TRIlong', 'TRIlat', 'TRImed']

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Top: individual activation traces
    ax = axes[0]
    for fma in FMA_SCORES:
        act = data[fma]['act']
        pct = np.linspace(0, 100, len(act))
        bic = act[[m for m in bicep_muscles if m in act.columns]].mean(axis=1).values
        tri = act[[m for m in tricep_muscles if m in act.columns]].mean(axis=1).values
        ax.plot(pct, bic, color=FMA_COLORS[fma], lw=2, ls='-', alpha=0.8)
        ax.plot(pct, tri, color=FMA_COLORS[fma], lw=2, ls='--', alpha=0.8)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_fma = [Line2D([0], [0], color=FMA_COLORS[f], lw=2) for f in FMA_SCORES]
    legend_type = [Line2D([0], [0], color='gray', lw=2, ls='-'),
                   Line2D([0], [0], color='gray', lw=2, ls='--')]
    leg1 = ax.legend(legend_fma, [FMA_SHORT[f] for f in FMA_SCORES],
                     loc='upper right', fontsize=8, ncol=3, title='FMA Score')
    ax.add_artist(leg1)
    ax.legend(legend_type, ['Biceps (mean)', 'Triceps (mean)'],
              loc='upper left', fontsize=10)

    ax.set_ylabel('Activation', fontsize=11)
    ax.set_title('Biceps vs Triceps Activation', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)

    # Bottom: co-contraction index
    ax2 = axes[1]
    for fma in FMA_SCORES:
        act = data[fma]['act']
        pct = np.linspace(0, 100, len(act))
        bic = act[[m for m in bicep_muscles if m in act.columns]].mean(axis=1).values
        tri = act[[m for m in tricep_muscles if m in act.columns]].mean(axis=1).values
        # Co-contraction index: 2 * min(bic, tri) / (bic + tri + eps)
        cci = 2 * np.minimum(bic, tri) / (bic + tri + 1e-6)
        # Smooth
        kernel = 11
        cci_smooth = np.convolve(cci, np.ones(kernel)/kernel, mode='same')
        ax2.plot(pct, cci_smooth, color=FMA_COLORS[fma], lw=2.5, label=FMA_SHORT[fma])

    # Phase shading
    phases_ref = data[50]['phases']
    n_ref = len(phases_ref)
    for phase_name, color in PHASE_COLORS.items():
        mask = phases_ref['phase'].values == phase_name
        if mask.any():
            s = np.where(mask)[0][0] / n_ref * 100
            e = (np.where(mask)[0][-1] + 1) / n_ref * 100
            for a in axes:
                a.axvspan(s, e, alpha=0.06, color=color)

    ax2.set_ylabel('Co-contraction Index', fontsize=11)
    ax2.set_xlabel('Motion Progress (%)', fontsize=12)
    ax2.set_title('Elbow Co-contraction Index (higher = more simultaneous flexor/extensor activity)',
                  fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=9, loc='best', ncol=3)
    ax2.grid(True, alpha=0.2)

    plt.suptitle('Elbow Co-contraction: Arm Stabilization During Drinking Task\n'
                 'High co-contraction indicates simultaneous biceps+triceps for joint stability',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, 'cocontraction.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7: Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════
def plot_summary_dashboard(data):
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35)

    # --- Panel A: Mean activation by FMA ---
    ax = fig.add_subplot(gs[0, 0])
    muscle_cols_all = {}
    for fma in FMA_SCORES:
        act = data[fma]['act']
        muscle_cols = [c for c in act.columns if c != 'time']
        muscle_cols_all[fma] = muscle_cols
    means = [data[f]['act'][muscle_cols_all[f]].values.mean() for f in FMA_SCORES]
    bars = ax.bar(range(len(FMA_SCORES)), means,
                  color=[FMA_COLORS[f] for f in FMA_SCORES], edgecolor='white')
    ax.set_xticks(range(len(FMA_SCORES)))
    ax.set_xticklabels([f'FMA\n{f}' for f in FMA_SCORES], fontsize=8)
    ax.set_ylabel('Mean Activation')
    ax.set_title('A. Overall Muscle Effort', fontweight='bold')
    ax.set_ylim(0, 0.35)
    ax.grid(True, alpha=0.2, axis='y')
    for i, v in enumerate(means):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)

    # --- Panel B: Biceps peak activation ---
    ax = fig.add_subplot(gs[0, 1])
    bic_peaks = []
    for fma in FMA_SCORES:
        act = data[fma]['act']
        bic_cols = [c for c in ['BIClong', 'BICshort'] if c in act.columns]
        bic_peaks.append(act[bic_cols].max().max())
    ax.bar(range(len(FMA_SCORES)), bic_peaks,
           color=[FMA_COLORS[f] for f in FMA_SCORES], edgecolor='white')
    ax.set_xticks(range(len(FMA_SCORES)))
    ax.set_xticklabels([f'FMA\n{f}' for f in FMA_SCORES], fontsize=8)
    ax.set_ylabel('Peak Activation')
    ax.set_title('B. Peak Biceps Activation', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2, axis='y')

    # --- Panel C: Drink phase % ---
    ax = fig.add_subplot(gs[0, 2])
    drink_pcts = []
    for fma in FMA_SCORES:
        phases = data[fma]['phases']['phase']
        drink_pcts.append((phases == 'Drink').sum() / len(phases) * 100)
    ax.bar(range(len(FMA_SCORES)), drink_pcts,
           color=[FMA_COLORS[f] for f in FMA_SCORES], edgecolor='white')
    ax.set_xticks(range(len(FMA_SCORES)))
    ax.set_xticklabels([f'FMA\n{f}' for f in FMA_SCORES], fontsize=8)
    ax.set_ylabel('% of Motion')
    ax.set_title('C. Time in Drink Phase', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    for i, v in enumerate(drink_pcts):
        ax.text(i, v + 0.5, f'{v:.0f}%', ha='center', fontsize=8)

    # --- Panel D: Total motion duration ---
    ax = fig.add_subplot(gs[0, 3])
    durations = [len(data[f]['phases']) / 200 for f in FMA_SCORES]
    ax.bar(range(len(FMA_SCORES)), durations,
           color=[FMA_COLORS[f] for f in FMA_SCORES], edgecolor='white')
    ax.set_xticks(range(len(FMA_SCORES)))
    ax.set_xticklabels([f'FMA\n{f}' for f in FMA_SCORES], fontsize=8)
    ax.set_ylabel('Duration (s)')
    ax.set_title('D. Total Motion Duration', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    for i, v in enumerate(durations):
        ax.text(i, v + 0.05, f'{v:.1f}s', ha='center', fontsize=8)

    # --- Panel E: Elbow flexion torque (wide) ---
    ax = fig.add_subplot(gs[1, :2])
    for fma in FMA_SCORES:
        torques = data[fma]['torques']
        if 'elbow_flexion' in torques.columns:
            pct = np.linspace(0, 100, len(torques))
            ax.plot(pct, torques['elbow_flexion'].values, color=FMA_COLORS[fma],
                    lw=2.5, label=FMA_SHORT[fma])
    ax.set_xlabel('Motion Progress (%)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('E. Elbow Flexion Torque — Bell Curve Confirms Drinking Biomechanics',
                 fontweight='bold')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.2)
    ax.axhline(0, color='black', lw=0.5, alpha=0.3)

    # --- Panel F: Shoulder activation (wide) ---
    ax = fig.add_subplot(gs[1, 2:])
    shoulder_muscles = ['DELT1', 'DELT2', 'DELT3', 'SUPSP', 'INFSP', 'SUBSC']
    for fma in FMA_SCORES:
        act = data[fma]['act']
        avail = [m for m in shoulder_muscles if m in act.columns]
        pct = np.linspace(0, 100, len(act))
        ax.plot(pct, act[avail].mean(axis=1).values, color=FMA_COLORS[fma],
                lw=2.5, label=FMA_SHORT[fma])
    ax.set_xlabel('Motion Progress (%)')
    ax.set_ylabel('Mean Activation')
    ax.set_title('F. Shoulder Muscle Activation — Severity Gradient',
                 fontweight='bold')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.2)

    # --- Panel G: Synergy peak timing (wide) ---
    ax = fig.add_subplot(gs[2, :2])
    syn_names = ['Synergy_1', 'Synergy_2', 'Synergy_3', 'Synergy_4']
    bar_width = 0.12
    for j, syn in enumerate(syn_names):
        peaks = []
        for fma in FMA_SCORES:
            if 'syn_coeff' in data[fma] and syn in data[fma]['syn_coeff'].columns:
                vals = data[fma]['syn_coeff'][syn].values
                peak_pct = np.argmax(vals) / len(vals) * 100
                peaks.append(peak_pct)
            else:
                peaks.append(0)
        x = np.arange(len(FMA_SCORES))
        ax.bar(x + j * bar_width, peaks, bar_width, label=syn.replace('_', ' '),
               alpha=0.85, edgecolor='white')
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels([FMA_SHORT[f] for f in FMA_SCORES], fontsize=9)
    ax.set_ylabel('Peak Timing (% motion)')
    ax.set_title('G. Synergy Peak Timing — Delayed Recruitment in Impairment',
                 fontweight='bold')
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.2, axis='y')

    # --- Panel H: Co-contraction bar ---
    ax = fig.add_subplot(gs[2, 2:])
    ccis = []
    for fma in FMA_SCORES:
        act = data[fma]['act']
        bic = act[[m for m in ['BIClong', 'BICshort'] if m in act.columns]].mean(axis=1).values
        tri = act[[m for m in ['TRIlong', 'TRIlat', 'TRImed'] if m in act.columns]].mean(axis=1).values
        cci = (2 * np.minimum(bic, tri) / (bic + tri + 1e-6)).mean()
        ccis.append(cci)
    ax.bar(range(len(FMA_SCORES)), ccis,
           color=[FMA_COLORS[f] for f in FMA_SCORES], edgecolor='white')
    ax.set_xticks(range(len(FMA_SCORES)))
    ax.set_xticklabels([FMA_SHORT[f] for f in FMA_SCORES], fontsize=9)
    ax.set_ylabel('Mean CCI')
    ax.set_title('H. Elbow Co-contraction Index — Joint Stability',
                 fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.2, axis='y')
    for i, v in enumerate(ccis):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=9)

    plt.suptitle('Inverse Dynamics Summary Dashboard — CVAE-Generated Drinking Motions\n'
                 'Key validation: biceps dominance, severity-graded torques, '
                 'physiologically plausible synergies',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, 'summary_dashboard.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
def main():
    print("Loading inverse dynamics data...")
    data = load_all_data()
    print(f"  Loaded {len(data)} FMA scores: {list(data.keys())}")

    print("\nGenerating visualizations...")
    plot_muscle_dominance(data)
    plot_activation_profiles(data)
    plot_phase_proportions(data)
    plot_torque_profiles(data)
    plot_synergy_timing(data)
    plot_cocontraction(data)
    plot_summary_dashboard(data)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == '__main__':
    main()
