"""
Publication Statistics: Aggregate ID results across multi-sample generated motions.

Computes:
  A. Descriptive statistics (mean +/- std per FMA level) for ATI, CCI, TRR, peak torques
  B. Spearman rank correlations (FMA vs each metric)
  C. Group comparisons (Healthy vs Impaired) with Mann-Whitney U + Cohen's d
  D. Healthy subject baseline from 77 original sessions

Output:
  - output/generated/plots/publication_stats.md
  - output/generated/plots/publication_figures.png

Usage:
  python scripts/publication_stats.py
"""
import os
import sys
import re
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.stats import kruskal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.utils.config import get_path

# --- Clinical group definitions (from fma_trend_analysis.py) ---
CLINICAL_GROUPS = [
    ('Severe',   16, 25),
    ('Moderate', 26, 40),
    ('Mild',     41, 55),
    ('Healthy',  56, 66),
]
GROUP_COLORS = {'Severe': '#D7263D', 'Moderate': '#F39237', 'Mild': '#F9C846', 'Healthy': '#2E86AB'}


def cohens_d(g1, g2):
    """Cohen's d effect size (pooled standard deviation)."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    pooled_std = np.sqrt(((n1 - 1) * np.std(g1, ddof=1)**2 + (n2 - 1) * np.std(g2, ddof=1)**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / pooled_std


def sig_code(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def load_effort_metrics(base_dir):
    """Load effort_metrics.json from all subdirectories, return DataFrame with fma_score column."""
    rows = []
    if not os.path.isdir(base_dir):
        return pd.DataFrame()

    for name in sorted(os.listdir(base_dir)):
        d = os.path.join(base_dir, name)
        metrics_path = os.path.join(d, 'effort_metrics.json')
        if not os.path.isfile(metrics_path):
            continue

        with open(metrics_path) as f:
            m = json.load(f)

        row = {'session': name, 'ATI': m.get('ATI'), 'CCI': m.get('CCI')}

        # Flatten TRR, per-joint torque, per-joint ROM
        for joint, val in m.get('TRR', {}).items():
            row[f'TRR_{joint}'] = val
        for joint, val in m.get('per_joint_mean_torque', {}).items():
            row[f'torque_{joint}'] = val
        for joint, val in m.get('per_joint_rom', {}).items():
            row[f'rom_{joint}'] = val

        rows.append(row)

    return pd.DataFrame(rows)


def parse_fma_score(session_name):
    """Extract FMA score from session directory name like FMA_50, FMA_18_s03."""
    m = re.match(r'FMA_(\d+)(?:_s\d+)?$', session_name)
    if m:
        return int(m.group(1))
    return None


def load_generated_metrics(gen_id_dir):
    """Load all generated session metrics with FMA scores."""
    df = load_effort_metrics(gen_id_dir)
    if df.empty:
        return df

    fma_scores = []
    for name in df['session']:
        fma = parse_fma_score(name)
        fma_scores.append(fma)
    df['fma_score'] = fma_scores
    df = df.dropna(subset=['fma_score'])
    df['fma_score'] = df['fma_score'].astype(int)
    return df


def _load_scores_csv():
    """Load actual FMA scores from output/scores.csv."""
    scores_path = get_path("scores_file")
    if not os.path.isfile(scores_path):
        return {}
    sdf = pd.read_csv(scores_path)
    return {row['filename'].replace('.mot', ''): int(row['fma_score'])
            for _, row in sdf.iterrows()}


def load_original_baseline(orig_id_dir):
    """Load effort metrics from all original sessions with correct FMA scores.

    Stroke patients get their actual FMA from scores.csv, healthy subjects get 66.
    """
    df = load_effort_metrics(orig_id_dir)
    if df.empty:
        return df

    scores = _load_scores_csv()
    fma_scores = []
    for name in df['session']:
        if name in scores:
            fma_scores.append(scores[name])
        elif name.startswith('S'):
            fma_scores.append(18)  # fallback for unlisted stroke patients
        else:
            fma_scores.append(66)  # healthy
    df['fma_score'] = fma_scores
    df['source'] = ['stroke' if name.startswith('S') else 'healthy' for name in df['session']]
    return df


def assign_clinical_group(fma):
    for name, lo, hi in CLINICAL_GROUPS:
        if lo <= fma <= hi:
            return name
    return 'Unknown'


# ============================================================
# DESCRIPTIVE STATS TABLE
# ============================================================
def descriptive_stats_table(df):
    """Per-FMA-level mean +/- std for key metrics."""
    key_metrics = ['ATI', 'CCI']
    trr_cols = [c for c in df.columns if c.startswith('TRR_')]
    torque_cols = [c for c in df.columns if c.startswith('torque_')]
    all_metrics = key_metrics + sorted(trr_cols) + sorted(torque_cols)
    # Filter to columns that exist
    all_metrics = [m for m in all_metrics if m in df.columns]

    lines = ['## A. Descriptive Statistics (mean +/- std per FMA level)\n']
    lines.append('| FMA | N | ' + ' | '.join(all_metrics) + ' |')
    lines.append('|-----|---|' + '|'.join(['---'] * len(all_metrics)) + '|')

    for fma in sorted(df['fma_score'].unique()):
        sub = df[df['fma_score'] == fma]
        n = len(sub)
        vals = []
        for m in all_metrics:
            mean = sub[m].mean()
            std = sub[m].std()
            if std > 0:
                vals.append(f'{mean:.3f} +/- {std:.3f}')
            else:
                vals.append(f'{mean:.3f}')
        lines.append(f'| {fma} | {n} | ' + ' | '.join(vals) + ' |')

    return '\n'.join(lines)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================
def correlation_analysis(df):
    """Spearman rank correlations: FMA vs each metric."""
    key_metrics = ['ATI', 'CCI']
    trr_cols = sorted([c for c in df.columns if c.startswith('TRR_')])
    torque_cols = sorted([c for c in df.columns if c.startswith('torque_')])
    all_metrics = key_metrics + trr_cols + torque_cols
    all_metrics = [m for m in all_metrics if m in df.columns]

    lines = ['## B. Correlation Analysis (Spearman rank)\n']
    lines.append('| Metric | rho | p-value | Sig |')
    lines.append('|--------|-----|---------|-----|')

    results = {}
    for m in all_metrics:
        valid = df[['fma_score', m]].dropna()
        if len(valid) < 4:
            continue
        rho, p = stats.spearmanr(valid['fma_score'], valid[m])
        results[m] = (rho, p)
        lines.append(f'| {m} | {rho:+.3f} | {p:.4f} | {sig_code(p)} |')

    return '\n'.join(lines), results


# ============================================================
# GROUP COMPARISONS
# ============================================================
def group_comparisons(df):
    """Healthy (FMA >= 56) vs Impaired (FMA <= 26): Mann-Whitney U + Cohen's d."""
    healthy_mask = df['fma_score'] >= 56
    impaired_mask = df['fma_score'] <= 26

    healthy = df[healthy_mask]
    impaired = df[impaired_mask]

    lines = [f'## C. Group Comparisons\n']
    lines.append(f'Healthy group: FMA >= 56 (N={len(healthy)})')
    lines.append(f'Impaired group: FMA <= 26 (N={len(impaired)})\n')
    lines.append('| Metric | Healthy (mean +/- std) | Impaired (mean +/- std) | U | p-value | Cohen d | Sig |')
    lines.append('|--------|------------------------|-------------------------|---|---------|---------|-----|')

    for m in ['ATI', 'CCI'] + sorted([c for c in df.columns if c.startswith('torque_')]):
        if m not in df.columns:
            continue
        h = healthy[m].dropna().values
        imp = impaired[m].dropna().values
        if len(h) < 2 or len(imp) < 2:
            continue

        u_stat, p_val = stats.mannwhitneyu(h, imp, alternative='two-sided')
        d = cohens_d(h, imp)

        lines.append(
            f'| {m} | {np.mean(h):.3f} +/- {np.std(h):.3f} '
            f'| {np.mean(imp):.3f} +/- {np.std(imp):.3f} '
            f'| {u_stat:.0f} | {p_val:.4f} | {d:+.2f} | {sig_code(p_val)} |'
        )

    return '\n'.join(lines)


# ============================================================
# HEALTHY BASELINE
# ============================================================
def healthy_baseline_section(healthy_df):
    """Compute reference stats from original recordings (healthy + stroke)."""
    if healthy_df.empty:
        return '## D. Original Subject Baseline\n\nNo original ID data found.\n'

    healthy_only = healthy_df[healthy_df['source'] == 'healthy'] if 'source' in healthy_df.columns else healthy_df
    stroke_only = healthy_df[healthy_df['source'] == 'stroke'] if 'source' in healthy_df.columns else pd.DataFrame()

    lines = ['## D. Original Subject Baseline\n']
    lines.append(f'Total: {len(healthy_df)} sessions (Healthy: {len(healthy_only)}, Stroke: {len(stroke_only)})\n')

    for label, sub in [('Healthy Originals', healthy_only), ('Stroke Originals', stroke_only)]:
        if sub.empty:
            continue
        lines.append(f'\n### {label} (N={len(sub)})\n')
        lines.append('| Metric | Mean | Std | Min | Max |')
        lines.append('|--------|------|-----|-----|-----|')

        for m in ['ATI', 'CCI'] + sorted([c for c in sub.columns if c.startswith('torque_')]):
            if m not in sub.columns:
                continue
            vals = sub[m].dropna()
            if len(vals) == 0:
                continue
            lines.append(f'| {m} | {vals.mean():.3f} | {vals.std():.3f} | {vals.min():.3f} | {vals.max():.3f} |')

    return '\n'.join(lines)


# ============================================================
# E. BOOTSTRAP 95% CIs
# ============================================================
def bootstrap_confidence_intervals(df, n_bootstrap=10000):
    """Bootstrap 95% CIs for ATI and CCI per FMA level."""
    lines = ['## E. Bootstrap 95% Confidence Intervals (10,000 resamples)\n']
    lines.append('| FMA | N | Metric | Mean | 95% CI Lower | 95% CI Upper | CI Width |')
    lines.append('|-----|---|--------|------|-------------|-------------|----------|')

    rng = np.random.default_rng(seed=42)

    for fma in sorted(df['fma_score'].unique()):
        sub = df[df['fma_score'] == fma]
        n = len(sub)

        for metric in ['ATI', 'CCI']:
            vals = sub[metric].dropna().values
            if len(vals) < 2:
                continue

            # Bootstrap resampling
            boot_means = np.zeros(n_bootstrap)
            for i in range(n_bootstrap):
                sample = rng.choice(vals, size=len(vals), replace=True)
                boot_means[i] = np.mean(sample)

            ci_lo = np.percentile(boot_means, 2.5)
            ci_hi = np.percentile(boot_means, 97.5)
            ci_width = ci_hi - ci_lo

            lines.append(
                f'| {fma} | {n} | {metric} | {np.mean(vals):.4f} '
                f'| {ci_lo:.4f} | {ci_hi:.4f} | {ci_width:.4f} |'
            )

    return '\n'.join(lines)


# ============================================================
# F. KRUSKAL-WALLIS TEST
# ============================================================
def kruskal_wallis_test(df):
    """Omnibus non-parametric ANOVA across all FMA levels."""
    lines = ['## F. Kruskal-Wallis Test (Omnibus Non-parametric ANOVA)\n']
    lines.append('| Metric | H-statistic | p-value | Sig | N groups | Total N |')
    lines.append('|--------|-------------|---------|-----|----------|---------|')

    results = {}
    fma_levels = sorted(df['fma_score'].unique())

    for metric in ['ATI', 'CCI']:
        groups = []
        for fma in fma_levels:
            vals = df.loc[df['fma_score'] == fma, metric].dropna().values
            if len(vals) >= 2:
                groups.append(vals)

        if len(groups) < 3:
            lines.append(f'| {metric} | - | - | - | {len(groups)} | - |')
            continue

        total_n = sum(len(g) for g in groups)
        h_stat, p_val = kruskal(*groups)
        results[metric] = (h_stat, p_val)

        lines.append(
            f'| {metric} | {h_stat:.3f} | {p_val:.6f} | {sig_code(p_val)} '
            f'| {len(groups)} | {total_n} |'
        )

    return '\n'.join(lines), results


# ============================================================
# G. POST-HOC DUNN'S TEST (Mann-Whitney U with Bonferroni)
# ============================================================
def posthoc_dunns_test(df):
    """Pairwise Mann-Whitney U with Bonferroni correction + Cohen's d."""
    lines = ["## G. Post-hoc Pairwise Tests (Mann-Whitney U + Bonferroni)\n"]

    fma_levels = sorted(df['fma_score'].unique())
    n_pairs = len(fma_levels) * (len(fma_levels) - 1) // 2

    effect_sizes = {}  # for heatmap

    for metric in ['ATI', 'CCI']:
        lines.append(f'\n### {metric}\n')
        lines.append('| FMA A | FMA B | U stat | p (raw) | p (corrected) | Sig | Cohen d |')
        lines.append('|-------|-------|--------|---------|---------------|-----|---------|')

        effect_matrix = {}

        for fma_a, fma_b in itertools.combinations(fma_levels, 2):
            vals_a = df.loc[df['fma_score'] == fma_a, metric].dropna().values
            vals_b = df.loc[df['fma_score'] == fma_b, metric].dropna().values

            if len(vals_a) < 2 or len(vals_b) < 2:
                continue

            u_stat, p_raw = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
            p_corrected = min(p_raw * n_pairs, 1.0)  # Bonferroni
            d = cohens_d(vals_a, vals_b)

            effect_matrix[(fma_a, fma_b)] = d
            effect_matrix[(fma_b, fma_a)] = -d

            lines.append(
                f'| {fma_a} | {fma_b} | {u_stat:.0f} '
                f'| {p_raw:.6f} | {p_corrected:.6f} | {sig_code(p_corrected)} '
                f'| {d:+.3f} |'
            )

        effect_sizes[metric] = effect_matrix

    return '\n'.join(lines), effect_sizes


# ============================================================
# H. LINEAR MIXED-EFFECTS MODEL
# ============================================================
def linear_mixed_effects(df):
    """Fit metric ~ fma_score + (1|sample_id) using statsmodels if available."""
    lines = ['## H. Linear Mixed-Effects Model\n']
    lines.append('Model: `metric ~ fma_score + (1|sample_id)`\n')

    try:
        import statsmodels.formula.api as smf
        has_statsmodels = True
    except ImportError:
        has_statsmodels = False
        lines.append('*statsmodels not available — skipping LME analysis.*\n')
        return '\n'.join(lines)

    # Create sample_id from session names (FMA_X_sNN -> sNN group)
    df = df.copy()
    df['sample_id'] = df['session'].str.extract(r'_s(\d+)$').fillna('s00')

    lines.append('| Metric | Fixed Effect (FMA) | Coeff | Std Err | z | p-value | Sig |')
    lines.append('|--------|-------------------|-------|---------|---|---------|-----|')

    for metric in ['ATI', 'CCI']:
        valid = df[['fma_score', 'sample_id', metric]].dropna()
        if len(valid) < 10 or valid['sample_id'].nunique() < 2:
            lines.append(f'| {metric} | - | - | - | - | - | insufficient data |')
            continue

        try:
            model = smf.mixedlm(f'{metric} ~ fma_score', valid, groups=valid['sample_id'])
            result = model.fit(reml=True)

            coeff = result.params['fma_score']
            stderr = result.bse['fma_score']
            z_val = result.tvalues['fma_score']
            p_val = result.pvalues['fma_score']

            lines.append(
                f'| {metric} | fma_score | {coeff:.6f} | {stderr:.6f} '
                f'| {z_val:.3f} | {p_val:.6f} | {sig_code(p_val)} |'
            )
        except Exception as e:
            lines.append(f'| {metric} | error | - | - | - | - | {str(e)[:50]} |')

    return '\n'.join(lines)


# ============================================================
# EFFECT SIZE HEATMAP
# ============================================================
def plot_effect_size_heatmap(effect_sizes, fma_levels, output_dir):
    """Pairwise Cohen's d matrix heatmap for ATI and CCI."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Pairwise Effect Sizes (Cohen's d)", fontsize=14, fontweight='bold')

    for ax, metric in zip(axes, ['ATI', 'CCI']):
        matrix = effect_sizes.get(metric, {})
        n = len(fma_levels)
        d_matrix = np.zeros((n, n))

        for i, fma_a in enumerate(fma_levels):
            for j, fma_b in enumerate(fma_levels):
                if i == j:
                    d_matrix[i, j] = 0
                else:
                    d_matrix[i, j] = matrix.get((fma_a, fma_b), 0)

        im = ax.imshow(d_matrix, cmap='RdBu_r', vmin=-3, vmax=3, aspect='equal')

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        tick_labels = [str(f) for f in fma_levels]
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=45)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel('FMA Score')
        ax.set_ylabel('FMA Score')
        ax.set_title(f'{metric} Cohen\'s d', fontweight='bold')

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = d_matrix[i, j]
                if abs(val) > 0.01:
                    color = 'white' if abs(val) > 1.5 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                            fontsize=6, color=color)

        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, 'effect_size_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Effect size heatmap saved to: {path}')


# ============================================================
# PUBLICATION FIGURES
# ============================================================
def create_publication_figures(gen_df, healthy_df, corr_results, output_dir):
    """Create multi-panel publication figure."""
    fig = plt.figure(figsize=(14, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3, top=0.93, bottom=0.08, left=0.08, right=0.95)
    fig.suptitle('Publication Figures: Inverse Dynamics by FMA Score', fontsize=14, fontweight='bold')

    # --- Panel A: CCI vs FMA score (scatter + regression + 95% CI) ---
    ax_a = fig.add_subplot(gs[0, 0])
    _scatter_with_ci(ax_a, gen_df, 'CCI', corr_results, 'A. CCI vs FMA Score', '#2E86AB')

    # --- Panel B: ATI vs FMA score ---
    ax_b = fig.add_subplot(gs[0, 1])
    _scatter_with_ci(ax_b, gen_df, 'ATI', corr_results, 'B. ATI vs FMA Score', '#E94F37')

    # --- Panel C: Boxplot of CCI by clinical group ---
    ax_c = fig.add_subplot(gs[1, 0])
    _boxplot_by_group(ax_c, gen_df, 'CCI', 'C. CCI by Clinical Group')

    # --- Panel D: Healthy baseline comparison (generated vs originals) ---
    ax_d = fig.add_subplot(gs[1, 1])
    _baseline_comparison(ax_d, gen_df, healthy_df, 'D. Generated vs Original Healthy')

    out_path = os.path.join(output_dir, 'publication_figures.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Figures saved to: {out_path}')


def _scatter_with_ci(ax, df, metric, corr_results, title, color):
    """Scatter + linear regression + 95% CI band."""
    valid = df[['fma_score', metric]].dropna()
    x, y = valid['fma_score'].values, valid[metric].values

    ax.scatter(x, y, alpha=0.5, s=30, color=color, edgecolors='white', linewidth=0.5)

    # Regression line + CI
    if len(x) >= 4:
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = slope * x_line + intercept

        # 95% CI for regression line
        n = len(x)
        x_mean = np.mean(x)
        se = np.sqrt(np.sum((y - (slope * x + intercept))**2) / (n - 2))
        ci = stats.t.ppf(0.975, n - 2) * se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))

        ax.plot(x_line, y_line, '-', color='black', lw=1.5)
        ax.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.15, color=color)

        # Annotation with Spearman from corr_results if available
        if metric in corr_results:
            rho, p = corr_results[metric]
        else:
            rho, p = stats.spearmanr(x, y)
        ax.text(0.05, 0.95, f'rho={rho:.3f}, p={p:.4f} {sig_code(p)}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('FMA Score')
    ax.set_ylabel(metric)
    ax.set_title(title, fontweight='bold')


def _boxplot_by_group(ax, df, metric, title):
    """Boxplot of a metric by clinical group."""
    df = df.copy()
    df['group'] = df['fma_score'].apply(assign_clinical_group)
    group_order = [g[0] for g in CLINICAL_GROUPS]
    group_data = []
    group_labels = []
    colors = []
    for g in group_order:
        sub = df[df['group'] == g][metric].dropna().values
        if len(sub) > 0:
            group_data.append(sub)
            group_labels.append(g)
            colors.append(GROUP_COLORS[g])

    if not group_data:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        return

    bp = ax.boxplot(group_data, patch_artist=True, widths=0.6)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)

    ax.set_xticklabels(group_labels, fontsize=9)
    ax.set_ylabel(metric)
    ax.set_title(title, fontweight='bold')


def _baseline_comparison(ax, gen_df, healthy_df, title):
    """Bar chart comparing generated healthy vs original healthy for ATI and CCI."""
    metrics = ['ATI', 'CCI']
    gen_healthy = gen_df[gen_df['fma_score'] >= 56]

    x = np.arange(len(metrics))
    width = 0.35

    # Generated healthy
    gen_means = [gen_healthy[m].mean() if m in gen_healthy.columns else 0 for m in metrics]
    gen_stds = [gen_healthy[m].std() if m in gen_healthy.columns else 0 for m in metrics]

    # Original healthy
    orig_means = [healthy_df[m].mean() if m in healthy_df.columns else 0 for m in metrics]
    orig_stds = [healthy_df[m].std() if m in healthy_df.columns else 0 for m in metrics]

    bars1 = ax.bar(x - width/2, gen_means, width, yerr=gen_stds, label='Generated (FMA 56-66)',
                   color='#2E86AB', alpha=0.7, capsize=5)
    bars2 = ax.bar(x + width/2, orig_means, width, yerr=orig_stds, label='Originals (N=77)',
                   color='#F39237', alpha=0.7, capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Value')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8)


# ============================================================
# MAIN
# ============================================================
def main():
    gen_id_dir = os.path.join(get_path("output_generated"), "id")
    orig_id_dir = get_path("output_originals_id")
    output_dir = get_path("output_generated_plots")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Publication Statistics")
    print("=" * 60)

    # Load generated metrics
    print("\nLoading generated ID results...")
    gen_df = load_generated_metrics(gen_id_dir)
    if gen_df.empty:
        print(f"ERROR: No generated ID data found in {gen_id_dir}")
        print("Run the pipeline first: python scripts/run_generated_pipeline.py")
        return
    print(f"  Loaded {len(gen_df)} sessions across {gen_df['fma_score'].nunique()} FMA levels")

    # Load healthy baseline
    print("Loading healthy baseline...")
    healthy_df = load_original_baseline(orig_id_dir)
    print(f"  Loaded {len(healthy_df)} original sessions")

    # --- Compute all statistics ---

    # A. Descriptive stats
    desc_table = descriptive_stats_table(gen_df)

    # B. Correlations
    corr_text, corr_results = correlation_analysis(gen_df)

    # C. Group comparisons
    group_text = group_comparisons(gen_df)

    # D. Healthy baseline
    baseline_text = healthy_baseline_section(healthy_df)

    # E. Bootstrap 95% CIs
    print("Computing bootstrap CIs (10,000 resamples)...")
    bootstrap_text = bootstrap_confidence_intervals(gen_df)

    # F. Kruskal-Wallis
    print("Running Kruskal-Wallis tests...")
    kw_text, kw_results = kruskal_wallis_test(gen_df)

    # G. Post-hoc Dunn's test
    print("Running post-hoc pairwise tests...")
    dunn_text, effect_sizes = posthoc_dunns_test(gen_df)

    # H. Linear mixed-effects model
    print("Fitting linear mixed-effects models...")
    lme_text = linear_mixed_effects(gen_df)

    # --- Write markdown report ---
    report_lines = [
        '# Publication Statistics Report\n',
        f'Generated sessions: {len(gen_df)} | FMA levels: {sorted(gen_df["fma_score"].unique().tolist())}',
        f'Original healthy sessions: {len(healthy_df)}\n',
        '---\n',
        desc_table, '\n---\n',
        corr_text, '\n---\n',
        group_text, '\n---\n',
        baseline_text, '\n---\n',
        bootstrap_text, '\n---\n',
        kw_text, '\n---\n',
        dunn_text, '\n---\n',
        lme_text, '\n---\n',
        '### Significance codes: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant',
    ]
    report_path = os.path.join(output_dir, 'publication_stats.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"\nReport saved to: {report_path}")

    # --- Create figures ---
    print("Creating publication figures...")
    create_publication_figures(gen_df, healthy_df, corr_results, output_dir)

    # --- Effect size heatmap ---
    fma_levels = sorted(gen_df['fma_score'].unique())
    if effect_sizes and len(fma_levels) >= 3:
        print("Creating effect size heatmap...")
        plot_effect_size_heatmap(effect_sizes, fma_levels, output_dir)

    # --- Print key results to console ---
    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)
    for m in ['CCI', 'ATI']:
        if m in corr_results:
            rho, p = corr_results[m]
            print(f"  {m} vs FMA: rho={rho:+.3f}, p={p:.4f} {sig_code(p)}")

    healthy_mask = gen_df['fma_score'] >= 50
    impaired_mask = gen_df['fma_score'] <= 26
    if healthy_mask.any() and impaired_mask.any():
        h_cci = gen_df.loc[healthy_mask, 'CCI'].dropna().values
        i_cci = gen_df.loc[impaired_mask, 'CCI'].dropna().values
        if len(h_cci) >= 2 and len(i_cci) >= 2:
            d = cohens_d(i_cci, h_cci)
            print(f"  CCI Cohen's d (impaired vs healthy): {d:+.2f}")


if __name__ == '__main__':
    main()
