"""
Literature Validation: Automated pass/fail checks against published stroke biomechanics values.

Loads all effort_metrics.json files (generated + originals) and checks:
  - Torque magnitudes vs Dewald & Beer 2001 ranges
  - CCI ranges per severity group
  - CCI monotonicity (Spearman)
  - ATI healthy baseline ratio (gen/orig)
  - ATI monotonicity
  - Phase-specific muscle dominance patterns
  - Synergy VAF ranges

Output:
  - output/generated/plots/literature_validation.md
  - output/generated/plots/literature_validation.png

Usage:
  python scripts/literature_validation.py
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
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.utils.config import get_path

# =============================================================================
# LITERATURE REFERENCE VALUES
# =============================================================================

# Dewald & Beer 2001: torque ranges (Nm)
TORQUE_RANGES = {
    'shoulder_elv':   (1.0, 12.0),
    'elbow_flexion':  (1.0, 8.0),
    'elv_angle':      (1.0, 15.0),
    'shoulder_rot':   (0.5, 8.0),
    'pro_sup':        (0.1, 3.0),
}

# CCI ranges per clinical group (from published meta-analyses)
CCI_RANGES = {
    'Severe':   (0.45, 0.85),
    'Moderate': (0.35, 0.70),
    'Mild':     (0.25, 0.55),
    'Healthy':  (0.15, 0.55),
}

# Synergy VAF expected ranges
SYNERGY_VAF_RANGES = {
    'healthy':  (0.85, 0.99),
    'impaired': (0.90, 1.00),
}

# Phase-specific expected dominant muscles
PHASE_DOMINANT_MUSCLES = {
    'Pick':  ['DELT1', 'DELT2', 'BIClong', 'BICshort', 'BRA', 'SUPSP', 'INFSP'],
    'Drink': ['BIClong', 'BICshort', 'BRA', 'DELT1', 'SUP', 'SUPSP'],
    'Place': ['TRIlong', 'TRIlat', 'TRImed', 'DELT3', 'ANC', 'DELT2', 'LAT1'],
}

# Clinical group definitions
CLINICAL_GROUPS = [
    ('Severe',   0, 25),
    ('Moderate', 26, 40),
    ('Mild',     41, 55),
    ('Healthy',  56, 66),
]


def assign_group(fma):
    for name, lo, hi in CLINICAL_GROUPS:
        if lo <= fma <= hi:
            return name
    return 'Unknown'


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_scores_map():
    """Load actual FMA scores from output/scores.csv."""
    scores_path = get_path("scores_file")
    if not os.path.isfile(scores_path):
        return {}
    import pandas as pd
    sdf = pd.read_csv(scores_path)
    return {row['filename'].replace('.mot', ''): int(row['fma_score'])
            for _, row in sdf.iterrows()}


def load_all_metrics(gen_id_dir, orig_id_dir):
    """Load effort_metrics.json from all sessions."""
    rows = []
    scores_map = _load_scores_map()

    # Generated sessions
    if os.path.isdir(gen_id_dir):
        for name in sorted(os.listdir(gen_id_dir)):
            path = os.path.join(gen_id_dir, name, 'effort_metrics.json')
            if not os.path.isfile(path):
                continue
            m = re.match(r'FMA_(\d+)(?:_s\d+)?$', name)
            if m:
                fma = int(m.group(1))
            elif name in scores_map:
                fma = scores_map[name]
            elif name.startswith('S'):
                fma = 18
            else:
                fma = 66
            with open(path) as f:
                data = json.load(f)
            data['session'] = name
            data['fma_score'] = fma
            data['source'] = 'generated'
            rows.append(data)

    # Original sessions
    if os.path.isdir(orig_id_dir):
        for name in sorted(os.listdir(orig_id_dir)):
            path = os.path.join(orig_id_dir, name, 'effort_metrics.json')
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                data = json.load(f)
            data['session'] = name
            if name in scores_map:
                data['fma_score'] = scores_map[name]
            elif name.startswith('S'):
                data['fma_score'] = 18  # fallback for unlisted stroke
            else:
                data['fma_score'] = 66  # healthy
            data['source'] = 'original'
            rows.append(data)

    return rows


def load_synergy_vafs(gen_id_dir):
    """Load synergy VAF from synergy extraction (recomputed from weights/coefficients)."""
    vafs = {}
    if not os.path.isdir(gen_id_dir):
        return vafs
    scores_map = _load_scores_map()

    for name in sorted(os.listdir(gen_id_dir)):
        d = os.path.join(gen_id_dir, name)
        act_path = os.path.join(d, 'activations.csv')
        sw_path = os.path.join(d, 'synergy_weights.csv')
        sc_path = os.path.join(d, 'synergy_coefficients.csv')

        if not all(os.path.isfile(p) for p in [act_path, sw_path, sc_path]):
            continue

        m = re.match(r'FMA_(\d+)(?:_s\d+)?$', name)
        if m:
            fma = int(m.group(1))
        elif name in scores_map:
            fma = scores_map[name]
        elif name.startswith('S'):
            fma = 18
        else:
            fma = 66

        act_df = pd.read_csv(act_path)
        act_cols = [c for c in act_df.columns if c != 'time']
        act = act_df[act_cols].values

        sw = pd.read_csv(sw_path, index_col=0).values
        sc = pd.read_csv(sc_path)
        sc_cols = [c for c in sc.columns if c != 'time']
        coeffs = sc[sc_cols].values

        recon = coeffs @ sw
        ss_total = np.sum(act ** 2)
        ss_resid = np.sum((act - recon) ** 2)
        vaf = 1 - (ss_resid / ss_total) if ss_total > 0 else 0.0

        if fma not in vafs:
            vafs[fma] = []
        vafs[fma].append(vaf)

    return vafs


def load_phase_activations(gen_id_dir):
    """Load activations with phase labels for muscle dominance analysis."""
    results = {}
    if not os.path.isdir(gen_id_dir):
        return results
    scores_map = _load_scores_map()

    for name in sorted(os.listdir(gen_id_dir)):
        d = os.path.join(gen_id_dir, name)
        act_path = os.path.join(d, 'activations.csv')
        phase_path = os.path.join(d, 'phase_labels.csv')

        if not all(os.path.isfile(p) for p in [act_path, phase_path]):
            continue

        m = re.match(r'FMA_(\d+)(?:_s\d+)?$', name)
        if m:
            fma = int(m.group(1))
        elif name in scores_map:
            fma = scores_map[name]
        elif name.startswith('S'):
            fma = 18
        else:
            fma = 66

        act_df = pd.read_csv(act_path)
        phase_df = pd.read_csv(phase_path)
        muscle_cols = [c for c in act_df.columns if c != 'time']

        for phase in ['Pick', 'Drink', 'Place']:
            mask = phase_df['phase'] == phase
            if mask.sum() == 0:
                continue
            phase_act = act_df.loc[mask, muscle_cols]
            mean_act = phase_act.mean().sort_values(ascending=False)
            top5 = mean_act.head(5).index.tolist()

            key = (fma, phase)
            if key not in results:
                results[key] = []
            results[key].append(top5)

    return results


# =============================================================================
# VALIDATION CHECKS
# =============================================================================

class ValidationResult:
    def __init__(self, name, passed, detail, category):
        self.name = name
        self.passed = passed
        self.detail = detail
        self.category = category


def check_torque_ranges(metrics_list):
    """Check that mean torques fall within published ranges."""
    results = []
    gen_metrics = [m for m in metrics_list if m['source'] == 'generated']
    if not gen_metrics:
        return results

    for joint, (lo, hi) in TORQUE_RANGES.items():
        torques = []
        for m in gen_metrics:
            t = m.get('per_joint_mean_torque', {}).get(joint, None)
            if t is not None:
                torques.append(t)

        if not torques:
            continue

        mean_t = np.mean(torques)
        passed = lo <= mean_t <= hi
        detail = f"{joint}: mean={mean_t:.2f} Nm, range=[{lo}, {hi}]"
        results.append(ValidationResult(
            f"Torque range: {joint}", passed, detail, "Torques"))

    return results


def check_cci_ranges(metrics_list):
    """Check CCI per clinical group against published ranges."""
    results = []
    gen_metrics = [m for m in metrics_list if m['source'] == 'generated']

    for group_name, lo_fma, hi_fma in CLINICAL_GROUPS:
        group = [m for m in gen_metrics if lo_fma <= m['fma_score'] <= hi_fma]
        if not group:
            continue

        ccis = [m['CCI'] for m in group if 'CCI' in m]
        if not ccis:
            continue

        mean_cci = np.mean(ccis)
        expected_lo, expected_hi = CCI_RANGES[group_name]
        passed = expected_lo <= mean_cci <= expected_hi
        detail = f"{group_name} (N={len(ccis)}): CCI={mean_cci:.3f}, expected=[{expected_lo}, {expected_hi}]"
        results.append(ValidationResult(
            f"CCI range: {group_name}", passed, detail, "CCI"))

    return results


def check_cci_monotonicity(metrics_list):
    """Check CCI decreases with FMA (Spearman rho < -0.5)."""
    gen = [m for m in metrics_list if m['source'] == 'generated']
    if len(gen) < 6:
        return [ValidationResult("CCI monotonicity", False,
                                 "Too few samples", "CCI")]

    fma_scores = [m['fma_score'] for m in gen]
    ccis = [m['CCI'] for m in gen]
    rho, p = stats.spearmanr(fma_scores, ccis)

    passed = rho < -0.5
    detail = f"Spearman rho={rho:.3f}, p={p:.4f} (threshold: rho < -0.5)"
    return [ValidationResult("CCI monotonicity", passed, detail, "CCI")]


def check_ati_baseline(metrics_list):
    """Check ATI for generated healthy vs original healthy (ratio < 2.0)."""
    gen_healthy = [m for m in metrics_list
                   if m['source'] == 'generated' and m['fma_score'] >= 56]
    orig_healthy = [m for m in metrics_list if m['source'] == 'original']

    if not gen_healthy or not orig_healthy:
        return [ValidationResult("ATI baseline ratio", False,
                                 "Missing data", "ATI")]

    gen_ati = np.mean([m['ATI'] for m in gen_healthy])
    orig_ati = np.mean([m['ATI'] for m in orig_healthy])

    if orig_ati > 0:
        ratio = gen_ati / orig_ati
    else:
        ratio = float('inf')

    passed = ratio < 2.0
    detail = f"Gen healthy ATI={gen_ati:.3f}, Orig ATI={orig_ati:.3f}, ratio={ratio:.2f} (threshold: < 2.0)"
    return [ValidationResult("ATI baseline ratio", passed, detail, "ATI")]


def check_ati_monotonicity(metrics_list):
    """Check ATI decreases with FMA (more impaired = more effort)."""
    gen = [m for m in metrics_list if m['source'] == 'generated']
    if len(gen) < 6:
        return [ValidationResult("ATI monotonicity", False,
                                 "Too few samples", "ATI")]

    fma_scores = [m['fma_score'] for m in gen]
    atis = [m['ATI'] for m in gen]
    rho, p = stats.spearmanr(fma_scores, atis)

    # ATI should decrease with FMA (healthy uses less effort)
    passed = rho < -0.3
    detail = f"Spearman rho={rho:.3f}, p={p:.4f} (threshold: rho < -0.3)"
    return [ValidationResult("ATI monotonicity", passed, detail, "ATI")]


def check_synergy_vaf(vafs):
    """Check synergy VAF ranges (healthy ~0.90, impaired ~0.95+)."""
    results = []

    healthy_vafs = []
    impaired_vafs = []
    for fma, vaf_list in vafs.items():
        if fma >= 56:
            healthy_vafs.extend(vaf_list)
        elif fma <= 30:
            impaired_vafs.extend(vaf_list)

    if healthy_vafs:
        mean_h = np.mean(healthy_vafs)
        lo, hi = SYNERGY_VAF_RANGES['healthy']
        passed = lo <= mean_h <= hi
        detail = f"Healthy VAF={mean_h:.3f}, expected=[{lo}, {hi}]"
        results.append(ValidationResult("Synergy VAF: Healthy", passed, detail, "Synergies"))

    if impaired_vafs:
        mean_i = np.mean(impaired_vafs)
        lo, hi = SYNERGY_VAF_RANGES['impaired']
        passed = lo <= mean_i <= hi
        detail = f"Impaired VAF={mean_i:.3f}, expected=[{lo}, {hi}]"
        results.append(ValidationResult("Synergy VAF: Impaired", passed, detail, "Synergies"))

    # Check impaired > healthy (stereotyped patterns = higher VAF)
    if healthy_vafs and impaired_vafs:
        mean_h = np.mean(healthy_vafs)
        mean_i = np.mean(impaired_vafs)
        passed = mean_i >= mean_h
        detail = f"Impaired VAF ({mean_i:.3f}) >= Healthy VAF ({mean_h:.3f})"
        results.append(ValidationResult("Synergy VAF: impaired >= healthy", passed, detail, "Synergies"))

    return results


def check_phase_dominance(phase_activations):
    """Check that expected muscles appear in top-5 per phase."""
    results = []

    for phase in ['Pick', 'Drink', 'Place']:
        expected = set(PHASE_DOMINANT_MUSCLES[phase])
        all_top5 = []

        for (fma, p), top5_lists in phase_activations.items():
            if p != phase:
                continue
            for top5 in top5_lists:
                all_top5.extend(top5)

        if not all_top5:
            results.append(ValidationResult(
                f"Phase dominance: {phase}", False,
                "No phase activation data", "Phase Dominance"))
            continue

        # Count frequency of each muscle in top-5 across all sessions
        from collections import Counter
        counts = Counter(all_top5)
        top_muscles = set(m for m, _ in counts.most_common(7))

        overlap = expected & top_muscles
        overlap_pct = len(overlap) / len(expected) * 100 if expected else 0
        passed = overlap_pct >= 40  # At least 40% of expected muscles appear

        detail = (f"{phase}: overlap={len(overlap)}/{len(expected)} ({overlap_pct:.0f}%), "
                  f"expected={sorted(expected)[:5]}, found={sorted(top_muscles)[:5]}")
        results.append(ValidationResult(
            f"Phase dominance: {phase}", passed, detail, "Phase Dominance"))

    return results


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(all_results, output_dir):
    """Generate markdown validation report."""
    lines = ['# Literature Validation Report\n']

    # Summary
    n_pass = sum(1 for r in all_results if r.passed)
    n_total = len(all_results)
    pct = n_pass / n_total * 100 if n_total > 0 else 0
    lines.append(f'**Overall: {n_pass}/{n_total} checks passed ({pct:.0f}%)**\n')

    # Group by category
    categories = {}
    for r in all_results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    for cat, results in categories.items():
        n_cat_pass = sum(1 for r in results if r.passed)
        lines.append(f'\n## {cat} ({n_cat_pass}/{len(results)} passed)\n')
        lines.append('| Check | Status | Detail |')
        lines.append('|-------|--------|--------|')
        for r in results:
            status = 'PASS' if r.passed else '**FAIL**'
            lines.append(f'| {r.name} | {status} | {r.detail} |')

    lines.append('\n---')
    lines.append('*Generated by scripts/literature_validation.py*')

    report_path = os.path.join(output_dir, 'literature_validation.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Report saved to: {report_path}')
    return report_path


def generate_summary_figure(all_results, output_dir):
    """Create summary pass/fail visualization."""
    categories = {}
    for r in all_results:
        if r.category not in categories:
            categories[r.category] = {'pass': 0, 'fail': 0}
        if r.passed:
            categories[r.category]['pass'] += 1
        else:
            categories[r.category]['fail'] += 1

    cat_names = list(categories.keys())
    passes = [categories[c]['pass'] for c in cat_names]
    fails = [categories[c]['fail'] for c in cat_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Stacked bar by category
    ax = axes[0]
    x = np.arange(len(cat_names))
    ax.barh(x, passes, color='#2ecc71', label='Pass', height=0.6)
    ax.barh(x, fails, left=passes, color='#e74c3c', label='Fail', height=0.6)
    ax.set_yticks(x)
    ax.set_yticklabels(cat_names, fontsize=11)
    ax.set_xlabel('Number of Checks')
    ax.set_title('Validation Results by Category', fontweight='bold')
    ax.legend(loc='lower right')

    for i, (p, f_) in enumerate(zip(passes, fails)):
        total = p + f_
        pct = p / total * 100 if total > 0 else 0
        ax.text(p + f_ + 0.1, i, f'{pct:.0f}%', va='center', fontsize=10)

    # Panel 2: Overall pie chart
    ax2 = axes[1]
    total_pass = sum(passes)
    total_fail = sum(fails)
    if total_pass + total_fail > 0:
        ax2.pie([total_pass, total_fail],
                labels=[f'Pass ({total_pass})', f'Fail ({total_fail})'],
                colors=['#2ecc71', '#e74c3c'],
                autopct='%1.0f%%', startangle=90, textprops={'fontsize': 12})
    ax2.set_title('Overall Validation', fontweight='bold', fontsize=13)

    plt.tight_layout()
    path = os.path.join(output_dir, 'literature_validation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Figure saved to: {path}')


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-id-dir", default=None,
                        help="Path to generated ID directory (overrides config default)")
    parser.add_argument("--output-dir", default=None,
                        help="Path to write report/figures (overrides config default)")
    args = parser.parse_args()

    gen_id_dir = args.gen_id_dir or os.path.join(get_path("output_generated"), "id")
    orig_id_dir = get_path("output_originals_id")
    output_dir = args.output_dir or get_path("output_generated_plots")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Literature Validation")
    print("=" * 60)

    # Load data
    print("\nLoading effort metrics...")
    metrics_list = load_all_metrics(gen_id_dir, orig_id_dir)
    n_gen = sum(1 for m in metrics_list if m['source'] == 'generated')
    n_orig = sum(1 for m in metrics_list if m['source'] == 'original')
    print(f"  Generated: {n_gen} sessions, Originals: {n_orig} sessions")

    print("Loading synergy VAFs...")
    vafs = load_synergy_vafs(gen_id_dir)
    print(f"  VAFs for {len(vafs)} FMA levels")

    print("Loading phase activations...")
    phase_act = load_phase_activations(gen_id_dir)
    print(f"  Phase data for {len(phase_act)} (fma, phase) combinations")

    # Run all checks
    print("\nRunning validation checks...")
    all_results = []
    all_results.extend(check_torque_ranges(metrics_list))
    all_results.extend(check_cci_ranges(metrics_list))
    all_results.extend(check_cci_monotonicity(metrics_list))
    all_results.extend(check_ati_baseline(metrics_list))
    all_results.extend(check_ati_monotonicity(metrics_list))
    all_results.extend(check_synergy_vaf(vafs))
    all_results.extend(check_phase_dominance(phase_act))

    # Print summary to console
    n_pass = sum(1 for r in all_results if r.passed)
    n_total = len(all_results)
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {n_pass}/{n_total} checks passed ({n_pass/n_total*100:.0f}%)")
    print(f"{'=' * 60}")

    for r in all_results:
        status = "PASS" if r.passed else "FAIL"
        marker = "  " if r.passed else ">>"
        print(f"  {marker} [{status}] {r.name}: {r.detail}")

    # Generate outputs
    print("\nGenerating report...")
    generate_report(all_results, output_dir)
    generate_summary_figure(all_results, output_dir)

    return n_pass, n_total


if __name__ == '__main__':
    main()
