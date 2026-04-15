"""
EMG Validation Script: Compare computed muscle activations against published EMG data.

References:
  - Alt Murphy et al. 2011: Kinematic analysis of the upper extremity in the drinking task
  - Roh et al. 2013: Alterations in upper limb muscle synergy structure in chronic stroke survivors

Checks:
  1. Dominant muscles match literature (DELT, BIC, TRI for drinking task)
  2. Phase-specific peaks (DELT/BIC peak during reach, TRI during place)
  3. Co-contraction index falls in published range
  4. Synergy count: 4 synergies explain >90% VAF for healthy
  5. Fewer distinct synergies for impaired (synergy merging)

Usage:
  python scripts/viz/validate_emg.py [id_base_dir]
"""
import os
import sys
import re
import json
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from src.utils.config import get_path

# Published EMG reference data for the drinking task
# Muscles expected to be dominant during each phase
LITERATURE = {
    'reach_dominant': ['DELT1', 'DELT2', 'BIClong', 'BICshort', 'SUPSP'],
    'drink_dominant': ['BIClong', 'BICshort', 'BRA', 'DELT1'],
    'place_dominant': ['TRIlong', 'TRIlat', 'DELT2', 'DELT3'],
    'cci_healthy_range': (0.15, 0.55),
    'cci_impaired_range': (0.35, 0.85),
    'healthy_vaf_4syn': 0.90,
    'top_k_muscles': 5,  # how many muscles to check as "dominant"
}


def load_session(session_dir):
    """Load activations, phase labels, synergy data, and effort metrics for one session."""
    result = {}
    act_path = os.path.join(session_dir, 'activations.csv')
    if not os.path.exists(act_path):
        return None
    result['activations'] = pd.read_csv(act_path)
    result['phase_labels'] = pd.read_csv(os.path.join(session_dir, 'phase_labels.csv'))

    syn_w_path = os.path.join(session_dir, 'synergy_weights.csv')
    if os.path.exists(syn_w_path):
        result['synergy_weights'] = pd.read_csv(syn_w_path, index_col=0)
    else:
        result['synergy_weights'] = None

    syn_c_path = os.path.join(session_dir, 'synergy_coefficients.csv')
    if os.path.exists(syn_c_path):
        result['synergy_coefficients'] = pd.read_csv(syn_c_path)
    else:
        result['synergy_coefficients'] = None

    metrics_path = os.path.join(session_dir, 'effort_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            result['effort_metrics'] = json.load(f)
    else:
        result['effort_metrics'] = None

    return result


def get_phase_mask(phase_df, phase_name):
    """Return boolean mask for a specific phase."""
    return phase_df['phase'].values == phase_name


def check_dominant_muscles(act_df, phase_df, phase_name, expected_muscles, top_k=5):
    """Check if the top-k activated muscles during a phase match expectations."""
    mask = get_phase_mask(phase_df, phase_name)
    if not np.any(mask):
        return 'SKIP', [], f'No {phase_name} phase frames found'

    muscle_cols = [c for c in act_df.columns if c != 'time']
    phase_act = act_df.loc[mask, muscle_cols]
    mean_act = phase_act.mean().sort_values(ascending=False)
    top_muscles = list(mean_act.head(top_k).index)

    overlap = set(top_muscles) & set(expected_muscles)
    overlap_pct = len(overlap) / min(len(expected_muscles), top_k) * 100

    if overlap_pct >= 60:
        status = 'PASS'
    elif overlap_pct >= 40:
        status = 'WARN'
    else:
        status = 'FAIL'

    detail = (f'Top {top_k}: {top_muscles}, '
              f'Expected: {expected_muscles}, '
              f'Overlap: {len(overlap)}/{min(len(expected_muscles), top_k)} ({overlap_pct:.0f}%)')
    return status, top_muscles, detail


def check_cci_range(effort_metrics, fma_score):
    """Check if CCI falls in the expected range for the FMA level."""
    if effort_metrics is None:
        return 'SKIP', 'No effort_metrics.json found'

    cci = effort_metrics.get('CCI', None)
    if cci is None:
        return 'SKIP', 'CCI not computed'

    if fma_score >= 50:
        lo, hi = LITERATURE['cci_healthy_range']
        label = 'healthy'
    else:
        lo, hi = LITERATURE['cci_impaired_range']
        label = 'impaired'

    if lo <= cci <= hi:
        status = 'PASS'
    elif abs(cci - lo) < 0.1 or abs(cci - hi) < 0.1:
        status = 'WARN'
    else:
        status = 'FAIL'

    return status, f'CCI={cci:.3f}, expected [{lo:.2f}, {hi:.2f}] for {label} (FMA {fma_score})'


def check_synergy_vaf(session_dir, fma_score):
    """Check if 4 synergies explain >90% VAF for healthy subjects."""
    # Look for VAF in stdout logs or recompute from synergy data
    syn_w = os.path.join(session_dir, 'synergy_weights.csv')
    act_path = os.path.join(session_dir, 'activations.csv')
    if not os.path.exists(syn_w) or not os.path.exists(act_path):
        return 'SKIP', 'Synergy data not found'

    # Recompute VAF from synergy reconstruction
    from sklearn.decomposition import NMF
    act_df = pd.read_csv(act_path)
    muscle_cols = [c for c in act_df.columns if c != 'time']
    act_data = np.clip(act_df[muscle_cols].values, 0, None)

    if act_data.max() < 1e-10:
        return 'SKIP', 'All activations near zero'

    nmf = NMF(n_components=4, init='nndsvda', max_iter=500, random_state=42)
    coeffs = nmf.fit_transform(act_data)
    recon = coeffs @ nmf.components_

    ss_total = np.sum(act_data ** 2)
    ss_resid = np.sum((act_data - recon) ** 2)
    vaf = 1 - (ss_resid / ss_total) if ss_total > 0 else 0.0

    threshold = LITERATURE['healthy_vaf_4syn']
    if fma_score >= 50:
        if vaf >= threshold:
            status = 'PASS'
        elif vaf >= threshold - 0.05:
            status = 'WARN'
        else:
            status = 'FAIL'
        detail = f'VAF(4 syn)={vaf:.3f}, threshold={threshold:.2f} for healthy'
    else:
        # Impaired: expect lower VAF (merged synergies)
        if vaf < threshold:
            status = 'PASS'
            detail = f'VAF(4 syn)={vaf:.3f} < {threshold:.2f} (expected for impaired: synergy merging)'
        else:
            status = 'WARN'
            detail = f'VAF(4 syn)={vaf:.3f} >= {threshold:.2f} (impaired should show synergy merging)'

    return status, detail


def validate_session(session_dir, fma_score):
    """Run all validation checks on one session. Returns list of (check, status, detail)."""
    data = load_session(session_dir)
    if data is None:
        return [('load', 'FAIL', f'Could not load data from {session_dir}')]

    results = []
    act_df = data['activations']
    phase_df = data['phase_labels']
    top_k = LITERATURE['top_k_muscles']

    # Check 1: Reach-phase dominant muscles
    status, _, detail = check_dominant_muscles(
        act_df, phase_df, 'Pick', LITERATURE['reach_dominant'], top_k)
    results.append(('Reach dominant muscles', status, detail))

    # Check 2: Drink-phase dominant muscles
    status, _, detail = check_dominant_muscles(
        act_df, phase_df, 'Drink', LITERATURE['drink_dominant'], top_k)
    results.append(('Drink dominant muscles', status, detail))

    # Check 3: Place-phase dominant muscles
    status, _, detail = check_dominant_muscles(
        act_df, phase_df, 'Place', LITERATURE['place_dominant'], top_k)
    results.append(('Place dominant muscles', status, detail))

    # Check 4: CCI range
    status, detail = check_cci_range(data['effort_metrics'], fma_score)
    results.append(('CCI in published range', status, detail))

    # Check 5: Synergy VAF
    status, detail = check_synergy_vaf(session_dir, fma_score)
    results.append(('Synergy VAF (4 components)', status, detail))

    return results


def _parse_fma_dir(name):
    """Parse FMA score from directory name. Handles FMA_50, FMA_18_s03, etc."""
    m = re.match(r'FMA_(\d+)(?:_s\d+)?$', name)
    if m:
        return int(m.group(1))
    return None


def generate_report(id_base_dir, output_dir=None):
    """Run validation on all FMA sessions and generate a Markdown report.

    Handles multi-sample directories (FMA_{score}_s{idx}) by grouping
    samples per FMA level and reporting aggregate pass rates.
    """
    if output_dir is None:
        output_dir = os.path.join(id_base_dir, '..', 'plots')
    os.makedirs(output_dir, exist_ok=True)

    # Discover FMA sessions (supports FMA_50, FMA_18_s00, FMA_18_s01, etc.)
    sessions = []
    for name in sorted(os.listdir(id_base_dir)):
        if name.startswith('FMA_') and os.path.isdir(os.path.join(id_base_dir, name)):
            fma = _parse_fma_dir(name)
            if fma is not None:
                sessions.append((fma, os.path.join(id_base_dir, name)))

    # Also check for original patient sessions (like S5_12_1)
    for name in sorted(os.listdir(id_base_dir)):
        d = os.path.join(id_base_dir, name)
        if not name.startswith('FMA_') and os.path.isdir(d):
            act_path = os.path.join(d, 'activations.csv')
            if os.path.exists(act_path):
                fma = 66 if not name.startswith('S') else 30  # approximate
                sessions.append((fma, d))

    if not sessions:
        print(f'No ID sessions found in {id_base_dir}')
        return

    lines = ['# EMG Validation Report\n']
    lines.append(f'Base directory: `{id_base_dir}`\n')
    lines.append(f'Sessions validated: {len(sessions)}\n')

    total_pass = total_warn = total_fail = total_skip = 0

    # Group sessions by FMA score for aggregated reporting
    fma_groups = defaultdict(list)
    for fma, session_dir in sessions:
        fma_groups[fma].append(session_dir)

    # Per-session detail
    for fma, session_dir in sessions:
        session_name = os.path.basename(session_dir)
        lines.append(f'\n## {session_name} (FMA ~ {fma})\n')
        lines.append('| Check | Status | Detail |')
        lines.append('|-------|--------|--------|')

        results = validate_session(session_dir, fma)
        for check, status, detail in results:
            icon = {'PASS': 'PASS', 'WARN': 'WARN', 'FAIL': 'FAIL', 'SKIP': 'SKIP'}[status]
            lines.append(f'| {check} | **{icon}** | {detail} |')
            if status == 'PASS': total_pass += 1
            elif status == 'WARN': total_warn += 1
            elif status == 'FAIL': total_fail += 1
            else: total_skip += 1

    # Aggregated summary per FMA level
    lines.append(f'\n---\n')
    lines.append(f'## Aggregated Results by FMA Level\n')
    lines.append('| FMA | N samples | Check | PASS | WARN | FAIL | SKIP | Pass Rate |')
    lines.append('|-----|-----------|-------|------|------|------|------|-----------|')

    check_names = ['Reach dominant muscles', 'Drink dominant muscles',
                   'Place dominant muscles', 'CCI in published range',
                   'Synergy VAF (4 components)']

    for fma in sorted(fma_groups.keys()):
        dirs = fma_groups[fma]
        n_samples = len(dirs)

        for check_name in check_names:
            counts = {'PASS': 0, 'WARN': 0, 'FAIL': 0, 'SKIP': 0}
            for session_dir in dirs:
                results = validate_session(session_dir, fma)
                for check, status, detail in results:
                    if check == check_name:
                        counts[status] += 1

            total_valid = counts['PASS'] + counts['WARN'] + counts['FAIL']
            pass_rate = f"{counts['PASS']}/{total_valid}" if total_valid > 0 else 'N/A'
            lines.append(
                f"| {fma} | {n_samples} | {check_name} | {counts['PASS']} "
                f"| {counts['WARN']} | {counts['FAIL']} | {counts['SKIP']} | {pass_rate} |"
            )

    lines.append(f'\n---\n')
    lines.append(f'## Overall Summary\n')
    lines.append(f'- **PASS**: {total_pass}')
    lines.append(f'- **WARN**: {total_warn}')
    lines.append(f'- **FAIL**: {total_fail}')
    lines.append(f'- **SKIP**: {total_skip}')
    lines.append(f'\n### References\n')
    lines.append('- Alt Murphy M et al. (2011). Kinematic analysis of the upper extremity in the drinking task.')
    lines.append('- Roh J et al. (2013). Alterations in upper limb muscle synergy structure in chronic stroke survivors.')

    report_text = '\n'.join(lines)
    report_path = os.path.join(output_dir, 'emg_validation_report.md')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f'\nEMG Validation Report saved to: {report_path}')
    print(f'  PASS: {total_pass} | WARN: {total_warn} | FAIL: {total_fail} | SKIP: {total_skip}')

    return report_text


if __name__ == '__main__':
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # Default: check generated ID outputs
        base_dir = get_path("output_generated") + "/id"
        if not os.path.isdir(base_dir):
            # Fallback to originals
            base_dir = get_path("output_originals_id")

    generate_report(base_dir)
