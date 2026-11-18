"""
Experiment 3 (Full Cohort): SC Damping Hierarchy Validation

Compares IK reconstruction error and SC_r3 excursion between:
  A. No SC hierarchy  (SC=0.05) — fresh IK run
  B. SC hierarchy     (SC=0.20) — reads from existing output/neutral_start/

Runs across all available trials (24 stroke + 53 healthy), parallelised.

Output:
  output/experiments/exp3_sc_damping_full/
    results.csv
    summary.txt
    figure.png   → copied to paper_publication_assets/figures_and_tables/damping_comparison.png
"""
import os, sys, subprocess, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.config import get_path, get_project_root, get

ROOT      = get_project_root()
PYTHON    = get("conda_python", default=sys.executable)
MODEL_IK  = get_path("mujoco_arm_model").replace("myoarm.xml", "myoarm_trunk.xml")
SCRIPT_IK = os.path.join(ROOT, "src", "inverse_kinematics", "ik_clean.py")
TRC_BASE  = os.path.join(ROOT, "output", "neutral_start")
MOT_BASE  = os.path.join(ROOT, "output", "neutral_start")
OUT_DIR   = os.path.join(ROOT, "output", "experiments", "exp3_sc_damping_full")
PAPER_FIG = os.path.join(ROOT, "paper_publication_assets", "figures_and_tables",
                         "damping_comparison.png")
EXCLUDE_HEALTHY = {'01_12_1', '01_12_2', '01_12_3', '15_12_1'}
N_WORKERS = 6


def all_trials():
    trials = []
    for group in ['stroke', 'healthy']:
        trc_dir = os.path.join(TRC_BASE, group, 'trc')
        if not os.path.isdir(trc_dir):
            continue
        for fn in sorted(os.listdir(trc_dir)):
            if not fn.endswith('.trc'):
                continue
            stem = fn[:-4]
            if group == 'healthy' and stem in EXCLUDE_HEALTHY:
                continue
            trials.append((group, stem))
    return trials


def load_mot_col(mot_path, col):
    try:
        with open(mot_path) as f:
            lines = f.readlines()
        hend = next(i for i, l in enumerate(lines) if 'endheader' in l)
        cols = lines[hend + 1].strip().split('\t')
        arr  = np.array([l.strip().split('\t') for l in lines[hend + 2:] if l.strip()],
                        dtype=float)
        if col in cols:
            return arr[:, cols.index(col)]
    except Exception:
        pass
    return None


def run_sc05(args):
    group, trial = args
    trc_path = os.path.join(TRC_BASE, group, 'trc', f'{trial}.trc')
    mot_dir  = os.path.join(OUT_DIR, 'sc_05', group, 'mot')
    os.makedirs(mot_dir, exist_ok=True)
    mot_path = os.path.join(mot_dir, f'{trial}.mot')

    if os.path.exists(mot_path):
        sc_r3 = load_mot_col(mot_path, 'sternoclavicular_r3')
        sc_r3_peak = float(np.max(np.abs(sc_r3))) * 57.2958 if sc_r3 is not None else np.nan
        return group, trial, np.nan, sc_r3_peak  # IK error not cached; will be None

    env = os.environ.copy()
    env['IK_INTERACTIVE_ALIGN'] = 'False'
    cmd = [PYTHON, SCRIPT_IK, MODEL_IK, trc_path, mot_path, '0', '1.0', '0.05']
    t0  = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    elapsed = time.time() - t0

    ik_err = np.nan
    for line in proc.stdout.splitlines():
        if 'FINAL_MEAN_ERROR:' in line:
            try:
                ik_err = float(line.split(':')[1].strip())
            except Exception:
                pass

    sc_r3_peak = np.nan
    if os.path.exists(mot_path):
        sc_r3 = load_mot_col(mot_path, 'sternoclavicular_r3')
        if sc_r3 is not None:
            sc_r3_peak = float(np.max(np.abs(sc_r3))) * 57.2958

    print(f'  [SC=0.05] {trial} ({group}) {elapsed:.0f}s  '
          f'IK={ik_err:.2f}mm  SC_r3={sc_r3_peak:.1f}°', flush=True)
    return group, trial, ik_err, sc_r3_peak


def read_sc20_mot(group, trial):
    mot_path = os.path.join(MOT_BASE, group, 'mot', f'{trial}.mot')
    if not os.path.exists(mot_path):
        return np.nan, np.nan
    # Re-run IK to get error, OR just measure SC_r3 from existing MOT.
    # We only report SC_r3 for the SC=0.20 config; IK error for SC=0.20
    # comes from re-running with SC=0.20 in a cached subdir.
    sc_r3 = load_mot_col(mot_path, 'sternoclavicular_r3')
    sc_r3_peak = float(np.max(np.abs(sc_r3))) * 57.2958 if sc_r3 is not None else np.nan
    return np.nan, sc_r3_peak  # IK error not available without re-run


def run_sc20(args):
    group, trial = args
    trc_path = os.path.join(TRC_BASE, group, 'trc', f'{trial}.trc')
    mot_dir  = os.path.join(OUT_DIR, 'sc_20', group, 'mot')
    os.makedirs(mot_dir, exist_ok=True)
    mot_path = os.path.join(mot_dir, f'{trial}.mot')

    if os.path.exists(mot_path):
        sc_r3 = load_mot_col(mot_path, 'sternoclavicular_r3')
        sc_r3_peak = float(np.max(np.abs(sc_r3))) * 57.2958 if sc_r3 is not None else np.nan
        return group, trial, np.nan, sc_r3_peak

    env = os.environ.copy()
    env['IK_INTERACTIVE_ALIGN'] = 'False'
    cmd = [PYTHON, SCRIPT_IK, MODEL_IK, trc_path, mot_path, '0', '1.0', '0.20']
    t0  = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    elapsed = time.time() - t0

    ik_err = np.nan
    for line in proc.stdout.splitlines():
        if 'FINAL_MEAN_ERROR:' in line:
            try:
                ik_err = float(line.split(':')[1].strip())
            except Exception:
                pass

    sc_r3_peak = np.nan
    if os.path.exists(mot_path):
        sc_r3 = load_mot_col(mot_path, 'sternoclavicular_r3')
        if sc_r3 is not None:
            sc_r3_peak = float(np.max(np.abs(sc_r3))) * 57.2958

    print(f'  [SC=0.20] {trial} ({group}) {elapsed:.0f}s  '
          f'IK={ik_err:.2f}mm  SC_r3={sc_r3_peak:.1f}°', flush=True)
    return group, trial, ik_err, sc_r3_peak


def run_parallel(fn, trials, n_workers):
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(fn, t): t for t in trials}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as e:
                t = futs[fut]
                print(f'  ERROR {t}: {e}', flush=True)
    return results


def make_figure(df, out_path):
    plt.rcParams.update({
        'font.family':     'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size':        9,
        'axes.spines.top':  False,
        'axes.spines.right': False,
        'axes.linewidth':   0.8,
    })

    STROKE_COL  = '#d62728'
    HEALTHY_COL = '#4472c4'
    SC05_ALPHA  = 0.45
    SC20_ALPHA  = 0.90

    configs = ['No SC Hierarchy\n(SC = 0.05)', 'SC Hierarchy\n(SC = 0.20)']
    cfg_keys = ['SC=0.05', 'SC=0.20']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), facecolor='white')

    for ai, (metric, ylabel, title) in enumerate([
        ('ik_error_mm',    'IK reconstruction error (mm)',   'IK Reconstruction Error'),
        ('sc_r3_peak_deg', 'Peak SC_r3 excursion (°)',       'Spurious SC Excursion'),
    ]):
        ax = axes[ai]
        ax.set_facecolor('white')
        ax.yaxis.grid(True, linestyle=':', linewidth=0.6, alpha=0.45, zorder=0)
        ax.set_axisbelow(True)

        x = np.arange(len(configs))
        w = 0.30
        rng = np.random.default_rng(0)

        for gi, group in enumerate(['stroke', 'healthy']):
            col    = STROKE_COL if group == 'stroke' else HEALTHY_COL
            alphas = [SC05_ALPHA, SC20_ALPHA]
            offset = (gi - 0.5) * (w + 0.06)
            for ci, cfg_key in enumerate(cfg_keys):
                sub = df[(df['sc_config'] == cfg_key) & (df['group'] == group)][metric].dropna()
                if len(sub) == 0:
                    continue
                mn  = sub.mean()
                sem = sub.std() / np.sqrt(len(sub))
                bar = ax.bar(x[ci] + offset, mn, w, yerr=sem, capsize=3,
                             color=col, alpha=alphas[ci], linewidth=0,
                             error_kw=dict(elinewidth=0.9, ecolor='#555', capthick=0.9),
                             zorder=3)

        # swarm dots (per-subject)
        for gi, group in enumerate(['stroke', 'healthy']):
            col    = STROKE_COL if group == 'stroke' else HEALTHY_COL
            offset = (gi - 0.5) * (w + 0.06)
            for ci, cfg_key in enumerate(cfg_keys):
                sub = df[(df['sc_config'] == cfg_key) & (df['group'] == group)][metric].dropna()
                jitter = rng.uniform(-0.08, 0.08, len(sub))
                ax.scatter(x[ci] + offset + jitter, sub.values,
                           color=col, alpha=0.35, s=14, linewidths=0, zorder=4)

        # significance bracket between configs (pooled)
        a05 = df[df['sc_config'] == 'SC=0.05'][metric].dropna()
        a20 = df[df['sc_config'] == 'SC=0.20'][metric].dropna()
        n   = min(len(a05), len(a20))
        if n > 2:
            _, p = stats.wilcoxon(a05.values[:n], a20.values[:n])
            p_str = 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'
            # Use 95th-percentile cap so outliers don't collapse the visible bars
            p95   = float(np.percentile(df[metric].dropna(), 95))
            ymax  = max(p95 * 1.5, df[metric].dropna().mean() * 4)
            bh    = ymax * 0.88
            ax.plot([0, 0, 1, 1], [bh, bh*1.02, bh*1.02, bh],
                    color='#333', lw=0.9)
            ax.text(0.5, bh * 1.03, p_str, ha='center', va='bottom',
                    fontsize=7.5, color='#111', fontweight='bold')
            # show clipped outliers as triangles above axis
            clip_val = ymax * 1.05
            for gi, group in enumerate(['stroke', 'healthy']):
                col    = STROKE_COL if group == 'stroke' else HEALTHY_COL
                offset = (gi - 0.5) * (w + 0.06)
                for ci, cfg_key in enumerate(cfg_keys):
                    sub = df[(df['sc_config'] == cfg_key) &
                             (df['group'] == group)][metric].dropna()
                    n_clip = (sub > ymax).sum()
                    if n_clip > 0:
                        ax.scatter([x[ci] + offset], [clip_val * 0.97],
                                   marker='^', color=col, s=30, zorder=5,
                                   clip_on=False)
                        ax.text(x[ci] + offset, clip_val * 0.96,
                                f'+{n_clip}', ha='center', va='top',
                                fontsize=6.5, color=col)
            ax.set_ylim(bottom=0, top=clip_val)

        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5, color='#444', labelpad=3)
        ax.set_title(title, fontsize=9.5, fontweight='semibold', pad=5, color='#1a1a1a')
        ax.spines['left'].set_color('#888')
        ax.spines['bottom'].set_color('#888')
        ax.tick_params(colors='#555')
        ax.set_xlim(-0.6, 1.6)

    # legend
    handles = [
        mpatches.Patch(color=STROKE_COL,  label=f'Stroke  (n = {len(df[df.group=="stroke"].trial.unique())})'),
        mpatches.Patch(color=HEALTHY_COL, label=f'Healthy (n = {len(df[df.group=="healthy"].trial.unique())})'),
    ]
    fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.98, 0.06),
               fontsize=8.5, frameon=True, framealpha=0.95,
               edgecolor='#ccc', fancybox=False)

    plt.tight_layout(pad=1.2, w_pad=2.5)
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Figure saved: {out_path}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    trials = all_trials()
    print(f'Full cohort: {len(trials)} trials  ({N_WORKERS} workers)\n')

    print('=== Running SC=0.05 (no hierarchy) ===')
    res05 = run_parallel(run_sc05, trials, N_WORKERS)

    print('\n=== Running SC=0.20 (hierarchy) ===')
    res20 = run_parallel(run_sc20, trials, N_WORKERS)

    rows = []
    for group, trial, ik_err, sc_r3 in res05:
        rows.append({'sc_config': 'SC=0.05', 'group': group, 'trial': trial,
                     'ik_error_mm': ik_err, 'sc_r3_peak_deg': sc_r3})
    for group, trial, ik_err, sc_r3 in res20:
        rows.append({'sc_config': 'SC=0.20', 'group': group, 'trial': trial,
                     'ik_error_mm': ik_err, 'sc_r3_peak_deg': sc_r3})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'results.csv'), index=False)

    print('\n--- Summary ---')
    for cfg in ['SC=0.05', 'SC=0.20']:
        sub = df[df.sc_config == cfg]
        ik  = sub.ik_error_mm.dropna()
        sc3 = sub.sc_r3_peak_deg.dropna()
        print(f'{cfg}: IK={ik.mean():.2f}±{ik.std():.2f}mm  '
              f'SC_r3={sc3.mean():.2f}±{sc3.std():.2f}°  n={len(sc3)}')

    a05 = df[df.sc_config == 'SC=0.05'].sc_r3_peak_deg.dropna()
    a20 = df[df.sc_config == 'SC=0.20'].sc_r3_peak_deg.dropna()
    n   = min(len(a05), len(a20))
    if n > 2:
        _, p = stats.wilcoxon(a05.values[:n], a20.values[:n])
        print(f'Wilcoxon SC_r3: p={p:.4f}  n={n}')

    ik05 = df[df.sc_config == 'SC=0.05'].ik_error_mm.dropna()
    ik20 = df[df.sc_config == 'SC=0.20'].ik_error_mm.dropna()
    n2   = min(len(ik05), len(ik20))
    if n2 > 2:
        _, p2 = stats.wilcoxon(ik05.values[:n2], ik20.values[:n2])
        print(f'Wilcoxon IK error: p={p2:.4f}  n={n2}')

    with open(os.path.join(OUT_DIR, 'summary.txt'), 'w') as f:
        f.write('SC Damping Hierarchy Validation — Full Cohort\n')
        f.write('=' * 55 + '\n\n')
        f.write(f'Trials: {len(trials)} ({sum(1 for g,_ in trials if g=="stroke")} stroke, '
                f'{sum(1 for g,_ in trials if g=="healthy")} healthy)\n\n')
        for cfg in ['SC=0.05', 'SC=0.20']:
            sub = df[df.sc_config == cfg]
            ik  = sub.ik_error_mm.dropna()
            sc3 = sub.sc_r3_peak_deg.dropna()
            f.write(f'{cfg}:\n')
            f.write(f'  IK error:    {ik.mean():.2f} ± {ik.std():.2f} mm  (n={len(ik)})\n')
            f.write(f'  SC_r3 peak:  {sc3.mean():.2f} ± {sc3.std():.2f} °  (n={len(sc3)})\n\n')
        if n > 2:
            f.write(f'Wilcoxon SC_r3 (SC=0.05 vs SC=0.20): p={p:.4f}  n={n}\n')
        if n2 > 2:
            f.write(f'Wilcoxon IK error: p={p2:.4f}  n={n2}\n')

    fig_path = os.path.join(OUT_DIR, 'figure.png')
    make_figure(df, fig_path)

    import shutil
    shutil.copy(fig_path, PAPER_FIG)
    print(f'Copied to: {PAPER_FIG}')
    print(f'\nAll results: {OUT_DIR}/')


if __name__ == '__main__':
    main()
